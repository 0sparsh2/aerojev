"""Grow each domain's wording with an open model through NVIDIA NIM (plan, section 6.1, "phrase banks").

  python -m general.gen_banks --domains all --model nvidia/nemotron-3.5-lightning-30b-a3b --n 12 --out banks/
  python -m general.build --banks banks/ --records 2000 --out data/   # build with the generated wording

For every fact value and register, every domain's situations and distractors, and every question's paraphrases:
  1. generate  ask for n new items in the style of the spec's own (shown as examples), as a JSON list
  2. check     rule checks: only the allowed {slots}, {v} exactly once where a value goes in, under 25 words,
               not a duplicate of an existing or another new item, formats with sample values
  3. judge     a second call labels every candidate (which value a sentence states, or whether it states the fact
               at all); candidates the judge disagrees with are dropped
Writes <out>/<domain>.json ({bank key: [items]}) and <out>/report.json (counts per step). Banks only add wording:
labels still come from the facts, so a bad phrase can make a record unclear but never mislabels it silently, and the
judge step exists to keep even that rare.

The trial NIM terms allow internal evaluation only, so banks made this way are for the pilot (docs/
GENERAL_DECISION_PLAN.md). Nothing confidential goes into the prompts: they contain only the spec's fictional text.
"""

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from .domains import DOMAINS
from .nim import NIMClient

SLOT = re.compile(r"\{([^{}]*)\}")
REGISTERS = {False: "formal business English, as written in a work system or email",
             True: "informal chat shorthand, as typed quickly in a team chat (lowercase is fine, no full stop needed)"}


def has_registers(f):
    return isinstance(f.phrases, dict) and bool({"formal", "informal"} & set(f.phrases))


def fact_values(f):
    """The bank values of a fact: True / False, each category (per-value phrases), or "n" (one list with {v})."""
    if f.kind == "bool":
        return [True, False]
    if f.kind == "cat" and isinstance(f.phrases, dict) and not has_registers(f):
        return list(f.values)
    return ["n"]


def bank_keys(dom):
    """Every bank this domain has: (key, kind, details). Single-register facts get formal wording only."""
    out = []
    for f in dom.facts:
        for v in fact_values(f):
            for informal in ((False, True) if has_registers(f) else (False,)):
                sample = v if v != "n" else (f.values[0] if f.kind == "cat" else f.sample(random.Random(0)))
                out.append((f"fact|{f.name}|{v}|{int(informal)}", "fact",
                            dict(fact=f, value=v, informal=informal, sample=sample)))
    out.append(("situations", "situations", {}))
    out.append(("distractors", "distractors", {}))
    for q in dom.questions:
        out.append((f"q|{q.id}", "paraphrase", dict(question=q)))
    return out


def existing(dom, key, kind, d):
    if kind == "fact":
        return d["fact"].templates(d["sample"], d["informal"])
    if kind == "situations":
        return list(dom.situations)
    if kind == "distractors":
        return list(dom.distractors)
    return [d["question"].text, *d["question"].paraphrases]


NO_OUTCOME = ("State only the fact itself, as an observation. Never mention a decision, action, approval, refusal, "
              "refund, priority, escalation, assignment or what will happen next.")


def prompt(dom, kind, d, have, n):
    ex = "\n".join(f"- {t}" for t in have[:8])
    head = (f"You write text templates for synthetic records in this setting: {dom.title}. All names are fictional.\n"
            f"Existing items (do not repeat or lightly reword them):\n{ex}\n")
    tail = f"Write {n} new items, varied in structure and vocabulary. Output a JSON list of strings only."
    if kind == "fact":
        f, v = d["fact"], d["value"]
        reg = REGISTERS[d["informal"]]
        if v == "n":
            what = (f"Each item states {f.about}. The value is filled in at the slot {{v}} (for example "
                    f"\"{f.text(d['sample'])}\"); use {{v}} exactly once and no other braces.")
        else:
            val = ("YES" if v else "NO") if f.kind == "bool" else f"\"{v}\""
            what = (f"The fact is: {f.about}. Every existing item states the same value of it ({val}). Each new item "
                    f"must mean exactly what the existing items mean, in different words. Do not use any braces.")
        return head + f"{what} {NO_OUTCOME} Register: {reg}. One sentence, at most 20 words.\n" + tail
    if kind == "situations":
        slots = ", ".join("{" + s + "}" for s in sorted({m for t in have for m in SLOT.findall(t)}))
        return head + (f"Each item is the opening sentence of a record saying what the case is about. {NO_OUTCOME} "
                       f"You may use these slots: {slots}; no other braces. One sentence, at most 20 words.\n" + tail)
    if kind == "distractors":
        return head + ("Each item is an irrelevant aside that could appear in such a record (office news, reminders, "
                       "small talk) and says nothing about the case itself. No braces. One sentence.\n" + tail)
    return head + ("Each item asks exactly the same question as the first existing item, with the same meaning and "
                   "scope, in different words. Do not add answer options or rules. No braces.\n" + tail)


def parse_list(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    try:
        out = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in out if isinstance(s, str) and s.strip()]


def allowed_slots(dom, kind, d):
    if kind == "fact":
        return {"v"} if d["value"] == "n" else set()
    if kind == "situations":
        return set(dom.entities) | {f.name for f in dom.facts}
    return set()


def rule_check(dom, kind, d, cand, seen):
    slots = SLOT.findall(cand)
    if set(slots) - allowed_slots(dom, kind, d):
        return "slots"
    if kind == "fact" and d["value"] == "n" and slots.count("v") != 1:
        return "no {v}"
    if len(cand.split()) > 25:
        return "too long"
    norm = re.sub(r"\W+", " ", cand.lower()).strip()
    if norm in seen:
        return "duplicate"
    seen.add(norm)
    return None


def thinking_off(model):
    """Reasoning models answer in "content" only after thinking; for short structured lists, switch thinking off."""
    return {"chat_template_kwargs": {"enable_thinking": False}} if "nemotron" in model else None


def judge(nim, dom, kind, d, cands):
    """-> list of bools: keep each candidate? One call labels every candidate; a candidate is kept only if its label
    is the expected one. If the labels cannot be aligned with the candidates, nothing is kept."""
    if not cands:
        return []
    shown = cands
    if kind == "fact":
        f, v = d["fact"], d["value"]
        shown = [c.replace("{v}", f.text(d["sample"])) for c in cands]
        if v == "n":
            q = (f"For each sentence, answer \"yes\" if it states {f.about} (the value in it is that quantity) and "
                 f"nothing else, \"no\" otherwise.")
            want = "yes"
        elif f.kind == "bool":
            q = (f"Fact: {f.about}. For each sentence, answer \"true\" if the sentence says yes to this fact, "
                 f"\"false\" if it says no, or \"unclear\".")
            want = "true" if v else "false"
        else:
            q = (f"For each sentence, say which value of \"{f.about}\" it states: one of "
                 f"{[str(x) for x in f.values] + ['unclear']}.")
            want = str(v).lower()
        q += " If a sentence states a decision, action or outcome rather than only the fact, answer \"unclear\"."
    elif kind == "situations":
        q = ("For each sentence, answer \"ok\" if it only says what the case is about, or \"outcome\" if it states "
             "or implies a decision, approval, refusal, outcome, priority or next action.")
        want = "ok"
    elif kind == "distractors":
        q = (f"Setting: {dom.title}. For each sentence, answer \"ok\" if it is an irrelevant aside that says nothing "
             f"about any case, or \"relevant\" if it says anything about a case.")
        want = "ok"
    else:
        q = (f"For each question, answer \"same\" if it asks exactly the same thing as \"{d['question'].text}\" "
             f"(same meaning and scope), or \"different\".")
        want = "same"
    listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(shown))
    out = parse_list(nim.chat([{"role": "user", "content": f"{q}\n\n{listing}\n\nOutput a JSON list with one label per "
                                                           f"item, in order, and nothing else."}],
                              temperature=0.0, max_tokens=800, extra=thinking_off(nim.model)))
    if len(out) != len(cands):
        return [False] * len(cands)
    return [str(o).strip().lower() == want for o in out]


def grow(nim, judge_nim, dom, key, kind, d, n):
    have = existing(dom, key, kind, d)
    seen = {re.sub(r"\W+", " ", t.lower()).strip() for t in have}
    raw = parse_list(nim.chat([{"role": "user", "content": prompt(dom, kind, d, have, n)}], temperature=0.9,
                              max_tokens=1200, seed=0, extra=thinking_off(nim.model)))
    reasons, cands = {}, []
    for c in raw:
        r = rule_check(dom, kind, d, c, seen)
        if r:
            reasons[r] = reasons.get(r, 0) + 1
        else:
            cands.append(c)
    keep = judge(judge_nim, dom, kind, d, cands)
    kept = [c for c, k in zip(cands, keep) if k]
    return key, kept, {"asked": n, "returned": len(raw), "rule_rejected": reasons,
                       "judge_rejected": len(cands) - len(kept), "kept": len(kept)}


def apply_banks(dom, banks):
    """Add generated wording to a domain in place (render.py and spec.py then use it like the spec's own)."""
    for f in dom.facts:
        ph, reg = f.phrases, has_registers(f)
        for informal in (False, True):
            for v in fact_values(f):
                extra = banks.get(f"fact|{f.name}|{v}|{int(informal)}", [])
                if not extra:
                    continue
                if reg:
                    side = ph["informal" if informal else "formal"]
                    (side[v] if f.kind == "bool" else side).extend(extra)
                elif isinstance(ph, dict):
                    ph[v].extend(extra)  # per-value phrases (one register)
                else:
                    ph.extend(extra)
    dom.situations.extend(banks.get("situations", []))
    dom.distractors.extend(banks.get("distractors", []))
    for q in dom.questions:
        q.paraphrases = tuple(q.paraphrases) + tuple(banks.get(f"q|{q.id}", []))


def load_banks(dirpath):
    """Apply every <domain>.json in dirpath to its registered domain."""
    for name, dom in DOMAINS.items():
        p = os.path.join(dirpath, f"{name}.json")
        if os.path.exists(p):
            apply_banks(dom, json.load(open(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="all")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b",
                    help="generator: OpenMDW-1.1 (commercial use allowed); check any other model's licence first")
    ap.add_argument("--judge-model", help="model for the judge step (default: the generator)")
    ap.add_argument("--n", type=int, default=12, help="new items asked per bank")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="NIM response cache (default <out>/nim_cache.jsonl)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = args.cache or os.path.join(args.out, "nim_cache.jsonl")
    nim = NIMClient(args.model, cache=cache)
    judge_nim = NIMClient(args.judge_model, cache=cache) if args.judge_model else nim
    names = sorted(DOMAINS) if args.domains == "all" else args.domains.split(",")
    report = {"model": args.model, "domains": {}}
    for name in names:
        dom = DOMAINS[name]
        jobs = bank_keys(dom)
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda j: grow(nim, judge_nim, dom, j[0], j[1], j[2], args.n), jobs))
        banks = {k: kept for k, kept, _ in res if kept}
        json.dump(banks, open(os.path.join(args.out, f"{name}.json"), "w"), indent=1, ensure_ascii=False)
        report["domains"][name] = {k: s for k, _, s in res}
        tot = sum(s["kept"] for _, _, s in res)
        print(f"{name:16s} {len(jobs)} banks, {tot} new items kept "
              f"({sum(s['returned'] for _, _, s in res)} returned)", flush=True)
    report["calls"] = nim.calls + (judge_nim.calls if judge_nim is not nim else 0)
    report["cache_hits"] = nim.hits + (judge_nim.hits if judge_nim is not nim else 0)
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
