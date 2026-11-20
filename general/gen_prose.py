"""Records written whole by an open model from a fact sheet, then fact-checked (plan, section 6.1, "prose").

  python -m general.gen_prose --domains all --records train=200,val=25,test=25 --out prose/
  python -m general.build --banks banks/ --prose prose/ --records 2000 --out data/

Templated records (general/render.py) are varied but share a skeleton. Prose records add whole documents written by
the generator, with free structure. For each record:
  1. facts   sampled exactly as for templated records (Domain.sample), so labels follow the same rules
  2. write   the generator gets a fact sheet (every fact in plain words, plus fictional names) and a document format,
             and writes a record that states every fact and no decision
  3. check   numbers: every int / float value appears verbatim. judge: a second call reads only the record and
             answers every yes / no and categorical fact, plus whether the record states a decision or outcome.
             Any mismatch, "not stated" or stated decision drops the record.
Writes <out>/<domain>.<split>.jsonl ({"facts", "state", "format"} per kept record) and <out>/report.json.
"""

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from .domains import DOMAINS
from .gen_banks import parse_list, thinking_off
from .nim import NIMClient

FORMATS = {
    "email": "an email between colleagues, with a subject line, greeting and sign-off",
    "ticket": "a ticket or case record with a few labelled fields and a free-text description",
    "chat": "a short team-chat thread of 3-6 messages, each line \"[hh:mm] Name: message\", informal",
    "report": "a short status note of one or two paragraphs",
    "call_notes": "rough notes taken during a phone call, as terse bullet points",  # test only, as in render.py
}
TEST_ONLY = {"call_notes"}


def fact_sheet(dom, facts):
    lines = [f"- {f.about}: {f.text(facts[f.name])}" for f in dom.facts]
    names = "; ".join(f"{k}: {v}" for k, v in facts["_entities"].items())
    return "\n".join(lines) + (f"\n- names to use (fictional): {names}" if names else "")


def write_prompt(dom, facts, fmt):
    return (f"Setting: {dom.title}. Write {FORMATS[fmt]}, about one case, in realistic workplace language.\n"
            f"The record must state every fact below, in its own words. Write numbers and amounts exactly as given. "
            f"Do not state any decision, recommendation, approval, refusal, priority or next action: only the facts "
            f"and the situation. 50-160 words. Output only the record.\n\nFacts:\n{fact_sheet(dom, facts)}")


def numbers_ok(dom, facts, text):
    plain = re.sub(r"[,$%]", "", text)
    return [f.name for f in dom.facts if f.kind in ("int", "float")
            and re.sub(r"[,$%]", "", f.text(facts[f.name])) not in plain]


def judge_prompt(dom, text):
    qs, keys = [], []
    for f in dom.facts:
        if f.kind == "bool":
            qs.append(f"According to the record, is it the case that {f.claim()}? Answer yes, no or not stated.")
            keys.append(f.name)
        elif f.kind == "cat":
            qs.append(f"{f.about[:1].upper()}{f.about[1:]}? Answer one of {[f.text(v) for v in f.values]} or "
                      f"not stated.")
            keys.append(f.name)
    qs.append("Does the record state a decision, recommendation, approval, refusal, priority or next action? "
              "Answer yes or no.")
    listing = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(qs))
    return (f"Read this record and answer each question from the record alone.\n\nRecord:\n{text}\n\nQuestions:\n"
            f"{listing}\n\nOutput a JSON list with one answer per question, in order, and nothing else."), keys


def one(nim, judge_nim, dom, split, i, seed):
    rng = random.Random(f"prose|{dom.name}|{split}|{seed}|{i}")
    facts = dom.sample(rng)
    fmts = [f for f in FORMATS if f not in TEST_ONLY or split == "test"]
    fmt = rng.choice(fmts)
    try:
        text = nim.chat([{"role": "user", "content": write_prompt(dom, facts, fmt)}], temperature=0.9,
                        max_tokens=600, seed=i, extra=thinking_off(nim.model)).strip()
    except RuntimeError as e:
        return None, f"write failed: {str(e)[:60]}"
    missing = numbers_ok(dom, facts, text)
    if missing:
        return None, "number missing"
    prompt, keys = judge_prompt(dom, text)
    try:
        ans = parse_list(judge_nim.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400,
                                        extra=thinking_off(judge_nim.model)))
    except RuntimeError:
        return None, "judge failed"
    if len(ans) != len(keys) + 1:
        return None, "judge unaligned"
    for k, a in zip(keys, ans):
        f, a = dom.fact(k), str(a).strip().lower()
        want = ("yes" if facts[k] else "no") if f.kind == "bool" else f.text(facts[k]).lower()
        if a != want:
            return None, "fact mismatch" if a != "not stated" else "fact not stated"
    if str(ans[-1]).strip().lower() != "no":
        return None, "states a decision"
    return {"facts": facts, "state": text, "format": fmt}, "kept"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="all")
    ap.add_argument("--records", default="train=200,val=25,test=25", help="records attempted per domain and split")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    ap.add_argument("--judge-model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = args.cache or os.path.join(args.out, "nim_cache.jsonl")
    nim = NIMClient(args.model, cache=cache)
    judge_nim = NIMClient(args.judge_model, cache=cache) if args.judge_model else nim
    plan = {s: int(n) for s, n in (x.split("=") for x in args.records.split(","))}
    names = sorted(DOMAINS) if args.domains == "all" else args.domains.split(",")
    report = {"model": args.model, "judge": args.judge_model or args.model, "domains": {}}
    for name in names:
        dom = DOMAINS[name]
        for split, n in plan.items():
            with ThreadPoolExecutor(args.workers) as ex:
                res = list(ex.map(lambda i: one(nim, judge_nim, dom, split, i, args.seed), range(n)))
            kept = [r for r, _ in res if r]
            with open(os.path.join(args.out, f"{name}.{split}.jsonl"), "w") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            why = {}
            for _, w in res:
                why[w] = why.get(w, 0) + 1
            report["domains"][f"{name}/{split}"] = why
            print(f"{name:16s} {split:5s} kept {len(kept)}/{n}  {why}", flush=True)
    report["calls"], report["cache_hits"] = nim.calls, nim.hits
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
