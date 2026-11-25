"""Tier C: public benchmarks that already publish Jev's answers (docs/GENERAL_DECISION_PLAN.md, section 7.1).

  python -m general.benchmarks --check                  # build every task, recompute each published score
  python -m general.benchmarks --out data/bench/tasks   # write every task as examples for the eval jobs

Sources are pinned to a commit and fetched at run time into --cache (default data/bench/, git-ignored and left out
of the Kaggle bundle). Nothing from them is committed or redistributed.

  system-one  github.com/yanng981/system-one-benchmark (MIT, Copyright (c) 2026 yanng981). Fixed samples (seed 42):
              SST-2, TREC and Banking77, 300 each; MASSIVE scenario, the same 150 utterances in 8 languages.
              Published answers: Jev 1.13, Kev-0.8B, Von 1.2, Laya (several settings), GLiNER2.5-Decide.
  elcronos    github.com/elcronos/jev-vs-open-decision-models (no licence file, so it is read, never copied). Full
              evaluation splits: emotion (2,000; "plain" and "defined" variants), tweet_topic (1,693), fin_topic
              (4,117), daily_dialog (7,740). Published answers: Jev 1.13 (typesafe/jev-1.13-20260917), PrismNLI-0.4B,
              Laya.

Every task becomes examples in the generated data's format (general/compose.py), so each system reads it through
general/contract.py. The question is rebuilt exactly as the benchmark asked it: the same instruction, option keys,
option descriptions (empty for elcronos's plain variant) and option order. Each example also carries
"refs": {system: {"pred", "conf"}}, every published system's answer on that row (pred None = the call failed).
score() computes accuracy, macro-F1 and ECE with one piece of code for every system, ours and theirs.
"""

import argparse
import json
import os
import re
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "bench")

S1 = ("yanng981/system-one-benchmark", "ed4544869a9480aa98db6572a21733803eadcbb8")
EL = ("elcronos/jev-vs-open-decision-models", "a1901bc3d520e73936de8d4326545c0cdcf742fb")

# Option descriptions exactly as system-one-benchmark's bench.py asks them.
# MIT License, Copyright (c) 2026 yanng981 (https://github.com/yanng981/system-one-benchmark).
S1_TREC = {
    "ABBR": "an abbreviation or the expansion of one",
    "ENTY": "an entity such as an object, animal, colour, product, event or term",
    "DESC": "a description, definition, reason or manner",
    "HUM": "a person, group or organisation",
    "LOC": "a location such as a city, country, mountain or other place",
    "NUM": "a number, date, amount, distance, speed or other numeric value",
}
S1_MASSIVE = {
    "alarm": "setting, checking or removing an alarm",
    "audio": "device volume or sound settings",
    "calendar": "calendar events, meetings or reminders",
    "cooking": "recipes or cooking",
    "datetime": "the current date, time or time zone",
    "email": "reading, writing or managing email",
    "general": "small talk, jokes or general requests to the assistant",
    "iot": "smart-home devices such as lights, plugs, vacuum or coffee machine",
    "lists": "to-do or shopping lists",
    "music": "music preferences or information about music",
    "news": "the news",
    "play": "playing music, radio, podcasts, audiobooks or games",
    "qa": "a factual question, definition, calculation, stock or currency",
    "recommendation": "recommendations for places, events or films",
    "social": "social media or complaints to a company",
    "takeaway": "ordering takeaway food",
    "transport": "trains, taxis, tickets or traffic",
    "weather": "the weather",
}
S1_LANGS = ["en", "zh-CN", "ja", "ko", "es", "de", "ar", "hi"]
S1_SETS = ["sst2", "trec", "banking77"] + [f"massive_{lang}" for lang in S1_LANGS]

# elcronos: the instruction strings of its frozen protocol (PROTOCOL.md section 1, PROTOCOL_ADDENDUM_v2.md section
# 2). Label order comes from the prediction file's probability columns and is checked against the gold ids.
EL_SETS = {
    "emotion": ("results/raw_predictions.parquet", "Which single primary emotion is expressed in this text?"),
    "tweet_topic": ("results/tweet_topic/raw_predictions.parquet", "Which single topic does this tweet belong to?"),
    "fin_topic": ("results/fin_topic/raw_predictions.parquet",
                  "Which single topic does this financial news tweet belong to?"),
    "daily_dialog": ("results/daily_dialog/raw_predictions.parquet",
                     "Which single emotion is expressed in this utterance?"),
}
EL_SYSTEMS = {"jev": "jev-1.13", "prismnli": "prismnli-0.4b", "laya": "laya"}
EL_NAMES = {"jev-1.13": "Jev 1.13", "prismnli-0.4b": "PrismNLI-0.4B", "laya": "Laya"}  # cross_dataset_summary.json

TASKS = [f"s1/{d}" for d in S1_SETS] + [f"el/{d}" for d in EL_SETS] + ["el/emotion-defined"]


# ---------------------------------------------------------------- fetching (pinned commits, cached)

def fetch(src, path, cache=CACHE):
    repo, sha = src
    local = os.path.join(cache, repo.replace("/", "__") + "@" + sha[:12], path)
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}", timeout=120) as r:
            data = r.read()
        with open(local + ".part", "wb") as f:
            f.write(data)
        os.replace(local + ".part", local)
    return local


def listing(src, cache=CACHE):
    """Every file path in the repository at the pinned commit (one GitHub API call, then cached)."""
    repo, sha = src
    local = os.path.join(cache, repo.replace("/", "__") + "@" + sha[:12], "_tree.json")
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1",
                                     headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            paths = [t["path"] for t in json.load(r)["tree"] if t["type"] == "blob"]
        json.dump(paths, open(local, "w"))
    return json.load(open(local))


# ---------------------------------------------------------------- examples

def example(task, i, text, instructions, options, label, refs):
    return {"id": f"{task}:{i}", "domain": task, "split": "bench", "rid": f"{task}:{i}", "state": text,
            "format": "bench", "injected": False,
            "question": {"id": task.split("/")[1], "type": "choice", "instructions": instructions,
                         "options": [{"key": k, "description": d} for k, d in options]},
            "label": label, "rule": None, "rule_in_text": False, "pair": None, "facts": {}, "refs": refs}


def s1_question(name, labels):
    """-> (instructions, [(key, description)]) as system-one-benchmark's bench.py builds it."""
    if name.startswith("massive_"):
        return "What is this request to a voice assistant about?", [(k, S1_MASSIVE[k]) for k in labels]
    if name == "sst2":
        return ("What is the sentiment of this movie review sentence?",
                [("negative", "negative sentiment"), ("positive", "positive sentiment")])
    if name == "trec":
        return "What kind of answer does this question ask for?", [(k, S1_TREC[k]) for k in labels]
    if name == "banking77":
        return "Which banking customer-support intent matches this message?", [(k, k.replace("_", " ")) for k in labels]
    raise KeyError(name)


def load_s1(name, cache=CACHE):
    data = json.load(open(fetch(S1, f"data/{name}.json", cache)))
    instructions, options = s1_question(name, data["labels"])
    systems = sorted(p[len("results/"):-len(f"_{name}.jsonl")] for p in listing(S1, cache)
                     if p.startswith("results/") and p.endswith(f"_{name}.jsonl"))
    refs = [{} for _ in data["rows"]]
    for system in systems:
        rows = [json.loads(line) for line in open(fetch(S1, f"results/{system}_{name}.jsonl", cache))]
        assert len(rows) == len(data["rows"]), (name, system)
        for i, (row, out) in enumerate(zip(data["rows"], rows)):
            assert out["text"] == row["text"] and out["gold"] == row["label"], (name, system, i)
            refs[i][system] = {"pred": out.get("pred"), "conf": out.get("conf")}
    return [example(f"s1/{name}", row.get("id", i), row["text"], instructions, options, row["label"], refs[i])
            for i, row in enumerate(data["rows"])]


def el_definitions(cache=CACHE):
    """The emotion 'defined' variant's label definitions, read from the pinned PROTOCOL.md (section 4)."""
    text = open(fetch(EL, "PROTOCOL.md", cache), encoding="utf-8").read()
    block = text.split("DEFINITION (frozen now):", 1)[1].split("```")[1]
    return dict(re.findall(r"^\s*([a-z]+):\s+(.+?)\s*$", block, flags=re.M))


def load_el(name, cache=CACHE):
    import pandas as pd
    base, variant = (name.split("-")[0], "defined") if name.endswith("-defined") else (name, "plain")
    path, instructions = EL_SETS[base]
    df = pd.read_parquet(fetch(EL, path, cache))
    if "variant" in df:
        df = df[df["variant"] == variant]
    labels = [c[len("jev_p_"):] for c in df.columns if c.startswith("jev_p_")]
    gold = dict(zip(df["gold_id"], df["gold_label"]))
    assert all(labels[i] == lab for i, lab in gold.items()), (name, "label order")
    desc = el_definitions(cache) if variant == "defined" else {}
    options = [(lab, desc.get(lab, "")) for lab in labels]
    out = []
    for r in df.sort_values("dataset_index").to_dict("records"):
        refs = {}
        for prefix, system in EL_SYSTEMS.items():
            if f"{prefix}_pred" not in r:
                continue
            failed = r.get(f"{prefix}_error") is not None or r[f"{prefix}_pred"] is None \
                or r[f"{prefix}_pred"] != r[f"{prefix}_pred"]  # NaN
            refs[system] = {"pred": None if failed else labels[int(r[f"{prefix}_pred"])],
                            "conf": None if failed else float(r[f"{prefix}_confidence"])}
        out.append(example(f"el/{name}", int(r["dataset_index"]), r["text"], instructions, options,
                           r["gold_label"], refs))
    return out


def load(task, cache=CACHE):
    source, name = task.split("/", 1)
    return load_s1(name, cache) if source == "s1" else load_el(name, cache)


# ---------------------------------------------------------------- scoring

def score(examples, preds, bins=10):
    """preds: {example id: (option key or None, confidence or None)}; a missing or None answer counts as wrong.
    -> {"n", "accuracy", "macro_f1", "ece"} (macro-F1 over the task's options; ECE on the top answer, equal-width bins)."""
    n = len(examples)
    pairs = [(e["label"], *(preds.get(e["id"]) or (None, None))) for e in examples]
    acc = sum(g == p for g, p, _ in pairs) / n
    keys = [o["key"] for o in examples[0]["question"]["options"]]
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p, _ in pairs:
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            if p is not None:
                fp[p] += 1
    f1 = [2 * tp[k] / (2 * tp[k] + fp[k] + fn[k]) if tp[k] + fp[k] + fn[k] else 0.0 for k in keys
          if tp[k] + fn[k] + fp[k]]
    rated = [(c, g == p) for g, p, c in pairs if c is not None]
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in rated if lo < c <= hi or (b == 0 and c == 0)]
        if bucket:
            ece += len(bucket) / len(rated) * abs(sum(ok for _, ok in bucket) / len(bucket)
                                                  - sum(c for c, _ in bucket) / len(bucket))
    return {"n": n, "accuracy": round(acc, 4), "macro_f1": round(sum(f1) / len(f1), 4) if f1 else 0.0,
            "ece": round(ece, 4) if rated else None}


def published(task, cache=CACHE):
    """-> {system: accuracy} as each benchmark reports it, to check our rebuild against."""
    source, name = task.split("/", 1)
    if source == "s1":
        out = {}
        for p in listing(S1, cache):
            if p.startswith("results/") and p.endswith("_summary.json"):
                s = json.load(open(fetch(S1, p, cache)))
                if name in s:
                    out[p[len("results/"):-len("_summary.json")]] = s[name]["accuracy"]
        return out
    if name.endswith("-defined"):
        return {}  # the cross-dataset summary covers the plain variant only
    rows = json.load(open(fetch(EL, "results/cross_dataset_summary.json", cache)))["models"]
    by_name = {v: k for k, v in EL_NAMES.items()}
    return {by_name[r["model"]]: r["accuracy"] for r in rows if r["dataset"] == name and r["model"] in by_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all", help="comma-separated task names (see TASKS), or all")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--check", action="store_true", help="recompute every published accuracy and compare")
    ap.add_argument("--out", help="write <out>/<source>__<name>.jsonl per task")
    args = ap.parse_args()
    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")
    bad = 0
    for task in tasks:
        ex = load(task, args.cache)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, task.replace("/", "__") + ".jsonl"), "w") as f:
                for e in ex:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        line = f"{task:22s} n={len(ex):5d} options={len(ex[0]['question']['options']):3d}"
        if args.check:
            pub = published(task, args.cache)
            for system in sorted(ex[0]["refs"]):
                s = score(ex, {e["id"]: (e["refs"][system]["pred"], e["refs"][system]["conf"]) for e in ex})
                p = pub.get(system)
                ok = p is None or abs(s["accuracy"] - p) < 5e-4
                bad += not ok
                line += f"\n    {system:28s} acc {s['accuracy']:.4f} published {p if p is not None else '-':<8} " \
                        f"{'ok' if ok else 'MISMATCH'}  macro-F1 {s['macro_f1']:.4f}  ECE {s['ece']}"
        print(line, flush=True)
    if args.check:
        print("all published accuracies reproduced" if not bad else f"{bad} mismatches")
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
