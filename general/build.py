"""Build general decision data from domain specs, with the checks the plan requires.

  python -m general.build --check-only                          # validate every domain spec
  python -m general.build --records 300 --out /tmp/general-dry  # dry run: template-only data for every domain
  python -m general.build --domains it_incident,ap_invoice --records 2000 --out data/general/v0

Writes <out>/<split>.jsonl (one example per line, general/compose.py) and <out>/stats.json. Training domains fill
train / val / test (80 / 10 / 10 of --records); a held-out domain (Domain.heldout) fills only dev_unseen or
test_unseen (--unseen-records), so nothing from it reaches training. stats.json:
  spec      Domain.validate() for each domain (the build stops if any spec has problems, unless --force)
  labels    label shares per domain, question and rule
  pairs     counterfactual pairs per split
  facts     every number a label needs must appear in the text (a check on the phrase templates)
  shortcut  per domain and question, bag-of-words logistic regression from the record text alone (default rule,
            rule not stated). Near-perfect accuracy means the label leaks through wording rather than the facts,
            and that question needs more varied phrasing.
  leakage   share of test (and unseen) 8-grams that also occur in train text
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from . import compose
from .domains import DOMAINS
from .render import number_in_text, small_count

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
UNSEEN = ("dev_unseen", "test_unseen")


def plan_splits(doms, records, unseen_records):
    """-> {split: [(domain, number of records)]}, only splits that get records."""
    out = {s: [] for s in (*SPLITS, *UNSEEN)}
    for d in doms:
        if d.heldout:
            out[f"{d.heldout}_unseen"].append((d, unseen_records))
        else:
            for s, frac in SPLITS.items():
                out[s].append((d, max(1, round(records * frac))))
    return {s: v for s, v in out.items() if v}


def numbers_present(e, dom):
    """Every int / float fact a label needs is written somewhere in the text (digits, ignoring separators). In
    model-written prose a small count may be in words; gen_prose.py had the judge read those back, so they are not
    checked here."""
    needs = next(q for q in dom.questions if q.id == e["question"]["id"]).needs
    prose = e["format"].startswith("prose-")
    return [n for n in needs if dom.fact(n).kind in ("int", "float") and not (prose and small_count(dom.fact(n)))
            and not number_in_text(dom.fact(n).text(e["facts"][n]), e["state"])]


def ngrams(text, n=8):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def shortcut_check(train, test):
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {"skipped": "scikit-learn not installed"}
    out = {}
    by = defaultdict(lambda: {"train": [], "test": []})
    for split, rows in (("train", train), ("test", test)):
        for e in rows:
            if e["rule"] == "default" and not e["rule_in_text"]:
                by[(e["domain"], e["question"]["id"])][split].append((e["state"], _canon(e)))
    for (d, q), s in sorted(by.items()):
        if len(s["train"]) < 50 or len(s["test"]) < 20 or len({y for _, y in s["train"]}) < 2:
            continue
        vec = CountVectorizer(min_df=2, ngram_range=(1, 2))
        xtr = vec.fit_transform([x for x, _ in s["train"]])
        clf = LogisticRegression(max_iter=2000).fit(xtr, [y for _, y in s["train"]])
        ys = [y for _, y in s["test"]]
        acc = (clf.predict(vec.transform([x for x, _ in s["test"]])) == ys).mean()
        major = Counter(ys).most_common(1)[0][1] / len(ys)
        out[f"{d}/{q}"] = {"bow_acc": round(float(acc), 3), "majority": round(major, 3),
                           "flag": bool(acc >= 0.95 and major < 0.9)}
    return out


def _canon(e):
    """The label under the original keys (opaque option_N keys are mapped back through the description)."""
    q = e["question"]
    if not e["label"].startswith("option_"):
        return e["label"]
    return next(o["description"] for o in q["options"] if o["key"] == e["label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="all", help="comma-separated names, or all")
    ap.add_argument("--records", type=int, default=300, help="records per training domain (split 80/10/10)")
    ap.add_argument("--unseen-records", type=int, default=300, help="records per held-out domain")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-rules", type=float, default=0.5, help="P(default rule stated in the instructions)")
    ap.add_argument("--p-pair", type=float, default=0.5, help="P(a question with variants is asked as a pair)")
    ap.add_argument("--banks", help="directory of generated wording (general/gen_banks.py) to add to the specs")
    ap.add_argument("--prose", help="directory of model-written records (general/gen_prose.py) to add to each split")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="build even if a spec has problems")
    args = ap.parse_args()

    if args.banks:
        from .gen_banks import load_banks
        load_banks(args.banks)
    names = sorted(DOMAINS) if args.domains == "all" else args.domains.split(",")
    doms = [DOMAINS[n] for n in names]
    stats = {"spec": {}}
    ok = True
    for d in doms:
        v = d.validate(seed=args.seed)
        stats["spec"][d.name] = v
        print(f"{d.name:16s} {'ok' if v['ok'] else 'PROBLEMS'}")
        for p in v["problems"]:
            print(f"    {p}")
        ok &= v["ok"]
    if args.check_only:
        raise SystemExit(0 if ok else 1)
    if not ok and not args.force:
        raise SystemExit("spec problems: fix them or pass --force")
    if not args.out:
        raise SystemExit("--out is required to build")

    os.makedirs(args.out, exist_ok=True)
    data = {}
    for split, plan in plan_splits(doms, args.records, args.unseen_records).items():
        rows = []
        for d, n in plan:
            rows += compose.build_domain(d, n, split, args.seed, p_rules=args.p_rules, p_pair=args.p_pair)
            path = os.path.join(args.prose or "", f"{d.name}.{split}.jsonl")
            if args.prose and os.path.exists(path):
                rng = random.Random(f"prose|{d.name}|{split}|{args.seed}")
                for i, line in enumerate(open(path)):
                    r = json.loads(line)
                    rows += compose.record_examples(rng, d, split, f"{d.name}-{split}-prose-{i:05d}",
                                                    p_rules=args.p_rules, p_pair=args.p_pair, facts=r["facts"],
                                                    written=(r["state"], "prose-" + r["format"]))
        data[split] = rows
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    data.setdefault("train", [])
    data.setdefault("test", [])
    labels = defaultdict(Counter)
    for e in data["train"]:
        labels[f"{e['domain']}/{e['question']['id']}/{e['rule']}"][_canon(e)] += 1
    stats["labels"] = {k: dict(v) for k, v in sorted(labels.items())}
    stats["examples"] = {s: len(r) for s, r in data.items()}
    stats["pairs"] = {s: len({e["pair"] for e in r if e["pair"]}) for s, r in data.items()}
    stats["formats"] = {s: dict(Counter(e["format"] for e in r if e["id"].endswith(":0"))) for s, r in data.items()}
    miss = Counter(n for r in data.values() for e in r for n in numbers_present(e, DOMAINS[e["domain"]]))
    stats["facts_missing_from_text"] = dict(miss)
    train_ng = set().union(set(), *(ngrams(e["state"]) for e in data["train"] if e["id"].endswith(":0")))
    for split in ("test", *UNSEEN):
        if data.get(split):
            grams = [ngrams(e["state"]) for e in data[split] if e["id"].endswith(":0")]
            stats[f"leakage_{split}_8gram_overlap"] = round(sum(len(g & train_ng) for g in grams) /
                                                            max(1, sum(len(g) for g in grams)), 3)
    stats["shortcut"] = shortcut_check(data["train"], data["test"])
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1)

    overlap = {k[len("leakage_"):-len("_8gram_overlap")]: v for k, v in stats.items() if k.startswith("leakage_")}
    print(f"examples {stats['examples']}  pairs {stats['pairs']}  8-gram overlap with train {overlap}")
    print(f"numbers missing from text: {stats['facts_missing_from_text'] or 'none'}")
    flagged = [k for k, v in stats["shortcut"].items() if isinstance(v, dict) and v.get("flag")]
    print(f"shortcut flags: {flagged or 'none'}")


if __name__ == "__main__":
    main()
