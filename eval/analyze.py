"""Error analysis over saved predictions.

  python -m eval.analyze kaggle/outputs/train/out/results/ft-qwen3-0.6b --temps
Prints per-format accuracy (test), confusion matrices (gold), reliability bins,
and the gold records with the most wrong core decisions. Writes analysis.json.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from aerojev import schema
from aerojev.metrics import softmax

CORE = [q.id for q in schema.TRAIN_QUESTIONS]


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def pred_of(p, T=1.0):
    pr = softmax(p["logits"], T)
    return p["keys"][int(np.argmax(pr))], float(pr.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--temps", action="store_true", help="apply temperatures from metrics.json")
    args = ap.parse_args()
    m = json.load(open(os.path.join(args.run_dir, "metrics.json")))
    T = m.get("temperatures", {}) if args.temps else {}
    out = {}

    test = [p for p in load(os.path.join(args.run_dir, "preds_test.jsonl")) if p["qid"] in CORE]
    by_fmt = defaultdict(list)
    for p in test:
        by_fmt[p.get("format")].append(pred_of(p, T.get(p["type"], 1.0))[0] == p["gold"])
    out["test_acc_by_format"] = {f: round(float(np.mean(v)), 3) for f, v in sorted(by_fmt.items())}
    print("test accuracy by record format (call_notes is never seen in training):")
    for f, a in out["test_acc_by_format"].items():
        print(f"  {f:14s} {a:.3f}  (n={len(by_fmt[f]) // len(CORE)} records)")

    gold = load(os.path.join(args.run_dir, "preds_gold.jsonl"))
    for qid in ["impact_severity", "recommended_action", "event_type"]:
        rows = [p for p in gold if p["qid"] == qid]
        keys = rows[0]["keys"]
        cm = Counter((p["gold"], pred_of(p)[0]) for p in rows)
        out[f"confusion_{qid}"] = {f"{g}->{pr}": n for (g, pr), n in cm.items()}
        print(f"\nconfusion {qid} (rows gold, cols pred):")
        short = [k[:6] for k in keys]
        print("        " + " ".join(f"{s:>6s}" for s in short))
        for g in keys:
            if any(cm[(g, pr)] for pr in keys):
                print(f"{g[:7]:>7s} " + " ".join(f"{cm[(g, pr)]:6d}" for pr in keys))

    conf, corr = [], []
    for p in gold + test:
        if p["qid"] in CORE:
            a, c = pred_of(p, T.get(p["type"], 1.0))
            conf.append(c)
            corr.append(a == p["gold"])
    conf, corr = np.array(conf), np.array(corr, dtype=float)
    bins = []
    for lo in np.arange(0, 1, 0.1):
        msk = (conf > lo) & (conf <= lo + 0.1)
        if msk.any():
            bins.append({"bin": f"{lo:.1f}-{lo + 0.1:.1f}", "n": int(msk.sum()), "confidence": round(float(conf[msk].mean()), 3),
                         "accuracy": round(float(corr[msk].mean()), 3)})
    out["reliability"] = bins
    print("\nreliability (gold+test, core):")
    for b in bins:
        print(f"  {b['bin']}  n={b['n']:5d}  conf={b['confidence']:.3f}  acc={b['accuracy']:.3f}")

    wrong = defaultdict(list)
    for p in gold:
        if p["qid"] in CORE and pred_of(p)[0] != p["gold"]:
            wrong[p["rid"]].append(f"{p['qid']}: gold={p['gold']} pred={pred_of(p)[0]}")
    out["gold_errors"] = dict(sorted(wrong.items(), key=lambda kv: -len(kv[1])))
    print(f"\ngold records with errors: {len(wrong)}/80")
    for rid, errs in list(out["gold_errors"].items())[:8]:
        print(f"  {rid}: " + "; ".join(errs))
    json.dump(out, open(os.path.join(args.run_dir, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
