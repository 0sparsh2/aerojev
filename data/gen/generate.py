"""Build the AeroJev synthetic datasets.

  python -m data.gen.generate --out data/processed

Splits
  train / val   : training templates, six formats
  test          : held-out templates (~25% of every phrase bank) + test-only call_notes format
  injection     : test templates, every record carries a prompt-injection line
Each row: {"id", "split", "format", "state", "labels", "meta"}
"""

import argparse
import collections
import json
import os
import random

from . import policy
from .policy import derive_labels, sample_scenario
from .render import Bank, Ctx, FORMAT_FNS, choose_format


def make_record(rng, split, idx, inject_p=0.03):
    s = sample_scenario(rng)
    s.extra["inject_p"] = inject_p
    fmt = choose_format(rng, s, split if split != "injection" else "test")
    bank = Bank("test" if split in ("test", "injection") else "train")
    ctx = Ctx(s, rng, bank)
    text = FORMAT_FNS[fmt](ctx)
    labels = derive_labels(s)  # after rendering: formats may pin facts (e.g. ERP always shows a promise date)
    return {
        "id": f"{split}-{idx:05d}",
        "split": split,
        "format": fmt,
        "state": text,
        "labels": labels,
        "meta": {"event": s.event, "sub": s.sub, "days_late": s.days_late, "coverage_days": s.coverage_days,
                 "aog": s.aog, "injected": bool(s.extra.get("injected"))},
    }


def build(n, split, seed, inject_p=0.03):
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        r = make_record(rng, split, len(rows), inject_p)
        if r["state"] in seen:
            continue
        seen.add(r["state"])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--n-injection", type=int, default=200)
    ap.add_argument("--v2", action="store_true", help="data v2: extra phrase banks, 6%% injections")
    ap.add_argument("--only", default="train,val,test,injection")
    args = ap.parse_args()
    policy.FEATURES["v2"] = args.v2
    inj = 0.06 if args.v2 else 0.03
    os.makedirs(args.out, exist_ok=True)
    seed0 = 100 if args.v2 else 0
    plan = {"train": (args.n_train, 1, inj), "val": (args.n_val, 2, inj), "test": (args.n_test, 3, inj),
            "injection": (args.n_injection, 4, 1.0)}
    sets = {name: build(n, name, seed0 + seed, inject_p=p) for name, (n, seed, p) in plan.items()
            if name in args.only.split(",")}
    train_states = {r["state"] for r in sets.get("train", [])}
    for name, rows in sets.items():
        if name != "train":
            rows[:] = [r for r in rows if r["state"] not in train_states]
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        fm = collections.Counter(r["format"] for r in rows)
        ev = collections.Counter(r["labels"]["event_type"] for r in rows)
        print(f"{name:10s} n={len(rows):5d} formats={dict(fm)}")
        print(f"{'':10s} events={dict(ev)}")
    for q in ["impact_severity", "line_stop_risk", "recommended_action", "delay_bucket",
              "export_controlled", "counterfeit_suspect", "sole_source"]:
        print(q, dict(sorted(collections.Counter(r["labels"][q] for r in sets.get("train", [])).items())))


if __name__ == "__main__":
    main()
