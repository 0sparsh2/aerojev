"""Evaluate a decision model on one or more datasets.

  python -m eval.evaluate --model Qwen/Qwen3-0.6B --sets gold,test --out results/zs-qwen3-0.6b
  python -m eval.evaluate --model Qwen/Qwen3-0.6B --adapter runs/qwen3/best --sets val,test,gold,injection

Writes <out>/preds_<set>.jsonl (raw label logits) and <out>/metrics.json.
Questions use canonical phrasing and canonical option order. A permutation
pass re-asks every choice question with options shuffled (gold set) to
measure whether the model reads options or memorises positions.
"""

import argparse
import json
import os
import random
import time

import torch

from aerojev import schema
from aerojev.engine import DecisionEngine, GGUFEngine
from aerojev.metrics import fit_temperatures, summarize
from aerojev.prompt import materialize

CORE = [q.id for q in schema.TRAIN_QUESTIONS]
ALLQ = schema.CATALOGUE


def load(path, limit=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def run_set(eng, rows, questions, shuffle_seed=None):
    preds, t0 = [], time.time()
    for i, r in enumerate(rows):
        rng = random.Random(f"{shuffle_seed}-{r['id']}") if shuffle_seed is not None else None
        qds, keysets = [], []
        for q in questions:
            qd, keys = materialize(q, rng=rng, shuffle=shuffle_seed is not None)
            qds.append(qd)
            keysets.append(keys)
        logits = eng.readout_logits(r["state"], qds)
        for qd, keys, lg in zip(qds, keysets, logits):
            preds.append({"rid": r["id"], "qid": qd["id"], "type": qd["type"], "keys": keys,
                          "logits": [round(x, 4) for x in lg.tolist()], "gold": r["labels"][qd["id"]],
                          "format": r.get("format")})
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(rows)}  {el / (i + 1):.3f}s/record", flush=True)
    return preds, (time.time() - t0) / max(1, len(rows))


def permutation_consistency(base_preds, perm_preds):
    import numpy as np
    base = {(p["rid"], p["qid"]): p["keys"][int(np.argmax(p["logits"]))] for p in base_preds if p["type"] == "choice"}
    agree = [base[(p["rid"], p["qid"])] == p["keys"][int(np.argmax(p["logits"]))]
             for p in perm_preds if p["type"] == "choice"]
    acc = [p["keys"][int(np.argmax(p["logits"]))] == p["gold"] for p in perm_preds if p["type"] == "choice"]
    return {"choice_agreement_with_canonical": float(np.mean(agree)), "choice_accuracy_shuffled": float(np.mean(acc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model id (also the tokenizer for --gguf)")
    ap.add_argument("--gguf", help="evaluate a GGUF file with llama.cpp on CPU instead of HF")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--layout", choices=["branch", "catalogue", "nocat"], default="branch")
    ap.add_argument("--adapter")
    ap.add_argument("--mode", choices=["tree", "unpacked"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--sets", default="gold,test")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", help="json file with per-type temperatures; default: fit on val if present")
    ap.add_argument("--no-perm", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.gguf:
        eng = GGUFEngine(args.gguf, args.model, n_threads=args.threads, layout=args.layout)
    else:
        eng = DecisionEngine(args.model, device=args.device, dtype=getattr(torch, args.dtype),
                             adapter=args.adapter, mode=args.mode, layout=args.layout)
    print(f"model={args.model} adapter={args.adapter} mode={eng.mode} device={args.device}", flush=True)

    results = {"model": args.model, "adapter": args.adapter, "gguf": args.gguf, "mode": eng.mode,
               "layout": args.layout, "sets": {}}
    all_preds = {}
    # a nocat model only knows its trained questions; the held-out ones need their text in the prompt
    questions = schema.TRAIN_QUESTIONS if args.layout == "nocat" else ALLQ
    for name in args.sets.split(","):
        rows = load(os.path.join(args.data, f"{name}.jsonl"), args.limit)
        print(f"[{name}] {len(rows)} records", flush=True)
        preds, spr = run_set(eng, rows, questions)
        all_preds[name] = preds
        with open(os.path.join(args.out, f"preds_{name}.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        results["sets"][name] = {"sec_per_record": spr}

    temps = {}
    if args.temps:
        temps = json.load(open(args.temps))
    elif "val" in all_preds:
        temps = fit_temperatures([p for p in all_preds["val"] if p["qid"] in CORE])
    results["temperatures"] = temps

    for name, preds in all_preds.items():
        results["sets"][name]["raw"] = summarize(preds, CORE)
        if temps:
            results["sets"][name]["calibrated"] = summarize(preds, CORE, temps)
        holdout = [p for p in preds if p["qid"] not in CORE]
        if holdout:
            results["sets"][name]["holdout_questions"] = summarize(holdout, [q.id for q in schema.HOLDOUT_QUESTIONS])["per_question"]

    # nocat never shows the options, so shuffling them has no meaning there
    if "gold" in all_preds and not args.no_perm and args.layout != "nocat":
        rows = load(os.path.join(args.data, "gold.jsonl"), args.limit)
        perm, _ = run_set(eng, rows, ALLQ, shuffle_seed=1)
        results["gold_permutation"] = permutation_consistency(all_preds["gold"], perm)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    for name, r in results["sets"].items():
        raw = r["raw"]
        print(f"== {name}: core acc {raw['core_mean_accuracy']:.3f}  macroF1 {raw['core_mean_macro_f1']:.3f}  "
              f"exact {raw['record_exact_match']:.3f}  ECE {raw['core_ece']:.3f}  {r['sec_per_record']:.2f}s/rec")
        for q, m in raw["per_question"].items():
            extra = f" auroc={m['auroc']:.3f}" if "auroc" in m else (f" mae={m['mae_expected']:.2f}" if "mae_expected" in m else "")
            print(f"   {q:20s} acc={m['accuracy']:.3f} (maj {m['majority_rate']:.2f}) f1={m['macro_f1']:.3f}{extra}")
    if "gold_permutation" in results:
        print("permutation:", results["gold_permutation"])


if __name__ == "__main__":
    main()
