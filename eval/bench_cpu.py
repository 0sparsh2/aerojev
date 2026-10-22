"""CPU latency / memory benchmark for one engine configuration (run one config per process).

  python -m eval.bench_cpu --engine hf --model Qwen/Qwen3-0.6B --threads 4
  python -m eval.bench_cpu --engine gguf --gguf models/aerojev-qwen3-0.6b-q8_0.gguf --tokenizer Qwen/Qwen3-0.6B

Appends one JSON line to results/bench_cpu.jsonl.
"""

import argparse
import json
import os
import statistics
import time

import torch

from aerojev import benchrec, schema
from aerojev.engine import DecisionEngine, GGUFEngine
from aerojev.prompt import materialize, render_branch, render_prefix, render_readout, render_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "gguf"], required=True)
    ap.add_argument("--mode", choices=["tree", "unpacked"])
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--gguf")
    ap.add_argument("--tokenizer")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--questions", choices=["all", "core", "one"], default="all")
    ap.add_argument("--set", default="gold")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--label", default="")
    ap.add_argument("--acc-key", help="results key whose accuracy this configuration corresponds to")
    ap.add_argument("--layout", choices=["branch", "catalogue"], default="branch")
    ap.add_argument("--out", default="results/bench_cpu.jsonl")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    t0 = time.time()
    if args.engine == "hf":
        eng = DecisionEngine(args.model, adapter=args.adapter, device="cpu", layout=args.layout, mode=args.mode)
    else:
        eng = GGUFEngine(args.gguf, args.tokenizer, n_threads=args.threads, layout=args.layout)
    load_s = time.time() - t0

    qs = {"all": schema.CATALOGUE, "core": schema.TRAIN_QUESTIONS, "one": [schema.IMPACT_SEVERITY]}[args.questions]
    qds = [materialize(q)[0] for q in qs]
    rows = [json.loads(l) for l in open(f"data/processed/{args.set}.jsonl")][: args.limit]

    eng.readout_logits(rows[0]["state"], qds)  # warm-up
    lat, toks = [], []
    enc = lambda t: len(eng.tok.encode(t, add_special_tokens=False))
    for r in rows:
        if args.layout == "catalogue":  # tokens actually processed per record (static catalogue is cached)
            n_prefix = enc(render_record(r["state"], eng.think))
            n_branch = sum(enc(render_readout(q)) for q in qds)
        else:
            n_prefix = enc(render_prefix(r["state"]))
            n_branch = sum(enc(render_branch(q, eng.think)) for q in qds)
        toks.append(n_prefix + n_branch)
        t1 = time.time()
        eng.readout_logits(r["state"], qds)
        lat.append(time.time() - t1)
    engine = "torch fp32" if args.engine == "hf" else "llama.cpp"
    rec = benchrec.record(
        args.label, engine, lat, len(qds), args.threads, layout=args.layout, mode=eng.mode,
        model=args.model or args.gguf, adapter=args.adapter, questions=args.questions, load_s=round(load_s, 2),
        mean_tokens=round(statistics.mean(toks)), tokens_per_s=round(sum(toks) / sum(lat)),
        file_mb=round(os.path.getsize(args.gguf) / 1e6) if args.gguf else None, acc_key=args.acc_key)
    benchrec.append(args.out, rec)


if __name__ == "__main__":
    main()
