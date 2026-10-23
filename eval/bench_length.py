"""CPU latency against record length, plus start-up time, for one GGUF model (catalogue or nocat layout).

  python -m eval.bench_length --gguf models/aerojev-qwen3-0.6b-cat-v2-q8_0.gguf --tokenizer Qwen/Qwen3-0.6B

Pads a real record with a quoted reply chain to several lengths and times the nine core decisions per
record. The longest case exceeds the engine's record budget, so it also checks that over-long records
are trimmed rather than overflowing the context. Appends JSON lines to results/latency_length.jsonl.
"""

import argparse
import json
import statistics
import time

from aerojev import benchrec, schema
from aerojev.engine import GGUFEngine
from aerojev.prompt import materialize

BASE = ("FW: PO 4521730088 - stage 2 HPT disk forgings\n"
        "Buyer note: Halvorsen says the forgings ship five weeks late. We have 12 days of disks; they are our only source.\n")
QUOTED = ("\n> On a previous day the supplier wrote: thank you for the forecast update, we will review capacity for the next "
          "quarter and confirm the delivery schedule for the remaining line items on the purchase order shortly. "
          "Please note our office will be closed for the regional holiday next week.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--layout", choices=["catalogue", "nocat"], default="catalogue")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="results/latency_length.jsonl")
    args = ap.parse_args()

    t0 = time.time()
    eng = GGUFEngine(args.gguf, args.tokenizer, n_threads=args.threads, layout=args.layout)
    startup = time.time() - t0  # model load + tokenizer + the catalogue's KV cache
    qds = [materialize(q)[0] for q in schema.TRAIN_QUESTIONS]
    print(json.dumps({"gguf": args.gguf, "startup_s": round(startup, 2)}))
    for target in (100, 300, 600, 1200, 2000, 5000):
        text = BASE
        while len(eng.tok.encode(text, add_special_tokens=False)) < target:
            text += QUOTED
        n_tok = len(eng.tok.encode(text, add_special_tokens=False))
        eng.readout_logits(text, qds)  # warm-up at this length
        lat = []
        for _ in range(args.runs):
            t1 = time.time()
            eng.readout_logits(text, qds)
            lat.append(time.time() - t1)
        rec = benchrec.record(f"record of ~{n_tok} tokens", "llama.cpp", lat, len(qds), args.threads,
                              gguf=args.gguf, record_tokens=n_tok, startup_s=round(startup, 2))
        benchrec.append(args.out, rec)


if __name__ == "__main__":
    main()
