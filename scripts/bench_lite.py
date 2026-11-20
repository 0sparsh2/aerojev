"""Latency and memory of the torch-free encoder runtime (aerojev.lite) on this CPU.

  python scripts/bench_lite.py --model models/aerojev-ettin-150m --weights int8 --threads 1 2 4

For each thread count: load time, per-record latency (all 9 questions; median / p95 over gold + test records,
after a warm-up) and the process's peak resident memory. Run once per weights file: peak memory is per process.
"""

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KiB on Linux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", default="int8", choices=["int8", "int4", "fp32"])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--records", type=int, help="time only the first N records (default: all gold + test)")
    ap.add_argument("--out", help="append one json line per thread count")
    args = ap.parse_args()
    base = peak_mb()
    from aerojev.lite import LiteEngine
    rows = [json.loads(l)["state"] for s in ("gold", "test") for l in open(os.path.join(args.data, f"{s}.jsonl"))]
    rows = rows[: args.records] if args.records else rows
    size = os.path.getsize(os.path.join(args.model, {"int8": "model.int8.onnx", "int4": "model.int4.onnx",
                                                     "fp32": "model.onnx"}[args.weights])) / 1e6
    for th in args.threads:
        t0 = time.time()
        eng = LiteEngine(args.model, threads=th, weights=args.weights)
        load_s = time.time() - t0
        for r in rows[:20]:
            eng.decide(r)
        lat = []
        for r in rows:
            t = time.perf_counter()
            eng.decide(r)
            lat.append(time.perf_counter() - t)
        lat.sort()
        rec = {"model": args.model, "weights": args.weights, "file_mb": round(size, 1), "threads": th,
               "load_s": round(load_s, 2), "ms_median": round(1e3 * lat[len(lat) // 2], 1),
               "ms_p95": round(1e3 * lat[int(len(lat) * 0.95)], 1), "records": len(lat),
               "peak_rss_mb": round(peak_mb()), "python_baseline_mb": round(base), "torch_loaded": "torch" in sys.modules}
        print(json.dumps(rec), flush=True)
        if args.out:
            open(args.out, "a").write(json.dumps(rec) + "\n")
        del eng


if __name__ == "__main__":
    main()
