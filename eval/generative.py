"""Baseline: ask a local GGUF chat model to *generate* a JSON object with all answers.

  python -m eval.generative --gguf models/base/Qwen3-0.6B-Q8_0.gguf --out results/gen-qwen3-0.6b

Two variants:
  free   : plain chat completion, temperature 0, parse the first JSON object
  schema : same prompt, decoding constrained by a JSON schema (enum per question)
Reports accuracy, schema validity and wall-clock latency on CPU, for comparison
with single-forward-pass logit readout.
"""

import argparse
import json
import os
import re
import time

from aerojev import schema
from aerojev.metrics import summarize

CORE = schema.TRAIN_QUESTIONS


def keys_for(q):
    return ["yes", "no"] if q.type == "yesno" else [o.key for o in q.options]


def build_prompt(state):
    lines = ["Read the supply chain record and answer every question.",
             "Reply with only a JSON object mapping each question id to one allowed value.", ""]
    for q in CORE:
        if q.type == "yesno":
            lines.append(f'- "{q.id}": {q.text} Allowed: "yes" | "no"')
        else:
            opts = "; ".join(f'"{o.key}" = {o.description}' for o in q.options)
            lines.append(f'- "{q.id}": {q.text} Allowed: {opts}')
    lines += ["", "Record:", state]
    return "\n".join(lines)


def json_schema():
    return {"type": "object", "properties": {q.id: {"type": "string", "enum": keys_for(q)} for q in CORE},
            "required": [q.id for q in CORE]}


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="gold")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bench-out", help="also append unified CPU latency lines here")
    ap.add_argument("--label", default="Qwen3-0.6B Q8_0 base")
    args = ap.parse_args()
    from llama_cpp import Llama

    llm = Llama(model_path=args.gguf, n_ctx=4096, n_threads=args.threads, n_threads_batch=args.threads,
                n_gpu_layers=0, offload_kqv=False, op_offload=False, verbose=False)
    rows = [json.loads(l) for l in open(f"data/processed/{args.set}.jsonl")][: args.limit]
    os.makedirs(args.out, exist_ok=True)
    report = {"gguf": args.gguf, "set": args.set, "threads": args.threads}
    for variant in ["free", "schema"]:
        preds, valid, lat, out_tokens = [], 0, [], []
        for r in rows:
            msgs = [{"role": "system", "content": "You are a precise supply chain analyst. /no_think"},
                    {"role": "user", "content": build_prompt(r["state"])}]
            kw = {"response_format": {"type": "json_object", "schema": json_schema()}} if variant == "schema" else {}
            t0 = time.time()
            res = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=400, **kw)
            lat.append(time.time() - t0)
            text = res["choices"][0]["message"]["content"]
            out_tokens.append(res["usage"]["completion_tokens"])
            ans = parse(re.sub(r"<think>.*?</think>", "", text, flags=re.S)) or {}
            ok = all(ans.get(q.id) in keys_for(q) for q in CORE)
            valid += ok
            for q in CORE:
                keys = keys_for(q) + ["<invalid>"]  # missing / out-of-schema answers always count as wrong
                a = ans.get(q.id)
                a = a if a in keys[:-1] else "<invalid>"
                logits = [0.0 if k == a else -20.0 for k in keys]
                preds.append({"rid": r["id"], "qid": q.id, "type": q.type, "keys": keys, "logits": logits,
                              "gold": r["labels"][q.id]})
        m = summarize(preds, [q.id for q in CORE])
        report[variant] = {"core_mean_accuracy": m["core_mean_accuracy"], "core_mean_macro_f1": m["core_mean_macro_f1"],
                           "record_exact_match": m["record_exact_match"], "schema_valid_rate": valid / len(rows),
                           "sec_per_record": sum(lat) / len(lat), "mean_output_tokens": sum(out_tokens) / len(out_tokens),
                           "per_question_accuracy": {q: v["accuracy"] for q, v in m["per_question"].items()}}
        print(variant, json.dumps({k: v for k, v in report[variant].items() if k != "per_question_accuracy"}), flush=True)
        if args.bench_out:
            from aerojev import benchrec
            benchrec.append(args.bench_out, benchrec.record(
                f"{args.label}: generate JSON ({variant})", "llama.cpp generate", lat, len(CORE), args.threads,
                file_mb=round(os.path.getsize(args.gguf) / 1e6), mean_output_tokens=report[variant]["mean_output_tokens"],
                accuracy_on_these=report[variant]["core_mean_accuracy"], acc_key=f"gen:gen-qwen3-0.6b-base:{variant}"))
    json.dump(report, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
