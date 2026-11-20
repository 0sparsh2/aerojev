"""CPU training throughput of the LoRA decision recipe, one configuration per process.

  python scripts/bench_train_cpu.py --model Qwen/Qwen2.5-0.5B-Instruct --threads 10
  python scripts/bench_train_cpu.py --model Qwen/Qwen2.5-0.5B-Instruct --top-layers 8   # LoRA on the top 8 blocks only
  python scripts/bench_train_cpu.py --layout nocat --shared --grad-accum 8 --records 24  # catalogue-free, 8 per batch
  python scripts/bench_train_cpu.py --model <pruned dir> --layout nocat --shared --full-ft  # pruned, full fine-tuning

Times forward + backward (+ an optimizer step every --grad-accum records) on real training records, after a short
warm-up, and appends one JSON line to results/train_cpu.jsonl with the projected time for a full or reduced run.
Run it on an otherwise idle machine.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerojev import benchrec  # noqa: E402
from aerojev.prompt import think_block  # noqa: E402
from train.train import build_example, build_group, record_loss, shared_step  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layout", choices=["branch", "catalogue", "nocat"], default="catalogue")
    ap.add_argument("--shared", action="store_true", help="--shared-catalogue training: --grad-accum records per batch")
    ap.add_argument("--full-ft", action="store_true", help="train the block weights instead of LoRA")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 = CPU autocast")
    ap.add_argument("--top-layers", type=int, default=0, help="put LoRA only on the top N blocks (0 = all)")
    ap.add_argument("--records", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--data", default="data/processed_v2/train.jsonl")
    ap.add_argument("--out", default="results/train_cpu.jsonl")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    think = think_block(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers - args.top_layers, n_layers)) if args.top_layers else None
    if args.full_ft:
        for p in model.parameters():
            p.requires_grad = False
        for p in list(model.model.layers.parameters()) + list(model.model.norm.parameters()):
            p.requires_grad = True
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0 if args.shared else 0.05, bias="none", task_type="CAUSAL_LM",
            layers_to_transform=layers,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4)

    rows = [json.loads(l) for l in open(args.data)][: args.warmup + args.records]
    rng = random.Random(0)
    lat, toks = [], []
    if args.shared:  # one timed step per --grad-accum records; per-record time = step time / records
        import types
        nsa = types.SimpleNamespace(micro_batch=args.grad_accum, grad_accum=args.grad_accum, ordinal_weight=0.5,
                                    distill_weight=0.0, distill_temp=1.0)
        off = lambda: torch.autocast("cpu", enabled=False)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        for i in range(0, len(rows) - args.grad_accum + 1, args.grad_accum):
            g = build_group(tok, think, rows[i:i + args.grad_accum], rng, 9, 9, args.layout)
            t0 = time.time()
            shared_step(model, g, "cpu", nsa, off, scaler)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if i >= args.warmup:
                lat += [(time.time() - t0) / args.grad_accum] * args.grad_accum
                toks += [len(g["static_ids"]) / args.grad_accum + len(p["input_ids"]) for p in g["records"]]
        rows = []
    for i, row in enumerate(rows):
        p = build_example(tok, think, row, rng, 5, 9, args.layout)
        t0 = time.time()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
            loss, _ = record_loss(model, p, "cpu", 0.5)
        (loss / args.grad_accum).backward()
        if (i + 1) % args.grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if i >= args.warmup:
            lat.append(time.time() - t0)
            toks.append(len(p["input_ids"]))

    sec = statistics.mean(lat)
    name = os.path.basename(args.model.rstrip("/"))
    tuning = "full fine-tuning" if args.full_ft else f"LoRA {'all' if not args.top_layers else f'top {args.top_layers}'}"
    label = (f"{name} · {args.layout}{' · 8 per batch' if args.shared else ''} · {args.dtype} · {args.threads} threads · "
             f"{tuning} of {n_layers} blocks · FFN {model.config.intermediate_size if hasattr(model, 'config') else ''}")
    rec = benchrec.record(label, "torch training", lat, 1, args.threads, layout=args.layout, dtype=args.dtype,
                          top_layers=args.top_layers or n_layers, n_layers=n_layers, mean_tokens=round(statistics.mean(toks)),
                          sec_per_record=round(sec, 3), records_per_hour=round(3600 / sec),
                          hours_for={n: round(n * sec / 3600, 1) for n in (2400, 4800, 7200, 12000)})
    benchrec.append(args.out, rec)


if __name__ == "__main__":
    main()
