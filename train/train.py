"""LoRA decision training on label-token logits (RFDT-style).

  python -m train.train --model Qwen/Qwen3-0.6B --out runs/qwen3-0.6b

Loss per record = mean over its questions of
  CE(softmax over the question's label tokens at its readout, gold label)
  + ordinal_weight * E[|level - gold|] / (levels - 1)        (score questions)
No loss anywhere else: no LM loss on the prompt, no generated JSON.

Each record is one tree-packed sequence holding a random subset of the core
questions, with option order shuffled, question text paraphrased and option
sets occasionally subsampled, so the model must read options, not positions.
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from aerojev import schema
from aerojev.prompt import materialize, pack, think_block

CORE = schema.TRAIN_QUESTIONS


def build_example(tok, think, row, rng, min_q, max_q, layout="branch"):
    k = rng.randint(min_q, min(max_q, len(CORE)))
    if layout == "catalogue":
        # the catalogue always lists every core question (as at inference), in random order with
        # shuffled options; only a random subset is asked
        cat, gold_idx = [], {}
        for q in rng.sample(CORE, len(CORE)):
            gold = row["labels"][q.id]
            qd, keys = materialize(q, rng=rng, shuffle=True, paraphrase=True, subset=True, answer=gold)
            cat.append(qd)
            gold_idx[q.id] = keys.index(gold)
        asked = rng.sample(cat, k)
        p = pack(tok, row["state"], asked, think, layout="catalogue", catalogue=cat)
        p["golds"] = [gold_idx[q["id"]] for q in asked]
        p["types"] = [q["type"] for q in asked]
        return p
    qs = rng.sample(CORE, k)
    qds, golds = [], []
    for q in qs:
        gold = row["labels"][q.id]
        qd, keys = materialize(q, rng=rng, shuffle=True, paraphrase=True, subset=True, answer=gold)
        qds.append(qd)
        golds.append(keys.index(gold))
    p = pack(tok, row["state"], qds, think)
    p["golds"] = golds
    p["types"] = [qd["type"] for qd in qds]
    return p


def record_loss(model, p, device, ordinal_weight):
    out = model(
        input_ids=p["input_ids"][None].to(device),
        position_ids=p["position_ids"][None].to(device),
        attention_mask=p["mask"][None, None].to(device),
        logits_to_keep=torch.tensor(p["readouts"], device=device),
    ).logits[0].float()
    losses, correct = [], []
    for i, (lab, g, t) in enumerate(zip(p["label_ids"], p["golds"], p["types"])):
        lg = out[i, lab]
        logp = F.log_softmax(lg, -1)
        loss = -logp[g]
        if t == "score" and ordinal_weight > 0:
            n = len(lab)
            dist = (torch.arange(n, device=lg.device) - g).abs().float() / (n - 1)
            loss = loss + ordinal_weight * (logp.exp() * dist).sum()
        losses.append(loss)
        correct.append(int(lg.argmax().item() == g))
    return torch.stack(losses).mean(), correct


@torch.no_grad()
def quick_eval(model, tok, think, rows, device, amp_dtype, layout="branch"):
    model.eval()
    hits, n = 0, 0
    for r in rows:
        qds, golds = [], []
        for q in CORE:
            qd, keys = materialize(q)
            qds.append(qd)
            golds.append(keys.index(r["labels"][q.id]))
        p = pack(tok, r["state"], qds, think, layout=layout)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            out = model(input_ids=p["input_ids"][None].to(device), position_ids=p["position_ids"][None].to(device),
                        attention_mask=p["mask"][None, None].to(device),
                        logits_to_keep=torch.tensor(p["readouts"], device=device)).logits[0]
        for i, (lab, g) in enumerate(zip(p["label_ids"], golds)):
            hits += int(out[i, lab].argmax().item() == g)
            n += 1
    model.train()
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--min-q", type=int, default=5)
    ap.add_argument("--max-q", type=int, default=9)
    ap.add_argument("--ordinal-weight", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=150, help="optimizer steps")
    ap.add_argument("--val-limit", type=int, default=150)
    ap.add_argument("--limit", type=int, help="train on the first N records (smoke tests)")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", default="fp16" if torch.cuda.is_available() else "fp32",
                    choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=["branch", "catalogue"], default="branch")
    ap.add_argument("--top-layers", type=int, default=0,
                    help="LoRA on the top N transformer blocks only (0 = all). Backprop then stops early (about 2x "
                         "faster on a CPU for N = a third of the blocks), but in our tests accuracy fell by ~30 points "
                         "(Qwen2.5-0.5B, catalogue layout): not recommended")
    ap.add_argument("--base-dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                    help="dtype of the frozen base weights (LoRA weights stay fp32); float16 halves GPU memory")
    ap.add_argument("--grad-checkpointing", action="store_true", help="recompute activations in backward to save memory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    think = think_block(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.base_dtype),
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_layers = model.config.num_hidden_layers
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     layers_to_transform=list(range(n_layers - args.top_layers, n_layers)) if args.top_layers else None,
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)  # adapter weights are created in fp32 whatever the base dtype
    model.print_trainable_parameters()
    model.to(args.device).train()

    train = [json.loads(l) for l in open(os.path.join(args.data, "train.jsonl"))]
    if args.limit:
        train = train[: args.limit]
    val = [json.loads(l) for l in open(os.path.join(args.data, "val.jsonl"))][: args.val_limit]

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    dev_type = args.device.split(":")[0]
    scaler = torch.amp.GradScaler(dev_type, enabled=args.precision == "fp16")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_records = int(len(train) * args.epochs)
    total_steps = args.max_steps or math.ceil(total_records / args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    best, step, seen, t0 = -1.0, 0, 0, time.time()
    run_loss, run_acc, run_n = 0.0, 0, 0
    acc0 = quick_eval(model, tok, think, val[: min(50, len(val))], args.device, amp_dtype, args.layout)
    print(f"step 0 val@50 acc={acc0:.3f}", flush=True)
    order = []
    while step < total_steps:
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        row = train[order.pop()]
        p = build_example(tok, think, row, rng, args.min_q, args.max_q, args.layout)
        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=args.precision != "fp32"):
            loss, correct = record_loss(model, p, args.device, args.ordinal_weight)
        if not torch.isfinite(loss):
            print("non-finite loss, skipping record", flush=True)
            continue
        scaler.scale(loss / args.grad_accum).backward()
        run_loss += loss.item()
        run_acc += sum(correct)
        run_n += len(correct)
        seen += 1
        if seen % args.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                rec = {"step": step, "records": seen, "loss": run_loss / args.grad_accum / 25,
                       "train_acc": run_acc / max(1, run_n), "lr": sched.get_last_lr()[0], "elapsed_s": el,
                       "rec_per_s": seen / el}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss, run_acc, run_n = 0.0, 0, 0
            if step % args.eval_every == 0 or step == total_steps:
                acc = quick_eval(model, tok, think, val, args.device, amp_dtype, args.layout)
                rec = {"step": step, "val_acc": acc}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                if acc > best:
                    best = acc
                    model.save_pretrained(os.path.join(args.out, "best"))
                    json.dump({"step": step, "val_acc": acc}, open(os.path.join(args.out, "best", "info.json"), "w"))
    model.save_pretrained(os.path.join(args.out, "last"))
    json.dump({"best_val_acc": best, "steps": step, "records": seen, "elapsed_s": time.time() - t0, "args": vars(args)},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"done. best val acc {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
