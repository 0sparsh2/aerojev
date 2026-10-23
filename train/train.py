"""LoRA decision training on label-token logits (RFDT-style).

  python -m train.train --model Qwen/Qwen3-0.6B --out runs/qwen3-0.6b

Loss per record = mean over its questions of
  CE(softmax over the question's label tokens at its readout, gold label)
  + ordinal_weight * E[|level - gold|] / (levels - 1)        (score questions)
  + distill_weight * T^2 * KL(teacher_T || student_T)       (with --teacher)
No loss anywhere else: no LM loss on the prompt, no generated JSON.

Each record is one tree-packed sequence holding a random subset of the core
questions, with option order shuffled, question text paraphrased and option
sets occasionally subsampled, so the model must read options, not positions.

--shared-catalogue: the --grad-accum records of one optimizer step share one
static prefix (catalogue layout: one shuffled catalogue; nocat: the system line).
It is encoded once; the records run as a padded batch on top of its per-layer K/V. Gradients are exactly those of the
records trained separately with that catalogue: the catalogue K/V are leaf
tensors during the record passes, and their accumulated gradient is sent back
through the catalogue once at the end of the step.

--layout nocat: no catalogue in the prompt; each question id has fixed labels in
canonical option order (see aerojev.prompt).
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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from aerojev import schema
from aerojev.prompt import (fit_state, label_token_ids, labels_for, materialize, pack, pack_ids, render_readout,
                            render_record, render_static, render_static_nocat, think_block)

CORE = schema.TRAIN_QUESTIONS
MAX_PREFIX_TOKENS = 1536  # record trimming used by pack() during training


def build_example(tok, think, row, rng, min_q, max_q, layout="branch"):
    k = rng.randint(min_q, min(max_q, len(CORE)))
    if layout == "catalogue":
        # the catalogue always lists every core question (as at inference), in random order with
        # shuffled options; only a random subset is asked
        cat, gold_idx, keymap = [], {}, {}
        for q in rng.sample(CORE, len(CORE)):
            gold = row["labels"][q.id]
            qd, keys = materialize(q, rng=rng, shuffle=True, paraphrase=True, subset=True, answer=gold)
            cat.append(qd)
            gold_idx[q.id] = keys.index(gold)
            keymap[q.id] = keys
        asked = rng.sample(cat, k)
        p = pack(tok, row["state"], asked, think, layout="catalogue", catalogue=cat)
        p["golds"] = [gold_idx[q["id"]] for q in asked]
        p["types"] = [q["type"] for q in asked]
        p["qids"] = [q["id"] for q in asked]
        p["keys"] = [keymap[q["id"]] for q in asked]
        p["rid"] = row["id"]
        return p
    qs = rng.sample(CORE, k)
    qds, golds, keysets = [], [], []
    for q in qs:
        gold = row["labels"][q.id]
        if layout == "nocat":  # options are never shown: canonical order, no paraphrase or subset
            qd, keys = materialize(q)
        else:
            qd, keys = materialize(q, rng=rng, shuffle=True, paraphrase=True, subset=True, answer=gold)
        qds.append(qd)
        golds.append(keys.index(gold))
        keysets.append(keys)
    p = pack(tok, row["state"], qds, think, layout="nocat" if layout == "nocat" else "branch")
    p["golds"] = golds
    p["types"] = [qd["type"] for qd in qds]
    p["qids"] = [qd["id"] for qd in qds]
    p["keys"] = keysets
    p["rid"] = row["id"]
    return p


def build_group(tok, think, rows, rng, min_q, max_q, layout="catalogue"):
    """--shared-catalogue: one static prefix for all rows, then each row as record + readouts at offset S.
    catalogue: one shuffled catalogue, whose option subsets keep the gold option of every row in the group.
    nocat: the short system line; questions in canonical form, never shown."""
    enc = lambda t: tok.encode(t, add_special_tokens=False)
    cat, keymap = [], {}
    for q in (CORE if layout == "nocat" else rng.sample(CORE, len(CORE))):
        golds = {r["labels"][q.id] for r in rows}
        qd, keys = materialize(q) if layout == "nocat" else \
            materialize(q, rng=rng, shuffle=True, paraphrase=True, subset=True, answer=golds)
        cat.append(qd)
        keymap[q.id] = keys
    static_ids = enc(render_static_nocat() if layout == "nocat" else render_static(cat))
    S = len(static_ids)
    recs = []
    for row in rows:
        k = rng.randint(min_q, min(max_q, len(CORE)))
        asked = rng.sample(cat, k)
        labs = [label_token_ids(tok, labels_for(q)) for q in asked]
        state = fit_state(tok, row["state"], MAX_PREFIX_TOKENS)
        p = pack_ids(enc(render_record(state, think)), [enc(render_readout(q)) for q in asked], labs, offset=S)
        p["golds"] = [keymap[q["id"]].index(row["labels"][q["id"]]) for q in asked]
        p["types"] = [q["type"] for q in asked]
        p["qids"] = [q["id"] for q in asked]
        p["keys"] = [keymap[q["id"]] for q in asked]
        p["rid"] = row["id"]
        recs.append(p)
    return {"static_ids": static_ids, "records": recs}


def readout_loss(lgs, p, ordinal_weight, teacher=None, distill_weight=0.0, distill_temp=1.0):
    """lgs: [n_readouts, vocab] logits at p's readouts. Returns (mean loss over the questions, correct flags)."""
    losses, correct = [], []
    for i, (lab, g, t) in enumerate(zip(p["label_ids"], p["golds"], p["types"])):
        lg = lgs[i, lab]
        logp = F.log_softmax(lg, -1)
        loss = -logp[g]
        if t == "score" and ordinal_weight > 0:
            n = len(lab)
            dist = (torch.arange(n, device=lg.device) - g).abs().float() / (n - 1)
            loss = loss + ordinal_weight * (logp.exp() * dist).sum()
        if teacher is not None and distill_weight > 0:
            tl = teacher[(p["rid"], p["qids"][i])]
            pt = F.softmax(torch.tensor([tl[k] for k in p["keys"][i]], device=lg.device) / distill_temp, -1)
            logps = F.log_softmax(lg / distill_temp, -1)
            loss = loss + distill_weight * distill_temp ** 2 * (pt * (pt.clamp_min(1e-12).log() - logps)).sum()
        losses.append(loss)
        correct.append(int(lg.argmax().item() == g))
    return torch.stack(losses).mean(), correct


def record_loss(model, p, device, ordinal_weight, teacher=None, distill_weight=0.0, distill_temp=1.0):
    out = model(
        input_ids=p["input_ids"][None].to(device),
        position_ids=p["position_ids"][None].to(device),
        attention_mask=p["mask"][None, None].to(device),
        logits_to_keep=torch.tensor(p["readouts"], device=device),
    ).logits[0].float()
    return readout_loss(out, p, ordinal_weight, teacher, distill_weight, distill_temp)


def shared_step(model, g, device, args, autocast, scaler, teacher=None):
    """Forward + backward for one --shared-catalogue group. Returns (per-record losses, correct flags, cost)."""
    base = model.get_base_model()
    body, head = base.model, base.lm_head
    S = len(g["static_ids"])
    with autocast():
        out = body(input_ids=torch.tensor([g["static_ids"]], device=device), use_cache=True)
    kv_graph = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
    kv_leaf = [(k.detach().requires_grad_(), v.detach().requires_grad_()) for k, v in kv_graph]
    cost = {"q_tokens": S, "attn_pairs": S * S, "useful_q_tokens": S, "useful_attn_pairs": S * S}
    pad = base.config.pad_token_id if base.config.pad_token_id is not None else 0
    rec_losses, correct = [], []
    recs = g["records"]
    for c in range(0, len(recs), args.micro_batch):
        chunk = recs[c:c + args.micro_batch]
        m, L = len(chunk), max(len(p["input_ids"]) for p in chunk)
        ids = torch.full((m, L), pad, dtype=torch.long)
        pos = torch.full((m, L), S, dtype=torch.long)
        mask = torch.zeros(m, 1, L, S + L, dtype=torch.bool)
        rows, cols = [], []
        for i, p in enumerate(chunk):
            n = len(p["input_ids"])
            ids[i, :n] = p["input_ids"]
            pos[i, :n] = p["position_ids"]
            mask[i, 0, :n, :S + n] = p["mask"]
            mask[i, 0, n:, :S] = True  # padding rows see only the catalogue, so no row is fully masked
            rows += [i] * len(p["readouts"])
            cols += p["readouts"]
            cost["useful_q_tokens"] += n
            cost["useful_attn_pairs"] += n * (S + n)
        cost["q_tokens"] += m * L
        cost["attn_pairs"] += m * L * (S + L)
        cache = DynamicCache(ddp_cache_data=[(k.expand(m, -1, -1, -1), v.expand(m, -1, -1, -1)) for k, v in kv_leaf],
                             config=base.config)
        with autocast():
            h = body(input_ids=ids.to(device), position_ids=pos.to(device), attention_mask=mask.to(device),
                     past_key_values=cache, use_cache=True).last_hidden_state
            lgs = head(h[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]).float()
        j, chunk_loss = 0, 0.0
        for p in chunk:
            n = len(p["readouts"])
            loss, corr = readout_loss(lgs[j:j + n], p, args.ordinal_weight, teacher, args.distill_weight,
                                      args.distill_temp)
            j += n
            chunk_loss = chunk_loss + loss
            rec_losses.append(loss.item())
            correct += corr
        if not torch.isfinite(chunk_loss):
            return None, None, cost
        scaler.scale(chunk_loss / args.grad_accum).backward()
    # the catalogue's own backward pass, once, with the K/V gradients summed over all records
    torch.autograd.backward([t for kv in kv_graph for t in kv], [t.grad for kv in kv_leaf for t in kv])
    return rec_losses, correct, cost


@torch.no_grad()
def quick_eval(model, tok, think, rows, device, amp_dtype, layout="branch", cached=False):
    model.eval()
    hits, n = 0, 0
    enc = lambda t: tok.encode(t, add_special_tokens=False)
    qds, keysets = zip(*[materialize(q) for q in CORE])
    autocast = lambda: torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                      enabled=amp_dtype != torch.float32)
    cache = None
    if cached and layout in ("catalogue", "nocat"):
        # the validation catalogue is fixed: encode it once with the current weights, as DecisionEngine does
        static = enc(render_static_nocat() if layout == "nocat" else render_static(list(qds)))
        S = len(static)
        with autocast():
            cache = model(input_ids=torch.tensor([static], device=device), use_cache=True,
                          logits_to_keep=1).past_key_values
        labs = [label_token_ids(tok, labels_for(q)) for q in qds]
    for r in rows:
        golds = [keys.index(r["labels"][q["id"]]) for q, keys in zip(qds, keysets)]
        if cache is not None:
            state = fit_state(tok, r["state"], MAX_PREFIX_TOKENS)
            p = pack_ids(enc(render_record(state, think)), [enc(render_readout(q)) for q in qds], labs, offset=S)
            kw = {"past_key_values": cache, "use_cache": True}
        else:
            p = pack(tok, r["state"], list(qds), think, layout=layout)
            kw = {}
        with autocast():
            out = model(input_ids=p["input_ids"][None].to(device), position_ids=p["position_ids"][None].to(device),
                        attention_mask=p["mask"][None, None].to(device),
                        logits_to_keep=torch.tensor(p["readouts"], device=device), **kw).logits[0]
        if cache is not None:
            cache.crop(S - cache.get_seq_length())  # drop this record's KV, keep the catalogue
        for i, (lab, g) in enumerate(zip(p["label_ids"], golds)):
            hits += int(out[i, lab].argmax().item() == g)
            n += 1
    model.train()
    return hits / n


def load_teacher(path):
    """eval.evaluate preds file -> {(record id, question id): {option key: logit}}."""
    t = {}
    for line in open(path):
        p = json.loads(line)
        t[(p["rid"], p["qid"])] = dict(zip(p["keys"], p["logits"]))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
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
    ap.add_argument("--layout", choices=["branch", "catalogue", "nocat"], default="branch")
    ap.add_argument("--shared-catalogue", action="store_true",
                    help="the --grad-accum records of a step share one static prefix (catalogue, or the nocat system "
                         "line), encoded once, and run as one batch on top of it")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="--shared-catalogue: records per forward pass on top of the catalogue (bounds memory)")
    ap.add_argument("--cached-val", action="store_true",
                    help="validation encodes the fixed catalogue once per pass (same results, less compute)")
    ap.add_argument("--teacher", help="eval.evaluate preds file over the training set: distill its label logits")
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--top-layers", type=int, default=0,
                    help="LoRA on the top N transformer blocks only (0 = all). Backprop then stops early (about 2x "
                         "faster on a CPU for N = a third of the blocks), but in our tests accuracy fell by ~30 points "
                         "(Qwen2.5-0.5B, catalogue layout): not recommended")
    ap.add_argument("--base-dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                    help="dtype of the frozen base weights (LoRA weights stay fp32); float16 halves GPU memory")
    ap.add_argument("--grad-checkpointing", action="store_true", help="recompute activations in backward to save memory")
    args = ap.parse_args()
    if args.shared_catalogue and args.layout not in ("catalogue", "nocat"):
        ap.error("--shared-catalogue needs --layout catalogue or nocat")
    if args.shared_catalogue and args.grad_checkpointing:
        ap.error("--shared-catalogue needs the KV cache, which gradient checkpointing turns off")
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
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
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
    teacher = load_teacher(args.teacher) if args.teacher else None

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    dev_type = args.device.split(":")[0]
    autocast = lambda: torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=args.precision != "fp32")
    scaler = torch.amp.GradScaler(dev_type, enabled=args.precision == "fp16")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_records = int(len(train) * args.epochs)
    total_steps = args.max_steps or math.ceil(total_records / args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    def validate(rows):
        nonlocal val_s
        t = time.time()
        acc = quick_eval(model, tok, think, rows, args.device, amp_dtype, args.layout, args.cached_val)
        val_s += time.time() - t
        return acc

    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    best, step, seen, t0, val_s = -1.0, 0, 0, time.time(), 0.0
    run_loss, run_acc, run_n = 0.0, 0, 0
    # hardware-independent cost: query tokens and attention (query, key) pairs that went through forward + backward
    cost = {"q_tokens": 0, "attn_pairs": 0, "useful_q_tokens": 0, "useful_attn_pairs": 0}
    acc0 = validate(val[: min(50, len(val))])
    print(f"step 0 val@50 acc={acc0:.3f}", flush=True)
    order = []

    def next_row():
        nonlocal order
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        return train[order.pop()]

    def optimizer_step():
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

    while step < total_steps:
        if args.shared_catalogue:
            g = build_group(tok, think, [next_row() for _ in range(args.grad_accum)], rng, args.min_q, args.max_q,
                            args.layout)
            losses, correct, c = shared_step(model, g, args.device, args, autocast, scaler, teacher)
            for k in cost:
                cost[k] += c[k]
            if losses is None:
                print("non-finite loss, skipping step", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            run_loss += sum(losses)
            run_acc += sum(correct)
            run_n += len(correct)
            seen += len(losses)
            optimizer_step()
            step += 1
        else:
            p = build_example(tok, think, next_row(), rng, args.min_q, args.max_q, args.layout)
            with autocast():
                loss, correct = record_loss(model, p, args.device, args.ordinal_weight, teacher, args.distill_weight,
                                            args.distill_temp)
            T = len(p["input_ids"])
            for k, v in (("q_tokens", T), ("attn_pairs", T * T)):
                cost[k] += v
                cost["useful_" + k] += v
            if not torch.isfinite(loss):
                print("non-finite loss, skipping record", flush=True)
                continue
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += loss.item()
            run_acc += sum(correct)
            run_n += len(correct)
            seen += 1
            if seen % args.grad_accum != 0:
                continue
            optimizer_step()
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
            acc = validate(val)
            rec = {"step": step, "val_acc": acc}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            if acc > best:
                best = acc
                model.save_pretrained(os.path.join(args.out, "best"))
                json.dump({"step": step, "val_acc": acc}, open(os.path.join(args.out, "best", "info.json"), "w"))
    model.save_pretrained(os.path.join(args.out, "last"))
    elapsed = time.time() - t0
    json.dump({"best_val_acc": best, "steps": step, "records": seen, "elapsed_s": elapsed,
               "train_s": elapsed - val_s, "val_s": val_s, "cost": cost,
               "device": torch.cuda.get_device_name() if dev_type == "cuda" else args.device, "args": vars(args)},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"done. best val acc {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
