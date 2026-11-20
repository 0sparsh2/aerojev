"""Encoder decision model: one pass over the record, one classification head per trained question.

  python -m train.train_encoder --model jhu-clsp/ettin-encoder-150m --out runs/ettin-150m --results results/ft-ettin-150m

The catalogue-free task as plain multi-head classification: the record text goes through a bidirectional encoder
(ModernBERT / Ettin), the token states are pooled, and each of the 9 questions has its own linear head over its
options in schema order. Loss per record = mean over questions of cross-entropy, plus the ordinal penalty on score
questions, as in train.train. Every weight trains (the encoders are small).

Pooling (--pool): "mean" averages the token states; "attn" gives every question its own learned attention over the
tokens (one scoring vector per question, initialised to zero so training starts from the mean), so e.g. the delay
question can look at the dates and the event question at the verbs.

Distillation (--teacher): eval.evaluate preds files over the training set (comma-separated: several teachers are
averaged in log-probability). Adds distill_weight * T^2 * KL(teacher_T || student_T) per question, as train.train.

Writes, like train.train + eval.evaluate do:
  <out>/summary.json, <out>/train_log.jsonl, <out>/model/ (fp16 weights, heads, tokenizer, label map)
  <results>/preds_<set>.jsonl and <results>/metrics.json (with temperatures fitted on val)
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from aerojev import schema
from aerojev.metrics import fit_temperatures, summarize
from aerojev.prompt import materialize

CORE = schema.TRAIN_QUESTIONS
KEYS = {q.id: materialize(q)[1] for q in CORE}  # option keys in schema order = head output order
TYPES = {q.id: q.type for q in CORE}


class EncoderDecider(nn.Module):
    def __init__(self, name, pool="mean", **load_kw):
        super().__init__()
        load_kw.setdefault("dtype", torch.float32)  # train in fp32 (autocast on GPU) even from an fp16 checkpoint
        self.encoder = AutoModel.from_pretrained(name, **load_kw)
        h = self.encoder.config.hidden_size
        self.pool = pool
        self.heads = nn.ModuleDict({qid: nn.Linear(h, len(keys)) for qid, keys in KEYS.items()})
        if pool == "attn":
            self.attn = nn.Linear(h, len(KEYS))  # token score per question
            nn.init.zeros_(self.attn.weight)
            nn.init.zeros_(self.attn.bias)

    @classmethod
    def load(cls, path, **load_kw):
        """A saved model/ directory (fp16 on disk) -> fp32 module."""
        meta = json.load(open(os.path.join(path, "decider.json")))
        m = cls(path, meta.get("pool", "mean"), **load_kw)
        state = torch.load(os.path.join(path, "heads.pt"), map_location="cpu")
        m.heads.load_state_dict({k: v.float() for k, v in state.items() if not k.startswith("_attn.")})
        if m.pool == "attn":
            m.attn.load_state_dict({k[len("_attn."):]: v.float() for k, v in state.items() if k.startswith("_attn.")})
        return m.eval()

    def head_params(self):
        return list(self.heads.parameters()) + (list(self.attn.parameters()) if self.pool == "attn" else [])

    def forward(self, input_ids, attention_mask):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pool == "attn":
            sc = self.attn(hs).float().masked_fill(attention_mask[..., None] == 0, -1e4)  # [B, T, Q]
            pooled = torch.einsum("btq,bth->bqh", sc.softmax(1).to(hs.dtype), hs)
            return {qid: head(pooled[:, i]).float() for i, (qid, head) in enumerate(self.heads.items())}
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * m).sum(1) / m.sum(1)
        return {qid: head(pooled).float() for qid, head in self.heads.items()}


def load_teachers(paths):
    """eval.evaluate preds files -> {(record id, question id): {option key: log-prob}}, averaged over the files."""
    files, t = paths.split(","), {}
    for path in files:
        for line in open(path):
            p = json.loads(line)
            d = t.setdefault((p["rid"], p["qid"]), {})
            for k, v in zip(p["keys"], F.log_softmax(torch.tensor(p["logits"]), -1).tolist()):
                d[k] = d.get(k, 0.0) + v / len(files)
    return t


def batch_loss(logits, rows, ordinal_weight, teacher=None, distill_weight=0.0, distill_temp=2.0):
    losses, correct = [], 0
    for qid, lg in logits.items():
        gold = torch.tensor([KEYS[qid].index(r["labels"][qid]) for r in rows], device=lg.device)
        logp = F.log_softmax(lg, -1)
        loss = F.nll_loss(logp, gold, reduction="none")
        if TYPES[qid] == "score" and ordinal_weight > 0:
            n = lg.shape[-1]
            dist = (torch.arange(n, device=lg.device)[None] - gold[:, None]).abs().float() / (n - 1)
            loss = loss + ordinal_weight * (logp.exp() * dist).sum(-1)
        if teacher is not None and distill_weight > 0:
            tl = torch.tensor([[teacher[(r["id"], qid)][k] for k in KEYS[qid]] for r in rows], device=lg.device)
            pt = F.softmax(tl / distill_temp, -1)
            kl = (pt * (pt.clamp_min(1e-12).log() - F.log_softmax(lg / distill_temp, -1))).sum(-1)
            loss = loss + distill_weight * distill_temp ** 2 * kl
        losses.append(loss)
        correct += int((lg.argmax(-1) == gold).sum())
    return torch.stack(losses).mean(), correct  # mean over questions and records


@torch.no_grad()
def predict(model, tok, rows, device, max_len, batch=64):
    model.eval()
    preds, t0 = [], time.time()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([r["state"] for r in chunk], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
            logits = model(enc["input_ids"], enc["attention_mask"])
        for j, r in enumerate(chunk):
            for qid in KEYS:
                preds.append({"rid": r["id"], "qid": qid, "type": TYPES[qid], "keys": KEYS[qid],
                              "logits": [round(x, 4) for x in logits[qid][j].tolist()], "gold": r["labels"][qid],
                              "format": r.get("format")})
    model.train()
    return preds, (time.time() - t0) / max(1, len(rows))


def save_decider(model, tok, mdir, base, max_len):
    """Save in fp16 with everything needed to run it: encoder weights, heads (+ pooling scores), tokenizer and
    decider.json (question ids, option keys in head order, max length, pooling). EncoderDecider.load reads it back;
    scripts/export_encoder.py turns it into ONNX."""
    os.makedirs(mdir, exist_ok=True)
    model.encoder.half().save_pretrained(mdir)
    model.encoder.float()
    heads = dict(model.heads.state_dict())
    if model.pool == "attn":
        heads.update({f"_attn.{k}": v for k, v in model.attn.state_dict().items()})
    torch.save({k: v.half().cpu() for k, v in heads.items()}, os.path.join(mdir, "heads.pt"))
    tok.save_pretrained(mdir)
    json.dump({"base": base, "keys": KEYS, "types": TYPES, "max_len": max_len, "pool": model.pool},
              open(os.path.join(mdir, "decider.json"), "w"), indent=1)


def accuracy(preds):
    return sum(p["keys"][max(range(len(p["logits"])), key=p["logits"].__getitem__)] == p["gold"] for p in preds) / len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3, help="the freshly initialized heads learn faster")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--ordinal-weight", type=float, default=0.5)
    ap.add_argument("--pool", choices=["mean", "attn"], default="mean")
    ap.add_argument("--teacher", help="eval.evaluate preds file(s) over the training set, comma-separated")
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps")
    ap.add_argument("--val-limit", type=int, default=150)
    ap.add_argument("--sets", default="val,test,test_v2,gold,gold2,injection")
    ap.add_argument("--limit", type=int, help="train on the first N records (smoke tests)")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = EncoderDecider(args.model, args.pool).to(args.device).train()
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    emb = model.encoder.get_input_embeddings().weight.numel()
    print(f"{args.model}: {enc_params / 1e6:.1f}M encoder parameters ({emb / 1e6:.1f}M embeddings)", flush=True)

    load = lambda name: [json.loads(l) for l in open(os.path.join(args.data, f"{name}.jsonl"))]
    train = load("train")[: args.limit] if args.limit else load("train")
    val = load("val")[: args.val_limit]
    teacher = load_teachers(args.teacher) if args.teacher else None
    if teacher:
        miss = sum((r["id"], q) not in teacher for r in train for q in KEYS)
        print(f"teacher: {len(teacher)} (record, question) targets, {miss} missing", flush=True)
        assert miss == 0, "the teacher must cover every training record"
    total_steps = args.max_steps or math.ceil(len(train) * args.epochs / args.batch)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                             {"params": model.head_params(), "lr": args.head_lr}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    use_amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    best, best_state, step, seen, t0, val_s = -1.0, None, 0, 0, time.time(), 0.0
    cost = {"q_tokens": 0, "attn_pairs": 0}
    order, run_loss, run_acc, run_n, bad = [], 0.0, 0, 0, 0
    while step < total_steps:
        if len(order) < args.batch:
            order += rng.sample(range(len(train)), len(train))
        rows = [train[order.pop()] for _ in range(args.batch)]
        enc = tok([r["state"] for r in rows], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        lens = enc["attention_mask"].sum(1).tolist()
        cost["q_tokens"] += sum(lens)
        cost["attn_pairs"] += sum(n * n for n in lens)  # bidirectional attention over each record
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device))
        loss, correct = batch_loss(logits, rows, args.ordinal_weight, teacher, args.distill_weight, args.distill_temp)
        bad = 0 if torch.isfinite(loss) else bad + 1  # fp16 autocast on a bf16-pretrained model can overflow
        if bad >= 20:
            raise RuntimeError(f"loss not finite for {bad} steps in a row (step {step}): fp16 overflow?")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        seen += len(rows)
        run_loss += loss.item()
        run_acc += correct
        run_n += len(rows) * len(KEYS)
        if step % 50 == 0:
            rec = {"step": step, "records": seen, "loss": run_loss / 50, "train_acc": run_acc / run_n,
                   "elapsed_s": time.time() - t0}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            run_loss, run_acc, run_n = 0.0, 0, 0
        if step % args.eval_every == 0 or step == total_steps:
            t = time.time()
            acc = accuracy(predict(model, tok, val, args.device, args.max_len)[0])
            val_s += time.time() - t
            log.write(json.dumps({"step": step, "val_acc": acc}) + "\n")
            print(json.dumps({"step": step, "val_acc": acc}), flush=True)
            if acc > best:
                best = acc
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    save_decider(model, tok, os.path.join(args.out, "model"), args.model, args.max_len)

    results = {"model": args.model, "adapter": None, "layout": "encoder", "sets": {}}
    all_preds = {}
    for name in args.sets.split(","):
        preds, spr = predict(model, tok, load(name), args.device, args.max_len)
        all_preds[name] = preds
        with open(os.path.join(args.results, f"preds_{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in preds)
        results["sets"][name] = {"sec_per_record": spr}
    temps = fit_temperatures(all_preds["val"]) if "val" in all_preds else {}
    results["temperatures"] = temps
    core = [q.id for q in CORE]
    for name, preds in all_preds.items():
        results["sets"][name]["raw"] = summarize(preds, core)
        if temps:
            results["sets"][name]["calibrated"] = summarize(preds, core, temps)
        raw = results["sets"][name]["raw"]
        print(f"== {name}: core acc {raw['core_mean_accuracy']:.3f}  macroF1 {raw['core_mean_macro_f1']:.3f}  "
              f"exact {raw['record_exact_match']:.3f}", flush=True)
    json.dump(results, open(os.path.join(args.results, "metrics.json"), "w"), indent=1)

    cfg = model.encoder.config
    json.dump({"best_val_acc": best, "steps": step, "records": seen, "elapsed_s": elapsed, "train_s": elapsed - val_s,
               "val_s": val_s, "cost": {**cost, "useful_q_tokens": cost["q_tokens"], "useful_attn_pairs": cost["attn_pairs"]},
               "model_dims": {"num_hidden_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
                              "intermediate_size": cfg.intermediate_size, "attn_width": cfg.hidden_size,
                              "block_params": enc_params - emb, "full_ft": True, "encoder": True},
               "device": torch.cuda.get_device_name() if use_amp else args.device, "args": vars(args)},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"done. best val acc {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
