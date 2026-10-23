"""Frozen-model probes: no LoRA, no backprop through the model.

  python scripts/probe.py --model Qwen/Qwen2.5-0.5B-Instruct --data data/processed_v2 --out results/probe

One forward pass per record (short system line + record, no catalogue) collects the last-token and
mean-pooled hidden states of a few layers. A linear head per core question is then trained on the
training set's features, with weight decay picked on val. Writes preds_<set>.jsonl and metrics.json in
the same format as eval.evaluate, plus the extraction cost (forward-only tokens) for CPU projections.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerojev import schema  # noqa: E402
from aerojev.metrics import summarize  # noqa: E402
from aerojev.prompt import MAX_RECORD_TOKENS, fit_state, materialize, render_record, render_static_nocat, think_block  # noqa: E402

CORE = schema.TRAIN_QUESTIONS
SETS = ["train", "val", "test", "test_v2", "gold", "gold2", "injection"]


@torch.no_grad()
def features(model, tok, think, rows, layers, device, dtype, batch):
    enc = lambda t: tok.encode(t, add_special_tokens=False)
    static = enc(render_static_nocat())
    seqs = [static + enc(render_record(fit_state(tok, r["state"], MAX_RECORD_TOKENS), think)) for r in rows]
    out, tokens, pairs = [], 0, 0
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # length-sorted batches waste little padding
    feats = [None] * len(seqs)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        L = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros(len(idx), L, dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
            att[j, :len(seqs[i])] = 1
            tokens += len(seqs[i])
            pairs += len(seqs[i]) * (len(seqs[i]) + 1) // 2
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
            hs = model(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True,
                       logits_to_keep=1).hidden_states
        att = att.to(device).unsqueeze(-1).float()
        last = att.squeeze(-1).sum(1).long() - 1
        parts = []
        for l in layers:
            h = hs[l].float()
            parts.append(h[torch.arange(len(idx), device=device), last])
            parts.append((h * att).sum(1) / att.sum(1))
        f = torch.cat(parts, -1).cpu()
        for j, i in enumerate(idx):
            feats[i] = f[j]
    return torch.stack(feats), tokens, pairs


def fit_head(x, y, n_cls, wd, steps, lr, device):
    head = torch.nn.Linear(x.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(head(x), y).backward()
        opt.step()
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="6,12,18,24", help="hidden_states indices (0 = embeddings)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400, help="full-batch AdamW steps per head")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wds", default="0.0001,0.001,0.01,0.1")
    ap.add_argument("--limit", type=int, help="first N records per set (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    think = think_block(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    model.to(args.device).eval()
    layers = [int(x) for x in args.layers.split(",")]

    data, feats, cost = {}, {}, {"tokens": 0, "attn_pairs": 0, "records": 0}
    t0 = time.time()
    for s in SETS:
        rows = [json.loads(l) for l in open(os.path.join(args.data, f"{s}.jsonl"))][: args.limit]
        data[s] = rows
        t = time.time()
        feats[s], tok_n, pairs = features(model, tok, think, rows, layers, args.device, dtype, args.batch)
        cost["tokens"] += tok_n
        cost["attn_pairs"] += pairs
        cost["records"] += len(rows)
        print(f"[{s}] {len(rows)} records, {time.time() - t:.0f}s, {tok_n / len(rows):.0f} tokens/record", flush=True)
    extract_s = time.time() - t0
    mu, sd = feats["train"].mean(0), feats["train"].std(0) + 1e-6
    X = {s: ((f - mu) / sd).to(args.device) for s, f in feats.items()}

    t1 = time.time()
    keysets = {q.id: materialize(q)[1] for q in CORE}
    heads, chosen = {}, {}
    for q in CORE:
        keys = keysets[q.id]
        y = {s: torch.tensor([keys.index(r["labels"][q.id]) for r in data[s]], device=args.device) for s in ("train", "val")}
        best = None
        for wd in [float(w) for w in args.wds.split(",")]:
            h = fit_head(X["train"], y["train"], len(keys), wd, args.steps, args.lr, args.device)
            with torch.no_grad():
                acc = (h(X["val"]).argmax(-1) == y["val"]).float().mean().item()
            if best is None or acc > best[0]:
                best = (acc, wd, h)
        heads[q.id], chosen[q.id] = best[2], {"weight_decay": best[1], "val_acc": best[0]}
        print(f"  {q.id:20s} wd={best[1]:g} val acc={best[0]:.3f}", flush=True)
    fit_s = time.time() - t1

    results = {"model": args.model, "adapter": None, "layout": "probe", "sets": {},
               "probe": {"layers": layers, "heads": chosen, "extract_s": extract_s, "fit_s": fit_s, "cost": cost,
                         "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device}}
    for s in SETS[1:]:
        preds = []
        with torch.no_grad():
            logits = {q.id: heads[q.id](X[s]).float().cpu() for q in CORE}
        for i, r in enumerate(data[s]):
            for q in CORE:
                preds.append({"rid": r["id"], "qid": q.id, "type": q.type, "keys": keysets[q.id],
                              "logits": [round(x, 4) for x in logits[q.id][i].tolist()], "gold": r["labels"][q.id],
                              "format": r.get("format")})
        with open(os.path.join(args.out, f"preds_{s}.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        results["sets"][s] = {"raw": summarize(preds, [q.id for q in CORE])}
        print(f"== {s}: core acc {results['sets'][s]['raw']['core_mean_accuracy']:.3f}", flush=True)
    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
