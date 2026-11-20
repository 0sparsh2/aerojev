"""Structured pruning of a Qwen2 decision model: drop transformer blocks and/or narrow the FFNs.

  python scripts/prune.py --drop top:8 --ffn 2560 --out /tmp/pruned/d16-w2560
  python scripts/prune.py --adapter runs/l4-nocat-12k/best --drop auto:8 --out /tmp/pruned/tuned-d16
  python scripts/prune.py --from-indices <pruned dir>/prune_info.json --out models/pruned   # rebuild exactly

The result is a standard Qwen2 checkpoint (fewer layers, smaller intermediate_size), so train.train, eval.evaluate
and the GGUF export work on it unchanged. Calibration uses the catalogue-free prompt (system line + record + the 9
readouts) of --calib training records, i.e. the tokens the model will actually see.

--drop top:N   remove the last N blocks
--drop auto:N  remove the N consecutive blocks whose input and output are most alike (angular distance at the
               readout positions, Gromov et al. 2024); the final block is always kept
--ffn K        keep the K FFN channels per block with the largest ||activation||_2 * ||down_proj column||_2
               (Wanda-style), measured on the calibration prompts after any block removal
--adapter      merge a LoRA adapter first (prune a task-tuned model instead of the base)

prune_info.json records the kept blocks and kept FFN channels, so --from-indices rebuilds the same model from the
same base (and adapter) without calibration.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerojev import schema  # noqa: E402
from aerojev.prompt import materialize, pack, think_block  # noqa: E402


def load(model_name, adapter, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        adapter = os.path.dirname(adapter) if os.path.isfile(adapter) else adapter
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.to(device).eval()


def calibration(tok, rows):
    """Catalogue-free prompts (tree-packed, all 9 readouts) for the calibration records."""
    think = think_block(tok)
    qs = [materialize(q)[0] for q in schema.TRAIN_QUESTIONS]
    return [pack(tok, r["state"], qs, think, layout="nocat") for r in rows]


def forward(model, p, device, **kw):
    return model(input_ids=p["input_ids"][None].to(device), position_ids=p["position_ids"][None].to(device),
                 attention_mask=p["mask"][None, None].to(device),
                 logits_to_keep=torch.tensor(p["readouts"], device=device), **kw)


@torch.no_grad()
def block_distances(model, calib, n, device):
    """Mean angular distance between the input of block l and the input of block l+n, at the readout positions,
    for every start l whose block of n leaves the final layer in place."""
    L = model.config.num_hidden_layers
    starts = list(range(0, L - n))  # blocks l .. l+n-1, all before the final layer
    tot = torch.zeros(len(starts), dtype=torch.float64)
    for p in calib:
        hs = forward(model, p, device, output_hidden_states=True).hidden_states  # hs[k] = input of block k
        idx = torch.tensor(p["readouts"], device=device)
        for j, l in enumerate(starts):
            a, b = hs[l][0, idx].float(), hs[l + n][0, idx].float()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
            tot[j] += (torch.arccos(cos) / math.pi).mean().item()
    return {l: tot[j].item() / len(calib) for j, l in enumerate(starts)}


def drop_blocks(model, keep):
    """Keep the given block indices, renumbering them so the KV cache and per-layer config line up."""
    layers = torch.nn.ModuleList([model.model.layers[i] for i in keep])
    for new, layer in enumerate(layers):
        layer.self_attn.layer_idx = new
    model.model.layers = layers
    cfg = model.config
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = [cfg.layer_types[i] for i in keep]
    cfg.num_hidden_layers = len(keep)
    if hasattr(cfg, "max_window_layers"):
        cfg.max_window_layers = min(cfg.max_window_layers, len(keep))


@torch.no_grad()
def ffn_scores(model, calib, device):
    """Per block: sqrt(sum over calibration tokens of act^2) * ||down_proj column||, act = down_proj's input."""
    sq = [None] * len(model.model.layers)
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def hook(mod, inp, out, i=i):
            a = inp[0].reshape(-1, inp[0].shape[-1]).float()
            s = (a * a).sum(0)
            sq[i] = s if sq[i] is None else sq[i] + s
        hooks.append(layer.mlp.down_proj.register_forward_hook(hook))
    for p in calib:
        forward(model, p, device)
    for h in hooks:
        h.remove()
    return [sq[i].sqrt() * layer.mlp.down_proj.weight.float().norm(dim=0)
            for i, layer in enumerate(model.model.layers)]


def narrow_ffn(model, keep_per_layer):
    for layer, keep in zip(model.model.layers, keep_per_layer):
        mlp = layer.mlp
        idx = torch.tensor(keep, device=mlp.down_proj.weight.device)
        for name, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
            old = getattr(mlp, name)
            w = old.weight.index_select(dim, idx).clone()
            new = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype, device=w.device)
            new.weight.data = w
            setattr(mlp, name, new)
        mlp.intermediate_size = len(keep)
    model.config.intermediate_size = len(keep_per_layer[0])


def counts(model):
    layers = sum(p.numel() for p in model.model.layers.parameters()) + sum(p.numel() for p in model.model.norm.parameters())
    emb = model.model.embed_tokens.weight.numel()
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    return {"blocks_and_norm": layers, "embeddings": emb, "lm_head": 0 if tied else model.lm_head.weight.numel(),
            "total": layers + emb + (0 if tied else model.lm_head.weight.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter to merge before pruning (directory or a file inside it)")
    ap.add_argument("--drop", default="none", help="none | top:N | auto:N")
    ap.add_argument("--ffn", type=int, default=0, help="FFN channels to keep per block (0 = all)")
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--calib", type=int, default=256, help="calibration records from the training split")
    ap.add_argument("--from-indices", help="prune_info.json of an earlier run: rebuild it without calibration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    info = json.load(open(args.from_indices)) if args.from_indices else None
    if info:
        args.model, args.adapter = info["model"], args.adapter or info.get("adapter")
    tok, model = load(args.model, args.adapter, args.device)
    before = counts(model)
    L = model.config.num_hidden_layers

    if info:
        drop_blocks(model, info["kept_blocks"])
        if info.get("ffn_channels"):
            narrow_ffn(model, info["ffn_channels"])
    else:
        rows = [json.loads(l) for l in open(os.path.join(args.data, "train.jsonl"))][: args.calib]
        calib = calibration(tok, rows)
        info = {"model": args.model, "adapter": args.adapter, "drop": args.drop, "ffn": args.ffn,
                "calib_records": len(calib), "kept_blocks": list(range(L)), "ffn_channels": None}
        kind, _, n = args.drop.partition(":")
        if kind in ("top", "auto"):
            n = int(n)
            if kind == "top":
                start = L - n
            else:
                dist = block_distances(model, calib, n, args.device)
                start = min(dist, key=dist.get)
                info["block_distances"] = {str(k): round(v, 4) for k, v in dist.items()}
            info["kept_blocks"] = [i for i in range(L) if not start <= i < start + n]
            drop_blocks(model, info["kept_blocks"])
            print(f"dropped blocks {start}..{start + n - 1}; kept {info['kept_blocks']}", flush=True)
        if args.ffn:
            scores = ffn_scores(model, calib, args.device)
            info["ffn_channels"] = [sorted(torch.topk(s, args.ffn).indices.tolist()) for s in scores]
            kept_share = [float(torch.topk(s, args.ffn).values.sum() / s.sum()) for s in scores]
            info["ffn_score_share_kept"] = [round(x, 4) for x in kept_share]
            narrow_ffn(model, info["ffn_channels"])
            print(f"FFN {model.config.intermediate_size} channels per block; share of importance kept: "
                  f"min {min(kept_share):.3f}, mean {sum(kept_share) / len(kept_share):.3f}", flush=True)

    info["params_before"], info["params_after"] = before, counts(model)
    info["config"] = {"num_hidden_layers": model.config.num_hidden_layers,
                      "intermediate_size": model.config.intermediate_size}
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump(info, open(os.path.join(args.out, "prune_info.json"), "w"))

    # the saved checkpoint must reload as a plain Qwen2 model and give the same readout logits
    from transformers import AutoModelForCausalLM
    re = AutoModelForCausalLM.from_pretrained(args.out, dtype=torch.float32, attn_implementation="sdpa").to(args.device).eval()
    p = calibration(tok, [json.loads(open(os.path.join(args.data, "val.jsonl")).readline())])[0]
    with torch.no_grad():
        diff = (forward(model, p, args.device).logits - forward(re, p, args.device).logits).abs().max().item()
    a, b = before["blocks_and_norm"], info["params_after"]["blocks_and_norm"]
    print(f"saved {args.out}: {info['config']}, block parameters {a / 1e6:.1f}M -> {b / 1e6:.1f}M "
          f"({a / b:.2f}x fewer), total {before['total'] / 1e6:.1f}M -> {info['params_after']['total'] / 1e6:.1f}M; "
          f"reload max logit diff {diff:.1e}", flush=True)
    assert diff < 1e-3, "reloaded checkpoint differs from the pruned model"


if __name__ == "__main__":
    main()
