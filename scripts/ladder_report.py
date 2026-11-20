"""Speed-up ladder report: quality, GPU time and projected CPU time per level.

  python scripts/ladder_report.py --root kaggle/outputs/ladder/out --out results/ladder
  python scripts/ladder_report.py --levels prune --root kaggle/outputs/prune/out kaggle/outputs/ladder/out \
      --out results/prune

Reads runs/<name>/summary.json (train.py) and results/ft-<name>/metrics.json (eval.evaluate), or
results/<name>/metrics.json for the probe. Training compute per record comes from the token and
attention-pair counts train.py logs, so it does not depend on the GPU the run used. CPU hours scale the
measured CPU training times (docs/TRAIN_ON_CPU.md: 20.2 s and 4.35 s per record at 1,055 tokens) by that
compute, so they assume the CPU sustains the same FLOP/s on every configuration. Training only: validation
passes are not included. Pruned models are costed with their own dimensions (summary.json "model_dims"), and
full fine-tuning with 6 instead of 4 FLOPs per weight and token (weight gradients).
"""

import argparse
import json
import os

# Qwen2.5-0.5B-Instruct
LAYERS, HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 24, 896, 4864, 14, 2, 64
N = LAYERS * (2 * HIDDEN * HEADS * HEAD_DIM + HEADS * HEAD_DIM            # q (+bias), o
              + 2 * (HIDDEN * KV_HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM)  # k, v (+bias)
              + 3 * HIDDEN * INTER + 2 * HIDDEN)                           # mlp, norms
D_ATTN = HEADS * HEAD_DIM
CPU_REF = {"regular CPU": 20.2, "M4": 4.35}  # s per training record, measured at
REF_TOKENS = 1055                            # tokens per record (old catalogue path, one sequence)


def train_flops(q_tokens, attn_pairs, dims=None):
    """Forward + backward: matmuls 4N per token with frozen base weights (LoRA), 6N with full fine-tuning; attention
    ~12 d per (query, key) pair per layer. LoRA adapters and the readout-only lm_head add under 2 % and are left
    out. dims: summary.json "model_dims" of the run (default: the unpruned Qwen2.5-0.5B)."""
    n, layers, d = (dims["block_params"], dims["num_hidden_layers"], dims["attn_width"]) if dims else (N, LAYERS, D_ATTN)
    return (6 if dims and dims.get("full_ft") else 4) * n * q_tokens + 12 * layers * d * attn_pairs


def forward_flops(q_tokens, attn_pairs):
    return 2 * N * q_tokens + 4 * LAYERS * D_ATTN * attn_pairs


REF_FLOPS = train_flops(REF_TOKENS, REF_TOKENS ** 2)

LEVELS = [
    ("l0-baseline", "L0", "today's recipe: one catalogue per record, 5-9 questions, 12k records"),
    ("l1-shared", "L1", "one catalogue per 8-record step (exact gradients)"),
    ("l2-shared-all9", "L2", "L1 + all 9 questions, LoRA dropout 0, cached-catalogue validation"),
    ("l3-6k-lr2e-4", "L3", "L2 on 6k records, LR 2e-4"),
    ("l3-6k-lr5e-4", "L3", "L2 on 6k records, LR 5e-4"),
    ("l3-6k-lr1e-3", "L3", "L2 on 6k records, LR 1e-3"),
    ("l3-6k-distill", "L3", "L2 on 6k records, LR 5e-4, distilled from fine-tuned Qwen3-1.7B"),
    ("l4-nocat-12k", "L4", "no catalogue in the prompt, all 9 questions, 12k records"),
    ("l4-nocat-6k", "L4", "no catalogue in the prompt, all 9 questions, 6k records, LR 5e-4"),
    ("l4-nocat-6k-seed1", "L4", "the 6k run again with another seed (run-to-run noise)"),
    ("l4-nocat-4.8k", "L4", "no catalogue, 4.8k records, LR 5e-4"),
    ("l4-nocat-3k", "L4", "no catalogue, 3k records, LR 5e-4"),
    ("l6-probe", "L6", "frozen model + linear heads (no LoRA), features from one forward pass"),
]
# pruning study: the reference is the reliable catalogue-free recipe; eval-only rows were not retrained
PRUNE_LEVELS = [
    ("l4-nocat-12k", "P0", "reference: full Qwen2.5-0.5B, catalogue-free, 12k records, LoRA"),
    ("l4-nocat-12k-seed1", "P0", "the reference again with another seed (run-to-run noise)"),
    ("p-d16top", "P1", "16 blocks: the top 8 removed; LoRA, 12k records"),
    ("p-d16auto", "P1", "16 blocks: the 8 most alike consecutive blocks removed (final block kept); LoRA, 12k"),
    ("p-w2560", "P2", "24 blocks, FFN narrowed 4864 -> 2560 (activation x weight scores); LoRA, 12k"),
    ("p-d16w", "P3", "16 blocks (top 8 removed) + FFN 2560; LoRA, 12k"),
    ("p-d16w-full-2e-5", "P3", "16 blocks + FFN 2560; full fine-tuning, LR 2e-5, 12k"),
    ("p-d16w-full-6e-5", "P3", "16 blocks + FFN 2560; full fine-tuning, LR 6e-5, 12k"),
    ("t-d16-noheal", "T1", "the trained reference with its top 8 blocks removed, no retraining"),
    ("t-d16w-noheal", "T2", "the trained reference, top 8 blocks removed + FFN 2560, no retraining"),
    ("t-d16w-heal3k", "T3", "T2 retrained ('healed') with 3k records of LoRA (cost on top of the reference run)"),
]
# smallest accurate model; references from the ladder and pruning runs
BAKEOFF_LEVELS = [
    ("l4-nocat-12k", "R", "reference: Qwen2.5-0.5B (494M), catalogue-free, LoRA"),
    ("l4-nocat-12k-seed1", "R", "the reference, second seed"),
    ("p-d16w-full-2e-5", "Q", "Qwen2.5-0.5B pruned to 16 blocks + FFN 2560 (276M), full fine-tuning"),
    ("p-d16w-full-2e-5-seed1", "Q", "the pruned model, second seed"),
    ("b-gemma3-270m-3e-5", "D", "Gemma 3 270M (268M, 168M of it vocabulary), full fine-tuning, LR 3e-5"),
    ("b-gemma3-270m-1e-4", "D", "Gemma 3 270M, LR 1e-4"),
    ("b-smollm2-135m-3e-5", "D", "SmolLM2-135M (135M), full fine-tuning, LR 3e-5"),
    ("b-smollm2-135m-1e-4", "D", "SmolLM2-135M, LR 1e-4"),
    ("b-ettin-150m", "E", "Ettin encoder 150M + 9 heads, 2 epochs, LR 5e-5"),
    ("b-ettin-68m", "E", "Ettin encoder 68M + 9 heads, 2 epochs, LR 8e-5"),
    ("b-ettin-32m", "E", "Ettin encoder 32M + 9 heads, 2 epochs, LR 1e-4"),
]
# Ettin-150M encoder round; the bake-off encoder (2 epochs, LR 5e-5, mean pooling) is the starting point
ENCODER_LEVELS = [
    ("l4-nocat-12k", "R", "reference: Qwen2.5-0.5B (494M), catalogue-free, LoRA"),
    ("l4-nocat-12k-seed1", "R", "the reference, second seed"),
    ("b-ettin-150m", "E0", "Ettin-150M + 9 heads, 2 epochs, LR 5e-5, mean pooling (bake-off)"),
    ("e150-seed1", "E0", "the same, second seed"),
    ("e150-ep4", "E1", "4 epochs"),
    ("e150-ep4-seed1", "E1", "4 epochs, second seed"),
    ("e150-ep8", "E1", "8 epochs"),
    ("e150-ep4-lr1e-4", "E2", "4 epochs, LR 1e-4"),
    ("e150-ep4-attn", "E3", "4 epochs, one attention pooling per question"),
    ("e150-ep4-kd", "E4", "4 epochs, distilled from the two reference runs (averaged)"),
    ("e150-ep4-attn-kd", "E5", "4 epochs, attention pooling + distillation"),
]
ULTRA_LEVELS = [
    ("e150-ep8", "E", "Ettin-150M, 8 epochs, LR 5e-5 (shipped, from the encoder round)"),
    ("e150-ep8-seed1", "E", "the same, second seed"),
    ("u150-ep8-lr1e-5", "U1", "UltraBERT-150M, 8 epochs, LR 1e-5 (its card's MNLI best)"),
    ("u150-ep8-lr1e-5-seed1", "U1", "the same, second seed"),
    ("u150-ep8-lr3e-5", "U3", "UltraBERT-150M, 8 epochs, LR 3e-5"),
    ("u150-ep8-lr3e-5-seed1", "U3", "the same, second seed"),
]
SETS = ["gold", "gold2", "test", "test_v2", "val", "injection"]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, nargs="+", help="kernel out/ directories; the first that has a run wins")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", choices=["ladder", "prune", "bakeoff", "encoder", "ultra"], default="ladder")
    ap.add_argument("--ref", help="reference run for the 'change against' list (default: the level set's own)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    levels, ref_name = {"prune": (PRUNE_LEVELS, "l4-nocat-12k"), "bakeoff": (BAKEOFF_LEVELS, "l4-nocat-12k"),
                        "encoder": (ENCODER_LEVELS, "l4-nocat-12k"), "ultra": (ULTRA_LEVELS, "e150-ep8"), "ladder": (LEVELS, "l0-baseline")}[args.levels]
    ref_name = args.ref or ref_name
    rows = []
    for name, level, what in levels:
        probe = name == "l6-probe"
        eval_only = name.endswith("-noheal")  # scored as pruned, with no training run of its own
        res = name if (probe or eval_only) else f"ft-{name}"
        root = next((r for r in args.root if os.path.isdir(os.path.join(r, "results", res))), args.root[0])
        m = load(os.path.join(root, "results", res, "metrics.json"))
        s = None if (probe or eval_only) else load(os.path.join(root, "runs", name, "summary.json"))
        if eval_only and m is not None:
            rows.append({"run": name, "level": level, "what": what, "eval_only": True,
                         "acc": {k: v["raw"]["core_mean_accuracy"] for k, v in m["sets"].items() if k in SETS}})
            continue
        if m is None or (s is None and not probe):
            rows.append({"run": name, "level": level, "what": what, "missing": True})
            continue
        r = {"run": name, "level": level, "what": what,
             "acc": {k: v["raw"]["core_mean_accuracy"] for k, v in m["sets"].items() if k in SETS}}
        if "gold_permutation" in m:
            r["gold_choice_shuffled"] = m["gold_permutation"]["choice_accuracy_shuffled"]
        if probe:
            p = m["probe"]
            c = p["cost"]
            per = forward_flops(c["tokens"], c["attn_pairs"]) / c["records"]
            r.update(records=10000, gpu_train_min=(p["extract_s"] + p["fit_s"]) / 60, gpu_val_min=0.0,
                     gflop_per_record=per / 1e9, forward_only=True)
        else:
            c = s["cost"]
            per = train_flops(c["useful_q_tokens"], c["useful_attn_pairs"], s.get("model_dims")) / s["records"]
            if s.get("model_dims"):
                r["block_params_m"] = s["model_dims"]["block_params"] / 1e6
            r.update(records=s["records"], gpu_train_min=s["train_s"] / 60, gpu_val_min=s["val_s"] / 60,
                     gflop_per_record=per / 1e9, tokens_per_record=c["useful_q_tokens"] / s["records"],
                     padding_overhead=c["q_tokens"] / c["useful_q_tokens"] - 1, best_val=s["best_val_acc"])
        r["cpu_hours"] = {k: r["records"] * sec * per / REF_FLOPS / 3600 for k, sec in CPU_REF.items()}
        rows.append(r)
    json.dump(rows, open(os.path.join(args.out, f"{args.levels}.json"), "w"), indent=1)

    base = next((r for r in rows if r["run"] == ref_name and not r.get("missing")), None)
    f = lambda x: "–" if x is None else f"{x:.3f}"
    lines = ["| Level | Run | Records | Gold | Gold2 | Test | Test v2 | Val | Injection | Gold, shuffled options "
             "| T4 train min | T4 val min | GFLOP/record | Regular CPU h (est.) | M4 h (est.) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['level']} | {r['run']} | missing | | | | | | | | | | | | |")
            continue
        if r.get("eval_only"):
            a = r["acc"]
            lines.append(f"| {r['level']} | {r['run']} | no training | " + " | ".join(f(a.get(k)) for k in SETS)
                         + " | – | – | – | – | – | – |")
            continue
        a = r["acc"]
        lines.append(
            f"| {r['level']} | {r['run']} | {r['records']:,} | " + " | ".join(f(a.get(k)) for k in SETS)
            + f" | {f(r.get('gold_choice_shuffled'))} | {r['gpu_train_min']:.1f} | {r['gpu_val_min']:.1f} | "
              f"{r['gflop_per_record']:.0f} | {r['cpu_hours']['regular CPU']:.1f} | {r['cpu_hours']['M4']:.1f} |")
    lines += ["", "Levels:"] + [f"* **{r['run']}**: {r['what']}" for r in rows]
    if base:
        lines += ["", f"Change against {base['level']} (accuracy points; compute ratio):", ""]
        for r in rows:
            if r.get("missing") or r is base:
                continue
            d = " ".join(f"{k} {100 * (r['acc'].get(k, 0) - base['acc'].get(k, 0)):+.1f}" for k in ("gold", "gold2", "test"))
            x = 1.0 if r.get("eval_only") else \
                base['gflop_per_record'] * base['records'] / (r['gflop_per_record'] * r['records'])
            ratio = "" if r.get("eval_only") else (
                "; the same training compute in total" if 0.95 <= x <= 1.05 else
                f"; {x:.1f}× less training compute in total" if x > 1 else
                f"; {1 / x:.1f}× more training compute in total")
            lines.append(f"* {r['run']}: {d}{ratio}")
    open(os.path.join(args.out, f"{args.levels}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
