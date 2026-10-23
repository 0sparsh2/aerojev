"""Speed-up ladder report: quality, GPU time and projected CPU time per level.

  python scripts/ladder_report.py --root kaggle/outputs/ladder/out --out results/ladder

Reads runs/<name>/summary.json (train.py) and results/ft-<name>/metrics.json (eval.evaluate), or
results/<name>/metrics.json for the probe. Training compute per record comes from the token and
attention-pair counts train.py logs, so it does not depend on the GPU the run used. CPU hours scale the
measured CPU training times (docs/TRAIN_ON_CPU.md: 20.2 s and 4.35 s per record at 1,055 tokens) by that
compute, so they assume the CPU sustains the same FLOP/s on every configuration. Training only: validation
passes are not included.
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


def train_flops(q_tokens, attn_pairs):
    """Forward + backward with frozen base weights: matmuls 4N per token, attention ~12 d per (query, key) pair
    per layer. LoRA adapters and the readout-only lm_head add under 2 % and are left out."""
    return 4 * N * q_tokens + 12 * LAYERS * D_ATTN * attn_pairs


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
SETS = ["gold", "gold2", "test", "test_v2", "val", "injection"]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, nargs="+", help="kernel out/ directories; the first that has a run wins")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for name, level, what in LEVELS:
        probe = name == "l6-probe"
        root = next((r for r in args.root if os.path.isdir(os.path.join(r, "results", name if probe else f"ft-{name}"))),
                    args.root[0])
        m = load(os.path.join(root, "results", name if probe else f"ft-{name}", "metrics.json"))
        s = None if probe else load(os.path.join(root, "runs", name, "summary.json"))
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
            per = train_flops(c["useful_q_tokens"], c["useful_attn_pairs"]) / s["records"]
            r.update(records=s["records"], gpu_train_min=s["train_s"] / 60, gpu_val_min=s["val_s"] / 60,
                     gflop_per_record=per / 1e9, tokens_per_record=c["useful_q_tokens"] / s["records"],
                     padding_overhead=c["q_tokens"] / c["useful_q_tokens"] - 1, best_val=s["best_val_acc"])
        r["cpu_hours"] = {k: r["records"] * sec * per / REF_FLOPS / 3600 for k, sec in CPU_REF.items()}
        rows.append(r)
    json.dump(rows, open(os.path.join(args.out, "ladder.json"), "w"), indent=1)

    base = next((r for r in rows if r["run"] == "l0-baseline" and not r.get("missing")), None)
    f = lambda x: "–" if x is None else f"{x:.3f}"
    lines = ["| Level | Run | Records | Gold | Gold2 | Test | Test v2 | Val | Injection | Gold, shuffled options "
             "| T4 train min | T4 val min | GFLOP/record | Regular CPU h (est.) | M4 h (est.) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['level']} | {r['run']} | missing | | | | | | | | | | | | |")
            continue
        a = r["acc"]
        lines.append(
            f"| {r['level']} | {r['run']} | {r['records']:,} | " + " | ".join(f(a.get(k)) for k in SETS)
            + f" | {f(r.get('gold_choice_shuffled'))} | {r['gpu_train_min']:.1f} | {r['gpu_val_min']:.1f} | "
              f"{r['gflop_per_record']:.0f} | {r['cpu_hours']['regular CPU']:.1f} | {r['cpu_hours']['M4']:.1f} |")
    lines += ["", "Levels:"] + [f"* **{r['run']}**: {r['what']}" for r in rows]
    if base:
        lines += ["", "Change against L0 (accuracy points; compute ratio):", ""]
        for r in rows:
            if r.get("missing") or r is base:
                continue
            d = " ".join(f"{k} {100 * (r['acc'].get(k, 0) - base['acc'].get(k, 0)):+.1f}" for k in ("gold", "gold2", "test"))
            lines.append(f"* {r['run']}: {d}; {base['gflop_per_record'] * base['records'] / (r['gflop_per_record'] * r['records']):.1f}× "
                         f"less training compute in total")
    open(os.path.join(args.out, "ladder.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
