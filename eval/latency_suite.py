"""Run every system through the same CPU latency protocol on this host.

  python -m eval.latency_suite --out results/latency_mac.jsonl            # laptop
  python -m eval.latency_suite --profile kaggle --out /kaggle/working/... # x86 server CPU

Protocol: 40 gold records, warm process, one record at a time, the 9 core
questions, 4 threads (classical models are single-threaded by nature), one
process per configuration so peak RSS is per system. Configurations whose
files are not present on this host are skipped. Run on an otherwise idle machine.
"""

import argparse
import glob
import os
import subprocess
import sys

ROOTS = ["models", "runs", "kaggle/outputs", "/kaggle/input"]  # local first, then downloaded runs


def find(name):
    for root in ROOTS:
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def configs(profile, threads):
    py = sys.executable
    C = []
    bc = [py, "-m", "eval.bench_cpu", "--questions", "core", "--threads", str(threads)]

    # classical, supervised on the same train split
    for name in ["tfidf", "minilm"]:
        C.append((f"{name}-lr", [py, "-m", "eval.classical", "--only", name, "--out", "/tmp/aerojev-classical"], True))

    # generative JSON with the base model (the "just prompt an LLM" approach)
    base_gguf = find("Qwen3-0.6B-Q8_0.gguf")
    if base_gguf:
        C.append(("gen-json", [py, "-m", "eval.generative", "--gguf", base_gguf, "--limit", "40", "--threads", str(threads),
                               "--out", "/tmp/aerojev-gen", "--label", "Qwen3-0.6B Q8_0 (base)"], True))

    # zero-shot logit readout, HF torch fp32, branch layout
    zs = [("Qwen/Qwen3-0.6B", None)]
    if profile == "kaggle":
        zs += [("Qwen/Qwen2.5-0.5B-Instruct", None), ("Qwen/Qwen3-1.7B", None), ("Qwen/Qwen3.5-0.8B", "unpacked")]
    zs_key = {"Qwen/Qwen3-0.6B": "qwen3-0.6b", "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5-0.5b", "Qwen/Qwen3-1.7B": "qwen3-1.7b",
              "Qwen/Qwen3.5-0.8B": "qwen3.5-0.8b"}
    for model, mode in zs:
        cmd = bc + ["--engine", "hf", "--model", model, "--label", f"{model.split('/')[1]} zero-shot",
                    "--acc-key", f"llm:zs-{zs_key[model]}"]
        C.append((f"zs {model}", cmd + (["--mode", mode] if mode else []), True))

    # decision-trained models: (base, run name, layout, label); HF fp32 rows only where the base
    # weights are local (Qwen3-0.6B on the laptop) or downloadable (Kaggle)
    trained = [("Qwen/Qwen3-0.6B", "qwen3-0.6b", "branch", "Qwen3-0.6B"),
               ("Qwen/Qwen3-0.6B", "qwen3-0.6b-cat", "catalogue", "Qwen3-0.6B"),
               ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b", "branch", "Qwen2.5-0.5B"),
               ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-cat", "catalogue", "Qwen2.5-0.5B"),
               ("Qwen/Qwen3-0.6B", "qwen3-0.6b-cat-v2", "catalogue", "Qwen3-0.6B v2"),
               ("Qwen/Qwen3-1.7B", "qwen3-1.7b-cat-v2", "catalogue", "Qwen3-1.7B v2")]
    for base, run, layout, tag in trained:
        adapter = find(os.path.join(run, "best", "adapter_model.safetensors"))
        if adapter and (profile == "kaggle" or base == "Qwen/Qwen3-0.6B"):
            C.append((f"ft {run} hf", bc + ["--engine", "hf", "--model", base, "--adapter", os.path.dirname(adapter),
                                            "--layout", layout, "--label", f"{tag} LoRA (fp32)",
                                            "--acc-key", f"llm:ft-{run}"], True))
        for q in ["q8_0", "q4_k_m"]:
            g = find(f"aerojev-{run}-{q}.gguf")
            if g:
                C.append((f"ft {run} {q}", bc + ["--engine", "gguf", "--gguf", g, "--tokenizer", base,
                                                 "--layout", layout, "--label", f"{tag} LoRA {q.upper()}",
                                                 "--acc-key", f"llm:gguf-{run}-{q}|llm:ft-{run}"], True))

    # thread scaling and single-question floor for the deployment candidate
    g = find("aerojev-qwen3-0.6b-cat-q4_k_m.gguf")
    if g:
        for t in [1, 2]:
            C.append((f"threads={t}", [py, "-m", "eval.bench_cpu", "--questions", "core", "--threads", str(t), "--engine", "gguf",
                                       "--gguf", g, "--tokenizer", "Qwen/Qwen3-0.6B", "--layout", "catalogue",
                                       "--label", "Qwen3-0.6B LoRA Q4_K_M"], True))
        C.append(("one-question", bc + ["--engine", "gguf", "--gguf", g, "--tokenizer", "Qwen/Qwen3-0.6B", "--layout", "catalogue",
                                        "--label", "Qwen3-0.6B LoRA Q4_K_M"] + ["--questions", "one"], True))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["local", "kaggle"], default="local")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", help="substring filter on config names")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    for name, cmd, _ in configs(args.profile, args.threads):
        if args.only and args.only not in name:
            continue
        if args.list:
            print(name, " ".join(cmd))
            continue
        # every script appends its own unified record to `out`
        if "eval.bench_cpu" in cmd:
            cmd = cmd + ["--out", out]
        else:
            cmd = cmd + ["--bench-out", out]
        print(f"== {name}", flush=True)
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith("{") or "Error" in l]
        print("\n".join(lines[-3:]) if lines else r.stdout[-800:], flush=True)


if __name__ == "__main__":
    main()
