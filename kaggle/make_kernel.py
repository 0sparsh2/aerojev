"""Bundle code+data into the private Kaggle dataset and write a kernel folder.

  python kaggle/make_kernel.py bundle                 # (re)build kaggle/dataset/aerojev_bundle.tar.gz
  python kaggle/make_kernel.py kernel baselines       # write kaggle/kernels/baselines/
then
  kaggle datasets create -p kaggle/dataset   (first time)  |  kaggle datasets version -p kaggle/dataset -m msg
  kaggle kernels push -p kaggle/kernels/baselines
"""

import json
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "sparshnagpal"
DATASET = f"{USER}/aerojev-bundle"

MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

def _ladder(smoke=False):
    """CPU speed-up ladder (docs/CPU_SPEEDUP_RESEARCH.md), Qwen2.5-0.5B only. Every level is trained and evaluated
    the same way; scripts/ladder_report.py turns the outputs into one table (quality, GPU time, compute).
    smoke: the same jobs on 64 records for a few steps, to catch GPU-only failures before the full run."""
    m = MODELS["qwen2.5-0.5b"]
    tiny = ["--limit", "64", "--max-steps", "4", "--val-limit", "8", "--eval-every", "2", "--warmup", "2"] if smoke else []
    lim = {"limit": 16} if smoke else {}
    fast = ["--shared-catalogue", "--min-q", "9", "--lora-dropout", "0", "--cached-val"]
    nocat = ["--layout", "nocat", "--shared-catalogue", "--min-q", "9", "--lora-dropout", "0", "--cached-val"]
    teacher = "results/teacher-qwen3-1.7b"

    def level(run, args, layout="catalogue"):
        lay = [] if layout == "nocat" else ["--layout", "catalogue"]
        return [{"kind": "train", "model": m, "out": f"runs/{run}",
                 "args": lay + ["--data", "data/processed_v2"] + args + tiny},
                {"kind": "eval", "model": m, "adapter": f"runs/{run}/best", "out": f"results/ft-{run}", "layout": layout,
                 "data": "data/processed_v2", "sets": "val,gold" if smoke else "val,test,test_v2,gold,gold2,injection",
                 **lim}]

    return {
        "_kernel_sources": [f"{USER}/aerojev-train-extra"],  # its fine-tuned Qwen3-1.7B adapter is the distillation teacher
        # queues balanced on the smoke run's T4 speeds: ~2.4 h each
        "0": level("l0-baseline", ["--epochs", "1.2"])
             + level("l3-6k-lr5e-4", ["--epochs", "0.6", "--lr", "5e-4"] + fast)
             + level("l3-6k-lr1e-3", ["--epochs", "0.6", "--lr", "1e-3"] + fast)
             + level("l4-nocat-12k", ["--epochs", "1.2"] + nocat, layout="nocat")
             + [{"kind": "cmd", "cmd": "{PY} scripts/probe.py --model Qwen/Qwen2.5-0.5B-Instruct --data data/processed_v2 "
                                       "--out {OUT}/results/l6-probe" + (" --limit 64 --steps 50" if smoke else "")}],
        "1": level("l1-shared", ["--epochs", "1.2", "--shared-catalogue"])
             + level("l2-shared-all9", ["--epochs", "1.2"] + fast)
             + level("l3-6k-lr2e-4", ["--epochs", "0.6"] + fast)
             + [{"kind": "eval", "model": MODELS["qwen3-1.7b"], "adapter_input": "runs/qwen3-1.7b-cat-v2/best",
                 "out": teacher, "layout": "catalogue", "data": "data/processed_v2", "sets": "train", "dtype": "float16",
                 **({"limit": 64} if smoke else {})}]
             + level("l3-6k-distill", ["--epochs", "0.6", "--lr", "5e-4",
                                       "--teacher", f"{{OUT}}/{teacher}/preds_train.jsonl"] + fast)
             + level("l4-nocat-6k", ["--epochs", "0.6", "--lr", "5e-4"] + nocat, layout="nocat"),
    }


def _ladder_floor():
    m = MODELS["qwen2.5-0.5b"]
    nocat = ["--layout", "nocat", "--shared-catalogue", "--min-q", "9", "--lora-dropout", "0", "--cached-val",
             "--data", "data/processed_v2", "--lr", "5e-4"]

    def level(run, args):
        return [{"kind": "train", "model": m, "out": f"runs/{run}", "args": nocat + args},
                {"kind": "eval", "model": m, "adapter": f"runs/{run}/best", "out": f"results/ft-{run}", "layout": "nocat",
                 "data": "data/processed_v2", "sets": "val,test,test_v2,gold,gold2,injection"}]

    return {"0": level("l4-nocat-3k", ["--epochs", "0.3"]) + level("l4-nocat-4.8k", ["--epochs", "0.48"]),
            "1": level("l4-nocat-6k-seed1", ["--epochs", "0.6", "--seed", "1"])}


KERNELS = {
    # zero-shot logit readout, no training: the JEV-CPU / SemIf setting
    "baselines": {
        "0": [{"kind": "eval", "model": MODELS["qwen2.5-0.5b"], "out": "results/zs-qwen2.5-0.5b"},
              {"kind": "eval", "model": MODELS["qwen3-1.7b"], "out": "results/zs-qwen3-1.7b"}],
        "1": [{"kind": "eval", "model": MODELS["qwen3-0.6b"], "out": "results/zs-qwen3-0.6b"},
              {"kind": "eval", "model": MODELS["qwen3.5-0.8b"], "out": "results/zs-qwen3.5-0.8b", "mode": "unpacked"}],
    },
    # LoRA decision training, one model per GPU, then eval + GGUF export
    "train": {
        "0": [{"kind": "train", "model": MODELS["qwen3-0.6b"], "out": "runs/qwen3-0.6b", "required": True,
               "args": ["--epochs", "2"]},
              {"kind": "eval", "model": MODELS["qwen3-0.6b"], "adapter": "runs/qwen3-0.6b/best",
               "out": "results/ft-qwen3-0.6b"},
              {"kind": "export", "model": MODELS["qwen3-0.6b"], "adapter": "runs/qwen3-0.6b/best",
               "name": "aerojev-qwen3-0.6b"}],
        "1": [{"kind": "train", "model": MODELS["qwen2.5-0.5b"], "out": "runs/qwen2.5-0.5b", "required": True,
               "args": ["--epochs", "2"]},
              {"kind": "eval", "model": MODELS["qwen2.5-0.5b"], "adapter": "runs/qwen2.5-0.5b/best",
               "out": "results/ft-qwen2.5-0.5b"},
              {"kind": "export", "model": MODELS["qwen2.5-0.5b"], "adapter": "runs/qwen2.5-0.5b/best",
               "name": "aerojev-qwen2.5-0.5b"}],
    },
    # same recipe, catalogue-first layout (static question catalogue, KV-cacheable on CPU)
    "train-cat": {
        "0": [{"kind": "train", "model": MODELS["qwen3-0.6b"], "out": "runs/qwen3-0.6b-cat", "required": True,
               "args": ["--epochs", "2", "--layout", "catalogue"]},
              {"kind": "eval", "model": MODELS["qwen3-0.6b"], "adapter": "runs/qwen3-0.6b-cat/best",
               "out": "results/ft-qwen3-0.6b-cat", "layout": "catalogue"},
              {"kind": "export", "model": MODELS["qwen3-0.6b"], "adapter": "runs/qwen3-0.6b-cat/best",
               "name": "aerojev-qwen3-0.6b-cat"}],
        "1": [{"kind": "train", "model": MODELS["qwen2.5-0.5b"], "out": "runs/qwen2.5-0.5b-cat", "required": True,
               "args": ["--epochs", "2", "--layout", "catalogue"]},
              {"kind": "eval", "model": MODELS["qwen2.5-0.5b"], "adapter": "runs/qwen2.5-0.5b-cat/best",
               "out": "results/ft-qwen2.5-0.5b-cat", "layout": "catalogue"},
              {"kind": "export", "model": MODELS["qwen2.5-0.5b"], "adapter": "runs/qwen2.5-0.5b-cat/best",
               "name": "aerojev-qwen2.5-0.5b-cat"}],
    },
    # data v2 (10k records, extra phrase banks), catalogue layout; 0.6B deployment candidate + 1.7B scale point
    "train-v2": {
        "0": [{"kind": "train", "model": MODELS["qwen3-0.6b"], "out": "runs/qwen3-0.6b-cat-v2", "required": True,
               "args": ["--epochs", "2", "--layout", "catalogue", "--data", "data/processed_v2"]},
              {"kind": "eval", "model": MODELS["qwen3-0.6b"], "adapter": "runs/qwen3-0.6b-cat-v2/best",
               "out": "results/ft-qwen3-0.6b-cat-v2", "layout": "catalogue", "data": "data/processed_v2",
               "sets": "val,test,test_v2,gold,gold2,injection"},
              {"kind": "export", "model": MODELS["qwen3-0.6b"], "adapter": "runs/qwen3-0.6b-cat-v2/best",
               "name": "aerojev-qwen3-0.6b-cat-v2"}],
        "1": [{"kind": "eval", "model": MODELS[m], "out": f"results/zs-{m}-gold2", "sets": "gold2",
               **({"mode": "unpacked"} if m == "qwen3.5-0.8b" else {})} for m in MODELS] + [
              {"kind": "train", "model": MODELS["qwen3-1.7b"], "out": "runs/qwen3-1.7b-cat-v2", "required": True,
               "args": ["--epochs", "1", "--layout", "catalogue", "--data", "data/processed_v2"]},
              {"kind": "eval", "model": MODELS["qwen3-1.7b"], "adapter": "runs/qwen3-1.7b-cat-v2/best",
               "out": "results/ft-qwen3-1.7b-cat-v2", "layout": "catalogue", "data": "data/processed_v2",
               "sets": "val,test,test_v2,gold,gold2,injection"},
              {"kind": "export", "model": MODELS["qwen3-1.7b"], "adapter": "runs/qwen3-1.7b-cat-v2/best",
               "name": "aerojev-qwen3-1.7b-cat-v2"}],
    },
    # blind gold2 for the v1 decision-trained models (adapters read from the training kernels' outputs)
    "gold2-v1": {
        "_kernel_sources": [f"{USER}/aerojev-train", f"{USER}/aerojev-train-cat"],
        "0": [{"kind": "eval", "model": MODELS[base], "adapter_input": f"runs/{run}/best", "out": f"results/ft-{run}-gold2",
               "sets": "gold2", **({"layout": "catalogue"} if run.endswith("-cat") else {})}
              for base, run in [("qwen3-0.6b", "qwen3-0.6b"), ("qwen2.5-0.5b", "qwen2.5-0.5b"),
                                ("qwen3-0.6b", "qwen3-0.6b-cat"), ("qwen2.5-0.5b", "qwen2.5-0.5b-cat")]],
    },
    # 1.7B scale point (fp16 base weights + checkpointing to fit a 16 GB T4) and the CPU-friendly
    # Qwen2.5-0.5B variants: same recipe as the notebook except LoRA depth and/or number of records
    "train-extra": {
        "0": [{"kind": "train", "model": MODELS["qwen3-1.7b"], "out": "runs/qwen3-1.7b-cat-v2", "required": True,
               "args": ["--epochs", "1", "--layout", "catalogue", "--data", "data/processed_v2",
                        "--base-dtype", "float16", "--grad-checkpointing"]},
              {"kind": "eval", "model": MODELS["qwen3-1.7b"], "adapter": "runs/qwen3-1.7b-cat-v2/best",
               "out": "results/ft-qwen3-1.7b-cat-v2", "layout": "catalogue", "data": "data/processed_v2",
               "sets": "val,test,test_v2,gold,gold2,injection"},
              {"kind": "export", "model": MODELS["qwen3-1.7b"], "adapter": "runs/qwen3-1.7b-cat-v2/best",
               "name": "aerojev-qwen3-1.7b-cat-v2"}],
        "1": [job for run, extra in [("qwen2.5-0.5b-cat-v2-top8", ["--epochs", "1.2", "--top-layers", "8"]),
                                     ("qwen2.5-0.5b-cat-v2-4800", ["--epochs", "0.48"]),
                                     ("qwen2.5-0.5b-cat-v2-top8-4800", ["--epochs", "0.48", "--top-layers", "8"])]
              for job in [{"kind": "train", "model": MODELS["qwen2.5-0.5b"], "out": f"runs/{run}",
                           "args": ["--layout", "catalogue", "--data", "data/processed_v2"] + extra},
                          {"kind": "eval", "model": MODELS["qwen2.5-0.5b"], "adapter": f"runs/{run}/best",
                           "out": f"results/ft-{run}", "layout": "catalogue", "data": "data/processed_v2",
                           "sets": "val,test,test_v2,gold,gold2,injection"}]],
    },
    # execute the end-to-end notebook unchanged (it has no platform-specific code) and keep the executed copy
    "notebook-exec": {
        "0": [{"kind": "cmd", "needs_llama": True,
               "cmd": "cd {PROJ}/notebooks && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 "
                      "aerojev_qwen2.5-0.5b_lora.ipynb --output {OUT}/aerojev_qwen2.5-0.5b_lora.executed.ipynb"},
              {"kind": "cmd", "cmd": "mkdir -p {OUT}/notebook && cp -r {PROJ}/results/notebook/. {OUT}/notebook/ "
                                     "&& rm -rf {OUT}/notebook/data"}],
    },
    # CPU training throughput on a small x86 VM (4 vCPU): what training costs without a GPU
    "cpu-train-bench": {
        "_gpu": False,
        "0": [{"kind": "cmd", "cmd": "{PY} scripts/bench_train_cpu.py --model Qwen/Qwen2.5-0.5B-Instruct --records 8 --warmup 2 "
                                     f"{extra} --out {{OUT}}/train_cpu_x86.jsonl"}
              for extra in ["--layout catalogue --threads 4",
                            "--layout catalogue --threads 4 --top-layers 8",
                            "--layout catalogue --threads 2",
                            "--layout branch --threads 4"]],
    },
    # CPU speed-up ladder on the T4s: quality, GPU time and compute per level (Qwen2.5-0.5B only)
    "ladder": _ladder(),
    # the catalogue-free recipe's floor: fewer records, and a second seed of the 6k run to size run-to-run noise
    "ladder-floor": _ladder_floor(),
    "ladder-smoke": _ladder(smoke=True),
    # x86 CPU latency suite (no GPU); inputs are the training kernels' outputs (adapters + GGUFs)
    "cpu-bench": {
        "_gpu": False,
        "_kernel_sources": [f"{USER}/aerojev-train", f"{USER}/aerojev-train-cat"],
        "0": [{"kind": "cmd", "needs_llama": True,
               "cmd": "{PY} -c \"from huggingface_hub import hf_hub_download as d; "
                      "d('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf', local_dir='models/base')\""},
              {"kind": "cmd", "cmd": "{PY} -m eval.latency_suite --profile kaggle --out {OUT}/latency_x86.jsonl"}],
    },
}


def bundle():
    os.makedirs(os.path.join(ROOT, "kaggle/dataset"), exist_ok=True)
    path = os.path.join(ROOT, "kaggle/dataset/aerojev_bundle.tar.gz")
    with tarfile.open(path, "w:gz") as t:
        for d in ["aerojev", "eval", "train", "data", "notebooks", "scripts"]:
            t.add(os.path.join(ROOT, d), arcname=d,
                  filter=lambda ti: None if "__pycache__" in ti.name else ti)
    json.dump({"title": "aerojev-bundle", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(ROOT, "kaggle/dataset/dataset-metadata.json"), "w"), indent=1)
    print(f"bundle -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def kernel(name):
    kdir = os.path.join(ROOT, "kaggle/kernels", name)
    os.makedirs(kdir, exist_ok=True)
    src = open(os.path.join(ROOT, "kaggle/runner.py")).read()
    spec = dict(KERNELS[name])
    gpu = spec.pop("_gpu", True)
    sources = spec.pop("_kernel_sources", [])
    src = src.replace("JOBS = {}  # filled in by make_kernel.py", f"JOBS = {spec!r}  # generated")
    slug = f"aerojev-{name}"
    open(os.path.join(kdir, f"{slug}.py"), "w").write(src)
    json.dump({
        "id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [DATASET], "kernel_sources": sources,
        "competition_sources": [], "model_sources": [], **({"machine_shape": "NvidiaTeslaT4"} if gpu else {}),
    }, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    print(f"kernel -> {kdir}")


if __name__ == "__main__":
    {"bundle": lambda: bundle(), "kernel": lambda: kernel(sys.argv[2])}[sys.argv[1]]()
