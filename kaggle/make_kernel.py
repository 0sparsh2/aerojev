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
