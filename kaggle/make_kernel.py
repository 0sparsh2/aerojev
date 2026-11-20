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


def _prune_study():
    """Prune-then-tune (scripts/prune.py) on Qwen2.5-0.5B with the catalogue-free 12k recipe: depth only (two ways of
    choosing blocks), FFN width only, both (LoRA vs full fine-tuning), plus pruning the already trained model with and
    without retraining. Pruned bases live in /tmp (rebuildable from the prune_info.json copied to the output)."""
    m, P, W = MODELS["qwen2.5-0.5b"], "/tmp/aerojev/pruned", "/kaggle/working/out"
    recipe = ["--layout", "nocat", "--shared-catalogue", "--min-q", "9", "--lora-dropout", "0", "--cached-val",
              "--data", "data/processed_v2"]
    sets = "val,test,test_v2,gold,gold2,injection"
    tuned = "{GGUF:runs/l4-nocat-12k/best/adapter_model.safetensors}"  # the trained reference, from aerojev-ladder

    def prune(name, args):
        return [{"kind": "cmd", "cmd": f"{{PY}} scripts/prune.py {args} --out {P}/{name} && mkdir -p {{OUT}}/pruned/{name} "
                                       f"&& cp {P}/{name}/prune_info.json {{OUT}}/pruned/{name}/"}]

    def score(name, model, adapter=None):
        return [{"kind": "eval", "model": model, "out": f"results/{name}", "layout": "nocat", "data": "data/processed_v2",
                 "sets": sets, **({"adapter": adapter} if adapter else {})}]

    def run(name, model, extra, full=False):
        jobs = [{"kind": "train", "model": model, "out": f"runs/{name}", "args": recipe + extra + (["--full-ft"] if full else [])}]
        if not full:
            return jobs + score(f"ft-{name}", model, f"runs/{name}/best")
        # a full checkpoint is 1.1 GB: score it, export a Q8_0 GGUF for CPU tests, then drop the weights
        return jobs + score(f"ft-{name}", f"{W}/runs/{name}/best") + [
            {"kind": "cmd", "cmd": f"{{PY}} -m aerojev.export --model {{OUT}}/runs/{name}/best --out {{OUT}}/gguf "
                                   f"--name aerojev-{name} --quants q8_0 && rm -f {{OUT}}/runs/{name}/best/model.safetensors"}]

    e12 = ["--epochs", "1.2"]
    return {
        "_kernel_sources": [f"{USER}/aerojev-ladder"],
        "0": prune("d16top", "--drop top:8") + run("p-d16top", f"{P}/d16top", e12)
             + prune("d16w", "--drop top:8 --ffn 2560") + run("p-d16w", f"{P}/d16w", e12)
             + run("p-d16w-full-2e-5", f"{P}/d16w", e12 + ["--lr", "2e-5"], full=True)
             + run("p-d16w-full-6e-5", f"{P}/d16w", e12 + ["--lr", "6e-5"], full=True),
        "1": run("l4-nocat-12k-seed1", m, e12 + ["--seed", "1"])
             + prune("d16auto", "--drop auto:8") + run("p-d16auto", f"{P}/d16auto", e12)
             + prune("w2560", "--ffn 2560") + run("p-w2560", f"{P}/w2560", e12)
             + prune("t-d16", f"--adapter {tuned} --drop top:8") + score("t-d16-noheal", f"{P}/t-d16")
             + prune("t-d16w", f"--adapter {tuned} --drop top:8 --ffn 2560") + score("t-d16w-noheal", f"{P}/t-d16w")
             + run("t-d16w-heal3k", f"{P}/t-d16w", ["--epochs", "0.3"]),
    }


def _bakeoff():
    """Smallest accurate decision model: small decoders (full fine-tuning, catalogue-free 12k recipe), encoders with one
    head per question (train/train_encoder.py), and a second seed of the pruned Qwen winner. Decoder checkpoints are
    exported to GGUF and their weights dropped; encoders keep fp16 weights for CPU tests."""
    W = "/kaggle/working/out"
    recipe = ["--layout", "nocat", "--shared-catalogue", "--min-q", "9", "--lora-dropout", "0", "--cached-val",
              "--data", "data/processed_v2", "--epochs", "1.2", "--full-ft"]
    sets = "val,test,test_v2,gold,gold2,injection"

    def decoder(name, model, extra, quants="q8_0"):
        return [{"kind": "train", "model": model, "out": f"runs/{name}", "args": recipe + extra},
                {"kind": "eval", "model": f"{W}/runs/{name}/best", "out": f"results/ft-{name}", "layout": "nocat",
                 "data": "data/processed_v2", "sets": sets},
                {"kind": "cmd", "cmd": f"{{PY}} -m aerojev.export --model {{OUT}}/runs/{name}/best --out {{OUT}}/gguf "
                                       f"--name {name} --quants {quants} && rm -f {{OUT}}/runs/{name}/best/model.safetensors"}]

    def encoder(name, model, lr, epochs):
        return [{"kind": "cmd", "cmd": f"{{PY}} -m train.train_encoder --model {model} --out {{OUT}}/runs/{name} "
                                       f"--results {{OUT}}/results/ft-{name} --lr {lr} --epochs {epochs} --sets {sets}"}]

    smol, gemma = "HuggingFaceTB/SmolLM2-135M-Instruct", "unsloth/gemma-3-270m-it"
    return {
        "0": decoder("b-smollm2-135m-3e-5", smol, ["--lr", "3e-5"]) + decoder("b-smollm2-135m-1e-4", smol, ["--lr", "1e-4"])
             + decoder("b-gemma3-270m-3e-5", gemma, ["--lr", "3e-5"]) + decoder("b-gemma3-270m-1e-4", gemma, ["--lr", "1e-4"]),
        "1": [{"kind": "cmd", "cmd": "{PY} scripts/prune.py --drop top:8 --ffn 2560 --out /tmp/aerojev/pruned/d16w"}]
             + decoder("p-d16w-full-2e-5-seed1", "/tmp/aerojev/pruned/d16w", ["--lr", "2e-5", "--seed", "1"], "f16,q8_0")
             + encoder("b-ettin-150m", "jhu-clsp/ettin-encoder-150m", 5e-5, 2)
             + encoder("b-ettin-68m", "jhu-clsp/ettin-encoder-68m", 8e-5, 2)
             + encoder("b-ettin-32m", "jhu-clsp/ettin-encoder-32m", 1e-4, 2),
    }


def _encoder_round():
    """Ettin-150M encoder (train/train_encoder.py): close the gap to the Qwen reference. One change at a time against
    the bake-off recipe (2 epochs, LR 5e-5, mean pooling): longer training, a higher LR, one attention pooling per
    question, distillation from the two reference Qwen runs (averaged), plus second seeds for run-to-run noise.
    The teachers score the 10k training records with the reference adapters from aerojev-ladder / aerojev-prune."""
    m, sets = "jhu-clsp/ettin-encoder-150m", "val,test,test_v2,gold,gold2,injection"
    q = MODELS["qwen2.5-0.5b"]
    kd = "{OUT}/results/teacher-ref0/preds_train.jsonl,{OUT}/results/teacher-ref1/preds_train.jsonl"

    def teacher(name, adapter):
        return [{"kind": "eval", "model": q, "adapter_input": adapter, "out": f"results/{name}", "layout": "nocat",
                 "data": "data/processed_v2", "sets": "train"}]

    def enc(name, *args):
        return [{"kind": "cmd", "cmd": f"{{PY}} -m train.train_encoder --model {m} --out {{OUT}}/runs/{name} "
                                       f"--results {{OUT}}/results/ft-{name} --sets {sets} " + " ".join(args)}]

    # the other GPU's teacher must be finished before a distillation run starts (its metrics.json is written last)
    wait = [{"kind": "cmd", "cmd": "for i in $(seq 120); do [ -f {OUT}/results/teacher-ref1/metrics.json ] && exit 0; "
                                   "sleep 30; done; exit 1"}]
    e4 = ("--epochs", "4")
    return {
        "_kernel_sources": [f"{USER}/aerojev-ladder", f"{USER}/aerojev-prune"],
        "0": teacher("teacher-ref0", "runs/l4-nocat-12k/best")
             + enc("e150-ep4", *e4)
             + wait
             + enc("e150-ep4-kd", *e4, "--teacher", kd)
             + enc("e150-ep4-attn-kd", *e4, "--pool", "attn", "--teacher", kd)
             + enc("e150-ep4-seed1", *e4, "--seed", "1"),
        "1": teacher("teacher-ref1", "runs/l4-nocat-12k-seed1/best")
             + enc("e150-seed1", "--epochs", "2", "--seed", "1")
             + enc("e150-ep4-lr1e-4", *e4, "--lr", "1e-4")
             + enc("e150-ep4-attn", *e4, "--pool", "attn")
             + enc("e150-ep8", "--epochs", "8"),
    }


def _ultra_round():
    """UltraBERT-150M (dmis-lab, Oct 2026, Apache-2.0) against Ettin-150M at the same size and the shipped recipe:
    8 epochs, mean pooling, the same heads, loss, data and evaluation. Only the learning rate is per model: UltraBERT's
    card says it prefers LRs up to 10x lower than other encoders (MNLI best 1e-5), so it gets 1e-5 and 3e-5, two seeds
    each; Ettin keeps 5e-5 and gets a second 8-epoch seed, so both sides are two-seed means.
    Decision rule, fixed before the run: UltraBERT's LR = the one with the higher two-seed mean val accuracy; it
    replaces Ettin only if its two-seed int8 mean of (gold, gold2, test) is >= 1 point higher and its CPU latency is
    <= 1.3x Ettin's. Every run is exported to ONNX int8 and scored again; latency is timed on the kernel's CPU after
    all training has stopped (Kaggle's Xeon, not the M4: a relative comparison)."""
    sets = "val,test,test_v2,gold,gold2,injection"
    U, E = "dmis-lab/UltraBERT-150M", "jhu-clsp/ettin-encoder-150m"

    def enc(name, model, lr, seed):
        return [{"kind": "cmd", "cmd": f"{{PY}} -m train.train_encoder --model {model} --out {{OUT}}/runs/{name} "
                                       f"--results {{OUT}}/results/ft-{name} --sets {sets} --epochs 8 --lr {lr} "
                                       f"--seed {seed}"}]

    def int8(*names):
        return [{"kind": "cmd", "cmd": f"{{PY}} scripts/export_encoder.py --model {{OUT}}/runs/{n}/model "
                                       f"--out {{OUT}}/onnx/{n} --temps {{OUT}}/results/ft-{n}/metrics.json && "
                                       f"{{PY}} -m eval.evaluate --onnx {{OUT}}/onnx/{n} --weights int8 --threads 4 "
                                       f"--data data/processed_v2 --sets {sets} --out {{OUT}}/results/onnx-{n}-int8"}
                for n in names]

    pip = [{"kind": "cmd", "cmd": "{PY} -m pip install -q onnx==1.23.2 onnxscript==0.7.2 onnxruntime==1.32.0",
            "required": True}]
    wait = [{"kind": "cmd", "cmd": "for i in $(seq 360); do [ -f {OUT}/results/ft-e150-ep8-seed1/metrics.json ] "
                                   "&& exit 0; sleep 30; done; exit 1"}]
    bench = [{"kind": "cmd", "cmd": f"{{PY}} scripts/bench_lite.py --model {{OUT}}/onnx/{n} --weights int8 "
                                    f"--threads 1 2 4 --out {{OUT}}/results/latency_kaggle_cpu.jsonl"}
             for n in ("e150-ep8-seed1", "u150-ep8-lr1e-5", "e150-ep8-seed1", "u150-ep8-lr1e-5")]  # twice, interleaved
    # 60 steps on the new architecture first: a load / fp16 failure stops both queues within minutes, not hours
    smoke = lambda g: [{"kind": "cmd", "required": True,
                        "cmd": f"{{PY}} -m train.train_encoder --model {U} --out /tmp/smoke{g} --results /tmp/smoke{g}r "
                               f"--max-steps 60 --eval-every 60 --val-limit 50 --sets gold --lr 3e-5"}]
    return {
        "0": smoke(0) + enc("u150-ep8-lr1e-5", U, 1e-5, 0) + enc("u150-ep8-lr1e-5-seed1", U, 1e-5, 1)
             + enc("e150-ep8-seed1", E, 5e-5, 1),
        "1": smoke(1) + enc("u150-ep8-lr3e-5", U, 3e-5, 0) + enc("u150-ep8-lr3e-5-seed1", U, 3e-5, 1)
             + pip + int8("u150-ep8-lr3e-5", "u150-ep8-lr3e-5-seed1")
             + wait + int8("u150-ep8-lr1e-5", "u150-ep8-lr1e-5-seed1", "e150-ep8-seed1") + bench,
    }


def _int8_diag():
    """Why did every int8 file built in aerojev-ultra lose 30-55 points, when the M4-built Ettin file lost nothing?
    That kernel's CPU (Skylake Xeon) has AVX-512 but no VNNI, where ONNX Runtime's u8 x s8 kernels can saturate.
    CPU only: for both Ettin 8-epoch seeds and both UltraBERT LR-3e-5 seeds (the LR the val rule picked), score
    the fp32 ONNX graph (rules out a shape fixed at export) and four int8 recipes on gold, gold2 and test, then time
    every recipe. Weights come from the aerojev-encoder and aerojev-ultra outputs."""
    sets, variants = "gold,gold2,test", ["s8", "s8-rr", "u8", "s8-constb"]
    models = ["e150-ep8", "e150-ep8-seed1", "u150-ep8-lr3e-5", "u150-ep8-lr3e-5-seed1"]
    ev = lambda d, w, out: (f"{{PY}} -m eval.evaluate --onnx {d} --weights {w} --threads 4 --data data/processed_v2 "
                            f"--sets {sets} --out {{OUT}}/results/{out}")
    jobs = [{"kind": "cmd", "cmd": "{PY} -m pip install -q onnx==1.23.2 onnxscript==0.7.2 onnxruntime==1.32.0",
             "required": True}]
    for m in models:
        d = f"{{OUT}}/onnx/{m}"
        jobs += [{"kind": "cmd", "cmd": f"{{PY}} scripts/export_encoder.py --model {{GGUF:runs/{m}/model}} --out {d} "
                                        f"--keep-fp32 --temps {{GGUF:results/ft-{m}/metrics.json}} && "
                                        f"{{PY}} scripts/int8_variants.py --dir {d}"},
                 {"kind": "cmd", "cmd": ev(d, "fp32", f"onnx-{m}-fp32")}]
        jobs += [{"kind": "cmd", "cmd": ev(f"{d}/v-{v}", "int8", f"onnx-{m}-{v}")} for v in variants]
    jobs += [{"kind": "cmd", "cmd": f"{{PY}} scripts/bench_lite.py --model {{OUT}}/onnx/{m}/v-{v} --weights int8 "
                                    f"--threads 1 4 --records 300 --out {{OUT}}/results/latency_variants.jsonl"}
             for m in ("e150-ep8-seed1", "u150-ep8-lr3e-5") for v in variants]
    jobs += [{"kind": "cmd", "cmd": f"{{PY}} scripts/bench_lite.py --model {{OUT}}/onnx/{m} --weights fp32 "
                                    f"--threads 1 4 --records 300 --out {{OUT}}/results/latency_variants.jsonl"}
             for m in ("e150-ep8-seed1", "u150-ep8-lr3e-5")]
    return {"_gpu": False, "_kernel_sources": [f"{USER}/aerojev-encoder", f"{USER}/aerojev-ultra"], "0": jobs}


def _int8_fix():
    """Follow-up to int8diag: an int8 recipe that is accurate on x86 CPUs without VNNI. CPU only. Starts from the
    fp32 graphs int8diag left in its output (no re-export). Variants: weight-only 8-bit (fp32 compute, and int8
    compute with per-block activation scales), weight-only 4-bit, and u8 with the FFN down-projections in fp32. Also
    scores the shipped M4-built int8 file (dataset aerojev-shipped-ettin) on this CPU."""
    sets, variants = "gold,gold2,test", ["mnb8", "mnb8-a4", "mnb4-a4", "u8-keepdown"]
    models = ["e150-ep8", "e150-ep8-seed1", "u150-ep8-lr3e-5", "u150-ep8-lr3e-5-seed1"]
    ev = lambda d, out: (f"{{PY}} -m eval.evaluate --onnx {d} --weights int8 --threads 4 --data data/processed_v2 "
                         f"--sets {sets} --out {{OUT}}/results/{out}")
    jobs = [{"kind": "cmd", "cmd": "{PY} -m pip install -q onnx==1.23.2 onnxscript==0.7.2 onnxruntime==1.32.0",
             "required": True},
            {"kind": "cmd", "cmd": ev("{GGUF:aerojev-shipped-ettin}", "onnx-shipped-m4-int8")}]
    for m in models:
        jobs += [{"kind": "cmd", "cmd": f"{{PY}} scripts/int8_variants.py --dir {{GGUF:onnx/{m}}} "
                                        f"--out {{OUT}}/onnx/{m} --variants {','.join(variants)}"}]
        jobs += [{"kind": "cmd", "cmd": ev(f"{{OUT}}/onnx/{m}/v-{v}", f"onnx-{m}-{v}")} for v in variants]
    jobs += [{"kind": "cmd", "cmd": f"{{PY}} scripts/bench_lite.py --model {{OUT}}/onnx/{m}/v-{v} --weights int8 "
                                    f"--threads 1 4 --records 300 --out {{OUT}}/results/latency_fix.jsonl"}
             for m in ("e150-ep8", "u150-ep8-lr3e-5") for v in variants]
    jobs += [{"kind": "cmd", "cmd": "{PY} scripts/bench_lite.py --model {GGUF:aerojev-shipped-ettin} --weights int8 "
                                    "--threads 1 4 --records 300 --out {OUT}/results/latency_fix.jsonl"}]
    return {"_gpu": False, "_kernel_sources": [f"{USER}/aerojev-int8diag"],
            "_datasets": [f"{USER}/aerojev-shipped-ettin"], "0": jobs}


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
    # pruning study: fewer blocks and/or narrower FFNs, healed by training, each level scored (Qwen2.5-0.5B only)
    "prune": _prune_study(),
    # smallest accurate model: SmolLM2-135M, Gemma 3 270M, Ettin encoders, pruned Qwen second seed
    "bakeoff": _bakeoff(),
    # Ettin-150M encoder round: epochs, LR, attention pooling, distillation from the Qwen reference, seeds
    "encoder": _encoder_round(),
    # UltraBERT-150M against the shipped Ettin-150M: two LRs x two seeds, plus an Ettin second seed (rule fixed before)
    "ultra": _ultra_round(),
    # CPU kernel: why int8 ONNX is wrong on Kaggle's x86 CPUs (no VNNI); four int8 recipes against fp32
    "int8diag": _int8_diag(),
    # CPU kernel: weight-only (MatMulNBits) recipes that are lossless on x86, and the shipped M4-built file itself
    "int8fix": _int8_fix(),
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
    datasets = [DATASET] + spec.pop("_datasets", [])
    src = src.replace("JOBS = {}  # filled in by make_kernel.py", f"JOBS = {spec!r}  # generated")
    slug = f"aerojev-{name}"
    open(os.path.join(kdir, f"{slug}.py"), "w").write(src)
    json.dump({
        "id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": datasets, "kernel_sources": sources,
        "competition_sources": [], "model_sources": [], **({"machine_shape": "NvidiaTeslaT4"} if gpu else {}),
    }, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    print(f"kernel -> {kdir}")


if __name__ == "__main__":
    {"bundle": lambda: bundle(), "kernel": lambda: kernel(sys.argv[2])}[sys.argv[1]]()
