# Kaggle tooling

The development machine has no CUDA GPU and little free disk. So every GPU training run, every large evaluation and
every x86 CPU measurement in this repository ran as a **private Kaggle kernel**, using these two scripts. The
project code does not depend on them: any machine with a CUDA GPU can run the same commands directly (README,
section 8).

| Path | What it is | In git? |
|---|---|---|
| `make_kernel.py` | builds the code + data bundle, and writes a kernel folder from a job spec; **every round's exact jobs are defined here** | yes |
| `runner.py` | the kernel's entry point: one job queue per GPU, run in parallel threads | yes |
| `kernels/<name>/` | generated kernel folders (`kernel-metadata.json` + the script Kaggle runs), kept as a record of what ran | yes |
| `outputs/<name>/` | what came back: logs, `status.json`, metrics, predictions, training curves, small configs; never weights | yes |
| `dataset/` | the bundle archive (`aerojev_bundle.tar.gz`), rebuilt on demand | no (`.gitignore`) |

## How a round runs

```bash
# 1. pack code + data into the private dataset sparshnagpal/aerojev-bundle
python kaggle/make_kernel.py bundle
kaggle datasets version -p kaggle/dataset -m "what changed"      # first time: kaggle datasets create -p kaggle/dataset

# 2. write the kernel folder for a round defined in KERNELS (make_kernel.py), then start it
python kaggle/make_kernel.py kernel encoder
kaggle kernels push -p kaggle/kernels/encoder

# 3. follow it, then download only what you need (outputs are available once the run has ended)
kaggle kernels status sparshnagpal/aerojev-encoder
kaggle kernels output sparshnagpal/aerojev-encoder -p kaggle/outputs/encoder \
    --file-pattern '^(out/(status\.json|setup\.log|gpu\d\.log)|out/results/.*|out/runs/[^/]+/(summary\.json|train_log\.jsonl))$'
```

**Inside the kernel** (`runner.py`):
1. It copies the bundle to `/tmp/aerojev/proj`.
2. It installs pinned `transformers` / `peft`, and removes the image's `torchao`.
3. It logs the CPU and GPUs to `out/setup.log`.
4. It runs each queue (`"0"`, `"1"` = GPU 0 and GPU 1) in its own thread, logging to `out/gpu<N>.log`.

Every finished job is appended to `out/status.json` with its exit code and duration. Model weights are downloaded
from Hugging Face inside Kaggle, and large files stay in the kernel's output. Only what is selected with
`--file-pattern` comes back.

## Job format

A kernel spec in `KERNELS` maps queue ids to lists of jobs. Keys starting with `_` configure the kernel itself.

| Job `kind` | Runs | Main fields |
|---|---|---|
| `train` | `python -m train.train` | `model`, `out` (under `out/`), `args` (list; `{OUT}` is substituted), `required` |
| `eval` | `python -m eval.evaluate` | `model`, `adapter` (from this kernel) or `adapter_input` (from a mounted kernel), `out`, `layout`, `data`, `sets`, `dtype`, `limit`, `mode` |
| `export` | merge LoRA, then GGUF f16 → Q8_0 (+ Q4_K_M when llama.cpp is built) | `model`, `adapter`, `name`, `keep_f16` |
| `cmd` | any shell command in the project directory | `cmd`, `needs_llama` |

**Placeholders in `cmd`:**
* `{PY}`: the kernel's Python.
* `{OUT}`: `/kaggle/working/out`.
* `{PROJ}`: the project copy.
* `{GGUF:<path>}`: the first file or directory under `/kaggle/input/**/<path>`. Use it for anything a mounted
  kernel or dataset produced, not only GGUF files.

**Flags:**
* `required: True` stops that queue if the job fails. Use it for smoke tests and training steps.
* `needs_llama: True` installs the CPU wheel of `llama-cpp-python`.

**Kernel options:**
* `_gpu: False` makes a CPU-only kernel.
* `_kernel_sources` mounts other kernels' outputs (for example, adapters from a training kernel).
* `_datasets` mounts extra private datasets.

## Kernels

All are private, under `sparshnagpal/aerojev-<name>`. The rounds are described in `docs/EXPERIMENTS.md`.

| Kernel | Machine | What it ran | Inputs besides the bundle | Results |
|---|---|---|---|---|
| `baselines` | 2× T4 | zero-shot logit readout, 4 Qwen models | – | `results/REPORT.md` |
| `train` | 2× T4 | LoRA training, branch layout, Qwen3-0.6B + Qwen2.5-0.5B; eval; GGUF export | – | `results/REPORT.md` |
| `train-cat` | 2× T4 | the same with the catalogue layout | – | `results/REPORT.md` |
| `gold2-v1` | T4 | blind gold2 for the four v1 adapters | `train`, `train-cat` | `results/REPORT.md` |
| `train-v2` | 2× T4 | data v2: Qwen3-0.6B (`best`); zero-shot gold2 | – | `results/REPORT.md` |
| `train-extra` | 2× T4 | Qwen3-1.7B (`large`); CPU-sized Qwen2.5-0.5B runs | – | `results/REPORT.md`, `docs/TRAIN_ON_CPU.md` |
| `notebook-exec` | T4 | the end-to-end notebook, executed unchanged | – | `notebooks/executed/` |
| `cpu-bench` | 4-vCPU x86 | latency suite for every system | `train`, `train-cat` | `docs/RUN_ON_CPU.md` |
| `cpu-train-bench` | 4-vCPU x86 | CPU training throughput | – | `docs/TRAIN_ON_CPU.md` |
| `ladder-smoke` | 2× T4 | the ladder's jobs on 64 records (plumbing check; outputs not kept) | `train-extra` | – |
| `ladder` | 2× T4 | CPU speed-up ladder L0–L6 | `train-extra` (distillation teacher) | `results/ladder/ladder.md` |
| `ladder-floor` | 2× T4 | catalogue-free with fewer records; a second seed | – | `results/ladder/ladder.md` |
| `prune` | 2× T4 | structured pruning + healing | `ladder` | `results/prune/prune.md` |
| `bakeoff` | 2× T4 | SmolLM2-135M, Gemma 3 270M, Ettin 32M/68M/150M; pruned Qwen second seed | – | `results/bakeoff/bakeoff.md` |
| `encoder` | 2× T4 | Ettin-150M: epochs, LR, pooling, distillation, seeds | `ladder`, `prune` (the reference runs, as distillation teachers) | `results/encoder/encoder.md` |
| `ultra` | 2× T4 | UltraBERT-150M against Ettin-150M; int8 export and CPU bench | – | `results/ultra/ultra.md` |
| `int8diag` | x86 CPU | why int8 fails on x86 without VNNI | `encoder`, `ultra` | `results/int8diag/int8_x86.md` |
| `int8fix` | x86 CPU | lossless weight-only recipes; the shipped int8 file | `int8diag`; dataset `aerojev-shipped-ettin` | `results/int8diag/int8_x86.md` |
| `general-gen` | x86 CPU | general decision data: NIM-grown wording (`general/gen_banks.py`), then pilot builds and checks | Kaggle Secret `NVIDIA_API_KEY` | `general/banks/`, kernel output |

**Private datasets:**
* `sparshnagpal/aerojev-bundle`: code + data, rebuilt per round.
* `sparshnagpal/aerojev-shipped-ettin`: the shipped `tiny` model's int8 file, uploaded so it could be scored on x86.

## What stays on Kaggle

The trained weights stay in the kernels' outputs and are downloaded only when needed:
* LoRA adapters;
* encoder checkpoints;
* fp32 / int8 / 4-bit ONNX files;
* GGUF files.

Examples:

```bash
# the shipped encoder's x86 file (rename model.int8.onnx -> model.int4.onnx; docs/ENCODER.md)
kaggle kernels output sparshnagpal/aerojev-int8fix -p /tmp/x86 --file-pattern 'onnx/e150-ep8/v-mnb4-a4/.*'
# a LoRA adapter
kaggle kernels output sparshnagpal/aerojev-train-v2 -p /tmp/v2 --file-pattern 'out/runs/qwen3-0.6b-cat-v2/best/.*'
```

`.gitignore` also keeps out copies of public tokenizers, encoder head weights saved next to (ignored) checkpoints,
and the distillation teachers' logits over the training set (regenerable).

## Gotchas

* **Archives are auto-extracted.** Kaggle unpacks an uploaded `.tar.gz` in a dataset, so `runner.py` looks for the
  extracted tree first.
* **The image's `torchao` breaks `peft`.** It makes `peft>=0.21` refuse to build LoRA layers, so the runner
  uninstalls it.
* **Embed job specs with `repr()`, not `json.dumps`.** Otherwise `true` / `True` breaks the generated Python.
* **`kaggle kernels output` only works after the run ends.** Until then use `kaggle kernels status`.
* **`/kaggle/input` is read-only.** Tools that write next to their input (e.g. ONNX Runtime's `quantize_dynamic`)
  need a copy in `/kaggle/working` or `/tmp` first.
* **Kaggle's CPUs lack VNNI.** Every Kaggle CPU seen so far (Skylake and AVX2 Xeons) lacks it, so dynamic int8 ONNX
  is wrong there (`results/int8diag/int8_x86.md`). Measure x86 CPU latency with fp32 or the 4-bit file.
* **Limits:**
  * about 30 GPU-hours a week;
  * 12-hour sessions;
  * 20 GB of output per kernel.

  Long rounds are split across two GPU queues and balanced on smoke-run speeds.
* **Secrets.** API keys go in as Kaggle Secrets, never in the bundle: the hosted Jev key and the NVIDIA NIM key.
  * **Account-wide but attached per kernel:** a secret is stored once in your account, and each kernel needs it
    attached once in the editor (Add-ons → Secrets).
  * **Whether attachments survive** new versions pushed with `kaggle kernels push` is not yet confirmed here. If a
    run reports the key missing, re-attach it.
  * **Code reads them through `general.nim.secret(name)`.** It tries the environment, then the Kaggle Secret, then
    the local `.env`. `.env` is git-ignored, and the bundle holds only `aerojev/`, `eval/`, `train/`, `data/`,
  `notebooks/` and `scripts/`.
