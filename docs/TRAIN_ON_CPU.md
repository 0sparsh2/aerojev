# Training AeroJev on a CPU

The models in this project were trained on one NVIDIA T4 GPU: the full recipe (Qwen2.5-0.5B-Instruct + LoRA,
12,000 training records) took 68 minutes. Training works on a CPU too, just slowly. This page shows what that run
looks like, measured on two CPUs:

* a **regular CPU**: a small x86 cloud VM with an Intel Xeon at 2.2 GHz, 2 cores / 4 threads, similar to an
  entry-level cloud server or an older laptop;
* an **Apple M4 laptop**, a fast laptop CPU (four performance cores).

Every timing is a training step on real training records (forward and backward pass over ~1,050 tokens, plus an
optimizer step every 8 records), measured with `scripts/bench_train_cpu.py` on an otherwise idle machine. Accuracies
come from the same configurations trained on a GPU (the result does not depend on the hardware) and are the mean over
the nine questions.

## Time per training record

Qwen2.5-0.5B-Instruct + LoRA (rank 16 on every block), catalogue layout, fp32:

| | Regular CPU (2 cores) | Apple M4 | T4 GPU |
|---|---|---|---|
| Seconds per training record | **20.2 s** | 4.4 s | 0.34 s |
| Peak RAM | **10.5 GB** | 6.0 GB | – |

The regular CPU is about 4.6× slower than the M4 and about 60× slower than the T4. A 16 GB machine is needed.

## How many records, how long, how accurate

Fewer training records is the way to shorten a CPU run. Accuracy on the hand-written gold sets and the held-out test set:

| Training records | Regular CPU | Apple M4 | Gold | Gold2 (blind) | Test |
|---|---|---|---|---|---|
| 2,400 | ~14 h | ~3 h | about 0.87 on validation (see note) | | |
| **4,800** | **~28 h** | **~6 h** | 0.881 | 0.873 | 0.908 |
| 12,000 (the full recipe) | ~70 h (≈ 3 days) | ~15 h | 0.924 | 0.912 | 0.948 |
| *for reference: the untrained model* | – | – | 0.204 | 0.181 | 0.252 |

Hours include the validation passes during training. The 4,800 and 12,000 rows are separately trained models. For
2,400 records there is no separate run: the 4,800-record run had reached 0.874 validation accuracy (the same nine
questions on 150 records) when it had seen 2,400 records.

## What the run looks like

* **One record at a time.** Each training record is one packed sequence holding 5–9 of the questions. Gradients from
  8 records are added up before each optimizer step, so on the regular CPU a step lands every ~2.7 minutes.
* **Progress** is printed every 25 steps: loss, training accuracy, records per second. Validation runs every
  `--eval-every` steps, prints the accuracy, and saves the adapter whenever it improves. If the run is
  interrupted, the best adapter so far is already in `<out>/best`. There is no resume.
* **Validation is not free.** Scoring one validation record costs about a third of a training step, so use a smaller
  `--val-limit` (50–60) and fewer checks on a CPU.
* **Threads.** 2 and 4 threads ran at the same speed on the 2-core VM, because hyperthreads add no compute. Use as many
  threads as physical (performance) cores, for example `OMP_NUM_THREADS=2`.
* **The machine is busy for hours** at full load: on a laptop, stay on mains power, keep it cool, and stop it from
  sleeping (`caffeinate -i` on macOS).

## A CPU run: 4,800 records

About 28 hours on the regular CPU, or 6 hours on an M4, for 0.88 gold accuracy:

```bash
OMP_NUM_THREADS=2 nohup python -m train.train --model Qwen/Qwen2.5-0.5B-Instruct \
    --layout catalogue --data data/processed_v2 --epochs 0.48 \
    --device cpu --precision fp32 --eval-every 75 --val-limit 60 \
    --out runs/qwen2.5-cpu > runs/qwen2.5-cpu.log 2>&1 &
tail -f runs/qwen2.5-cpu.log
```

`--epochs 0.48` over the 10,000 data-v2 records is 4,800 records (600 optimizer steps). Use `--epochs 0.24` for
2,400 records in half the time, or `--epochs 1.2` for the full recipe. Set `OMP_NUM_THREADS` to your number of
physical cores. Then evaluate and export as usual (README, steps 4–5) and serve the GGUF on any CPU
(`docs/RUN_ON_CPU.md`).

## Other CPUs

Training speed scales with physical cores, clock speed and vector width (AVX2, AVX-512, Arm NEON). The two machines
measured here bracket most laptops: a 2-core VM at 2.2 GHz at the slow end, an M4 at the fast end. To measure your
machine in about five minutes, run

```bash
python scripts/bench_train_cpu.py --model Qwen/Qwen2.5-0.5B-Instruct --threads <physical cores>
```

It prints seconds per record and the projected hours for 2,400, 4,800, 7,200 and 12,000 records.

## What does not help

| Idea | Speed | Accuracy | Verdict |
|---|---|---|---|
| LoRA on the top 8 blocks only (`--top-layers 8`) | 11.2 s per record on the regular CPU, 2.2 s on the M4 (about 2× faster) | gold 0.597, gold2 0.551, test 0.616 (12,000 records) | **Do not use.** The model ends up answering mostly by option position |
| Branch prompt layout (shorter training records) | 14.5 s / 3.8 s per record | no better than the catalogue layout in any comparison, worse in most | Not worth it |
| bf16 / fp16 on the CPU | On the M4, bf16 autocast used under two cores and had not finished 14 records after 18 minutes | – | Avoid. CPUs with bf16 matrix units (e.g. Intel AMX) were not tested |
| Apple's GPU (MPS) | Slower than the CPU: packed sequences change length every step and MPS recompiled per shape | – | Avoid |
| More threads than physical cores | No change | – | Avoid |

With LoRA on only the top blocks, the loss plateaued around 0.87, against about 0.2 with LoRA on every block. In
this prompt layout the lower blocks also have to adapt.

## When a GPU is the better choice

A single T4 runs the full recipe about 60× faster than the regular CPU (68 minutes against roughly 3 days) and reaches
0.924 gold accuracy, against 0.881 for the 4,800-record CPU-sized run. Free and low-cost GPU notebooks are enough.
Training is a one-off cost; the trained model then runs on any CPU with no GPU at all (`docs/RUN_ON_CPU.md`).
