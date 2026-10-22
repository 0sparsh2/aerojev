# Training AeroJev on a CPU

The models in this project were trained on one NVIDIA T4 GPU: the full recipe (Qwen2.5-0.5B-Instruct + LoRA,
12,000 training records) took 68 minutes. Training works on a CPU too, just slowly. This page shows what that run
looks like, measured on two CPUs:

* a **regular CPU**: a small x86 cloud VM with an Intel Xeon at 2.2 GHz, 2 cores / 4 threads, similar to an
  entry-level cloud server or an older laptop;
* an **Apple M4 laptop**, a fast laptop CPU (four performance cores).

Every timing is a training step on real training records (forward and backward pass over ~1,050 tokens, plus an
optimizer step every 8 records), measured with `scripts/bench_train_cpu.py` on an otherwise idle machine.

## Time per training record

Qwen2.5-0.5B-Instruct + LoRA (rank 16), catalogue layout, fp32:

| Setup | Regular CPU (2 cores) | Apple M4 | T4 GPU |
|---|---|---|---|
| The notebook's recipe (LoRA on all 24 blocks) | **20.2 s** | 4.4 s | 0.34 s |
| LoRA on the top 8 blocks only (`--top-layers 8`) | **11.2 s** | 2.2 s | – |
| Branch layout (shorter records, less accurate model) | 14.5 s | 3.8 s | – |
| Peak RAM: recipe / top 8 blocks | **10.5 GB / 5.2 GB** | 6.0 GB / 4.1 GB | – |

The regular CPU is about 4.6× slower than the M4 and about 60× slower than the T4. LoRA on only the top 8 blocks
roughly halves the time: backpropagation stops at the lowest adapted block instead of running through the whole
network. Its effect on accuracy is being measured; the other numbers here are final.

## Hours for a full or shorter run

| Training records | Regular CPU, recipe | Regular CPU, top 8 | M4, recipe | M4, top 8 | Validation accuracy at that point of the GPU run |
|---|---|---|---|---|---|
| 2,400 | 13.4 h | **7.4 h** | 2.9 h | 1.5 h | 0.877 |
| 4,800 | 26.9 h | 14.9 h | 5.8 h | 3.0 h | 0.932 |
| 12,000 (the full recipe) | 67 h (2.8 days) | 37 h | 14.5 h | 7.5 h | 0.970 |

The last column is the mean accuracy over the nine questions on 150 validation records, read off the full GPU run
of the same model at that many records. A run planned for fewer records finishes its learning-rate schedule sooner,
so it usually ends somewhat above the matching point of a longer run. Add 5–10% to the hours for the validation
passes during training.

## What the run looks like

* **One record at a time.** Each training record is one packed sequence holding 5–9 of the questions. Gradients from
  8 records are added up before each optimizer step, so on the regular CPU a step lands every ~2.7 minutes (recipe)
  or ~1.5 minutes (top 8 blocks).
* **Progress** is printed every 25 steps: loss, training accuracy, records per second. Validation runs every
  `--eval-every` steps, prints the accuracy, and saves the adapter whenever it improves. If the run is
  interrupted, the best adapter so far is already in `<out>/best`. There is no resume.
* **Validation is not free.** Scoring one validation record costs about a third of a training step, so use a smaller
  `--val-limit` (50–60) and fewer checks on a CPU.
* **Memory.** On the regular CPU the full recipe peaked at 10.5 GB of RAM, which needs a 16 GB machine. LoRA on the top
  8 blocks peaked at 5.2 GB and fits an 8 GB machine.
* **Threads.** 2 and 4 threads ran at the same speed on the 2-core VM, because hyperthreads add no compute. Use as many
  threads as physical (performance) cores, for example `OMP_NUM_THREADS=2`.
* **The machine is busy for hours** at full load: on a laptop, stay on mains power, keep it cool, and stop it from
  sleeping (`caffeinate -i` on macOS).

## An overnight run on a regular CPU

LoRA on the top 8 blocks over 2,400 records: about **8 hours** on the regular CPU (1.5–2 h on an M4).

```bash
OMP_NUM_THREADS=2 nohup python -m train.train --model Qwen/Qwen2.5-0.5B-Instruct \
    --layout catalogue --data data/processed_v2 --epochs 0.24 --top-layers 8 \
    --device cpu --precision fp32 --warmup 20 --eval-every 50 --val-limit 60 \
    --out runs/qwen2.5-cpu > runs/qwen2.5-cpu.log 2>&1 &
tail -f runs/qwen2.5-cpu.log
```

`--epochs 0.24` over the 10,000 data-v2 records is 2,400 records (300 optimizer steps). For more accuracy, a weekend
run of 4,800 records (`--epochs 0.48`, about 15 h with top 8 blocks on the regular CPU) reached 0.93 validation
accuracy at that point of the GPU run. Then evaluate and export as usual (README, steps 4–5) and serve the GGUF on any
CPU (`docs/RUN_ON_CPU.md`).

## Other CPUs

Training speed scales with physical cores, clock speed and vector width (AVX2, AVX-512, Arm NEON). The two machines
measured here bracket most laptops: a 2-core VM at 2.2 GHz at the slow end, an M4 at the fast end. To measure your
machine in about five minutes, run

```bash
python scripts/bench_train_cpu.py --model Qwen/Qwen2.5-0.5B-Instruct --threads <physical cores>
```

It prints seconds per record and the projected hours for 2,400, 4,800, 7,200 and 12,000 records.

## What not to do

* **bf16 or fp16 on the CPU.** On the M4, PyTorch's bf16 autocast used under two cores' worth of CPU and had not
  finished 14 records after 18 minutes; fp32 took 4.4 s per record. CPUs with bf16 matrix units (for example Intel
  AMX on recent Xeons) may behave differently; that was not tested.
* **Apple's GPU (MPS).** Packed sequences change length every step and MPS recompiled per shape, which made it slower
  than the CPU in our tests.
* **More threads than physical cores.** No gain, as measured above.

## When a GPU is the better choice

A single T4 runs the full recipe about 60× faster than the regular CPU (68 minutes against roughly 2.8 days). Free and
low-cost GPU notebooks are enough. Training is a one-off cost; the trained model then runs on any CPU with no GPU at
all (`docs/RUN_ON_CPU.md`).
