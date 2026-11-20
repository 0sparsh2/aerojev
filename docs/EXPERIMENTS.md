# Experiment log

Every experiment round in this repository, in the order it ran. Each entry has:
* **the question** the round asked;
* **the setup;**
* **the decision rule**, where one was fixed before the run;
* **the outcome;**
* **where the numbers are.**

Every round ran as a private Kaggle kernel (`kaggle/README.md`). The kernel name is the round name, and the exact
jobs are in `kaggle/make_kernel.py`.

**Conventions:**
* **Accuracy** is the mean over the nine trained questions.
* **Gold** is 80 hand-written records and **gold2** 48 more, written blind. **Test** is 800 synthetic records with
  held-out wording.
* **Noise:** sampling error is about ±1.0 point on gold, ±1.3 on gold2 and ±0.3 on test. Two runs of the same recipe
  differ by up to about 1.5 points on gold.
* **"Reference"** is the catalogue-free Qwen2.5-0.5B recipe from the ladder round (two runs: gold 0.928 / 0.940).

## Overview

| # | Round | Date | Machine | Question | Outcome |
|---|---|---|---|---|---|
| 1 | `baselines` | 2026-10-21 | 2× T4 | How good is untrained logit readout? | 0.51–0.70 gold: not usable |
| 2 | `train` | 2026-10-21 | 2× T4 | Does LoRA decision training work? | 0.90–0.93 gold |
| 3 | `train-cat` | 2026-10-22 | 2× T4 | Catalogue-first layout? | as accurate or better, 3–6× faster on CPU |
| 4 | `gold2-v1` | 2026-10-22 | T4 | Blind gold2 for the v1 models | 0.90–0.93: no overfitting to gold |
| 5 | `train-v2` | 2026-10-22 | 2× T4 | More varied data (v2), bigger model | Qwen3-0.6B v2: 0.950 gold (`best`) |
| 6 | `train-extra` | 2026-10-22 | 2× T4 | Qwen3-1.7B; CPU-sized runs | 1.7B most robust (`large`); 4,800 records: 0.88 |
| 7 | `notebook-exec` | 2026-10-22 | T4 | Does the notebook run unchanged? | yes; executed copy in `notebooks/executed/` |
| 8 | `cpu-bench` | 2026-10-22 | 4-vCPU x86 | Inference cost on a regular CPU | 1.3–2.8 s per record (catalogue) |
| 9 | `cpu-train-bench` | 2026-10-22 | 4-vCPU x86 | Training cost on a regular CPU | 20.2 s per record: ~3 days for the full recipe |
| 10 | `ladder`, `ladder-floor` | 2026-10-22/23 | 2× T4 | How far can CPU training cost come down? | 3.6× exact; 6× catalogue-free at equal accuracy |
| 11 | `prune` | 2026-10-28 | 2× T4 | Cut the 0.5B model down | 16 blocks + FFN 2560 (`small`) |
| 12 | `bakeoff` | 2026-11-13 | 2× T4 | Smallest model that is still accurate | encoders beat small decoders |
| 13 | `encoder` | 2026-11-13 | 2× T4 | Close the encoder's gap | 8 epochs → `tiny`: 0.932 / 0.928 / 0.969 |
| 14 | `ultra` | 2026-11-19 | 2× T4 + x86 CPU | UltraBERT-150M vs Ettin-150M | +1 point at fp32, mostly test; not adopted |
| 15 | `int8diag` | 2026-11-20 | x86 CPU | Why is int8 wrong on Kaggle? | no-VNNI int16 saturation |
| 16 | `int8fix` | 2026-11-20 | x86 CPU | A lossless x86 file | weight-only 4-bit, lossless |

Hosted Jev was evaluated separately with `eval/jev.py`, in three prompt variants (`results/jev-*`). It is not a
Kaggle round.

## 1. `baselines`: zero-shot logit readout

* **Question.** Can a pretrained small LLM answer the questions by reading label-token logits, with no training? This
  is the Meanblock/JEV-CPU (SemIf) setting.
* **Setup.**
  * Models: Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Qwen3-1.7B and Qwen3.5-0.8B.
  * Branch layout, data v1.
  * Qwen3.5 runs `unpacked`, because its recurrent layers cannot be tree-packed.
* **Outcome.** Gold 0.508 / 0.632 / 0.704 / 0.518. Zero-shot readout is far from usable on this playbook, which
  needs arithmetic across facts.
* **Files:** `kaggle/outputs/baselines/`, `results/REPORT.md`.

## 2–4. `train`, `train-cat`, `gold2-v1`: decision training and the layout

* **Question.** Does RFDT-style LoRA decision training work, and does putting the question catalogue first (so its
  key-value cache can be reused on CPU) cost accuracy?
* **Setup.**
  * Qwen3-0.6B and Qwen2.5-0.5B-Instruct, LoRA r=16 on every linear layer, 2 epochs over 6,000 records.
  * Loss: cross-entropy over label logits plus an ordinal penalty.
  * Two layouts: branch, and catalogue.
  * Then a blind gold2 pass over all four adapters.
* **Outcome.**

  | Layout | Qwen2.5-0.5B, gold | Qwen3-0.6B, gold |
  |---|---|---|
  | branch | 0.896 | 0.929 |
  | catalogue | 0.932 | 0.935 |

  * **gold2:** 0.90–0.93, in line with gold.
  * **The catalogue layout won on both counts:** at least as accurate, and 3× (M4) to 6× (x86) faster at inference.
* **Files:** `kaggle/outputs/{train,train-cat,gold2-v1}/`, `results/REPORT.md`.

## 5–6. `train-v2`, `train-extra`: data v2, scale, CPU-sized runs

* **Question.** Does more varied data help? What does a 1.7B model add? How accurate are runs small enough for a CPU?
* **Setup.**
  * **Data v2:** 10,000 records, adding early deliveries, equipment outages, supplier-caught defects and more
    injection styles.
  * **Qwen3-0.6B,** catalogue layout, 2 epochs.
  * **Qwen3-1.7B,** 1 epoch, with fp16 base weights and gradient checkpointing to fit a T4.
  * **Qwen2.5-0.5B CPU-sized variants:** 4,800 records, and LoRA on the top 8 blocks only.
* **Outcome.**
  * **Qwen3-0.6B v2:** gold 0.950, gold2 0.931, test 0.959. This became the `best` preset.
  * **Qwen3-1.7B v2:** 0.946 / 0.949 / 0.965. It was the most robust to shuffled options (0.95 agreement), and
    became the `large` preset.
  * **4,800 records:** gold 0.881.
  * **Top-8-block LoRA:** gold 0.60. The model learned to answer by option position. Do not use it.
* **Files:** `kaggle/outputs/{train-v2,train-extra}/`, `results/REPORT.md`, `docs/TRAIN_ON_CPU.md`.

## 7–9. `notebook-exec`, `cpu-bench`, `cpu-train-bench`: reproducibility and CPU cost

* **notebook-exec:** the end-to-end notebook ran unchanged on a T4. The executed copy and its figures are in
  `notebooks/executed/`.
* **cpu-bench** (4-vCPU Xeon, 2.2 GHz): every system under one latency protocol.
  * **Catalogue models:** 1.3–2.8 s per record.
  * **Branch models:** 8.6–14.6 s.
  * **On this machine, PyTorch fp32 beat the prebuilt llama.cpp wheel.**
* **cpu-train-bench:** one training record takes 20.2 s on the 2-core VM and 4.4 s on an M4 (measured locally),
  against 0.34 s on a T4. The full recipe is about 3 days on the VM.
* **Files:**
  * `kaggle/outputs/{cpu-bench,cpu-train-bench}/`;
  * `docs/RUN_ON_CPU.md`, `docs/TRAIN_ON_CPU.md`;
  * `results/train_cpu.jsonl`.

## 10. `ladder`, `ladder-floor`: cutting CPU training cost

* **Question.** The desk research in `docs/CPU_SPEEDUP_RESEARCH.md` found that 83% of each training sequence
  re-encodes the question catalogue. Which proposed cut keeps accuracy?
* **Setup.** Qwen2.5-0.5B, data v2. Each "level" is trained and scored the same way. Compute per record is computed
  from logged token and attention-pair counts, so it does not depend on the GPU.

  | Level | Change | Gold | Compute |
  |---|---|---|---|
  | L0 | baseline | 0.922 | 1× |
  | L1 | one catalogue per 8-record step (exact gradients, `tests/test_shared_prefix.py`) | 0.925 | 3.6× less |
  | L2 | + all 9 questions per record, LoRA dropout 0, cached-catalogue validation | 0.918 | 3.5× less |
  | L3 | + 6,000 records, at LR 2e-4 / 5e-4 / 1e-3, and with distillation | 0.875–0.886 (1e-3 diverged) | 6.9× less |
  | L4 | no catalogue in the prompt (`nocat`), 12,000 records | 0.928 (second run 0.940) | 6.0× less |
  | L4 | `nocat`, 6,000 / 4,800 / 3,000 records | 0.885–0.921 / 0.854 / 0.863 | 12–24× less |
  | L6 | frozen model + linear probes | 0.799 | 14.7× less |

* **Outcome.**
  * **The catalogue-free recipe (L4, 12,000 records) keeps accuracy at a sixth of the compute.** It is the recipe of
    `notebooks/aerojev_qwen2.5-0.5b_nocat.ipynb`, and the reference for later rounds.
  * **Fewer records is noisy:** the two 6,000-record runs differed by 3.6 points.
* **Files:** `results/ladder/ladder.md`, `kaggle/outputs/{ladder,ladder-floor}/`.

## 11. `prune`: a smaller Qwen

* **Question.** Can the 494M reference be cut down structurally and then healed by training?
* **Setup.** `scripts/prune.py`, with the catalogue-free 12,000-record recipe:
  * **Depth only:** remove the top 8 blocks, or the 8 most alike consecutive blocks.
  * **Width only:** FFN 4864 → 2560, chosen by activation × weight scores.
  * **Both:** with LoRA or full fine-tuning, at two learning rates.
  * **Pruning the trained reference:** with no retraining, and with 3,000 records of healing.
* **Outcome.**
  * **Best: 16 blocks + FFN 2560, full fine-tuning at LR 2e-5** (276M parameters).
    * **Accuracy:** gold 0.914 / 0.899 over two runs, against the reference's 0.928 / 0.940. Gold2 and test are level.
    * **Speed:** about 2× faster at inference and 0.3 GB less RAM than `fast`.
  * **Its GGUF Q8_0 (299 MB) is the `small` preset:** 0.915 / 0.914 / 0.951 after export.
  * **Removing the most alike blocks was worse** than removing the top blocks.
  * **Pruning without retraining collapses** to 0.42–0.56.
* **Files:** `results/prune/prune.md`, `results/gguf-qwen2.5-0.5b-nocat-d16*`, `results/latency_length_prune.jsonl`.

## 12. `bakeoff`: the smallest accurate model

* **Question.** Below 276M parameters, which architecture keeps accuracy: a small decoder, or an encoder with one
  head per question?
* **Setup.**
  * **Decoders:** SmolLM2-135M and Gemma 3 270M, full fine-tuning with the catalogue-free recipe, two learning rates
    each.
  * **Encoders:** Ettin 32M / 68M / 150M with 9 linear heads (`train/train_encoder.py`), 2 epochs.
  * **Noise check:** a second run of the pruned Qwen winner.
* **Outcome.**
  * **Decoders were 3–13 points below the reference.** Gemma 3 270M is 168M parameters of vocabulary.
  * **Ettin-150M was 1–2 points short;** Ettin-68M was within 1–2 points, at a fifth of the compute.
  * **Ettin-32M lost 3–5.**
  * **The encoder was carried forward.**
* **Files:** `results/bakeoff/bakeoff.md`.

## 13. `encoder`: closing the encoder's gap

* **Question.** What closes Ettin-150M's remaining 1–2 points on the hand-written sets?
* **Setup.** One change at a time against the bake-off recipe (2 epochs, LR 5e-5, mean pooling), with second runs for
  noise:
  * 4 and 8 epochs;
  * LR 1e-4;
  * one learned attention pooling per question;
  * distillation from the two reference Qwen runs;
  * attention pooling plus distillation.
* **Outcome.**
  * **Training longer is what worked.** Going from 2 to 4 epochs raised every set in both runs. `delay_bucket` rose
    from 0.89–0.91 to 0.95–0.96 on test.
  * **8 epochs:** 0.929 / 0.926 / 0.969.
  * **No effect beyond noise:** the higher LR and attention pooling.
  * **Distillation:** it gave the best gold2 (0.940) but dropped test to the teachers' level.
  * **Exported to int8 ONNX** (151 MB): 0.932 / 0.928 / 0.969 on the M4, 28 ms per record. This is the `tiny`
    preset.
* **Files:** `results/encoder/encoder.md`, `docs/ENCODER.md`, `results/onnx-ettin-150m-ep8-int8/`,
  `results/latency_lite.jsonl`.

## 14. `ultra`: UltraBERT-150M against Ettin-150M

* **Question.** UltraBERT-150M (`dmis-lab`, October 2026, Apache-2.0) is a newer encoder of the same size. It has
  digit-level number tokens, which could help with dates. Is it better?
* **Setup.**
  * **Same recipe as `tiny`:** 8 epochs, mean pooling, same heads, loss and data.
  * **UltraBERT:** LR 1e-5 and 3e-5 (its card prefers low LRs), two runs each.
  * **Ettin:** a second run, so both sides are two-run means.
  * **Each run:** a 60-step smoke test first, and the queue stops if it fails. Then int8 export and CPU benchmarks on
    the kernel's CPU.
* **Decision rule, fixed before the run.**
  * **Learning rate:** UltraBERT's is the one with the higher validation mean.
  * **Replacement:** it replaces Ettin if its two-run mean of gold / gold2 / test is at least 1 point higher, and its
    CPU latency is at most 1.3× Ettin's.
* **Outcome.** The rule picked LR 3e-5 (validation 0.9936 against 0.9903).

  | | UltraBERT (two-run mean) | Ettin | Difference |
  |---|---|---|---|
  | PyTorch, gold / gold2 / test | 0.938 / 0.927 / 0.982 | 0.932 / 0.921 / 0.964 | +1.01 |
  | fp32 ONNX on x86, latency 1.10× / 1.06× | | | +1.04: passes |
  | 4-bit x86 file, latency 1.11× / 1.20× | | | +0.79: fails |

  * **Most of the gain is on the synthetic test set.** There, `delay_bucket` rose from 0.954 to 0.991. Gold and
    gold2 moved within noise.
  * **Training cost:** 34 minutes on a T4, against 28.
  * **Verdict: Ettin stays shipped.**
* **Side finding.** Every int8 file built in this kernel scored 0.31–0.61 on every set, with the shipped recipe. That led to
  rounds 15–16.
* **Files:** `results/ultra/ultra.md`, `results/ultra/latency_kaggle_cpu.jsonl`.

## 15. `int8diag`: why int8 is wrong on Kaggle's CPUs

* **Question.** The same int8 recipe is lossless on the M4. Why does it lose 30–55 points on Kaggle?
* **Setup.**
  * **Machine:** a CPU-only kernel on an Intel Xeon at 2.2 GHz (AVX2, no AVX-512, no VNNI). `aerojev-ultra` had run
    on a Skylake AVX-512 Xeon, also without VNNI.
  * **Models:** both Ettin runs and both UltraBERT LR 3e-5 runs, with fp32 kept as the control.
  * **Four int8 recipes,** via `scripts/int8_variants.py`:
    * s8 weights (the shipped recipe);
    * 7-bit weights (`reduce_range`);
    * u8 × u8;
    * s8 with the attention products in fp32.
* **Outcome.**
  * **fp32 is lossless,** so the export is fine.
  * **The shipped recipe collapses:** 0.43–0.61 gold. ONNX Runtime's u8 × s8 kernel (`VPMADDUBSW`) sums pairs of
    products into 16-bit integers, which saturate on CPUs without VNNI.
  * **Keeping attention in fp32 did not help,** so the loss is in the weight matrices.
  * **The saturation-free recipes only partly recover:** Ettin is still 2–5 points short, UltraBERT 15–20.
* **Files:** `results/int8diag/int8_x86.md`, `results/int8diag/latency_variants.jsonl`.

## 16. `int8fix`: a lossless x86 file

* **Question.** Which quantization is lossless on x86 CPUs without VNNI? Is the shipped M4-built file affected?
* **Setup.**
  * **The shipped file** was uploaded as a private dataset and scored.
  * **Weight-only `MatMulNBits` recipes,** built from int8diag's fp32 graphs:
    * 8-bit weights with fp32 compute;
    * 8-bit weights with int8 compute per block of 32;
    * 4-bit weights with int8 compute per block of 32.
* **Outcome.**
  * **The shipped file is affected:** 0.524 gold on x86.
  * **Every weight-only recipe is lossless** for both models and both runs.
  * **The 4-bit one is 109 MB, peaks at 0.33 GB,** and takes 584 / 320 ms (1 / 4 threads), against fp32's 478 / 258
    ms at 597 MB and 0.96 GB.
  * **Shipped as the x86 file:** `export_encoder.py --int4` now writes it, and `aerojev/lite.py` picks it on x86.
* **Still open:**
  * the 4-bit file on the M4;
  * int8 on x86 CPUs with VNNI;
  * a u8 recipe with fp32 down-projections, which failed to build because it writes next to its read-only input.
* **Files:** `results/int8diag/int8_x86.md`, `results/int8diag/latency_fix.jsonl`.

## Hosted Jev (`eval/jev.py`)

* **Setup.** TypeSafe's hosted Jev (jev-1.14.0) on the same splits and questions, in three variants:
  * the questions as-is;
  * with the playbook rules in the instructions;
  * with the rules plus a definition of "export controlled".

  Every response is cached in `results/jev-*/cache.jsonl`, so the numbers can be recomputed without calling the
  API.
* **Outcome.**

  | Variant | Gold | Test |
  |---|---|---|
  | questions as-is | 0.869 | 0.817 |
  | + playbook rules | 0.892 | 0.853 |
  | + playbook + export definition | 0.918 | 0.882 |

  * **Robustness:** best option-order agreement of all systems (0.99–1.00).
  * **Latency:** 0.12–0.18 s per record over the API.
  * **Fairness note:** Jev is zero-shot, while the AeroJev models are trained on the playbook.
* **Files:** `results/jev-plain/`, `results/jev-playbook/`, `results/jev-playbook-export/`, `results/REPORT.md`.

## Next

General, domain-agnostic decision models benchmarked against Jev: `docs/GENERAL_DECISION_PLAN.md`.
