# Results

Every number in the README and in `docs/` comes from a file here or under `kaggle/outputs/<kernel>/out/`. Nothing in
this folder is hand-edited, except the explanatory sections at the end of `ultra/ultra.md` and
`int8diag/int8_x86.md`.

## Reports

| File | What it is | Made by |
|---|---|---|
| `REPORT.md` | the cross-system report: accuracy against latency, accuracy and calibration per system, per question, held-out questions, option-order robustness, CPU latency, training runs | `python scripts/report.py` |
| `summary.json` | everything `REPORT.md` is built from, in one file | `scripts/report.py` |
| `dashboard/` | `index.html` plus CSVs (`systems`, `per_question`, `reliability`, `training`) | `python scripts/export_dashboard.py` |
| `ladder/ladder.md` | CPU speed-up ladder (round 10) | `scripts/ladder_report.py --levels ladder` |
| `prune/prune.md` | pruning study (round 11) | `--levels prune` |
| `bakeoff/bakeoff.md` | smallest-model bake-off (round 12) | `--levels bakeoff` |
| `encoder/encoder.md` | Ettin-150M encoder round (round 13) | `--levels encoder` |
| `ultra/ultra.md` | UltraBERT-150M against Ettin-150M, with the decision rule's outcome (round 14) | `--levels ultra`, plus a written outcome |
| `int8diag/int8_x86.md` | why int8 fails on x86 without VNNI, and the lossless fix (rounds 15–16) | written from `kaggle/outputs/int8{diag,fix}/` |

Round numbers refer to `docs/EXPERIMENTS.md`. Each `*.md` from `ladder_report.py` has a `.json` twin with the same
rows. The ladder reports are cross-round, so pass every kernel output they need to `--root`:

```bash
python scripts/ladder_report.py --levels ultra --root kaggle/outputs/ultra/out kaggle/outputs/encoder/out \
    --out results/ultra
```

Their columns:
* **Records:** training records seen, counting epochs.
* **Accuracy** on every split.
* **T4 minutes** for training and validation.
* **GFLOP per training record,** from logged token counts.
* **Projected CPU hours** on a 2-core x86 VM and an M4. These scale measured CPU times by compute: estimates, not
  measurements.

## Evaluation folders (`results/<run>/`, and `kaggle/outputs/<kernel>/out/results/<run>/`)

Written by `python -m eval.evaluate` (or `eval/jev.py`, `eval/generative.py`):

* **`metrics.json`.** The settings (`model`, `adapter`, `gguf`, `mode`, `layout`), the `temperatures` fitted on val,
  and `sets.<split>`. Each split holds:
  * `sec_per_record`;
  * `raw` metrics, and `calibrated` (after temperature scaling).

  Each metrics block has:
  * `core_mean_accuracy`, `core_mean_macro_f1`, `core_mean_balanced_acc`, `core_ece`;
  * `record_exact_match` (all nine right);
  * `n_records`;
  * `per_question.<id>` with `accuracy`, `macro_f1`, `balanced_acc`, `majority_rate`, `ece`, `nll`, and for some
    types `auroc`, `mae_expected`, `within_1`.

  `gold_permutation` (when present) holds the option-order test.
* **`preds_<split>.jsonl`**: one row per record and question:
  `{"rid", "qid", "type", "keys", "logits", "gold", "format"}`. Every metric can be recomputed from these with
  `aerojev/metrics.py`, and `eval/analyze.py` does error analysis over them.
* **`cache.jsonl`** (`jev-*` only): every hosted-API response, so the Jev numbers can be recomputed without calling
  the API.

**Run-name prefixes:**

| Prefix | Meaning |
|---|---|
| `zs-<model>` | zero-shot readout |
| `ft-<run>` | the trained run `<run>`, evaluated in PyTorch on the GPU |
| `gguf-<model>-<variant>-<quant>` | the exported GGUF file, evaluated on the M4 through llama.cpp |
| `onnx-<run>-<file>` | an exported encoder file, evaluated through ONNX Runtime on that machine's CPU |
| `jev-<variant>` | hosted Jev, prompt variant `plain` / `playbook` / `playbook_export` |
| `gen-<model>` | the generate-JSON baseline |
| `classical/` | TF-IDF and MiniLM + logistic regression |
| `notebook*/` | notebook outputs (git-ignored, regenerable) |

## Latency and training-cost files

| File | Rows | Made by |
|---|---|---|
| `latency_mac.jsonl` | every system on the M4: one row per system with `p50_s`, `p95_s`, `ms_per_decision`, `peak_rss_mb`, and `acc_key` (which accuracy to join) | `eval/latency_suite.py` |
| `kaggle/outputs/cpu-bench/out/latency_x86.jsonl` | the same, on the 4-vCPU x86 VM | `eval/latency_suite.py --profile kaggle` |
| `latency_length.jsonl`, `latency_length_v2.jsonl`, `latency_length_prune.jsonl` | time against record length, plus start-up time, per GGUF model | `eval/bench_length.py` |
| `latency_lite.jsonl` | the encoder (and the `small` GGUF for comparison) on the M4, per thread count: `ms_median`, `ms_p95`, `peak_rss_mb`, `load_s` | `scripts/bench_lite.py` |
| `ultra/latency_kaggle_cpu.jsonl`, `int8diag/latency_variants.jsonl`, `int8diag/latency_fix.jsonl` | the same on Kaggle's x86 CPU, per model and weights file | `scripts/bench_lite.py` |
| `bench_cpu.jsonl` | early per-engine CPU benchmarks on the M4 | `eval/bench_cpu.py` |
| `train_cpu.jsonl` | CPU training throughput on the M4 (`kaggle/outputs/cpu-train-bench/` has the x86 VM) | `scripts/bench_train_cpu.py` |

**Protocols differ between files.** Each file states its own:
* the latency suite times 40 gold records;
* `bench_lite.py` times 880 gold + test records.

Compare rows within one file, or read the protocol note in the doc that cites them.
