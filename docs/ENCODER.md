# The smallest accurate model: Ettin-150M encoder, int8 (`--preset tiny`)

The Qwen models in this project are text generators turned into decision engines. For a **fixed** set of nine
questions, the task is also plain classification, and a much smaller model built for classification does as well.

**Ettin-150M** (`jhu-clsp/ettin-encoder-150m`, a ModernBERT-style encoder, MIT licence) reads the record once. Each
of the nine questions has its own small output layer (a "head") over that question's options. Trained for 8 epochs
on the 10,000 training records and stored in int8, it:

* **is about as accurate as the Qwen2.5-0.5B model** this study was measured against. It is better on the synthetic
  test sets and inside the reference's range on the hand-written gold sets;
* **is a 151 MB file and runs without PyTorch.** The runtime is ONNX Runtime + `tokenizers` + numpy, about 100 MB of
  packages;
* **takes 28 ms per record** (4 threads) or 47 ms (1 thread) on an Apple M4, **using under 0.5 GB of RAM**.

On x86 servers it needs a different file, because int8 is wrong on x86 CPUs without VNNI. The replacement is a
109 MB 4-bit file that is lossless there and is picked automatically (see "On x86 CPUs" below).

The same limit applies as for the catalogue-free Qwen model. It answers only its nine trained questions, each with its
options in schema order. A new question or option needs retraining, and ad-hoc questions are refused.

## Where it stands

Mean accuracy over the nine questions. Gold and gold2 are the hand-written records (80 and 48); test is 800
synthetic records. The CPU columns are the median per record over the gold + test records, one record at a time,
warm, on the M4 (`scripts/bench_lite.py`; `results/latency_lite.jsonl`).

| Model | Weights | File | Gold | Gold2 | Test | M4 ms, 4 threads | M4 ms, 1 thread | Peak RAM |
|---|---|---|---|---|---|---|---|---|
| **Ettin-150M encoder, 8 epochs, ONNX int8** (`tiny`) | 150M | **151 MB** | 0.932 | 0.928 | **0.969** | **28** | **47** | **0.40–0.48 GB** |
| Qwen2.5-0.5B pruned to 16 blocks + FFN 2560, GGUF Q8_0 (`small`) | 276M | 299 MB | 0.915 | 0.914 | 0.951 | 83 | 153 | 1.0 GB |
| Qwen2.5-0.5B + LoRA, catalogue-free, 12k records (the reference, two seeds) | 494M | – | 0.928 / 0.940 | 0.905 / 0.919 | 0.944 / 0.954 | – | – | – |
| Qwen3-0.6B + LoRA, catalogue, data v2 (`best`) | 600M | 639 MB | **0.950** | 0.931 | 0.960 | – | – | 2.0 GB |

The repo's most accurate model, `best` (Qwen3-0.6B), is still about 2 points better on the 80 gold records. It can
also take ad-hoc questions. The encoder is the choice when size, memory and speed matter most.

Sampling error on these sets is about ±1.0 point on gold, ±1.3 on gold2 and ±0.3 on test. Two training runs of the
same recipe differ by up to about 1.5 points on gold. Differences smaller than that are noise.

## What closed the gap: training longer

The first encoder run in the bake-off (2 epochs) was 1–2 points short of the reference on the hand-written sets. One
change at a time was then tried on a Kaggle T4 (`kaggle/make_kernel.py` → `encoder`; full table:
`results/encoder/encoder.md`):

| Run | Gold | Gold2 | Test | T4 training |
|---|---|---|---|---|
| 2 epochs, LR 5e-5, mean pooling (two seeds) | 0.910 / 0.926 | 0.912 / 0.889 | 0.945 / 0.948 | 7 min |
| **4 epochs** (two seeds) | 0.929 / 0.943 | 0.914 / 0.907 | 0.960 / 0.964 | 14 min |
| **8 epochs** (shipped) | 0.929 | 0.926 | 0.969 | 28 min |
| 4 epochs, LR 1e-4 | 0.924 | 0.917 | 0.968 | 14 min |
| 4 epochs, one learned attention pooling per question | 0.940 | 0.903 | 0.961 | 14 min |
| 4 epochs, distilled from the two Qwen reference runs | 0.933 | 0.940 | 0.943 | 14 min + 2 × 13 min teacher |
| 4 epochs, attention pooling + distillation | 0.928 | 0.905 | 0.966 | 14 min |

* **More epochs is what worked.** Going from 2 to 4 epochs raised every set, in both seeds: gold +1.8, gold2 +1.0 and
  test +1.6 on average. The question that improved most was **delay_bucket**, which needs the record's dates
  compared: on test it went from 0.89–0.91 to 0.95–0.96, level with the reference. 8 epochs added a little more on gold2
  and test.
* **A higher learning rate and per-question attention pooling did not help** beyond run-to-run noise.
* **Distillation was mixed.** Learning from the Qwen runs' probabilities gave the best gold2 (0.940) but dropped test
  to the teachers' own level (0.943). It also needs the teachers run over all training records first. Not adopted.
* The shipped model is the 8-epoch run. It has the best average over gold, gold2 and test (0.941, against 0.926 and
  0.938 for the two reference runs). That choice is made on the same sets it is reported on, so read its edge over
  the 4-epoch runs as small. **For retraining on a CPU, 4 epochs gives the same quality within noise at half the
  time.**

## Smaller: int8 with ONNX Runtime (ARM, Apple Silicon)

`scripts/export_encoder.py` exports the encoder, its pooling and all nine heads as one ONNX graph. It checks that
graph against PyTorch before quantizing anything: a padded batch of 8 gold records, then single unpadded records of
other lengths, with a largest logit difference of 3e-5 (it must be under 1e-3). It then stores the encoder's weight
matrices and the token embedding table as int8, with one scale per output channel. The heads and pooling stay fp32.

| Weights (bake-off 2-epoch model, M4) | File | Gold | Gold2 | Test | M4 ms, 4 threads | M4 ms, 1 thread |
|---|---|---|---|---|---|---|
| fp32 ONNX | 597 MB | 0.910 | 0.912 | 0.945 | 38 | 49 |
| **int8 ONNX** (dynamic, per-channel) | **151 MB** | 0.907 | 0.912 | 0.946 | **26** | **45** |
| first 4-bit recipe (MatMulNBits, blocks of 128, fp32 compute) + int8 embeddings | 109 MB | 0.907 | 0.907 | 0.936 | – | 291 |
| PyTorch's own dynamic int8 | – | – | – | 0.867 | – | – |

The shipped 8-epoch model behaves the same way. Its int8 file scores gold 0.932, gold2 0.928 and test 0.969, against
0.929 / 0.926 / 0.969 before quantisation.

* **int8 costs no accuracy on the M4** and is the fastest option there.
* **The first 4-bit recipe was not worth it on the M4:** it saved 42 MB, cost about a point on test, and was 6×
  slower. `--int4` now uses a different recipe, the one below; that recipe is not yet measured on the M4.
* **PyTorch's quantiser is the wrong tool here.** Its dynamic int8 quantised the same encoder matrices and lost 8
  points on test. The likely cause, not verified, is its one scale per matrix; ONNX Runtime used one scale per output
  channel.

## On x86 CPUs: use the 4-bit file

**The problem.**
* **What happens:** the int8 file is accurate on ARM only. On x86 CPUs without VNNI it loses 30–45 points.
  * **Affected CPUs:** Intel before Ice Lake / Cascade Lake, AMD before Zen 4.
  * **The shipped file itself was tested** on Kaggle's AVX2 Xeon: gold 0.524.
  * **The same happened on a Skylake AVX-512 Xeon.**
* **Why:** ONNX Runtime multiplies u8 activations by s8 weights with `VPMADDUBSW`, which adds pairs of products into
  16-bit integers that saturate. VNNI CPUs, and ARM's int8 dot product, accumulate in 32 bits.
* **Ruled out:** keeping the attention products in fp32 did not help, so the loss is in the weight matrices.
* **Partial fixes:** saturation-free int8 recipes (7-bit weights, or u8 × u8) recovered most of Ettin, still 2–5
  points short. UltraBERT stayed 15–20 points short.

**The fix: weight-only quantization.** `export_encoder.py --int4` writes `model.int4.onnx`:
* 4-bit weights in blocks of 32 (ONNX Runtime's `MatMulNBits`);
* int8 compute with a scale per block of 32 activations (`accuracy_level=4`), so no 16-bit sums can overflow;
* int8 embeddings.

Kernel `aerojev-int8fix` (Kaggle's Intel Xeon at 2.2 GHz, AVX2, no VNNI). Gold / gold2 / test:

| Ettin-150M, 8 epochs | fp32 | int8 (shipped recipe) | 8-bit weights, fp32 compute | **4-bit weights, int8 per block** |
|---|---|---|---|---|
| shipped run | 0.929 / 0.926 / 0.969 | 0.522 / 0.532 / 0.496 | 0.929 / 0.921 / 0.969 | **0.933 / 0.935 / 0.966** |
| second seed | 0.933 / 0.917 / 0.958 | 0.606 / 0.558 / 0.591 | 0.935 / 0.917 / 0.959 | **0.926 / 0.912 / 0.958** |

| File, same CPU | ms per record, 1 / 4 threads | File | Peak RSS (1 thread) |
|---|---|---|---|
| fp32 | 478 / 258 | 597 MB | 956 MB |
| int8 (wrong answers here) | 378 / 200 | 151 MB | 431 MB |
| 8-bit weights, fp32 compute | 950 / 494 | 164 MB | 327 MB |
| **4-bit weights, int8 per block** | **584 / 320** | **109 MB** | **328 MB** |

* **Weight-only quantization is lossless on x86**, for both models and both seeds.
* **4-bit is the low-memory x86 file:** 5.5× smaller than fp32 with a third of the memory, at about 1.2× its CPU
  time.
* **fp32 is the faster lossless x86 file** when about 1 GB of RAM is acceptable.
* **8-bit weight-only is lossless but the slowest** on this CPU, so it is not useful here.
* **The machine is slow:** Kaggle's CPU is a 2.2 GHz cloud Xeon, about 10× slower per record than the M4 for the
  same model. Expect a modern x86 server to be much faster.

**Automatic choice.** `aerojev/lite.py` (and `server.py --preset tiny`) picks the file by CPU. On x86 the order is
4-bit, fp32, int8; on ARM it is int8, 4-bit, fp32. If only the int8 file exists on an x86 CPU without VNNI, it warns.
Pass `--weights int8|int4|fp32` (server, `eval.evaluate`, `scripts/bench_lite.py`) to override.

**Getting the shipped model's x86 file.** It was built on Kaggle and kept there, to save local disk. Download it,
then copy it next to the model's `decider.json` and `tokenizer.json`. The variant's file name is `model.int8.onnx`;
rename it to `model.int4.onnx`:

```bash
kaggle kernels output sparshnagpal/aerojev-int8fix -p /tmp/x86 --file-pattern 'onnx/e150-ep8/v-mnb4-a4/.*'
cp /tmp/x86/onnx/e150-ep8/v-mnb4-a4/model.int8.onnx models/aerojev-ettin-150m/model.int4.onnx
```

**Not measured yet:**
* the new 4-bit file on the M4;
* int8 on x86 CPUs that have VNNI;
* a source-built ONNX Runtime tuned for the CPU.

All variants and the diagnosis: `results/int8diag/int8_x86.md`. To test recipes on another CPU, run
`scripts/int8_variants.py --dir <export dir with model.onnx>`. It writes one folder per recipe, which you then
evaluate with `eval.evaluate --onnx`.

## UltraBERT-150M: tested, not adopted

UltraBERT-150M (`dmis-lab`, October 2026, Apache-2.0) is a newer encoder of the same size:
* 24 layers;
* SwiGLU feed-forward layers;
* sliding-window attention on two layers in three;
* digit-level number tokens.

The digit-level tokens looked promising for `delay_bucket`, which compares dates. It was trained exactly like the
shipped Ettin model, with the same data, heads, loss, 8 epochs and evaluation. Only the learning rate differed: its
card prefers lower ones, so it got 1e-5 and 3e-5, two runs each. The rule for replacing Ettin was fixed before the
run:
* **Learning rate:** the one with the higher validation mean (3e-5) is the one compared.
* **Accuracy:** at least 1 point higher, on the two-run mean of gold, gold2 and test.
* **Speed:** at most 1.3× Ettin's CPU latency.

| Two-run means (gold / gold2 / test) | UltraBERT, LR 3e-5 | Ettin | Difference (mean of 3) | Latency ratio, 1 / 4 threads | Rule |
|---|---|---|---|---|---|
| PyTorch (T4) | 0.938 / 0.927 / 0.982 | 0.932 / 0.921 / 0.964 | +1.01 | – | – |
| fp32 ONNX (x86) | 0.938 / 0.927 / 0.982 | 0.931 / 0.921 / 0.964 | +1.04 | 1.10 / 1.06 | passes |
| 8-bit weights (x86) | 0.938 / 0.931 / 0.982 | 0.932 / 0.919 / 0.964 | +1.20 | 1.01 / 1.08 | passes, but slow |
| 4-bit weights, the x86 file | 0.938 / 0.919 / 0.982 | 0.930 / 0.924 / 0.962 | +0.79 | 1.11 / 1.20 | fails |

* **Most of the lead is on the generated test set** (about +2 points). There, `delay_bucket` rose from 0.954 to
  0.991.
* **On the hand-written gold and gold2 sets the gap is within noise.**
* **Training cost:** UltraBERT took 34 minutes on a T4, against 28.
* **Verdict: Ettin stays the shipped `tiny` model.** UltraBERT is the more accurate choice where fp32 or 8-bit
  weights fit. Its int8 on ARM is untested.
* **UltraBERT is more sensitive to quantization.** It lost 15–20 points with the saturation-free int8 recipes,
  against Ettin's 2–5.

Tables: `results/ultra/ultra.md`, `results/int8diag/int8_x86.md`.

Memory settings (int8, 1 thread, M4): turning off ONNX Runtime's memory arena lowers the peak from 443 to 355 MB for
about 4 ms more per record. `aerojev/lite.py` does this. Turning off weight prepacking saves a little more memory but
makes inference 2.4× slower. Peak memory grows with the thread count: 0.40 GB at 1 thread, 0.48 GB at 4.

## Training it

| | T4 GPU | Apple M4 (measured) | Regular 2-core CPU (estimate) |
|---|---|---|---|
| Time per training record | 0.02 s | **0.20 s** | ~1 s |
| 4 epochs × 10,000 records | 14 min | **2.2 h** | ~11 h |
| 8 epochs | 28 min | 4.5 h | ~23 h |
| Peak RAM (batch 16, fp32) | – | 6.1 GB | not measured |

The M4 time was measured with 40 optimizer steps (640 records) of `train/train_encoder.py`. The regular-CPU figures
are FLOP projections calibrated on earlier measurements, not measured.

On a CPU the encoder costs about the same to train as the catalogue-free Qwen reference (2.4 h on the M4 for 12,000
records). Its gains are at decision time: about 3× faster, half the file size and less than half the memory of the
pruned Qwen.

```bash
# train (GPU: ~30 min on a T4; CPU: add --device cpu, see the table above)
python -m train.train_encoder --model jhu-clsp/ettin-encoder-150m --epochs 8 \
    --out runs/ettin-150m --results results/ft-ettin-150m

# export: int8 ONNX (ARM) + 4-bit ONNX (x86, --int4) + tokenizer + decider.json (temperatures fitted on val)
python scripts/export_encoder.py --model runs/ettin-150m/model --out models/aerojev-ettin-150m \
    --temps results/ft-ettin-150m/metrics.json --int4

# check accuracy and speed of the exported file
python -m eval.evaluate --onnx models/aerojev-ettin-150m --data data/processed_v2 \
    --sets val,test,gold,gold2 --out results/onnx-ettin-150m
python scripts/bench_lite.py --model models/aerojev-ettin-150m --threads 1 4
```

Training options in `train/train_encoder.py`:
* `--epochs` (8 shipped; 4 is nearly as good)
* `--lr 5e-5` for the encoder and `--head-lr 1e-3` for the new heads
* `--batch 16` and `--max-len 512`. Records here are at most ~260 tokens; Ettin itself reads up to 8k.
* `--pool mean|attn`
* `--teacher` (distillation)

## Running it

```bash
pip install -r requirements-lite.txt          # numpy, onnxruntime, tokenizers: no PyTorch
python server.py --preset tiny                # models/aerojev-ettin-150m; int8 on ARM, 4-bit on x86; same API and page
```

```python
from aerojev.lite import LiteEngine
eng = LiteEngine("models/aerojev-ettin-150m", threads=2)
eng.decide("Kestrel says the 718 disks slip 6 weeks; 2 weeks on hand; sole source.")
# {"event_type": {"answer": "delivery_delay", "confidence": 1.0, "probabilities": {...}}, ...}
```

The server process with this model holds about 400 MB. `GET /health` reports `"engine": "onnxruntime"`.

## Files

* `train/train_encoder.py`: training. Writes an fp16 checkpoint, the heads, `decider.json` and predictions.
* `scripts/export_encoder.py`: ONNX export, parity check against PyTorch, int8 (ARM) and 4-bit (`--int4`, x86) files.
* `scripts/int8_variants.py`: builds other quantization recipes from an fp32 export, to test them on a new CPU.
* `aerojev/lite.py`: `LiteEngine`, the torch-free runtime; picks the weights file by CPU.
* `aerojev/engine.py`: `EncoderEngine` runs the PyTorch checkpoint, for checks.
* `scripts/bench_lite.py`: latency and peak memory per thread count.
* `models/aerojev-ettin-150m/`: the shipped model (`model.int8.onnx`, `tokenizer.json`, `decider.json`; add
  `model.int4.onnx` for x86). It is not in git, like all weights.
* `results/encoder/encoder.md`, `results/ultra/ultra.md`, `results/int8diag/int8_x86.md`: the measurements.
