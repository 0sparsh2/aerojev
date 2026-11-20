# Dynamic int8 on x86 CPUs without VNNI

Kernel `sparshnagpal/aerojev-int8diag` (CPU: Intel Xeon @ 2.20GHz, AVX2, no AVX-512, no VNNI). Same results on
the Skylake AVX-512 Xeon (no VNNI) of `aerojev-ultra`. Accuracy on gold / gold2 / test, ONNX Runtime 1.32.0.

| Model | fp32 ONNX | int8 as shipped (s8) | s8, attention products fp32 | s8, 7-bit weights (reduce_range) | u8 weights (u8 x u8) |
|---|---|---|---|---|---|
| Ettin-150M, 8 epochs (shipped weights) | 0.929 / 0.926 / 0.969 | 0.522 / 0.532 / 0.496 | 0.524 / 0.528 / 0.484 | 0.879 / 0.868 / 0.925 | 0.897 / 0.868 / 0.929 |
| Ettin-150M, 8 epochs, second seed | 0.933 / 0.917 / 0.958 | 0.606 / 0.558 / 0.591 | 0.596 / 0.604 / 0.569 | 0.901 / 0.875 / 0.936 | 0.910 / 0.880 / 0.935 |
| UltraBERT-150M, 8 epochs, LR 3e-5 | 0.942 / 0.931 / 0.983 | 0.432 / 0.419 / 0.385 | 0.433 / 0.405 / 0.395 | 0.722 / 0.736 / 0.827 | 0.715 / 0.752 / 0.822 |
| UltraBERT-150M, second seed | 0.935 / 0.924 / 0.981 | 0.461 / 0.454 / 0.312 | 0.440 / 0.444 / 0.316 | 0.739 / 0.794 / 0.823 | 0.739 / 0.775 / 0.821 |

Latency on that CPU (median ms per record, 300 gold+test records, 1 / 4 threads):

| Model | file | 1 thread | 4 threads | peak RSS MB |
|---|---|---|---|---|
| e150-ep8-seed1/v-s8 | int8 | 378.1 | 199.6 | 431 |
| e150-ep8-seed1/v-s8-rr | int8 | 388.2 | 188.5 | 432 |
| e150-ep8-seed1/v-u8 | int8 | 444.0 | 227.5 | 432 |
| e150-ep8-seed1/v-s8-constb | int8 | 368.6 | 220.4 | 385 |
| u150-ep8-lr3e-5/v-s8 | int8 | 402.3 | 201.2 | 398 |
| u150-ep8-lr3e-5/v-s8-rr | int8 | 398.4 | 205.4 | 385 |
| u150-ep8-lr3e-5/v-u8 | int8 | 479.9 | 251.4 | 393 |
| u150-ep8-lr3e-5/v-s8-constb | int8 | 394.5 | 200.9 | 364 |
| e150-ep8-seed1 | fp32 | 478.3 | 258.5 | 956 |
| u150-ep8-lr3e-5 | fp32 | 528.3 | 274.4 | 907 |

Findings:

* The fp32 graph matches PyTorch (also on single records of other lengths) and scores like the GPU model: the export
  is fine. The loss is in int8 inference on this CPU.
* The shipped recipe (u8 activations x s8 weights) collapses: VPMADDUBSW's int16 saturation on CPUs without VNNI.
  Leaving the attention products in fp32 does not help, so it is the weight matrices.
* Saturation-free recipes (7-bit weights, u8 x u8) recover most of Ettin (2-5 points below fp32) but not UltraBERT
  (15-20 points below): per-tensor dynamic activation quantization itself hurts UltraBERT.
* The same s8 recipe is lossless on Apple Silicon (M4: Ettin int8 gold 0.932 vs 0.929 fp32).

## The fix: weight-only quantization (kernel `sparshnagpal/aerojev-int8fix`, same AVX2 Xeon)

The shipped int8 file, built on the M4, scores 0.524 / 0.528 / 0.484 on this CPU: the shipped model is affected.

| Model | fp32 | 8-bit weights, fp32 compute (mnb8) | 8-bit weights, int8 per-block compute (mnb8-a4) | 4-bit weights, int8 per-block compute (mnb4-a4) |
|---|---|---|---|---|
| Ettin-150M, 8 epochs (shipped weights) | 0.929 / 0.926 / 0.969 | 0.929 / 0.921 / 0.969 | 0.924 / 0.921 / 0.969 | 0.933 / 0.935 / 0.966 |
| Ettin-150M, second seed | 0.933 / 0.917 / 0.958 | 0.935 / 0.917 / 0.959 | 0.936 / 0.914 / 0.959 | 0.926 / 0.912 / 0.958 |
| UltraBERT-150M, LR 3e-5 | 0.942 / 0.931 / 0.983 | 0.942 / 0.933 / 0.982 | 0.942 / 0.933 / 0.983 | 0.946 / 0.921 / 0.982 |
| UltraBERT-150M, second seed | 0.935 / 0.924 / 0.981 | 0.935 / 0.928 / 0.981 | 0.935 / 0.926 / 0.981 | 0.931 / 0.917 / 0.981 |

Speed on this CPU (median ms per record over 300 gold + test records, 1 / 4 threads), file size, peak RSS at 1 thread:

| File | Ettin-150M | UltraBERT-150M | Size (Ettin / Ultra) | Peak RSS (Ettin / Ultra) |
|---|---|---|---|---|
| fp32 | 478 / 258 | 528 / 274 | 597 / 605 MB | 956 / 907 MB |
| dynamic int8, s8 (wrong answers here) | 378 / 200 | 402 / 201 | 151 / 153 MB | 431 / 398 MB |
| 8-bit weights (mnb8) | 950 / 494 | 961 / 532 | 164 / 166 MB | 327 / 368 MB |
| 8-bit weights, int8 per block (mnb8-a4) | 864 / 464 | 982 / 525 | 164 / 166 MB | 412 / 430 MB |
| **4-bit weights, int8 per block (mnb4-a4)** | 584 / 320 | 651 / 384 | 109 / 110 MB | 328 / 385 MB |

* Weight-only quantization is lossless on x86 for both models: every variant is within noise of fp32.
* On this CPU fp32 is still the fastest lossless file. 4-bit (mnb4-a4) is about 1.2-1.5x slower than fp32 but
  5.5x smaller and needs about a third of the memory. 8-bit weight-only is slower than both: not useful here.
* Recommended: model.int8.onnx on ARM / Apple Silicon (lossless there and fastest), model.int4.onnx (mnb4-a4) on
  x86; aerojev/lite.py picks by CPU. Not yet measured: mnb4-a4 on the M4, and dynamic int8 on x86 with VNNI.
* The u8 variant with the FFN down-projections in fp32 failed to build (quantize_dynamic writes next to its
  read-only input); not pursued, since weight-only quantization already recovers everything.
