| Level | Run | Records | Gold | Gold2 | Test | Test v2 | Val | Injection | Gold, shuffled options | T4 train min | T4 val min | GFLOP/record | Regular CPU h (est.) | M4 h (est.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E | e150-ep8 | 80,000 | 0.929 | 0.926 | 0.969 | 0.960 | 0.987 | 0.959 | – | 27.7 | 0.4 | 91 | 22.8 | 4.9 |
| E | e150-ep8-seed1 | 80,000 | 0.935 | 0.917 | 0.958 | 0.953 | 0.986 | 0.957 | – | 27.6 | 0.4 | 91 | 22.8 | 4.9 |
| U1 | u150-ep8-lr1e-5 | 80,000 | 0.942 | 0.926 | 0.980 | 0.978 | 0.991 | 0.977 | – | 34.0 | 0.5 | 108 | 26.9 | 5.8 |
| U1 | u150-ep8-lr1e-5-seed1 | 80,000 | 0.947 | 0.931 | 0.979 | 0.980 | 0.990 | 0.979 | – | 33.9 | 0.5 | 108 | 26.9 | 5.8 |
| U3 | u150-ep8-lr3e-5 | 80,000 | 0.942 | 0.931 | 0.983 | 0.980 | 0.994 | 0.982 | – | 33.9 | 0.5 | 108 | 26.9 | 5.8 |
| U3 | u150-ep8-lr3e-5-seed1 | 80,000 | 0.935 | 0.924 | 0.981 | 0.981 | 0.993 | 0.983 | – | 33.9 | 0.5 | 108 | 26.9 | 5.8 |

Levels:
* **e150-ep8**: Ettin-150M, 8 epochs, LR 5e-5 (shipped, from the encoder round)
* **e150-ep8-seed1**: the same, second seed
* **u150-ep8-lr1e-5**: UltraBERT-150M, 8 epochs, LR 1e-5 (its card's MNLI best)
* **u150-ep8-lr1e-5-seed1**: the same, second seed
* **u150-ep8-lr3e-5**: UltraBERT-150M, 8 epochs, LR 3e-5
* **u150-ep8-lr3e-5-seed1**: the same, second seed

Change against E (accuracy points; compute ratio):

* e150-ep8-seed1: gold +0.6 gold2 -0.9 test -1.1; the same training compute in total
* u150-ep8-lr1e-5: gold +1.2 gold2 +0.0 test +1.1; 1.2× more training compute in total
* u150-ep8-lr1e-5-seed1: gold +1.8 gold2 +0.5 test +1.0; 1.2× more training compute in total
* u150-ep8-lr3e-5: gold +1.2 gold2 +0.5 test +1.3; 1.2× more training compute in total
* u150-ep8-lr3e-5-seed1: gold +0.6 gold2 -0.2 test +1.2; 1.2× more training compute in total

## Decision rule outcome (fixed before the run)

Rule: UltraBERT's LR = the higher two-seed mean val accuracy (3e-5: 0.9936 vs 1e-5: 0.9903). It replaces Ettin
if its two-seed int8 mean of (gold, gold2, test) is >= 1 point higher and its CPU latency <= 1.3x Ettin's.

* Full precision (fp32 ONNX on a Kaggle CPU, two-seed means): UltraBERT 0.938 / 0.927 / 0.982 vs Ettin
  0.931 / 0.921 / 0.964, average +1.04 points; latency 1.06-1.10x (results/int8diag/latency_variants.jsonl). Passes.
* int8: undecidable on Kaggle. Every Kaggle CPU tested lacks VNNI, where dynamic int8 breaks both models
  (results/int8diag/int8_x86.md). With the saturation-free u8 recipe UltraBERT loses 15-20 points against Ettin's
  2-5, so with today's int8 methods UltraBERT does not qualify; Ettin stays shipped.

### Update: with lossless x86 quantization (kernel `sparshnagpal/aerojev-int8fix`, results/int8diag/int8_x86.md)

Two-seed means of (gold, gold2, test), UltraBERT LR 3e-5 against Ettin, on a Kaggle x86 CPU. The difference is in
accuracy points, from unrounded scores:

| File | UltraBERT | Ettin | Difference | Latency ratio (1 / 4 threads) | Rule (>= +1 point, <= 1.3x) |
|---|---|---|---|---|---|
| fp32 | 0.949 | 0.939 | +1.04 | 1.10 / 1.06 | passes |
| 8-bit weights (mnb8) | 0.950 | 0.938 | +1.20 | 1.01 / 1.08 | passes (but this file is slow on x86) |
| 4-bit weights (mnb4-a4, the x86 file) | 0.946 | 0.938 | +0.79 | 1.11 / 1.20 | does not pass |

The lead comes mostly from the generated test set (about +2 points); gold and gold2 differ by less than the noise.
Ettin stays the shipped `tiny` model. UltraBERT is the more accurate choice where fp32 or 8-bit weights fit. Its
dynamic int8 on ARM (the M4 file type) is untested.
