| Level | Run | Records | Gold | Gold2 | Test | Test v2 | Val | Injection | Gold, shuffled options | T4 train min | T4 val min | GFLOP/record | Regular CPU h (est.) | M4 h (est.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | l4-nocat-12k | 12,000 | 0.928 | 0.905 | 0.944 | 0.942 | 0.969 | 0.944 | – | 16.5 | 2.1 | 309 | 11.6 | 2.5 |
| P0 | l4-nocat-12k-seed1 | 12,000 | 0.940 | 0.919 | 0.954 | 0.953 | 0.964 | 0.946 | – | 17.8 | 2.3 | 309 | 11.6 | 2.5 |
| P1 | p-d16top | 12,000 | 0.907 | 0.900 | 0.959 | 0.956 | 0.978 | 0.958 | – | 11.6 | 1.6 | 206 | 7.7 | 1.7 |
| P1 | p-d16auto | 12,000 | 0.860 | 0.875 | 0.898 | 0.883 | 0.936 | 0.896 | – | 12.7 | 1.8 | 206 | 7.7 | 1.7 |
| P2 | p-w2560 | 12,000 | 0.875 | 0.896 | 0.952 | 0.946 | 0.972 | 0.945 | – | 16.3 | 2.1 | 186 | 7.0 | 1.5 |
| P3 | p-d16w | 12,000 | 0.893 | 0.884 | 0.949 | 0.940 | 0.974 | 0.947 | – | 10.5 | 1.8 | 124 | 4.7 | 1.0 |
| P3 | p-d16w-full-2e-5 | 12,000 | 0.914 | 0.917 | 0.951 | 0.943 | 0.981 | 0.953 | – | 8.8 | 1.0 | 182 | 6.8 | 1.5 |
| P3 | p-d16w-full-6e-5 | 12,000 | 0.872 | 0.870 | 0.916 | 0.912 | 0.939 | 0.908 | – | 8.4 | 0.9 | 182 | 6.8 | 1.5 |
| T1 | t-d16-noheal | no training | 0.561 | 0.551 | 0.550 | 0.548 | 0.547 | 0.526 | – | – | – | – | – | – |
| T2 | t-d16w-noheal | no training | 0.424 | 0.391 | 0.402 | 0.403 | 0.417 | 0.404 | – | – | – | – | – | – |
| T3 | t-d16w-heal3k | 3,000 | 0.874 | 0.845 | 0.942 | 0.936 | 0.963 | 0.943 | – | 2.4 | 0.5 | 125 | 1.2 | 0.3 |

Levels:
* **l4-nocat-12k**: reference: full Qwen2.5-0.5B, catalogue-free, 12k records, LoRA
* **l4-nocat-12k-seed1**: the reference again with another seed (run-to-run noise)
* **p-d16top**: 16 blocks: the top 8 removed; LoRA, 12k records
* **p-d16auto**: 16 blocks: the 8 most alike consecutive blocks removed (final block kept); LoRA, 12k
* **p-w2560**: 24 blocks, FFN narrowed 4864 -> 2560 (activation x weight scores); LoRA, 12k
* **p-d16w**: 16 blocks (top 8 removed) + FFN 2560; LoRA, 12k
* **p-d16w-full-2e-5**: 16 blocks + FFN 2560; full fine-tuning, LR 2e-5, 12k
* **p-d16w-full-6e-5**: 16 blocks + FFN 2560; full fine-tuning, LR 6e-5, 12k
* **t-d16-noheal**: the trained reference with its top 8 blocks removed, no retraining
* **t-d16w-noheal**: the trained reference, top 8 blocks removed + FFN 2560, no retraining
* **t-d16w-heal3k**: T2 retrained ('healed') with 3k records of LoRA (cost on top of the reference run)

Change against P0 (accuracy points; compute ratio):

* l4-nocat-12k-seed1: gold +1.2 gold2 +1.4 test +1.0; the same training compute in total
* p-d16top: gold -2.1 gold2 -0.5 test +1.4; 1.5× less training compute in total
* p-d16auto: gold -6.8 gold2 -3.0 test -4.6; 1.5× less training compute in total
* p-w2560: gold -5.3 gold2 -0.9 test +0.8; 1.7× less training compute in total
* p-d16w: gold -3.5 gold2 -2.1 test +0.5; 2.5× less training compute in total
* p-d16w-full-2e-5: gold -1.4 gold2 +1.2 test +0.7; 1.7× less training compute in total
* p-d16w-full-6e-5: gold -5.6 gold2 -3.5 test -2.9; 1.7× less training compute in total
* t-d16-noheal: gold -36.7 gold2 -35.4 test -39.4
* t-d16w-noheal: gold -50.4 gold2 -51.4 test -54.2
* t-d16w-heal3k: gold -5.4 gold2 -6.0 test -0.2; 9.9× less training compute in total
