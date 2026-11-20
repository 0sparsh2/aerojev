| Level | Run | Records | Gold | Gold2 | Test | Test v2 | Val | Injection | Gold, shuffled options | T4 train min | T4 val min | GFLOP/record | Regular CPU h (est.) | M4 h (est.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | l4-nocat-12k | 12,000 | 0.928 | 0.905 | 0.944 | 0.942 | 0.969 | 0.944 | – | 16.5 | 2.1 | 309 | 11.6 | 2.5 |
| R | l4-nocat-12k-seed1 | 12,000 | 0.940 | 0.919 | 0.954 | 0.953 | 0.964 | 0.946 | – | 17.8 | 2.3 | 309 | 11.6 | 2.5 |
| E0 | b-ettin-150m | 20,000 | 0.910 | 0.912 | 0.945 | 0.938 | 0.969 | 0.939 | – | 6.9 | 0.1 | 91 | 5.7 | 1.2 |
| E0 | e150-seed1 | 20,000 | 0.926 | 0.889 | 0.948 | 0.942 | 0.970 | 0.943 | – | 7.0 | 0.1 | 91 | 5.7 | 1.2 |
| E1 | e150-ep4 | 40,000 | 0.929 | 0.914 | 0.960 | 0.946 | 0.987 | 0.958 | – | 13.6 | 0.2 | 91 | 11.4 | 2.4 |
| E1 | e150-ep4-seed1 | 40,000 | 0.943 | 0.907 | 0.964 | 0.954 | 0.987 | 0.954 | – | 13.5 | 0.2 | 91 | 11.4 | 2.4 |
| E1 | e150-ep8 | 80,000 | 0.929 | 0.926 | 0.969 | 0.960 | 0.987 | 0.959 | – | 27.7 | 0.4 | 91 | 22.8 | 4.9 |
| E2 | e150-ep4-lr1e-4 | 40,000 | 0.924 | 0.917 | 0.968 | 0.958 | 0.985 | 0.963 | – | 13.9 | 0.2 | 91 | 11.4 | 2.4 |
| E3 | e150-ep4-attn | 40,000 | 0.940 | 0.903 | 0.961 | 0.953 | 0.985 | 0.953 | – | 13.9 | 0.2 | 91 | 11.4 | 2.4 |
| E4 | e150-ep4-kd | 40,000 | 0.933 | 0.940 | 0.943 | 0.932 | 0.972 | 0.938 | – | 13.7 | 0.2 | 91 | 11.4 | 2.4 |
| E5 | e150-ep4-attn-kd | 40,000 | 0.928 | 0.905 | 0.966 | 0.956 | 0.982 | 0.961 | – | 13.7 | 0.2 | 91 | 11.4 | 2.4 |

Levels:
* **l4-nocat-12k**: reference: Qwen2.5-0.5B (494M), catalogue-free, LoRA
* **l4-nocat-12k-seed1**: the reference, second seed
* **b-ettin-150m**: Ettin-150M + 9 heads, 2 epochs, LR 5e-5, mean pooling (bake-off)
* **e150-seed1**: the same, second seed
* **e150-ep4**: 4 epochs
* **e150-ep4-seed1**: 4 epochs, second seed
* **e150-ep8**: 8 epochs
* **e150-ep4-lr1e-4**: 4 epochs, LR 1e-4
* **e150-ep4-attn**: 4 epochs, one attention pooling per question
* **e150-ep4-kd**: 4 epochs, distilled from the two reference runs (averaged)
* **e150-ep4-attn-kd**: 4 epochs, attention pooling + distillation

Change against R (accuracy points; compute ratio):

* l4-nocat-12k-seed1: gold +1.2 gold2 +1.4 test +1.0; the same training compute in total
* b-ettin-150m: gold -1.8 gold2 +0.7 test +0.1; 2.0× less training compute in total
* e150-seed1: gold -0.1 gold2 -1.6 test +0.4; 2.0× less training compute in total
* e150-ep4: gold +0.1 gold2 +0.9 test +1.6; the same training compute in total
* e150-ep4-seed1: gold +1.5 gold2 +0.2 test +2.0; the same training compute in total
* e150-ep8: gold +0.1 gold2 +2.1 test +2.5; 2.0× more training compute in total
* e150-ep4-lr1e-4: gold -0.4 gold2 +1.2 test +2.4; the same training compute in total
* e150-ep4-attn: gold +1.3 gold2 -0.2 test +1.6; the same training compute in total
* e150-ep4-kd: gold +0.6 gold2 +3.5 test -0.1; the same training compute in total
* e150-ep4-attn-kd: gold +0.0 gold2 -0.0 test +2.2; the same training compute in total
