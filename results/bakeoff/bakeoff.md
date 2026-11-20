| Level | Run | Records | Gold | Gold2 | Test | Test v2 | Val | Injection | Gold, shuffled options | T4 train min | T4 val min | GFLOP/record | Regular CPU h (est.) | M4 h (est.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | l4-nocat-12k | 12,000 | 0.928 | 0.905 | 0.944 | 0.942 | 0.969 | 0.944 | – | 16.5 | 2.1 | 309 | 11.6 | 2.5 |
| R | l4-nocat-12k-seed1 | 12,000 | 0.940 | 0.919 | 0.954 | 0.953 | 0.964 | 0.946 | – | 17.8 | 2.3 | 309 | 11.6 | 2.5 |
| Q | p-d16w-full-2e-5 | 12,000 | 0.914 | 0.917 | 0.951 | 0.943 | 0.981 | 0.953 | – | 8.8 | 1.0 | 182 | 6.8 | 1.5 |
| Q | p-d16w-full-2e-5-seed1 | 12,000 | 0.899 | 0.910 | 0.953 | 0.947 | 0.973 | 0.942 | – | 8.6 | 0.9 | 182 | 6.8 | 1.5 |
| D | b-gemma3-270m-3e-5 | 12,000 | 0.849 | 0.847 | 0.882 | 0.862 | 0.955 | 0.869 | – | 12.4 | 1.2 | 154 | 5.8 | 1.2 |
| D | b-gemma3-270m-1e-4 | 12,000 | 0.796 | 0.819 | 0.838 | 0.843 | 0.927 | 0.837 | – | 12.4 | 1.2 | 154 | 5.8 | 1.2 |
| D | b-smollm2-135m-3e-5 | 12,000 | 0.839 | 0.822 | 0.860 | 0.863 | 0.916 | 0.854 | – | 10.8 | 1.4 | 160 | 6.0 | 1.3 |
| D | b-smollm2-135m-1e-4 | 12,000 | 0.871 | 0.882 | 0.912 | 0.912 | 0.959 | 0.911 | – | 10.8 | 1.3 | 160 | 6.0 | 1.3 |
| E | b-ettin-150m | 20,000 | 0.910 | 0.912 | 0.945 | 0.938 | 0.969 | 0.939 | – | 6.9 | 0.1 | 91 | 5.7 | 1.2 |
| E | b-ettin-68m | 20,000 | 0.924 | 0.900 | 0.933 | 0.927 | 0.964 | 0.928 | – | 3.6 | 0.1 | 36 | 2.2 | 0.5 |
| E | b-ettin-32m | 20,000 | 0.881 | 0.863 | 0.915 | 0.904 | 0.946 | 0.896 | – | 1.6 | 0.0 | 11 | 0.7 | 0.1 |

Levels:
* **l4-nocat-12k**: reference: Qwen2.5-0.5B (494M), catalogue-free, LoRA
* **l4-nocat-12k-seed1**: the reference, second seed
* **p-d16w-full-2e-5**: Qwen2.5-0.5B pruned to 16 blocks + FFN 2560 (276M), full fine-tuning
* **p-d16w-full-2e-5-seed1**: the pruned model, second seed
* **b-gemma3-270m-3e-5**: Gemma 3 270M (268M, 168M of it vocabulary), full fine-tuning, LR 3e-5
* **b-gemma3-270m-1e-4**: Gemma 3 270M, LR 1e-4
* **b-smollm2-135m-3e-5**: SmolLM2-135M (135M), full fine-tuning, LR 3e-5
* **b-smollm2-135m-1e-4**: SmolLM2-135M, LR 1e-4
* **b-ettin-150m**: Ettin encoder 150M + 9 heads, 2 epochs, LR 5e-5
* **b-ettin-68m**: Ettin encoder 68M + 9 heads, 2 epochs, LR 8e-5
* **b-ettin-32m**: Ettin encoder 32M + 9 heads, 2 epochs, LR 1e-4

Change against R (accuracy points; compute ratio):

* l4-nocat-12k-seed1: gold +1.2 gold2 +1.4 test +1.0; the same training compute in total
* p-d16w-full-2e-5: gold -1.4 gold2 +1.2 test +0.7; 1.7× less training compute in total
* p-d16w-full-2e-5-seed1: gold -2.9 gold2 +0.5 test +0.8; 1.7× less training compute in total
* b-gemma3-270m-3e-5: gold -7.9 gold2 -5.8 test -6.2; 2.0× less training compute in total
* b-gemma3-270m-1e-4: gold -13.2 gold2 -8.6 test -10.6; 2.0× less training compute in total
* b-smollm2-135m-3e-5: gold -8.9 gold2 -8.3 test -8.4; 1.9× less training compute in total
* b-smollm2-135m-1e-4: gold -5.7 gold2 -2.3 test -3.2; 1.9× less training compute in total
* b-ettin-150m: gold -1.8 gold2 +0.7 test +0.1; 2.0× less training compute in total
* b-ettin-68m: gold -0.4 gold2 -0.5 test -1.1; 5.2× less training compute in total
* b-ettin-32m: gold -4.7 gold2 -4.2 test -2.9; 17.2× less training compute in total
