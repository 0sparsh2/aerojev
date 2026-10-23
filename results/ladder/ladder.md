| Level | Run | Records | Gold | Gold2 | Test | Test v2 | Val | Injection | Gold, shuffled options | T4 train min | T4 val min | GFLOP/record | Regular CPU h (est.) | M4 h (est.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | l0-baseline | 12,000 | 0.922 | 0.919 | 0.948 | 0.944 | 0.966 | 0.949 | 0.772 | 62.4 | 3.5 | 1864 | 69.8 | 15.0 |
| L1 | l1-shared | 12,000 | 0.925 | 0.935 | 0.947 | 0.947 | 0.975 | 0.948 | 0.800 | 25.6 | 3.7 | 516 | 19.3 | 4.2 |
| L2 | l2-shared-all9 | 12,000 | 0.918 | 0.900 | 0.957 | 0.956 | 0.972 | 0.956 | 0.753 | 25.1 | 2.1 | 536 | 20.1 | 4.3 |
| L3 | l3-6k-lr2e-4 | 6,000 | 0.886 | 0.894 | 0.930 | 0.929 | 0.947 | 0.926 | 0.766 | 12.6 | 1.1 | 536 | 10.0 | 2.2 |
| L3 | l3-6k-lr5e-4 | 6,000 | 0.875 | 0.854 | 0.915 | 0.919 | 0.920 | 0.917 | 0.738 | 11.8 | 1.1 | 536 | 10.0 | 2.2 |
| L3 | l3-6k-lr1e-3 | 6,000 | 0.519 | 0.514 | 0.524 | 0.522 | 0.545 | 0.518 | 0.163 | 11.6 | 1.1 | 536 | 10.0 | 2.2 |
| L3 | l3-6k-distill | 6,000 | 0.886 | 0.843 | 0.916 | 0.914 | 0.929 | 0.904 | 0.734 | 12.9 | 1.1 | 536 | 10.0 | 2.2 |
| L4 | l4-nocat-12k | 12,000 | 0.928 | 0.905 | 0.944 | 0.942 | 0.969 | 0.944 | – | 16.5 | 2.1 | 309 | 11.6 | 2.5 |
| L4 | l4-nocat-6k | 6,000 | 0.921 | 0.912 | 0.953 | 0.946 | 0.956 | 0.945 | – | 8.7 | 1.1 | 309 | 5.8 | 1.2 |
| L4 | l4-nocat-6k-seed1 | 6,000 | 0.885 | 0.882 | 0.921 | 0.922 | 0.935 | 0.912 | – | 8.4 | 1.1 | 309 | 5.8 | 1.2 |
| L4 | l4-nocat-4.8k | 4,800 | 0.854 | 0.838 | 0.907 | 0.908 | 0.928 | 0.911 | – | 6.5 | 0.9 | 310 | 4.6 | 1.0 |
| L4 | l4-nocat-3k | 3,000 | 0.863 | 0.822 | 0.902 | 0.899 | 0.920 | 0.901 | – | 4.1 | 0.7 | 310 | 2.9 | 0.6 |
| L6 | l6-probe | 10,000 | 0.799 | 0.771 | 0.831 | 0.814 | 0.896 | 0.803 | – | 4.3 | 0.0 | 152 | 4.7 | 1.0 |

Levels:
* **l0-baseline**: today's recipe: one catalogue per record, 5-9 questions, 12k records
* **l1-shared**: one catalogue per 8-record step (exact gradients)
* **l2-shared-all9**: L1 + all 9 questions, LoRA dropout 0, cached-catalogue validation
* **l3-6k-lr2e-4**: L2 on 6k records, LR 2e-4
* **l3-6k-lr5e-4**: L2 on 6k records, LR 5e-4
* **l3-6k-lr1e-3**: L2 on 6k records, LR 1e-3
* **l3-6k-distill**: L2 on 6k records, LR 5e-4, distilled from fine-tuned Qwen3-1.7B
* **l4-nocat-12k**: no catalogue in the prompt, all 9 questions, 12k records
* **l4-nocat-6k**: no catalogue in the prompt, all 9 questions, 6k records, LR 5e-4
* **l4-nocat-6k-seed1**: the 6k run again with another seed (run-to-run noise)
* **l4-nocat-4.8k**: no catalogue, 4.8k records, LR 5e-4
* **l4-nocat-3k**: no catalogue, 3k records, LR 5e-4
* **l6-probe**: frozen model + linear heads (no LoRA), features from one forward pass

Change against L0 (accuracy points; compute ratio):

* l1-shared: gold +0.3 gold2 +1.6 test -0.1; 3.6× less training compute in total
* l2-shared-all9: gold -0.4 gold2 -1.9 test +0.9; 3.5× less training compute in total
* l3-6k-lr2e-4: gold -3.6 gold2 -2.5 test -1.8; 6.9× less training compute in total
* l3-6k-lr5e-4: gold -4.7 gold2 -6.5 test -3.3; 6.9× less training compute in total
* l3-6k-lr1e-3: gold -40.3 gold2 -40.5 test -42.3; 6.9× less training compute in total
* l3-6k-distill: gold -3.6 gold2 -7.6 test -3.2; 6.9× less training compute in total
* l4-nocat-12k: gold +0.6 gold2 -1.4 test -0.3; 6.0× less training compute in total
* l4-nocat-6k: gold -0.1 gold2 -0.7 test +0.6; 12.0× less training compute in total
* l4-nocat-6k-seed1: gold -3.7 gold2 -3.7 test -2.7; 12.1× less training compute in total
* l4-nocat-4.8k: gold -6.8 gold2 -8.1 test -4.0; 15.0× less training compute in total
* l4-nocat-3k: gold -6.0 gold2 -9.7 test -4.5; 24.1× less training compute in total
* l6-probe: gold -12.4 gold2 -14.8 test -11.7; 14.7× less training compute in total
