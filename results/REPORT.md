# AeroJev results

Core = the 9 trained questions; accuracy is the mean over questions.

## Accuracy vs latency

Same protocol everywhere: 40 gold records, one at a time, warm process, 9 core decisions per record, 4 threads (classical models single-threaded). Hosted API latency is the client-measured p50 round trip. `*` = accuracy of the fp32 model (quantized accuracy not measured separately).

| System | Engine | Layout | Gold acc | Gold2 acc | Test acc | api p50 s/record | api ms/decision | Peak RSS MB | File MB |
|---|---|---|---|---|---|---|---|---|---|
| Jev (TypeSafe API) (jev-1.14.0) | hosted API (GPU, network) | + playbook + export definition | 0.918 | – | 0.882 | 0.120 | 10.0 | – | – |
| Jev (TypeSafe API) (jev-1.14.0) | hosted API (GPU, network) | + playbook rules | 0.892 | – | 0.853 | 0.175 | 14.6 | – | – |
| Jev (TypeSafe API) (jev-1.14.0) | hosted API (GPU, network) | questions as-is | 0.869 | – | 0.817 | 0.171 | 14.3 | – | – |

## Accuracy (core questions)

| System | Method | Layout | Gold acc | Gold macro-F1 | Test acc | Injection acc | Gold exact | ECE raw→cal (gold) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | zero-shot readout | branch | 0.508 | 0.355 | 0.445 | 0.438 | 0.000 | 0.071→0.063 |
| Qwen3-0.6B | zero-shot readout | branch | 0.632 | 0.498 | 0.570 | 0.583 | 0.000 | 0.224→0.054 |
| Qwen3-1.7B | zero-shot readout | branch | 0.704 | 0.669 | 0.660 | 0.638 | 0.050 | 0.266→0.122 |
| Qwen3.5-0.8B | zero-shot readout | branch (unpacked) | 0.518 | 0.472 | 0.568 | 0.570 | 0.000 | 0.086→0.086 |
| Qwen2.5-0.5B-Instruct | LoRA decision-trained | catalogue | 0.932 | 0.922 | 0.944 | 0.941 | 0.525 | 0.041→0.032 |
| Qwen3-0.6B | LoRA decision-trained | catalogue | 0.935 | 0.930 | 0.954 | 0.951 | 0.613 | 0.049→0.042 |
| Qwen2.5-0.5B-Instruct | LoRA decision-trained | branch | 0.896 | 0.879 | 0.933 | 0.911 | 0.475 | 0.068→0.070 |
| Qwen3-0.6B | LoRA decision-trained | branch | 0.929 | 0.921 | 0.933 | 0.932 | 0.550 | 0.049→0.044 |
| Qwen/Qwen3-0.6B | gguf | branch | 0.928 | 0.919 | 0.933 | – | 0.537 | 0.050→0.044 |
| Jev (TypeSafe API) | hosted, zero-shot | questions as-is | 0.869 | 0.864 | 0.817 | 0.816 | 0.300 | 0.049→0.077 |
| Jev (TypeSafe API) | hosted, zero-shot | + playbook + export definition | 0.918 | 0.917 | 0.882 | 0.893 | 0.500 | 0.040→0.036 |
| Jev (TypeSafe API) | hosted, zero-shot | + playbook rules | 0.892 | 0.883 | 0.853 | 0.857 | 0.375 | 0.039→0.046 |
| minilm + logistic regression | supervised classical | – | 0.707 | 0.652 | 0.766 | 0.768 | 0.125 | 0.069 |
| tfidf + logistic regression | supervised classical | – | 0.797 | 0.786 | 0.852 | 0.842 | 0.200 | 0.035 |
| gen-qwen3-0.6b-base | generate JSON (free) | – | 0.590 | 0.513 | – | – | 0.013 | – |
| gen-qwen3-0.6b-base | generate JSON (schema) | – | 0.592 | 0.508 | – | – | 0.000 | – |

## Gold accuracy per question

| System | event_type | commodity | impact_severity | line_stop_risk | recommended_action | export_controlled | counterfeit_suspect | sole_source | delay_bucket |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (zero-shot readout, branch) | 0.25 | 0.35 | 0.26 | 0.60 | 0.10 | 0.86 | 0.90 | 0.85 | 0.40 |
| Qwen3-0.6B (zero-shot readout, branch) | 0.65 | 0.66 | 0.42 | 0.60 | 0.40 | 0.82 | 0.97 | 0.89 | 0.26 |
| Qwen3-1.7B (zero-shot readout, branch) | 0.72 | 0.75 | 0.51 | 0.71 | 0.47 | 0.94 | 0.97 | 0.97 | 0.28 |
| Qwen3.5-0.8B (zero-shot readout, branch (unpacked)) | 0.68 | 0.80 | 0.28 | 0.53 | 0.39 | 0.17 | 0.70 | 0.88 | 0.25 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, catalogue) | 0.95 | 0.93 | 0.78 | 0.96 | 0.93 | 0.99 | 1.00 | 1.00 | 0.86 |
| Qwen3-0.6B (LoRA decision-trained, catalogue) | 0.94 | 0.95 | 0.80 | 0.95 | 0.91 | 0.99 | 1.00 | 1.00 | 0.88 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, branch) | 0.88 | 0.95 | 0.70 | 0.88 | 0.81 | 1.00 | 0.99 | 1.00 | 0.86 |
| Qwen3-0.6B (LoRA decision-trained, branch) | 0.89 | 0.95 | 0.86 | 0.95 | 0.89 | 1.00 | 1.00 | 1.00 | 0.82 |
| Qwen/Qwen3-0.6B (gguf, branch) | 0.88 | 0.95 | 0.86 | 0.95 | 0.89 | 1.00 | 1.00 | 1.00 | 0.82 |
| Jev (TypeSafe API) (hosted, zero-shot, questions as-is) | 0.99 | 0.91 | 0.75 | 0.96 | 0.59 | 0.75 | 1.00 | 1.00 | 0.88 |
| Jev (TypeSafe API) (hosted, zero-shot, + playbook + export definition) | 0.97 | 0.91 | 0.79 | 0.97 | 0.84 | 1.00 | 1.00 | 1.00 | 0.78 |
| Jev (TypeSafe API) (hosted, zero-shot, + playbook rules) | 0.97 | 0.93 | 0.78 | 0.97 | 0.82 | 0.75 | 1.00 | 1.00 | 0.80 |
| minilm | 0.75 | 0.94 | 0.45 | 0.61 | 0.56 | 0.72 | 0.99 | 0.89 | 0.45 |
| tfidf | 0.81 | 0.99 | 0.56 | 0.70 | 0.68 | 0.97 | 1.00 | 0.97 | 0.49 |

## Held-out questions (never trained on) — gold accuracy

| System | supplier_region | mentions_new_date | military_program |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct (zero-shot readout, branch) | 0.75 | 0.65 | 0.81 |
| Qwen3-0.6B (zero-shot readout, branch) | 0.79 | 0.90 | 0.85 |
| Qwen3-1.7B (zero-shot readout, branch) | 0.94 | 0.94 | 0.86 |
| Qwen3.5-0.8B (zero-shot readout, branch (unpacked)) | 0.38 | 0.89 | 0.51 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, catalogue) | 0.53 | 0.61 | 0.84 |
| Qwen3-0.6B (LoRA decision-trained, catalogue) | 0.69 | 0.78 | 0.85 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, branch) | 0.81 | 0.90 | 0.82 |
| Qwen3-0.6B (LoRA decision-trained, branch) | 0.75 | 0.88 | 0.84 |
| Qwen/Qwen3-0.6B (gguf, branch) | 0.78 | 0.88 | 0.84 |
| Jev (TypeSafe API) (hosted, zero-shot, questions as-is) | 0.75 | 1.00 | 0.79 |
| Jev (TypeSafe API) (hosted, zero-shot, + playbook + export definition) | 0.76 | 1.00 | 0.80 |
| Jev (TypeSafe API) (hosted, zero-shot, + playbook rules) | 0.75 | 1.00 | 0.79 |

## Option-order robustness (gold, choice questions shuffled)

| System | agreement with canonical order | accuracy shuffled |
|---|---|---|
| Qwen2.5-0.5B-Instruct (zero-shot readout, branch) | 0.247 | 0.316 |
| Qwen3-0.6B (zero-shot readout, branch) | 0.666 | 0.603 |
| Qwen3-1.7B (zero-shot readout, branch) | 0.816 | 0.691 |
| Qwen3.5-0.8B (zero-shot readout, branch (unpacked)) | 0.662 | 0.556 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, catalogue) | 0.831 | 0.794 |
| Qwen3-0.6B (LoRA decision-trained, catalogue) | 0.894 | 0.894 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, branch) | 0.866 | 0.828 |
| Qwen3-0.6B (LoRA decision-trained, branch) | 0.938 | 0.909 |
| Qwen/Qwen3-0.6B (gguf, branch) | 0.941 | 0.909 |
| Jev (TypeSafe API) (hosted, zero-shot, questions as-is) | 0.994 | 0.816 |
| Jev (TypeSafe API) (hosted, zero-shot, + playbook + export definition) | 1.000 | 0.872 |
| Jev (TypeSafe API) (hosted, zero-shot, + playbook rules) | 0.994 | 0.875 |

## CPU latency (12 decisions per record unless noted)

| Host | Label | Engine | Layout | Threads | Questions | s/record p50 | ms/decision | tokens/record | Peak RSS MB | File MB |
|---|---|---|---|---|---|---|---|---|---|---|
| mac-m4 | hf-fp32 base | hf | branch | 4 | 12 | 1.969 | 166.7 | 1204 | 3066 | – |
| mac-m4 | gguf-q8_0 base | gguf | branch | 4 | 12 | 2.356 | 197.7 | 1204 | 1839 | 639 |
| mac-m4 | hf-fp32 base | hf | branch | 8 | 12 | 1.915 | 165.1 | 1204 | 3120 | – |
| mac-m4 | gguf-q8_0 base | gguf | branch | 8 | 12 | 2.561 | 217.0 | 1204 | 1833 | 639 |
| mac-m4 | gguf-q8_0 base, 1 question | gguf | branch | 4 | 1 | 0.408 | 424.6 | 250 | 1646 | 639 |
| mac-m4 | hf-fp32 base catalogue | hf | catalogue | 4 | 12 | 0.463 | 40.0 | 156 | 3347 | – |
| mac-m4 | gguf-q8_0 base catalogue | gguf | catalogue | 4 | 12 | 0.670 | 57.4 | 156 | 1962 | 639 |

## Training runs (Kaggle T4, LoRA r=16)

| Run | best val acc | steps | records | minutes |
|---|---|---|---|---|
| qwen3-0.6b-cat | 0.976 | 1500 | 12000 | 114 |
| qwen2.5-0.5b-cat | 0.971 | 1500 | 12000 | 67 |
| qwen3-0.6b | 0.966 | 1500 | 12000 | 89 |
| qwen2.5-0.5b | 0.960 | 1500 | 12000 | 55 |
