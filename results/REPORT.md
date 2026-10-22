# AeroJev results

Core = the 9 trained questions; accuracy is the mean over questions.

## Accuracy vs latency

Same protocol everywhere: 40 gold records, one at a time, warm process, 9 core decisions per record, 4 threads (classical models single-threaded). Hosted API latency is the client-measured p50 round trip. `*` = accuracy of the fp32 model (quantized accuracy not measured separately).

| System | Engine | Layout | Gold acc | Gold2 acc | Test acc | api p50 s/record | api ms/decision | mac p50 s/record | mac ms/decision | x86 p50 s/record | x86 ms/decision | Peak RSS MB | File MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B v2 LoRA (fp32) | torch fp32 | catalogue | 0.950 | 0.931 | 0.959 | – | – | 0.494 | 55.34 | – | – | 3561 | – |
| Qwen3-0.6B v2 LoRA Q8_0 | llama.cpp | catalogue | 0.950 | 0.931 | 0.960 | – | – | 0.491 | 56.46 | – | – | 1961 | 639 |
| Qwen3-0.6B v2 LoRA Q4_K_M | llama.cpp | catalogue | 0.940 | 0.921 | 0.951 | – | – | 0.510 | 57.46 | – | – | 1723 | 397 |
| Qwen3-0.6B LoRA (fp32) | torch fp32 | catalogue | 0.935 | 0.921 | 0.954 | – | – | 0.484 | 53.43 | 1.982 | 231.46 | 3359 | – |
| Qwen3-0.6B LoRA Q8_0 | llama.cpp | catalogue | 0.935 | 0.921 | 0.953 | – | – | 0.498 | 56.55 | 2.834 | 324.25 | 1962 | 639 |
| Qwen2.5-0.5B LoRA (fp32) | torch fp32 | catalogue | 0.932 | 0.933 | 0.944 | – | – | – | – | 1.264 | 145.55 | – | – |
| Qwen2.5-0.5B LoRA Q4_K_M | llama.cpp | catalogue | 0.932 | 0.942 | – | – | – | 0.376 | 42.78 | 2.263 | 261.33 | 1271 | 398 |
| Qwen2.5-0.5B LoRA Q8_0 | llama.cpp | catalogue | 0.931 | 0.938 | – | – | – | 0.324 | 36.91 | 1.802 | 214.48 | 1403 | 531 |
| Qwen3-0.6B LoRA (fp32) | torch fp32 | branch | 0.929 | 0.921 | 0.933 | – | – | 1.417 | 157.93 | 12.158 | 1362.36 | 3413 | – |
| Qwen3-0.6B LoRA Q8_0 | llama.cpp | branch | 0.928 | 0.919 | 0.933 | – | – | 1.742 | 196.11 | 14.631 | 1646.33 | 1840 | 639 |
| Qwen3-0.6B LoRA Q4_K_M | llama.cpp | catalogue | 0.918 | 0.919 | 0.951 | – | – | 0.498 | 57.57 | 2.407 | 284.67 | 1726 | 397 |
| Jev (TypeSafe API) (jev-1.14.0) | hosted API (GPU, network) | + playbook + export definition | 0.918 | – | 0.882 | 0.120 | 10.0 | – | – | – | – | – | – |
| Qwen3-0.6B LoRA Q4_K_M | llama.cpp | branch | 0.910 | 0.912 | 0.932 | – | – | 1.959 | 217.89 | 12.207 | 1367.62 | 1490 | 397 |
| Qwen2.5-0.5B LoRA (fp32) | torch fp32 | branch | 0.896 | 0.896 | 0.933 | – | – | – | – | 8.640 | 1006.88 | – | – |
| Qwen2.5-0.5B LoRA Q8_0 | llama.cpp | branch | 0.896* | 0.896* | 0.933* | – | – | – | – | 10.625 | 1189.73 | – | 531 |
| Qwen2.5-0.5B LoRA Q4_K_M | llama.cpp | branch | 0.896* | 0.896* | 0.933* | – | – | – | – | 13.864 | 1598.79 | – | 398 |
| Jev (TypeSafe API) (jev-1.14.0) | hosted API (GPU, network) | + playbook rules | 0.892 | – | 0.853 | 0.175 | 14.6 | – | – | – | – | – | – |
| Jev (TypeSafe API) (jev-1.14.0) | hosted API (GPU, network) | questions as-is | 0.869 | – | 0.817 | 0.171 | 14.3 | – | – | – | – | – | – |
| tfidf + logistic regression | sklearn | – | 0.797 | 0.806 | 0.852 | – | – | 0.001 | 0.09 | 0.009 | 0.95 | 721 | – |
| minilm + logistic regression | sklearn | – | 0.707 | 0.688 | 0.766 | – | – | 0.006 | 0.79 | 0.027 | 3.17 | 1029 | – |
| Qwen3-1.7B zero-shot | torch fp32 | branch | 0.704 | 0.725 | 0.660 | – | – | – | – | 32.025 | 3655.34 | – | – |
| Qwen3-0.6B zero-shot | torch fp32 | branch | 0.632 | 0.646 | 0.570 | – | – | 1.491 | 175.5 | 11.646 | 1308.19 | 3237 | – |
| Qwen3-0.6B Q8_0 (base): generate JSON (schema) | llama.cpp generate | – | 0.592 | – | – | – | – | 1.689 | 188.37 | 6.208 | 709.75 | 1274 | 639 |
| Qwen3-0.6B Q8_0 (base): generate JSON (free) | llama.cpp generate | – | 0.590 | – | – | – | – | 1.730 | 199.47 | 6.932 | 817.58 | 1274 | 639 |
| Qwen3.5-0.8B zero-shot | torch fp32 | branch | 0.518 | 0.512 | 0.568 | – | – | – | – | 27.313 | 3166.52 | – | – |
| Qwen2.5-0.5B-Instruct zero-shot | torch fp32 | branch | 0.508 | 0.532 | 0.445 | – | – | – | – | 8.517 | 974.34 | – | – |

## Accuracy (core questions)

| System | Method | Layout | Gold acc | Gold macro-F1 | Test acc | Injection acc | Gold exact | ECE raw→cal (gold) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | zero-shot readout | branch | 0.508 | 0.355 | 0.445 | 0.438 | 0.000 | 0.071→0.063 |
| Qwen3-0.6B | zero-shot readout | branch | 0.632 | 0.498 | 0.570 | 0.583 | 0.000 | 0.224→0.054 |
| Qwen3-1.7B | zero-shot readout | branch | 0.704 | 0.669 | 0.660 | 0.638 | 0.050 | 0.266→0.122 |
| Qwen3.5-0.8B | zero-shot readout | branch (unpacked) | 0.518 | 0.472 | 0.568 | 0.570 | 0.000 | 0.086→0.086 |
| Qwen2.5-0.5B-Instruct | LoRA decision-trained | catalogue | 0.932 | 0.922 | 0.944 | 0.941 | 0.525 | 0.041→0.032 |
| Qwen3-0.6B | LoRA decision-trained | catalogue | 0.935 | 0.930 | 0.954 | 0.951 | 0.613 | 0.049→0.042 |
| Qwen/Qwen3-0.6B | ? | catalogue | 0.950 | 0.945 | 0.959 | 0.958 | 0.650 | 0.035→0.036 |
| Qwen2.5-0.5B-Instruct | LoRA decision-trained | branch | 0.896 | 0.879 | 0.933 | 0.911 | 0.475 | 0.068→0.070 |
| Qwen3-0.6B | LoRA decision-trained | branch | 0.929 | 0.921 | 0.933 | 0.932 | 0.550 | 0.049→0.044 |
| Qwen/Qwen2.5-0.5B-Instruct | gguf | catalogue | 0.932 | 0.922 | – | – | 0.487 | 0.037→0.028 |
| Qwen/Qwen2.5-0.5B-Instruct | gguf | catalogue | 0.931 | 0.920 | – | – | 0.512 | 0.041→0.032 |
| Qwen/Qwen3-0.6B | gguf | catalogue | 0.918 | 0.915 | 0.951 | – | 0.512 | 0.060→0.056 |
| Qwen/Qwen3-0.6B | gguf | catalogue | 0.935 | 0.930 | 0.953 | – | 0.613 | 0.052→0.046 |
| Qwen/Qwen3-0.6B | gguf | catalogue | 0.940 | 0.931 | 0.951 | – | 0.575 | 0.037→0.032 |
| Qwen/Qwen3-0.6B | gguf | catalogue | 0.950 | 0.945 | 0.960 | – | 0.650 | 0.035→0.034 |
| Qwen/Qwen3-0.6B | gguf | branch | 0.910 | 0.899 | 0.932 | – | 0.487 | 0.058→0.054 |
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
| Qwen/Qwen3-0.6B (?, catalogue) | 0.93 | 0.96 | 0.82 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.89 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, branch) | 0.88 | 0.95 | 0.70 | 0.88 | 0.81 | 1.00 | 0.99 | 1.00 | 0.86 |
| Qwen3-0.6B (LoRA decision-trained, branch) | 0.89 | 0.95 | 0.86 | 0.95 | 0.89 | 1.00 | 1.00 | 1.00 | 0.82 |
| Qwen/Qwen2.5-0.5B-Instruct (gguf, catalogue) | 0.94 | 0.94 | 0.78 | 0.96 | 0.90 | 0.99 | 1.00 | 1.00 | 0.89 |
| Qwen/Qwen2.5-0.5B-Instruct (gguf, catalogue) | 0.95 | 0.93 | 0.78 | 0.95 | 0.93 | 0.99 | 1.00 | 1.00 | 0.86 |
| Qwen/Qwen3-0.6B (gguf, catalogue) | 0.93 | 0.94 | 0.74 | 0.95 | 0.88 | 0.99 | 1.00 | 1.00 | 0.85 |
| Qwen/Qwen3-0.6B (gguf, catalogue) | 0.94 | 0.95 | 0.80 | 0.95 | 0.91 | 0.99 | 1.00 | 1.00 | 0.88 |
| Qwen/Qwen3-0.6B (gguf, catalogue) | 0.93 | 0.94 | 0.85 | 0.99 | 0.94 | 0.99 | 1.00 | 1.00 | 0.84 |
| Qwen/Qwen3-0.6B (gguf, catalogue) | 0.93 | 0.96 | 0.82 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.89 |
| Qwen/Qwen3-0.6B (gguf, branch) | 0.86 | 0.95 | 0.76 | 0.93 | 0.89 | 1.00 | 1.00 | 1.00 | 0.80 |
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
| Qwen/Qwen3-0.6B (?, catalogue) | 0.57 | 0.90 | 0.94 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, branch) | 0.81 | 0.90 | 0.82 |
| Qwen3-0.6B (LoRA decision-trained, branch) | 0.75 | 0.88 | 0.84 |
| Qwen/Qwen2.5-0.5B-Instruct (gguf, catalogue) | 0.51 | 0.57 | 0.84 |
| Qwen/Qwen2.5-0.5B-Instruct (gguf, catalogue) | 0.54 | 0.64 | 0.84 |
| Qwen/Qwen3-0.6B (gguf, catalogue) | 0.47 | 0.78 | 0.89 |
| Qwen/Qwen3-0.6B (gguf, catalogue) | 0.69 | 0.75 | 0.85 |
| Qwen/Qwen3-0.6B (gguf, catalogue) | 0.51 | 0.81 | 0.94 |
| Qwen/Qwen3-0.6B (gguf, catalogue) | 0.56 | 0.90 | 0.94 |
| Qwen/Qwen3-0.6B (gguf, branch) | 0.79 | 0.88 | 0.84 |
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
| Qwen/Qwen3-0.6B (?, catalogue) | 0.866 | 0.869 |
| Qwen2.5-0.5B-Instruct (LoRA decision-trained, branch) | 0.866 | 0.828 |
| Qwen3-0.6B (LoRA decision-trained, branch) | 0.938 | 0.909 |
| Qwen/Qwen3-0.6B (gguf, branch) | 0.944 | 0.897 |
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
| qwen3-0.6b-cat-v2 | 0.980 | 2500 | 20000 | 178 |
| qwen3-0.6b-cat | 0.976 | 1500 | 12000 | 114 |
| qwen2.5-0.5b-cat | 0.971 | 1500 | 12000 | 67 |
| qwen3-0.6b | 0.966 | 1500 | 12000 | 89 |
| qwen2.5-0.5b | 0.960 | 1500 | 12000 | 55 |
