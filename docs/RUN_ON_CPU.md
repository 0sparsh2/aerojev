# Running AeroJev on a CPU

AeroJev answers nine supply-chain questions about a record (event type, commodity, severity, line-stop risk,
recommended action, export control, counterfeit, sole source, delay) in about **0.3–0.5 s on a laptop CPU**, using
**1.4–2 GB of RAM** and no GPU. This guide covers the fastest way to run it and the measurements behind each choice.

All timings are the median for one record at a time, in a warm process, with four threads, on short records like
the hand-written gold set (about 100 tokens each). They come from an Apple M4 laptop and a 4-vCPU x86 cloud VM
(Intel Xeon, 2.2 GHz). Accuracy is the mean over the nine questions on the 80 hand-written gold records.

## The recommended setup

| Choice | Recommendation | Why |
|---|---|---|
| Runtime | **llama.cpp** through `llama-cpp-python`, CPU-only | Same speed as PyTorch on Apple Silicon at about half the memory |
| Model file | **GGUF Q8_0** of a LoRA-trained model, adapter merged | Accuracy within 0.2 points of fp32 in every test |
| Prompt layout | **catalogue** | The question list is processed once at start-up, not for every record: 3× faster on the M4, 6× on x86 |
| Threads | **the number of performance cores** (4 on an M4) | More threads did not help; fewer cost a lot |
| Probabilities | **calibrated** (fitted temperatures loaded) | Confidence can be used to send uncertain decisions to an analyst |
| Process | **one long-running server** | Start-up takes 2–3.5 s; a decision takes a few hundred ms |

That is exactly what this command runs:

```bash
python server.py --preset best      # or --preset fast
open http://127.0.0.1:8080          # test page; the same server answers POST /v1/decide
```

### Which model

| Goal | Model | File | M4 s/record | RAM | Gold acc | Gold2 acc |
|---|---|---|---|---|---|---|
| **Fastest** (`--preset fast`) | Qwen2.5-0.5B + LoRA, catalogue, Q8_0 | 531 MB | **0.32** | 1.4 GB | 0.931 | 0.938 |
| **Most accurate** (`--preset best`) | Qwen3-0.6B + LoRA, data v2, catalogue, Q8_0 | 639 MB | 0.49 | 2.0 GB | **0.950** | 0.931 |
| Smallest | Qwen2.5-0.5B + LoRA, catalogue, Q4_K_M | 398 MB | 0.38 | 1.3 GB | 0.932 | 0.942 |

On a 4-vCPU x86 server the same models take 1.3–2.8 s per record (see the measurements below). The hosted Jev API,
which runs on GPUs, scored 0.918 on the same gold set.

## 1. Install (CPU only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`llama-cpp-python` compiles llama.cpp during install. Build it **without GPU support** so it can only use the CPU:

* **macOS (Apple Silicon).** The default build includes Metal, and llama.cpp can then move large prompt batches to
  the GPU even when no layers are offloaded. Rebuild it CPU-only:

  ```bash
  CMAKE_ARGS="-DGGML_METAL=OFF" pip install --force-reinstall --no-cache-dir llama-cpp-python==0.3.16
  ```

* **Linux or Windows (x86).** Use the prebuilt CPU wheel, which is what the x86 numbers below used:

  ```bash
  pip install llama-cpp-python==0.3.16 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  ```

  A source build tuned for the machine (`CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python …`)
  may be faster on modern x86 CPUs. It was not measured here.

To check the build:

```bash
python -c "import llama_cpp; print('GPU offload compiled in:', llama_cpp.llama_supports_gpu_offload())"
```

`False` is the CPU-only build. If it prints `True`, AeroJev still runs on the CPU: `GGUFEngine` sets
`n_gpu_layers=0`, `offload_kqv=False` and `op_offload=False`.

## 2. Model files

Weights are not in git. The server's presets expect these files in `models/`:

| Preset | Files |
|---|---|
| `best` | `aerojev-qwen3-0.6b-cat-v2-q8_0.gguf`, `temps-qwen3-0.6b-cat-v2.json` |
| `fast` | `aerojev-qwen2.5-0.5b-cat-q8_0.gguf`, `temps-qwen2.5-0.5b-cat.json` |

To make them, train the adapter (README, step 3; on a GPU in about an hour, or on a CPU as in `docs/TRAIN_ON_CPU.md`), then merge and export:

```bash
python -m aerojev.export --model Qwen/Qwen3-0.6B --adapter runs/qwen3-0.6b-cat-v2/best \
    --out models --name aerojev-qwen3-0.6b-cat-v2          # writes …-q8_0.gguf and …-q4_k_m.gguf
```

The temperature file is the `temperatures` entry that `eval.evaluate` fits on the validation set. Save it as
`{"choice": 1.3, "score": 1.0, "yesno": 1.0}`. The tokenizer (a few MB) is downloaded from Hugging Face on first use;
after that, `HF_HUB_OFFLINE=1` keeps everything offline.

## 3. Run it

**Test page and HTTP API.** `python server.py --preset best`, then open http://127.0.0.1:8080 or call the API:

```bash
curl -s localhost:8080/v1/decide -d '{"state": "Kestrel: 718 disks slip 6 weeks. 2 weeks on hand. Sole source."}'
curl -s localhost:8080/v1/decide -d '{"state": "…", "questions": ["impact_severity", "recommended_action"]}'
```

The server listens on `127.0.0.1` only. For your own model file use
`--gguf <file> --tokenizer <base model id> --layout catalogue --temps <file>`.

**From Python**, without HTTP:

```python
import json
from aerojev import schema
from aerojev.engine import GGUFEngine

engine = GGUFEngine("models/aerojev-qwen3-0.6b-cat-v2-q8_0.gguf", "Qwen/Qwen3-0.6B", n_threads=4,
                    layout="catalogue", temperatures=json.load(open("models/temps-qwen3-0.6b-cat-v2.json")))
answers = engine.decide(record_text, schema.TRAIN_QUESTIONS)
answers["impact_severity"]   # {'type': 'score', 'answer': …, 'confidence': …, 'probabilities': {'0': …, …}, 'expected': …}
```

Create the engine once and reuse it. Each `decide` call answers one record.

## 4. Why this setup: measurements

**The prompt layout matters most.** These are the same trained Qwen3-0.6B weights (s per record, nine decisions):

| Layout and runtime | Apple M4 | x86, 4 vCPU |
|---|---|---|
| branch, PyTorch fp32 | 1.42 | 12.2 |
| branch, llama.cpp Q8_0 | 1.74 | 14.6 |
| **catalogue, PyTorch fp32** | **0.48** | **1.98** |
| **catalogue, llama.cpp Q8_0** | **0.50** | 2.83 |
| catalogue, llama.cpp Q4_K_M | 0.50 | 2.41 |

In the *branch* layout every question's full text and options are processed again for every record (about a
thousand tokens per record). In the *catalogue* layout the question list comes first, its key-value cache is computed once
when the process starts, and each record costs only its own tokens plus about five tokens per question. The catalogue
model was also the more accurate one in training (0.935 vs 0.929 gold).

**Runtime.** On the M4, llama.cpp Q8_0 matches PyTorch fp32 (0.49 vs 0.49 s for the best model) at about half the
memory (2.0 vs 3.6 GB). On the x86 VM, PyTorch fp32 was faster than the prebuilt llama.cpp wheel (Qwen2.5-0.5B:
1.26 vs 1.80 s per record). Use PyTorch there if 3.5 GB of RAM is acceptable:
`python server.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/<run>/best --layout catalogue --temps <file>`.

**Quantization saves memory, not time.** Accuracy of the best model's files on gold / blind gold2 / test:

| Precision | File | Gold | Gold2 | Test |
|---|---|---|---|---|
| fp32 (PyTorch) | – | 0.950 | 0.931 | 0.959 |
| **Q8_0** | 639 MB | **0.950** | **0.931** | **0.960** |
| Q4_K_M | 397 MB | 0.940 | 0.921 | 0.951 |

Q8_0 matched fp32 within 0.2 points for every model tested. Q4_K_M cost up to about 2 points. It was never the
fastest option: slightly slower than Q8_0 on the M4, and on the x86 VM faster than Q8_0 but slower than PyTorch fp32.

**The alternatives are slower or less accurate.** All of these were timed on the M4:

| Approach | s/record | Gold acc |
|---|---|---|
| Decision-trained Qwen3-0.6B, catalogue, Q8_0 (this setup) | 0.49 | 0.950 |
| The same base model asked to *generate* a JSON answer (Q8_0) | 1.73 | 0.590 |
| The same base model, untrained, reading label logits | 1.49 | 0.632 |
| TF-IDF + logistic regression per question | 0.0008 | 0.797 |

TF-IDF is far faster, and a good fallback when accuracy matters less. It cannot answer a question it was not
trained on, and its probabilities are less informative.

## 5. Tuning

**Threads.** Use the performance cores and no more. On the M4 (four performance cores, Q4_K_M, catalogue): 1 thread
1.38 s, 2 threads 0.79 s, 4 threads 0.50 s; 8 threads were no faster than 4 in an earlier run. On the x86 VM: 4.80,
2.60 and 2.41 s.

**Keep records short.** Time grows about linearly with the record's length, so strip quoted reply chains, legal
footers and signatures before calling the model:

| Record tokens | Qwen2.5-0.5B, Q8_0 | Qwen3-0.6B v2, Q8_0 |
|---|---|---|
| ~110 | 0.37 s | 0.62 s |
| ~330 | 0.68 s | 1.13 s |
| ~650 | 1.21 s | 2.24 s |
| ~1,250 | 2.36 s | 4.70 s |
| ~2,000 | 4.02 s | 8.19 s |

That is about 2 ms per record token for Qwen2.5-0.5B and 4 ms for Qwen3-0.6B on the M4. Records longer than 2,048
tokens are trimmed (the beginning is kept and the cut is marked) so they always fit the model's 4,096-token context.

Long, noisy records also cost **accuracy**. The models were trained on records of about 50–300 tokens. In a test, the
same two-line buyer note with a 5,000-token quoted reply chain appended went from severity 3 / expedite to
severity 1 / monitor, although the facts were unchanged. Strip quoted replies and footers for accuracy as well as speed.

**Ask only the questions you need.** One question took 0.27 s against 0.50 s for nine (M4, Q4_K_M), because most of
the time goes into reading the record.

**Start once, decide many times.** Loading takes 2.2 s (Qwen2.5-0.5B) or 3.5 s (Qwen3-0.6B), including the
catalogue's cache. Keep the server or engine running.

**Throughput.** A process answers one record at a time, because the catalogue cache is shared state. For more records
per second, run several processes, each using its own performance cores and 1.4–2 GB of RAM. Scaling across processes
was not measured.

**Use the confidence.** With the fitted temperatures loaded, the best model's decisions with confidence of 0.9 or more
were right 98.5% of the time. Sending lower-confidence decisions to an analyst is the simplest way to trade a little
manual work for accuracy.

## 6. What not to do on a CPU

* **Don't generate text.** Asking the model to write a JSON answer is about 3.5× slower and far less accurate.
* **Don't use the branch layout** for a fixed question set. Keep it for ad-hoc questions. The server answers those
  anyway, without the cache.
* **Don't use bf16 or fp16** on the CPU. PyTorch fell back to slow kernels in our training tests.
* **Don't expect Q4_K_M to be the fastest option.** Choose it for memory or disk size.
* **Don't leave GPU offload on** if you need CPU-only behaviour (section 1).

## 7. Measure your own machine

```bash
python -m eval.latency_suite --out results/latency_local.jsonl     # every system available locally
python -m eval.bench_cpu --engine gguf --gguf models/<file>.gguf --tokenizer <base model id> --layout catalogue --threads 4
python -m eval.bench_length --gguf models/<file>.gguf --tokenizer <base model id>  # time vs record length
```

Close other heavy programs first. Background CPU load changes these numbers a lot.

## Troubleshooting

| Message | Meaning |
|---|---|
| `n_ctx_per_seq (4096) < n_ctx_train (…)` | Harmless: the engine uses a 4,096-token context. |
| `ggml_metal_init: …` on macOS | This llama.cpp build includes Metal. The engine still runs on the CPU; rebuild CPU-only to remove it. |
| `… not found: train and export a model first` | The preset's model file is missing from `models/` (section 2). |
| `Address already in use` | Another program uses port 8080: add `--port 8081`. |
