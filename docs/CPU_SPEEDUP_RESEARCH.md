# How far below 67 hours? CPU training and inference: research notes

October 2026. This is desk research: nothing here was trained or benchmarked for this page. Numbers marked
**measured** come from this repo (`results/train_cpu.jsonl`, `kaggle/outputs/*/train_log.jsonl`, or a tokenizer
count over 400 training records). Numbers marked **est.** are FLOP arithmetic. Accuracy effects are unknown until a
run is made. Sources are listed at the end.

The starting point (`docs/TRAIN_ON_CPU.md`): Qwen2.5-0.5B-Instruct + LoRA r16 on all 24 blocks, catalogue layout,
fp32, 12,000 records. That is **20.2 s per record and 67 h** on the regular CPU (2-core Xeon, 2.2 GHz), 4.35 s and
14.5 h on the M4, and 68 min on a T4.

## Summary

1. **Most of the time goes into re-reading the question catalogue.** A training sequence averages 1,089 tokens
   (**measured**): 898 are the catalogue (83 %), 150 the record, 42 the readouts. Every record re-encodes the
   catalogue with a full forward and backward pass, because each record gets its own random shuffle of it.
2. **The regular CPU has little headroom left.** One record is ~1.9 TFLOP of work. 20.2 s is ~90 GFLOP/s, about
   two thirds of a 2-core AVX2 Xeon's FP32 peak. Runtime and kernel tuning can give **at most ~1.5×** there. Anything
   larger has to come from doing less work: fewer tokens per record, fewer records, or fewer FLOPs per token.
3. **Largest exact win: one catalogue per optimizer step.** Encode the catalogue once and run the step's 8 records on
   top of it. The gradients are exactly those of training the 8 records separately with that catalogue. The inference
   engine already does this (`TorchEngine`: static KV once, records at `offset=S`); training needs the same path with
   autograd on. Est. **3.6×**, so the full recipe takes **~19 h** on the regular CPU and **~4 h** on the M4.
   Precedent: Tree Training (up to 6.2× for SFT and RL), Prefix Grouper (identical outputs and gradients).
4. **Two free fixes.** (a) Ask all 9 questions in every record. Readouts are independent tree branches, so the
   random 5–9 subset changes nothing a readout sees; it only throws away labels. All 9 gives +29 % labels for ~1 %
   more compute. (b) Cache the catalogue KV once per validation pass, as the engine does. That is exact and makes
   validation ~4× cheaper.
5. **Getting below ~10 h means changing what is trained.** Remove the catalogue from the prompt (fixed heads or a
   context-distilled LoRA: est. 7×). Use a smaller base (Gemma 3 270M has 100M transformer parameters against 358M:
   est. 3.6× per token). Train on fewer records by tuning the learning rate and batch size (literature: 1.3–2×; not
   shown for this task).
6. **Realistic floor on the regular CPU.** ~15–19 h with exact math and today's model. ~5–10 h with a catalogue-free
   model. ~2–4 h with a catalogue-free 270M model. ~3 h for a one-off forward-only feature pass if frozen-model probes
   turn out good enough, which is doubtful. The M4 runs each of these ~4.6× faster.

## The ladder

Each row includes the rows above it unless it says otherwise. All times are **est.** except the first and last rows.

| # | Change | What you give up | Regular CPU s/record | 12,000 records, regular CPU | M4 | Accuracy risk |
|---|---|---|---|---|---|---|
| 0 | Today (**measured**) | – | 20.2 | 67 h | 14.5 h | – |
| 1 | One catalogue per 8 records (§2.1) | 8 records share one catalogue shuffle | ~5.6 | **~19 h** | ~4 h | low |
| 2 | + all 9 questions, cached-catalogue validation, `lora_dropout=0`, runtime hygiene (§2.2–2.4) | nothing material | ~4.5–5 | ~15–17 h | ~3.5 h | none–low |
| 3 | + fewer records from a tuned LR/batch and distillation, if 12k → ~6k holds (§3) | sweeps on the T4 first | same | **~8 h** | ~2 h | medium: unproven |
| 4 | Instead of 1–3: no catalogue in the prompt (§4.1) | the model no longer reads its question list | ~2.9 | ~10 h (12k) / ~5 h (6k) | ~2 h / ~1 h | medium |
| 5 | Row 4 on Gemma 3 270M (§4.2) | new base model, re-export, re-check calibration | ~1.1–1.5 | ~4 h / **~2 h** | ~1 h / ~30 min | high: untested model |
| 6 | Frozen model + probe heads, no LoRA (§4.4) | probably several accuracy points | 1.2 (forward only, one pass over 10k) | ~3.3 h total | ~40 min | high |
| – | T4 GPU today (**measured**) | – | 0.34 | 68 min | – | – |

Rows 1–2 keep the current model, loss, prompt and inference path exactly. Row 3 changes only hyperparameters. Rows
4–6 change the product: the model stops being a "reads its questions" decision engine and becomes a fixed-question
classifier. Today's catalogue layout already fixes the question set at inference, so the two are functionally close.

## 1. Where the time goes

**Token budget** (**measured**, Qwen2.5 tokenizer, 400 data-v2 training records, `build_example` as in `train.py`):

| Part | Mean tokens | p90 |
|---|---|---|
| Catalogue, randomized per record | 898 | 956 |
| Catalogue, fixed (inference) | 969 | – |
| Record (`render_record`) | 150 | 199 |
| Readouts (~5.8 tokens × 7 asked on average) | ~42 | – |
| Whole packed sequence | 1,089 | 1,167 |

**FLOPs per record** (Qwen2.5-0.5B: N = 358M non-embedding parameters, T = 1,089):

* Matmuls: forward 2NT, plus backward 2NT for activation gradients only (frozen weights get no weight gradient).
  4NT ≈ **1.56 TFLOP**.
* Attention over the full 1,089 × 1,089 masked score matrix, forward + backward: ≈ **0.3 TFLOP**.
* LoRA adapters (r = 16, 168 modules) and the lm_head (readout rows only, via `logits_to_keep`): under 2 %.
* Total ≈ **1.87 TFLOP** per record. 20.2 s is ≈ 92 GFLOP/s on the regular CPU; 4.35 s is ≈ 430 GFLOP/s on the M4.

**Headroom.** The bench log does not record the CPU model or flags. "Intel Xeon @ 2.20GHz" on Google Cloud is
usually Broadwell: AVX2 with two 256-bit FMA units, so 2 cores × 2.2 GHz × 32 FLOP/cycle ≈ 140 GFLOP/s FP32 peak.
92 GFLOP/s is ~65 % of that (~33 % if it is an AVX-512 part). Dense fp32 GEMM rarely sustains much above 80 % of peak
end to end, so kernels have little left to give. Let `scripts/bench_train_cpu.py` log `lscpu` flags (AVX-512, AMX,
BF16) next time so the ceiling is known.

On the M4, PyTorch's fp32 GEMMs go through Apple Accelerate (this venv: `BLAS_INFO=accelerate`), which uses the M4's
SME matrix unit (`FEAT_SME2`, `SME_F32F32`, `SME_B16F32` present). The elementwise ops, the 168 small LoRA matmuls and
masked attention are the more likely bottleneck there, so the M4 probably gains more from fusion (`torch.compile`) than
the x86 VM does. Unmeasured.

**Learning curve** (**measured**, T4 runs, val accuracy on 150 records × 9 questions):

| Records seen | 1,200 | 2,400 | 3,600 | 4,800 | 6,000 | 7,200 | 8,400 | 12,000 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B cat, data v1, 12k schedule | 0.733 | 0.877 | 0.911 | 0.932 | 0.944 | 0.954 | 0.966 | 0.970 |
| Qwen2.5-0.5B cat, data v2, 4.8k schedule | 0.726 | 0.874 | 0.906 | 0.919 | | | | |
| Qwen3-0.6B cat, data v2, 20k schedule | 0.874 | 0.910 | 0.907 | 0.932 | 0.941 | 0.949 | 0.949 | 0.971 |

Validation is nearly flat after ~8,400 records (0.966 → 0.970), while gold moved more (0.881 at 4.8k to 0.924 at 12k
in the separate runs). Qwen3-0.6B learns faster per record but costs ~1.6× more per record (T4: 6,394 s against
3,982 s for 12k records), so per FLOP the two bases come out even.

## 2. Exact speed-ups (same model, same loss, same inference)

### 2.1 One catalogue per optimizer step

Today each of the 8 records in an optimizer step carries its own shuffled catalogue: 8 × 1,089 = 8,712 tokens of
forward and backward per step. Instead, shuffle one catalogue per step, encode it once, and run the 8 records
(record + readouts, ~192 tokens each) on top of its per-layer K/V:

* Tokens per step: 898 + 8 × 192 ≈ 2,436, against 8,712. Matmul FLOPs per record fall 3.6×.
* Attention per record falls from ~306 GFLOP to ~80: the catalogue's self-attention is shared, and each record's
  192 queries see 898 + 192 keys.
* Total ≈ 516 GFLOP per record, against 1,865: **est. 3.6×**. With 16 records per catalogue, est. 4.4×, but that
  doubles the effective batch (see §3).

**Why the gradient is exact.** Under the tree mask, record *i*'s tokens see only the catalogue and themselves. The
sum of the 8 losses therefore has exactly the gradient of running the 8 (catalogue + record) sequences separately
with that catalogue. The catalogue's own activations receive the summed gradient from all 8. This is the
Prefix Grouper result (decompose attention into one prefix self-attention plus per-suffix attention to the shared
prefix; identical forward outputs and backward gradients) and the Tree Training result (each shared prefix computed
once in forward and backward; log-probabilities match independent per-branch computation).

**How, reusing what exists.** `TorchEngine` already computes the static catalogue's KV once and evaluates
`pack_ids(record, readouts, offset=S)` on top of it. Training does the same with gradients enabled and the 8 records
as a padded batch. To bound memory, use the Tree Training partitioning trick: make the catalogue K/V leaf tensors,
run the records a few at a time to accumulate their K/V gradients, then backpropagate once through the catalogue.

```python
# sketch, one optimizer step; not tested
cat_out = model(input_ids=static_ids[None], use_cache=True)            # with grad
kv_graph = per_layer_kv(cat_out.past_key_values)                       # [(k, v)] * 24, in the autograd graph
kv_leaf = [(k.detach().requires_grad_(), v.detach().requires_grad_()) for k, v in kv_graph]
for chunk in chunks(records, 2):                                       # micro-batches bound peak RAM
    cache = cache_from([(k.expand(len(chunk), -1, -1, -1), v.expand(len(chunk), -1, -1, -1)) for k, v in kv_leaf])
    p = batch_pack(chunk, offset=S)                                     # pack_ids(record, readouts, offset=S), padded
    loss = chunk_loss(model(**p, past_key_values=cache))                # LoRA grads + dK/dV on the leaves
    (loss * len(chunk) / len(records)).backward()
torch.autograd.backward([t for kv in kv_graph for t in kv],             # one backward through the catalogue
                        [t.grad for kv in kv_leaf for t in kv])
opt.step()
```

Peak memory is then the catalogue's activations plus one micro-batch, about today's level. Avoid the tempting
alternative of one 2,436-token tree-packed sequence with a two-level mask. Masked SDPA on the CPU may take the math
path and keep a 2,436² score matrix per head and layer for backward (~8 GB in fp32), which does not fit next to
everything else on a 16 GB machine.

**What changes in the data.** 12,000 records see ~1,500 distinct catalogue shuffles instead of 12,000. That is still
plenty of variation in option order and paraphrase, but check it on the T4 before trusting it. The option-subset
augmentation (`subset=True`, which always keeps the gold option) has to keep the union of the batch's gold options.

**Not worth it:** a stop-gradient catalogue (catalogue KV under `no_grad`, refreshed per step). It removes the
catalogue's backward pass but saves only ~20 % on top of exact sharing, because the shared catalogue's cost is
already divided by 8.

### 2.2 Ask all 9 questions in every record

`build_example` asks a random 5–9 of the 9 questions (`--min-q 5 --max-q 9`). In both layouts, each readout branch
attends to the prefix and itself only. The subset therefore never changes what any readout sees, at training or
inference time; it only discards supervision. `--min-q 9` gives 9 labels per record instead of 7 on average (+29 %)
for 2 more readout branches (~12 tokens, ~1 %). Labels from the same record are correlated, so this will not cut the
record count by 29 %, but it costs nothing.

### 2.3 Validation with a cached catalogue

`quick_eval` packs the full catalogue for every validation record, so a validation record costs about a third of a
training step. The validation catalogue is fixed, so compute its KV once per validation pass with the current LoRA
weights, then score each record on top of it, exactly as `TorchEngine.decide` does. Same numbers, ~4× cheaper. On the
regular CPU that saves most of the ~3 h the 12k run spends validating.

### 2.4 Runtime hygiene (≤ 1.3× together, most of it on the M4)

* **`lora_dropout=0`.** At 0.05, PEFT applies dropout to the input of each of the 168 adapted modules: an extra
  activation copy each, in forward and backward. The data augmentation (shuffles, paraphrases, option subsets) already
  regularizes, and the T4 can confirm accuracy holds. Small speed and memory gain.
* **`torch.compile`.** Inductor's CPU backend fuses RMSNorm, SiLU·mul, RoPE and residual adds. The documented gains
  are for inference (1.25–2.35× over eager for bf16 inference on Xeon, PyTorch 2.1). Training gains are less documented, and lengths
  change per record, so use `dynamic=True` or pad to a few length buckets. Expect 1.1–1.3× on x86, possibly more on
  the M4. Unmeasured.
* **Threads and allocator (Linux).** Physical cores only (already documented), `OMP_PROC_BIND=close`, and jemalloc or
  tcmalloc via `LD_PRELOAD`. Usually a 5–15 % gain on x86 PyTorch CPU workloads.
* **Not IPEX.** Intel stopped developing the Intel Extension for PyTorch after 2.8 and it reaches end of life in
  March 2026. Its CPU optimizations are upstream in stock PyTorch.
* **bf16 only on hardware that has it.** On CPUs with AMX (Sapphire Rapids or newer: AWS c7i/m7i, GCP C3/N4, Azure
  v6), some oneDNN kernels ran more than 5× faster with AMX than when restricted to AVX-512 (Phoronix). Intel reports
  2.3–5.5× end-to-end for mixed-precision training on 4th-gen Xeon against fp32 on 3rd-gen. Use stock
  `torch.autocast("cpu", torch.bfloat16)`. Without AMX or AVX512_BF16, bf16 has no fast path and is slower. The M4
  result in `TRAIN_ON_CPU.md` fits this; the likely reason is that PyTorch's bf16 CPU matmul on macOS does not use the
  Accelerate (matrix-unit) path that fp32 uses.

## 3. Fewer records for the same accuracy

These change only hyperparameters. Run them on the T4: accuracy does not depend on the hardware, and a T4 run takes
an hour against days on a CPU.

* **Learning rate first.** The current LR is 2e-4 with an effective batch of 8. Thinking Machines ("LoRA Without
  Regret") find the optimal LoRA LR is consistently ~10× the full fine-tuning LR, and ~15× for runs of about 100 steps.
  Two 2026 re-evaluations ("Learning Rate Matters", ICML 2026; "Beware of the Batch Size") find that once LR and batch
  size are tuned, vanilla LoRA lands within 1–2 % of nine published variants. They also find a task-specific optimal
  batch size, not a monotonic "smaller is better". Sweep LR {2e-4, 5e-4, 1e-3} × records {2.4k, 4.8k, 7.2k} and
  look for gold ≥ 0.92 at ≤ 6k records.
* **Batch size.** LoRA is "less tolerant of large batch sizes than full fine-tuning", and more rank does not fix it
  (Thinking Machines). Marek et al. (NeurIPS 2025) show small batches match or beat large ones per FLOP if Adam's β2
  is rescaled to keep its half-life in tokens (β2 = 0.9999 at batch 1). They also call gradient accumulation wasteful
  on a single device. This pulls against §2.1, whose saving grows with records per catalogue: 8 records → 3.6×,
  4 → 2.6×, 2 → 1.7×. Keep 8 unless the sweep shows smaller batches converge much faster.
* **Initialization and LoRA variants (try after the LR sweep).** LoRA-GA reports 2–4× faster convergence than vanilla
  LoRA. PiSSA reports faster convergence, with an initialization that takes seconds. LoRA+ (larger LR for B) reports
  up to 2×. PEFT ships PiSSA/EVA/CorDA initializations and `create_loraplus_optimizer`. Per the 2026 re-evaluations,
  much of these gains disappears once the baseline LR is tuned.
* **Distill from the 1.7B model.** The fine-tuned Qwen3-1.7B is the most accurate model in the repo. Its label
  distributions for 10,000 records × 9 questions are a few MB, computed once on Kaggle. Soft targets carry more
  information per record than one-hot labels. A 2025 study finds higher-temperature soft labels help most when data
  is scarce and hard labels shrink the gain. Use CE(gold) + KL(teacher) at the readouts. Plausible 1.2–2× fewer
  records; nothing in the sources measures it for a task like this.
* **Stop at the plateau.** The 12k data-v1 run was at 0.966 val by 8,400 records, against 0.970 at the end. Ending the
  cosine schedule earlier is a 1.4× cut if gold holds; check on gold and gold2, not val.
* **Selective backprop (low priority).** Skipping the backward pass for records the model already gets right reached
  up to 3.5× on CIFAR/SVHN, but MosaicML Composer reports ~10 % in practice. With §2.1 the backward pass is shared by
  the batch, so a skipped record saves only its ~192 tokens. Not worth the complexity.

## 4. Fewer FLOPs per token

### 4.1 Take the catalogue out of the prompt

The catalogue is fixed at inference, so the model can learn it instead of reading it. Two forms:

* **Fixed heads.** Prompt = system line + record + 9 learned query tokens, one per question. Train a linear head per
  question on the query token's final hidden state, or keep the label-token readout and train the 9 query embeddings
  with PEFT's `trainable_token_indices`. ~180 tokens per record against 1,089: est. 266 GFLOP per record, **7×**. A
  Clarivate study (Dec 2025) found embedding-plus-head fine-tuning of causal LLMs matches or beats fine-tuned BERT on
  single-label classification, while instruction-style label generation was competitive only for multi-label tasks
  with much larger trainable budgets.
* **Context distillation.** Keep the label-token readout and the short readout prompt, and drop the catalogue. Snell
  et al. (2022) show instructions can be distilled into weights. Cartridges (ICLR 2026) train a compact KV cache in
  place of a long context and match in-context performance with 38.6× less memory. Gist tokens compress prompts up to
  26×. Here the "teacher" is today's catalogue model, so the student can be trained on its distributions (§3) with
  no extra labels.

Costs: the model can no longer answer a question that is not in its training set by reading it, and the option-order
robustness becomes moot because options are no longer shown. The catalogue layout already gives this up in practice,
since adding a question means retraining. Inference also gains (§6).

### 4.2 A smaller base model

| Model (release) | Non-embedding params | Attention | Notes for this repo |
|---|---|---|---|
| Qwen2.5-0.5B (current) | 358M | GQA, 24 layers | baseline |
| **Gemma 3 270M** (Aug 2025) | **~100M** (170M are embeddings, 256k vocab) | GQA with sliding-window layers | Google positions it for fine-tuned classification and extraction; est. 3.6× fewer matmul FLOPs per token; check that the sliding window composes with the tree mask and that labels stay single tokens |
| LFM2-350M / **LFM2.5-350M** (Mar 2026) | ~0.3B | short convolutions + a few GQA layers | LFM2 report: prefill 1,067 tok/s at 1k context against 318 for Qwen3-0.6B (hardware not given); the convolutions see the previous tokens regardless of the mask, so tree-packed readouts would leak into each other; it needs one row per readout or a per-branch conv state |
| Qwen3.5-0.8B (early 2026) | larger than today | 18 Gated DeltaNet + 6 attention layers | bigger, and the recurrent layers cannot be tree-packed (already found); no gain here |
| Gemma 4 E2B (Apr 2026) | ~2B effective | – | too big for this goal |

Gemma 3 270M with §2.1 is est. ~190 GFLOP per record (10× fewer than today), and with §4.1 ~79 GFLOP (24×). Smaller
GEMMs run less efficiently on a CPU, hence ~1.1–1.5 s per record in the ladder rather than ~0.85. Whether a 100M
transformer reaches 0.92 gold is the open question: the base-model curve above shows a stronger base learns faster per
record.

### 4.3 Drop the top layers

Gromov et al. (ICLR 2025) remove up to about half the layers of Llama-2-70B before collapse, after a short QLoRA
"heal". Simply dropping the deepest layers nearly matches their similarity-based choice. Skean et al. (ICML 2025)
find intermediate layers give stronger features than the last layer on 32 embedding tasks. A Qwen2.5-0.5B cut to
16–18 blocks would be 1.3–1.5× cheaper in both training and inference. Small models are less redundant than 70B ones,
so this is a T4 experiment, not a given. Note that LoRA on only the top 8 blocks failed here (gold 0.597). That is a
different change: it froze the lower blocks rather than removing the upper ones.

### 4.4 Do not backpropagate through the model at all

* **Probe heads on frozen features.** Run the frozen model forward once over each record. No catalogue is needed:
  about 150 tokens, 107 GFLOP, ~1.2 s on the regular CPU, ~0.25 s on the M4. Store mean-pooled and last-token states
  from a few layers (~200 MB for 10k records) and train 9 heads in seconds. Retrain at will. Evidence is mixed: in one
  informal SmolLM2-360M comparison LoRA beat a linear probe by 3.6 points, and probes are usually a few points behind.
  The top-8 LoRA failure is not evidence against probes. It failed at binding label letters to shuffled options,
  which needs the lower blocks; a head predicts the class directly, with no letters involved. The TF-IDF baseline
  (gold 0.797) is the floor a probe must clear. A one-hour pilot on the M4 settles it.
* **Side-tuning** (LST 2022; QST, ACL 2024). A small side network reads the frozen model's hidden states, and the
  backbone is forward-only and can be 4-bit. QST reports up to 3× faster fine-tuning and 2.3× less memory with
  competitive accuracy. It sits between probes and LoRA, and is a research project in its own right.
* **Encoders.** Ettin (2025), with encoders and decoders from 17M to 1B trained identically, finds encoders win at
  classification: a 400M encoder beats a 1B decoder on MNLI. A ~150M encoder (ModernBERT-base, Ettin-150M) fully
  fine-tuned with 9 heads on the record alone is ~0.1 TFLOP per record, 18× less than today. It probably needs 3+
  epochs, though, and it gives up the instruction-tuned prior that lets the current model learn in a few thousand
  records. The frozen-MiniLM baseline (gold 0.707) says nothing about a fully fine-tuned encoder.

## 5. Hardware you may already have

* **The M4's GPU through MLX.** The PyTorch MPS attempt failed because every new sequence length triggered a
  recompile. MLX does not need a fixed shape per graph (`mx.compile` is optional), and mlx-lm ships Qwen2 with LoRA
  layers. The custom readout loss and the tree mask need a ~150-line MLX training loop (additive mask into
  `mx.fast.scaled_dot_product_attention`). The M4's GPU (10-core version) is commonly rated at roughly 4 TFLOPS FP32,
  against the 430 GFLOP/s the CPU path reaches now. One community benchmark shows Qwen3-0.6B LoRA SFT at ~4.7 it/s (batch 1) on an
  M4 Pro, sequence length not stated. Plausibly 3–5× over the M4 CPU, unmeasured. Combined with §2.1, the full recipe
  could land near an hour on the laptop.
* **MPS with length buckets.** Padding sequences to a few lengths (multiples of 128) stops the per-shape recompiles.
  §2.1 also makes shapes regular. Cheaper to try than MLX, with a less certain payoff.
* **Rented CPUs with AMX.** See §2.4. If you rent anyway, a T4 is still cheaper per run.
* **The Neural Engine.** maderix/ANE trains transformers on the ANE through reverse-engineered private APIs, at
  ~5–9 % of peak utilization, and could break with any macOS update. Not for this project.
* **llama.cpp training.** ggml has CPU training (PR #10544, described as "extremely limited"). Tether's QVAC Fabric
  fork adds LoRA fine-tuning on CPU, Vulkan and Metal, including Qwen3, with an assistant-only masked loss. It is not
  upstream (Dec 2025) and has no readout-only loss or tree mask, so using it would mean writing ggml code. Revisit if
  it is upstreamed: it would allow training on a quantized base with the same kernels used for serving.

## 6. CPU inference

Today (**measured**, `docs/RUN_ON_CPU.md`): 0.32 s per record on the M4 (Qwen2.5-0.5B cat, Q8_0) and 1.3–2.8 s on
the 4-vCPU x86 VM. A decision is prefill only: ~100–150 record tokens plus 9 readouts × ~6 tokens, attending to the
969 cached catalogue tokens. Prefill is compute-bound, so the levers differ from chat inference (decode, which is
memory-bound).

| Idea | Est. effect | Effort | Notes |
|---|---|---|---|
| One token per readout instead of `{"<id>": "` (~6 tokens) | −25–30 % tokens per record | retrain | e.g. 9 new special tokens trained via `trainable_token_indices`; also trims training slightly |
| Catalogue-free model (§4.1) | ~1.2–1.4× per record; start-up and catalogue KV gone | retrain | follows from the training change |
| Build llama.cpp for the machine (`-DGGML_NATIVE=ON`) and compare Q8_0, Q4_0 (repacked online for AVX2/NEON) and F16 | x86: plausibly 1.2–2× | low | the llamafile sgemm merge (#6414) found F16/F32 prompt processing 2× faster than before and faster than quantized types; Q4_0/Q8_0 got ~40 % |
| ik_llama.cpp row-interleaved types (Q8_K_R8) | x86: possibly > the above | low | its fastest prompt-processing path, benchmarked on an AVX2 Broadwell Xeon, likely the same class as the "regular CPU" |
| M4: KleidiAI SME2 kernels (`GGML_CPU_KLEIDIAI=ON`) | unknown on macOS | low | ggml picks SME2 > I8MM > DotProd at runtime; no macOS numbers found |
| Several records per call, tree-packed after the cached catalogue | throughput, not latency | low | bigger GEMMs; for batch scoring jobs |
| Fewer blocks (§4.3) or a smaller base (§4.2) | 1.3–3.6× | retrain | same trade-offs as for training |
| OpenVINO on Intel CPUs | unknown | medium | int8/AMX paths, prefix caching in Model Server; an alternative engine on Intel servers |

Not useful here: speculative decoding (nothing is generated); decode-oriented activation sparsity; BitNet/T-MAC
low-bit kernels (big wins at 1–2 bits, which would need a ternary model or hurt accuracy); KV-cache quantization (the
KV is small).

## 7. Looked at and rejected for training

| Idea | Why not |
|---|---|
| Zeroth-order optimizers (MeZO; FZOO, ICLR 2026; AdaMeZO, ICML 2026) | Forward-only, so memory-cheap, but a ZO step (two forwards) costs about as much as a frozen-base LoRA step, and MeZO needs tens of times more steps. FZOO's "Adam-scale" convergence is shown on RoBERTa-large; across models it is 3× fewer forwards than MeZO, still far behind backprop |
| LoRA on top blocks only | measured here: gold 0.597 |
| Progressive layer freezing (FreezeOut and successors) | ≤ 20 % wall-clock in vision; the lower blocks are the ones this task needs |
| Approximate backprop (WTA-CRS) | targets activation memory, not FLOPs; code for T5/BERT only |
| Token dropping (Random-LTD) | 25.6 % wall-clock in GPT pretraining; here the only droppable tokens are the catalogue's, which §2.1 and §4.1 already handle |
| Stop-gradient catalogue KV | ~1.2× over exact sharing (§2.1), with an approximation |
| Intel Extension for PyTorch | end of life (March 2026); use stock PyTorch |
| Training on the ANE | private APIs, ~5–9 % utilization |

## 8. What to run first, when you run things

Run accuracy experiments on the T4 (~1 h each) and only speed measurements on the CPUs (`bench_train_cpu.py`).

1. Implement §2.1 + §2.2 + §2.3 in `train/train.py`. Check gradient equivalence: with the same catalogue, the new path
   and the old per-record path should give the same LoRA gradients to fp32 noise. Measure s/record on the M4 and x86.
2. On the T4: the 4,800-record recipe with §2.1/§2.2. Target: gold ≥ 0.881, the current 4.8k result.
3. On the T4: the LR × records sweep (§3), then distillation from Qwen3-1.7B. Target: gold ≥ 0.92 with ≤ 6k records.
4. M4 pilot: frozen features + probe heads (§4.4), ~1 h. This decides whether probes are a real option.
5. On the T4: the catalogue-free model (§4.1) on Qwen2.5-0.5B, then on Gemma 3 270M (§4.2).
6. Optional: an MLX port for the M4 GPU (§5) and the inference build matrix (§6).

## Sources

Shared-prefix training
* Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse (2025, v5 Apr 2026) — https://arxiv.org/abs/2511.00413
* Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward (2025) — https://arxiv.org/abs/2506.05433 ; verl implementation notes — https://deepwiki.com/volcengine/verl/10.7-prefixgrouper:-prefix-shared-attention-optimization
* PrefixFlow / Schedule-Level Shared-Prefix Reuse for LLM RL Training (2026) — https://arxiv.org/abs/2606.01143

LoRA optimization and sample efficiency
* Thinking Machines, LoRA Without Regret (2025) — https://thinkingmachines.ai/blog/lora/
* Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning (ICML 2026) — https://arxiv.org/abs/2602.04998
* Beware of the Batch Size: Hyperparameter Bias in Evaluating LoRA (2026) — https://arxiv.org/abs/2602.09492
* Marek et al., Small Batch Size Training for Language Models (NeurIPS 2025) — https://arxiv.org/abs/2507.07101
* LoRA-GA (NeurIPS 2024) — https://arxiv.org/abs/2407.05000 ; PiSSA (NeurIPS 2024) — https://arxiv.org/abs/2404.02948 ; LoRA+ (ICML 2024) — https://arxiv.org/abs/2402.12354
* PEFT LoRA guide (initializations, LoRA+, trainable tokens) — https://huggingface.co/docs/peft/developer_guides/lora
* Revisiting Knowledge Distillation: The Hidden Role of Dataset Size (2025) — https://arxiv.org/abs/2510.15516
* Selective-Backprop, Accelerating Deep Learning by Focusing on the Biggest Losers (2019) — https://arxiv.org/abs/1910.00762 ; Composer method card — https://docs.mosaicml.com/projects/composer/en/v0.3.1/method_cards/selective_backprop.html

Removing the prompt, probes, side networks, encoders
* Snell et al., Learning by Distilling Context (2022) — https://arxiv.org/abs/2209.15189 ; On-Policy Context Distillation (2026) — https://arxiv.org/abs/2602.12275
* Cartridges (ICLR 2026) — https://arxiv.org/abs/2506.06266 , https://github.com/HazyResearch/cartridges
* Gist tokens (NeurIPS 2023) — https://arxiv.org/abs/2304.08467
* Fine-Tuning Causal LLMs for Text Classification: Embedding-Based vs. Instruction-Based (2025) — https://arxiv.org/abs/2512.12677
* LST: Ladder Side-Tuning (NeurIPS 2022) — https://arxiv.org/abs/2206.06522 ; QST: Quantized Side Tuning (ACL 2024) — https://arxiv.org/abs/2401.07159
* Ettin, Seq vs Seq: paired encoders and decoders (2025) — https://arxiv.org/abs/2507.11412

Smaller and shallower models
* Gemma 3 270M — https://developers.googleblog.com/en/introducing-gemma-3-270m/
* LFM2 Technical Report — https://arxiv.org/abs/2511.23404 ; LFM2.5-350M (Mar 2026) — https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind
* Qwen3.5-0.8B — https://huggingface.co/Qwen/Qwen3.5-0.8B
* Gromov et al., The Unreasonable Ineffectiveness of the Deeper Layers (ICLR 2025) — https://arxiv.org/abs/2403.17887 ; ShortGPT — https://arxiv.org/abs/2403.03853
* Skean et al., Layer by Layer (ICML 2025) — https://arxiv.org/abs/2502.02013

CPU runtimes and hardware
* IPEX discontinuation — https://github.com/intel/intel-extension-for-pytorch/issues/867 ; EOL notice — https://pytorch-extension.intel.com/
* Intel AMX for PyTorch — https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-training-inference-on-amx.html ; Phoronix AMX review — https://www.phoronix.com/review/intel-xeon-amx/3
* PyTorch 2.6 on Intel (FlexAttention on x86 CPU) — https://pytorch.org/blog/unlocking-pt-2-6-intel/
* llama.cpp training support, PR #10544 — https://github.com/ggml-org/llama.cpp/pull/10544 ; QVAC Fabric LoRA fine-tuning — https://huggingface.co/blog/qvac/fabric-llm-finetune
* mlx-lm — https://github.com/ml-explore/mlx-lm ; mlx-lm-lora — https://github.com/Goekdeniz-Guelmez/mlx-lm-lora
* ANE training (maderix) — https://github.com/maderix/ane
* MeZO — https://arxiv.org/abs/2305.17333 ; FZOO (ICLR 2026) — https://arxiv.org/abs/2506.09034 ; AdaMeZO (ICML 2026) — https://icml.cc/virtual/2026/poster/63417
* FreezeOut — https://arxiv.org/abs/1706.04983 ; WTA-CRS — https://arxiv.org/abs/2305.15265 ; Random-LTD — https://arxiv.org/abs/2211.11586

CPU inference
* llamafile sgemm in llama.cpp, PR #6414 — https://github.com/ggml-org/llama.cpp/pull/6414
* ik_llama.cpp CPU comparison — https://github.com/ikawrakow/ik_llama.cpp/discussions/164
* KleidiAI + SME2 in llama.cpp (Arm) — https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/performance_llama_cpp_sme2/
* T-MAC (EuroSys 2025) — https://arxiv.org/abs/2407.00088
* OpenVINO Model Server, long-context and prefix caching — https://docs.openvino.ai/2026/model-server/ovms_demo_long_context.html
