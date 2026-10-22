# AeroJev: a Jev-style decision model on CPU for aerospace supply chain

AeroJev turns a ~0.6B Qwen model into a typed **decision engine** for an aerospace
supply-chain analytics team. It takes a messy record (forwarded supplier email,
ERP exception row, Teams message, NCR, risk-intel alert, portal notice, call
notes) and a set of questions, and returns a probability for every allowed answer.
It never generates text, and it runs on a laptop or server CPU.

It follows the same idea as TypeSafe Jev, featherless-ai/simple-jev (RFDT),
Meanblock/JEV-CPU (SemIf) and sting8k/qwen-jev-like:

1. **Logit readout.** Each question ends with an assistant prefill such as
   `{"answer": "`. The next-token logits are restricted to the question's label
   tokens (`A`..`K`, `0`..`4`, `yes`/`no`) and softmaxed.
2. **One forward pass per record (tree packing).** The record is laid out once,
   then every question branch follows it. A 4D attention mask lets each branch
   see the record and itself only, and position ids restart per branch. This is
   exactly equivalent to running the questions separately (`tests/test_tree_pack.py`).
3. **Decision training (RFDT-style).** LoRA on all linear layers, trained with
   cross-entropy over the label-token logits at the readout positions only, plus
   an ordinal penalty on score questions. There is no LM loss and no JSON to learn.
4. **Temperature scaling** per question type, fitted on validation.
5. **CPU serving** via HF torch (fp32) or llama.cpp GGUF (Q8_0 / Q4_K_M), with
   KV-cache reuse.

## Methods surveyed (and what was tested here)

| Approach | Source | Idea | Tested here? |
|---|---|---|---|
| Zero-shot logit readout | Meanblock/JEV-CPU (SemIf) | Pretrained model + letter options, softmax over option-letter logits, fp32 CPU | Yes: Qwen2.5-0.5B, Qwen3-0.6B, Qwen3.5-0.8B, Qwen3-1.7B |
| Prompt-policy search | featherless-ai/simple-jev | Pick a prompt layout (`baseline`, `repeat_state`, ...) per model family without training | Partly: two layouts (branch, catalogue) |
| Decision training on label logits (RFDT) | featherless-ai/simple-jev `RFDT/` | CE / KL over allowed-label logits at the readout only, LoRA or full FT, optional teacher soft labels | **Yes: the main method** (hard labels from the playbook, LoRA r=16) |
| Reserved option tokens, tree batching, per-type temperature | Oumi blog (Jev reverse-engineering) | Score `<opt_i>` tokens at one position; lay out the state once with a branch mask per question; calibrate per question type | Tree batching + per-type temperature: yes. Reserved tokens: no (plain letters/digits are single tokens in Qwen's vocab) |
| Catalogue-first prompt, cached prefix, trie readout | sting8k/qwen-jev-like | Question catalogue before the state, one forward pass, multi-token values via a trie | Catalogue-first with a persistent KV cache: yes (single-token labels, so no trie needed) |
| Supervised classical classifiers | standard | TF-IDF / sentence embeddings + logistic regression per question | Yes (strong baseline, but cannot answer new questions) |
| Generate JSON with a small LLM | standard | Prompt for a JSON object, optionally grammar-constrained | Yes: free and schema-constrained (llama.cpp) |
| Hosted decision model | TypeSafe Jev | Large closed decision model behind an API | Yes, as an external reference (`eval/jev.py`, run by a parallel session) |
| CPU runtimes | llama.cpp, torch | GGUF Q8_0 / Q4_K_M via llama.cpp vs torch fp32 | Yes, on an Apple M4 and a Kaggle x86 CPU |

## The decision schema

| id | type | what it decides |
|---|---|---|
| `event_type` | choice (11) | delay, quality escape, capacity, raw material, logistics, financial distress, compliance, force majeure, obsolescence, commercial, no issue |
| `commodity` | choice (10) | castings/forgings, fasteners, electronics, composites, machined, raw material, harnesses, chemicals, mechanical assemblies, other |
| `impact_severity` | score 0-4 | none → critical (line stop / AOG / escape onto delivered aircraft) |
| `line_stop_risk` | yes/no | stock-out, line stop or AOG within 30 days if nothing changes |
| `recommended_action` | choice (9) | monitor, expedite, recovery plan/SCAR, alternate source, quarantine, escalate, trade compliance, renegotiate, last-time buy |
| `export_controlled` | yes/no | ITAR / EAR 600-series items or data involved |
| `counterfeit_suspect` | yes/no | counterfeit / suspect unapproved part indicators |
| `sole_source` | yes/no | sole / single qualified source stated |
| `delay_bucket` | score 0-4 | not late, 1-7 d, 8-30 d, 31-90 d, >90 d / indefinite |
| `supplier_region`, `mentions_new_date`, `military_program` | held out | **never trained on**; used to measure transfer to new questions |

Labels come from a written playbook (`data/gen/policy.py`). For example, the
shortfall is days late minus days of stock, and severity, line-stop and action
follow from it. Several decisions therefore need arithmetic across facts
(e.g. "40 on hand, 10 per week" against "slip of 6 weeks").

## Two prompt layouts

| layout | prefix (computed per record) | per-question branch | when to use |
|---|---|---|---|
| `branch` | system + record | full question + options + prefill (~80 tokens) | arbitrary ad-hoc questions |
| `catalogue` | static question catalogue (**KV-cached once per process**) + record | `{"<id>": "` (~5 tokens) | a fixed schema; ~4x less CPU work per record |

The catalogue layout matches cached and uncached logits to 1e-4
(`tests/test_catalogue_cache.py`). It needs dense attention: Qwen3.5's Gated
DeltaNet layers carry recurrent state, so neither tree packing nor
prefix-branch reuse is valid there, and the engine falls back to `unpacked` mode.

## Data

Labels are sampled first, as latent facts: event, commodity, days late, stock
coverage, sole source, export marking, counterfeit signals and so on. The text
is then rendered from phrase banks in 7 formats. Every fact the playbook uses
is stated in the record.

| split | records | how it differs from train |
|---|---|---|
| `train` / `val` | 6000 / 500 | training templates, 6 formats (val shares 60% of 8-grams with train) |
| `test` | 800 | ~25% of every phrase bank held out, plus the test-only `call_notes` format (25% 8-gram overlap) |
| `injection` | 200 | test templates, each record carries a prompt-injection line |
| `gold` | 80 | hand-written messy records in formats the generator never produces (0.7% 8-gram overlap); labels from the same playbook |

All company and program names are fictional.

## Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch transformers==5.20.2 peft==0.21.2 accelerate scikit-learn sentence-transformers
CMAKE_ARGS="-DGGML_METAL=OFF" pip install llama-cpp-python==0.3.16      # CPU GGUF backend

python -m data.gen.generate && python -m data.gold.build_gold           # datasets
python -m eval.classical                                                # TF-IDF / MiniLM baselines
python -m eval.evaluate --model Qwen/Qwen3-0.6B --sets gold --out results/zs-local   # zero-shot readout

# GPU work runs as private Kaggle kernels (weights download inside Kaggle)
python kaggle/make_kernel.py bundle && kaggle datasets version -p kaggle/dataset -m update
python kaggle/make_kernel.py kernel train && kaggle kernels push -p kaggle/kernels/train

# CPU serving
python server.py --gguf models/aerojev-qwen3-0.6b-cat-q8_0.gguf --tokenizer Qwen/Qwen3-0.6B --layout catalogue
curl -s localhost:8080/v1/decide -d '{"state": "Kestrel: 718 disks slip 6 weeks. 2 weeks on hand. Sole source."}'
```

## Weights

Model files are not in git (GGUF files are 0.4–0.6 GB; LoRA adapters ~35–40 MB each). They are produced by
the private Kaggle kernels: `aerojev-train` (branch layout), `aerojev-train-cat` (catalogue layout),
`aerojev-train-v2` (data v2, Qwen3-0.6B and Qwen3-1.7B) and `aerojev-notebook`. Fetch them with, for example,
`kaggle kernels output sparshnagpal/aerojev-train -p kaggle/outputs/train --file-pattern ".*\.(gguf|safetensors)$"`,
or regenerate them with the `export` job in `kaggle/runner.py`.

## End-to-end notebook

`notebooks/aerojev_qwen2.5-0.5b_lora.ipynb` runs the whole pipeline for one base model
(Qwen2.5-0.5B-Instruct + LoRA). It covers the schema, playbook, data generation and leakage check, gold sets,
prompt layouts, a tree-packing check, baselines, training, evaluation, calibration, GGUF export, CPU latency,
and usage. Regenerate it with `python scripts/build_notebook.py`.

* **Kaggle (full run on a T4):** attach the `aerojev-bundle` dataset and run all cells, or
  `python kaggle/make_kernel.py notebook notebook gpu && kaggle kernels push -p kaggle/kernels/notebook`.
* **Locally / CPU only:** it switches to FAST mode, a smoke test with a few dozen records per step.
  Environment overrides: `AEROJEV_NB_MODEL` (another base model), `AEROJEV_NB_FAST=0|1`,
  `AEROJEV_NB_EXPORT=0` (skip the llama.cpp build).

## Layout

```
aerojev/    schema.py (questions) · prompt.py (layouts, tree packing) · engine.py (HF / GGUF engines) · metrics.py
data/gen/   world.py · policy.py (playbook) · render.py (7 formats, split-aware phrase banks) · generate.py
data/gold/  80 hand-written records
train/      train.py (LoRA decision training)
eval/       evaluate.py · classical.py · generative.py · bench_cpu.py · analyze.py
kaggle/     runner.py (per-GPU job queues) · make_kernel.py (bundle, script and notebook kernels)
notebooks/  aerojev_qwen2.5-0.5b_lora.ipynb (end-to-end walkthrough, built by scripts/build_notebook.py)
tests/      tree-pack, catalogue-cache and GGUF parity checks
server.py   stdlib HTTP API
```

## Results

See `results/REPORT.md` (generated by `python scripts/report.py`).
