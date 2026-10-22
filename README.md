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

## Building the dataset

There is no labelled history of supply-chain records, so the training data is **built** from facts. The order
matters, because labels must never be guessed from text:

1. **World** (`data/gen/world.py`): fictional suppliers (each tied to the commodities it plausibly makes), parts,
   programs (some military), materials, ports and defect types. Swap in your own supplier and part masters here.
2. **Facts** (`Scenario` in `data/gen/policy.py`): one record's ground truth: event, commodity, days late, days of
   stock, line down / AOG, sourcing, export marking, counterfeit signals, months to last-time buy.
3. **Playbook** (`derive_labels` in `data/gen/policy.py`): the team's rules, written as code, that turn facts into
   the answer to every question (for example, shortfall = days late − days of stock drives severity, line-stop risk
   and the action). Writing these down is where the team agrees on the right answers.
4. **Sampling** (`sample_scenario`): facts are drawn per event, steering toward a balanced mix of regimes (stock covers
   the delay / short shortfall / long shortfall / no recovery date) and rejecting draws near any rule's threshold.
5. **Rendering** (`data/gen/render.py`): per-fact phrase banks ("6 weeks late", "40 on hand, 10 a week", "sole source",
   "ITAR controlled"), per-event situations in formal and informal registers, and 7 formats (supplier email, ERP
   exception, Teams/Slack message, quality notification, risk alert, portal notice, call notes). About a third of
   records get distractor sentences, and a few percent get a prompt-injection line with unchanged labels.
6. **Splits**: each phrase bank is split by a stable hash of the template index, so ~25% of every bank's wording, and
   the whole `call_notes` format, appear only in test. `scripts/make_data.sh` builds every split deterministically.
7. **Leakage check**: the share of each evaluation set's 8-grams that also occur in training text.
8. **Gold sets** (`data/gold/`): records written by hand in real-world styles, facts first, labelled by the same
   playbook: `gold` (80, used for error analysis) and `gold2` (48, written blind before any data iteration).

| split | records | how it differs from train | 8-gram overlap with train |
|---|---|---|---|
| `train` / `val` (v1) | 6000 / 500 | training templates, 6 formats | val 60% |
| `train` / `val` (v2) | 10000 / 500 | adds early deliveries, equipment outages, supplier-caught defects, more injection styles | |
| `test` | 800 | held-out templates + the test-only `call_notes` format | 25% |
| `test_v2` | 800 | held-out templates of the v2 generator | |
| `injection` | 200 | a prompt-injection line in every record | |
| `gold` / `gold2` | 80 / 48 | hand-written, formats the generator never produces | 0.7% |

The notebook's section 2 walks through every step above with runnable code. All company and program names are fictional.

## Reproduce

Everything runs locally. Training needs a CUDA GPU (Qwen3-0.6B takes about 1.5 h for 2 epochs on a T4-class card);
everything else, including inference, runs on a CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # on Apple Silicon see the llama-cpp-python note inside

# 1. data: v1 + v2 splits and both gold sets
bash scripts/make_data.sh

# 2. baselines
python -m eval.classical                                             # TF-IDF / MiniLM + logistic regression
python -m eval.evaluate --model Qwen/Qwen3-0.6B --sets gold,test --out results/zs-qwen3-0.6b   # untrained readout

# 3. decision training (GPU)
python -m train.train --model Qwen/Qwen3-0.6B --layout catalogue --data data/processed_v2 --epochs 2 \
    --out runs/qwen3-0.6b-cat-v2

# 4. evaluation (fits per-type temperatures on val)
python -m eval.evaluate --model Qwen/Qwen3-0.6B --adapter runs/qwen3-0.6b-cat-v2/best --layout catalogue \
    --data data/processed_v2 --sets val,test,test_v2,gold,gold2,injection --out results/ft-qwen3-0.6b-cat-v2

# 5. export for CPU: merge LoRA, GGUF Q8_0 + Q4_K_M (builds llama.cpp's quantizer on first use)
python -m aerojev.export --model Qwen/Qwen3-0.6B --adapter runs/qwen3-0.6b-cat-v2/best --out models \
    --name aerojev-qwen3-0.6b-cat-v2

# 6. CPU latency, every system under the same protocol (run on an idle machine)
python -m eval.latency_suite --out results/latency_local.jsonl

# 7. report and serving
python scripts/report.py
python server.py --gguf models/aerojev-qwen3-0.6b-cat-v2-q8_0.gguf --tokenizer Qwen/Qwen3-0.6B --layout catalogue
curl -s localhost:8080/v1/decide -d '{"state": "Kestrel: 718 disks slip 6 weeks. 2 weeks on hand. Sole source."}'
```

## Weights

Model files are not in git (GGUF files are 0.4–0.6 GB; LoRA adapters ~35–40 MB each). Steps 3 and 5 above recreate
them: training writes `runs/<run>/best/` (the LoRA adapter) and the export writes `models/*.gguf`.

## Test UI

**Running on a CPU:** see [`docs/RUN_ON_CPU.md`](docs/RUN_ON_CPU.md) for the recommended setup, install steps and the measurements behind them.

A local page for trying the model on any record, served by `server.py` (standard library only):

```bash
python server.py --preset best     # Qwen3-0.6B + LoRA (data v2), GGUF Q8_0; or --preset fast for Qwen2.5-0.5B
open http://127.0.0.1:8080
```

Paste a supplier email, ERP row or chat message, or load one of eight example records (`ui/examples.py`), then press
Decide. Each of the twelve questions shows its answer, its probability distribution and, for the examples, ✓ or ✗
against the answer the playbook gives for the facts the record states. You can also add a yes/no question of your own.
The page is `ui/index.html`; the same server answers `POST /v1/decide` for programs.

## End-to-end notebook

`notebooks/aerojev_qwen2.5-0.5b_lora.ipynb` runs and explains the whole pipeline for one base model
(Qwen2.5-0.5B-Instruct + LoRA): the decisions, building the dataset step by step, the prompt layout and readout,
tree packing, baselines, training, evaluation, calibration, GGUF export, CPU latency and usage.
Regenerate it with `python scripts/build_notebook.py`.

* **On a CUDA GPU** it runs the full recipe (about an hour on a T4-class card).
* **On a CPU only** it switches to FAST mode, the same steps with a few dozen records each.
* Overrides: `AEROJEV_NB_MODEL` (another base model), `AEROJEV_NB_FAST=0|1`, `AEROJEV_NB_EXPORT=0`
  (skip the llama.cpp build), `AEROJEV_ROOT` (repo location if Jupyter starts elsewhere).

## Layout

```
aerojev/    schema.py (questions) · prompt.py (layouts, tree packing) · engine.py (HF / GGUF engines)
            metrics.py · export.py (merge + GGUF) · benchrec.py (latency record schema)
data/gen/   world.py · policy.py (facts + playbook) · render.py (7 formats, split-aware phrase banks) · generate.py
data/gold/  hand-written gold (80) and blind gold2 (48) records
train/      train.py (LoRA decision training)
eval/       evaluate.py · classical.py · generative.py · jev.py · bench_cpu.py · latency_suite.py · analyze.py
notebooks/  end-to-end walkthrough (built by scripts/build_notebook.py)
scripts/    make_data.sh · report.py · export_dashboard.py · build_notebook.py
tests/      tree-pack, catalogue-cache and GGUF parity checks
docs/       RUN_ON_CPU.md (CPU inference guide)
server.py   stdlib HTTP API and test page (ui/index.html, ui/examples.py)
kaggle/     optional: how the GPU runs for this report were executed (see below)
```

## Results

Mean accuracy over the nine trained questions. CPU time is per record of nine decisions (one record at a time, warm,
four threads): an Apple M4 laptop and a 4-vCPU x86 cloud VM. Full tables: `results/REPORT.md`.

| System | Gold (80) | Gold2 (48, blind) | Test (800) | M4 s/record | x86 s/record |
|---|---|---|---|---|---|
| **Qwen3-0.6B + LoRA, catalogue, data v2** (fp32) | **0.950** | 0.931 | **0.959** | 0.49 | – |
| same, GGUF Q8_0 (640 MB) | 0.950 | 0.931 | 0.960 | 0.49 | – |
| same, GGUF Q4_K_M (400 MB) | 0.940 | 0.921 | 0.951 | 0.51 | – |
| Qwen2.5-0.5B + LoRA, catalogue (GGUF Q8_0) | 0.931 | **0.938** | – | **0.32** | 1.80 |
| Qwen3-0.6B + LoRA, catalogue, data v1 (fp32) | 0.935 | 0.921 | 0.954 | 0.48 | 1.98 |
| Qwen3-0.6B + LoRA, branch layout (fp32) | 0.929 | 0.921 | 0.933 | 1.42 | 12.2 |
| Jev hosted API, best variant (GPU + network) | 0.918 | – | 0.882 | 0.12 (API) | – |
| TF-IDF + logistic regression | 0.797 | 0.806 | 0.852 | 0.0008 | 0.009 |
| Qwen3-0.6B untrained, logit readout | 0.632 | 0.646 | 0.570 | 1.49 | 11.6 |
| Qwen3-0.6B generating JSON (Q8_0) | 0.590 | – | – | 1.73 | 6.9 |

* **Decision training is what makes a sub-1B model usable**: the untrained readout is at 0.63, the trained one at 0.95.
* **The catalogue layout is the main CPU lever**: caching the question catalogue's KV makes the same model ~3× faster on
  the M4 and ~6× on the x86 VM, and it was also more accurate than the branch layout.
* **Quantization is nearly free for accuracy** (Q8_0 matches fp32; Q4_K_M costs about a point) and cuts memory ~3×, but
  it does not make decisions faster than torch fp32 on these CPUs.
* **Training on a CPU is possible but slow**: 4.4 s per training record on the M4 (the full 12,000-record recipe ≈ 14.5 h;
  LoRA on the top 8 blocks ≈ 2×  faster), against ~70 min on one T4 GPU (`scripts/bench_train_cpu.py`).

## Appendix: running the GPU steps on Kaggle (optional)

The runs in this report were executed as private Kaggle kernels, because the development machine had no CUDA
GPU and little free disk. `kaggle/` holds that tooling: `runner.py` runs one job queue per GPU (train, eval,
export, any command) and `make_kernel.py` packs the code and data into a private dataset and writes kernel
folders. None of the project code depends on it; any machine with a CUDA GPU can run steps 3–5 directly.
