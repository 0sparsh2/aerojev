# AeroJev: Jev-style decision models on a CPU, for aerospace supply chain

AeroJev reads a messy supply-chain record and answers a fixed set of operational questions about it. Each answer
comes with a **probability for every allowed option**. Records include forwarded supplier emails, ERP exception
rows, Teams messages, NCRs, risk-intel alerts, portal notices and call notes. AeroJev never writes free text, and it
runs on an ordinary laptop or server **CPU**, with no GPU.

The smallest model is a 151 MB file. It answers nine decisions per record in **28 ms** on an Apple M4, uses **under
0.5 GB of RAM**, runs **without PyTorch**, and scores **0.932** mean accuracy on 80 hand-written records. For
comparison, the hosted Jev API scored **0.918** on the same records.

```bash
curl -s localhost:8080/v1/decide -d '{"state": "Kestrel: 718 disks slip 6 weeks. 2 weeks on hand. Sole source."}'
```

This is the `tiny` model's actual answer on an M4 (7 ms for this short record; probabilities under 0.01 omitted):

```jsonc
{"answers": {
  "event_type":          {"type": "choice", "answer": "delivery_delay", "confidence": 0.905,
                          "probabilities": {"delivery_delay": 0.905, "force_majeure": 0.056, "capacity_constraint": 0.019, ...}},
  "commodity":           {"type": "choice", "answer": "castings_forgings", "confidence": 1.0, ...},
  "impact_severity":     {"type": "score", "answer": "3", "confidence": 0.986, "expected": 2.99, ...},
  "line_stop_risk":      {"type": "yesno", "answer": "yes", "p_yes": 1.0, ...},
  "recommended_action":  {"type": "choice", "answer": "expedite", "confidence": 0.977, ...},
  "export_controlled":   {"type": "yesno", "answer": "no", ...},
  "counterfeit_suspect": {"type": "yesno", "answer": "no", ...},
  "sole_source":         {"type": "yesno", "answer": "yes", ...},
  "delay_bucket":        {"type": "score", "answer": "3", "expected": 3.0, ...}},     // 3 = 31-90 days late
 "latency_ms": 7.2}
```

A 6-week slip against 2 weeks of stock is a 4-week shortfall. Under the team's playbook that means severity 3, a
line-stop risk, and expedite.

All company, supplier and program names in this repository are fictional.

## Contents

1. [Choose a model](#1-choose-a-model)
2. [Quick start](#2-quick-start)
3. [How it works](#3-how-it-works)
4. [The decisions](#4-the-decisions)
5. [The data](#5-the-data)
6. [Results](#6-results)
7. [What was tried, in order](#7-what-was-tried-in-order)
8. [Reproduce everything](#8-reproduce-everything)
9. [CPU deployment notes](#9-cpu-deployment-notes)
10. [Repository layout](#10-repository-layout)
11. [Weights and where they live](#11-weights-and-where-they-live)
12. [Running the heavy steps on Kaggle](#12-running-the-heavy-steps-on-kaggle)
13. [Tests](#13-tests)
14. [Next: general decision models](#14-next-general-decision-models)
15. [Limitations](#15-limitations)
16. [Documentation index](#16-documentation-index)
17. [Base models and licences](#17-base-models-and-licences)

## 1. Choose a model

Five trained models ship as server presets (`python server.py --preset <name>`). Accuracy is the mean over the nine
trained questions. **Gold** is 80 hand-written records; **gold2** is 48 more, written blind before any data
iteration; **test** is 800 synthetic records with held-out wording. CPU times are per record (all nine decisions),
warm, 4 threads, on an Apple M4.

| Preset | Model | File | CPU per record | Peak RAM | Gold | Gold2 | Test | Questions it can answer |
|---|---|---|---|---|---|---|---|---|
| **`tiny`** | Ettin-150M encoder + 9 heads, ONNX int8 | **151 MB** | **28 ms**ᵃ | **0.40–0.48 GB** | 0.932 | 0.928 | **0.969** | the 9 trained ones |
| `small` | Qwen2.5-0.5B pruned to 16 blocks + FFN 2560, catalogue-free, GGUF Q8_0 | 299 MB | 83 msᵃ | 1.0 GB | 0.915 | 0.914 | 0.951 | the 9 trained ones |
| `fast` | Qwen2.5-0.5B + LoRA, catalogue layout, GGUF Q8_0 | 531 MB | 0.32 s | 1.4 GB | 0.931 | 0.938 | – | trained + ad-hoc questions |
| **`best`** | Qwen3-0.6B + LoRA, data v2, catalogue layout, GGUF Q8_0 | 639 MB | 0.49 s | 2.0 GB | **0.950** | 0.931 | 0.960 | trained + ad-hoc questions |
| `large` | Qwen3-1.7B + LoRA, data v2, catalogue layout, GGUF Q8_0 | 1.8 GB | 1.17 s | 2.6 GB | 0.943 | **0.949** | 0.965 | trained + ad-hoc questions |

ᵃ Median over 880 gold + test records, one at a time (`results/latency_lite.jsonl`). The Qwen rows were measured
earlier on 40 gold records under the latency suite's protocol (`results/REPORT.md`).

* **Smallest, fastest, least memory: `tiny`.** It is the default recommendation when the nine questions are all you
  need. It needs only `numpy`, `onnxruntime` and `tokenizers`.
* **Most accurate, and it takes new questions: `best`.** The Qwen models can also answer a yes/no or multiple-choice
  question written at request time. Accuracy on such questions is lower than on trained ones (section 6).
* **Most robust to option order: `large`.** It gives the same answer for 0.95 of choice questions when their options
  are shuffled, against 0.87 for `best`. It costs about 2.4× the CPU time.
* **On an x86 server, `tiny` needs its 4-bit file.** The int8 file is accurate on Apple Silicon and ARM, but on x86
  CPUs without VNNI it is badly wrong: gold accuracy falls to 0.52. These are Intel CPUs before Ice Lake / Cascade
  Lake and AMD before Zen 4.
  * **The 4-bit file is lossless there:** gold 0.933 for the same model. It is 109 MB, peaks at 0.33 GB, and
    `aerojev.lite` picks it automatically on x86.
  * **Speed:** 320 ms per record with 4 threads on a 2.2 GHz cloud Xeon, against 28 ms on the M4.
  * **Details:** section 9.

## 2. Quick start

**The `tiny` model, no PyTorch** (about 100 MB of packages):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lite.txt        # numpy, onnxruntime, tokenizers
python server.py --preset tiny              # needs models/aerojev-ettin-150m/ (section 11)
open http://127.0.0.1:8080                  # test page
```

**Any model** (full install: training, evaluation, export, llama.cpp):

```bash
pip install -r requirements.txt             # on Apple Silicon see the llama-cpp-python note inside
python server.py --preset best              # or tiny / small / fast / large
```

**The HTTP API** (`server.py`, standard library only, listens on 127.0.0.1):

```bash
# every question the loaded model knows
curl -s localhost:8080/v1/decide -d '{"state": "Kestrel says the 718 disks slip 6 weeks; 2 weeks on hand; sole source."}'
# only some of them
curl -s localhost:8080/v1/decide -d '{"state": "...", "questions": ["impact_severity", "recommended_action"]}'
# an ad-hoc question (Qwen catalogue/branch models only; tiny and small refuse it)
curl -s localhost:8080/v1/decide -d '{"state": "...", "questions": [{"id": "call_today", "type": "yesno",
     "text": "Should a buyer call the supplier today?"}]}'
curl -s localhost:8080/health               # model, layout, engine (llama.cpp / onnxruntime / torch), calibrated?
curl -s localhost:8080/v1/questions         # the question catalogue the model answers by default
```

**From Python**, without HTTP:

```python
from aerojev.lite import LiteEngine                        # torch-free
eng = LiteEngine("models/aerojev-ettin-150m", threads=2)   # picks int8 on ARM, int4/fp32 on x86
eng.decide("Kestrel says the 718 disks slip 6 weeks; 2 weeks on hand; sole source.")

import json
from aerojev import schema
from aerojev.engine import GGUFEngine                      # Qwen models via llama.cpp
eng = GGUFEngine("models/aerojev-qwen3-0.6b-cat-v2-q8_0.gguf", "Qwen/Qwen3-0.6B", n_threads=4, layout="catalogue",
                 temperatures=json.load(open("models/temps-qwen3-0.6b-cat-v2.json")))
eng.decide(record_text, schema.TRAIN_QUESTIONS)
```

Create an engine once and reuse it. Loading takes 0.3 s (`tiny`) to 3.5 s (`best`), and a decision takes
milliseconds to hundreds of milliseconds.

**The test page** (`ui/index.html`) is served at `/`. Paste a record, or load one of eight examples (`ui/examples.py`),
then press Decide. Every question shows:
* its answer;
* the full probability distribution;
* for the examples, ✓ or ✗ against the answer the playbook gives for the stated facts.

With the Qwen models you can also add a yes/no question of your own.

## 3. How it works

AeroJev follows the idea behind TypeSafe's hosted **Jev** decision model, and open re-creations of it:
featherless-ai/simple-jev (RFDT), Meanblock/JEV-CPU (SemIf) and sting8k/qwen-jev-like. A question is answered by
**reading scores for its allowed options**, never by generating text. Two model families do this here.

### 3.1 Decoder models (Qwen + LoRA): `fast`, `best`, `large`, `small`

1. **Logit readout.** Each question ends with an assistant prefill such as `{"answer": "`. The next-token logits
   are restricted to the question's label tokens and softmaxed. The labels are `A`..`K` for choices, `0`..`4` for
   scores, and `yes`/`no`. There is no sampling and no JSON parsing.
2. **One forward pass per record (tree packing).** The record is laid out once, then every question's branch follows
   it.
   * A 4D attention mask lets each branch see the record and itself only.
   * Position ids restart per branch.
   * This is exactly equivalent to running each question separately (`tests/test_tree_pack.py`).
3. **Decision training (RFDT-style).**
   * LoRA r=16 on every linear layer.
   * The loss is cross-entropy over the label-token logits at the readout positions only, plus an ordinal penalty on
     score questions.
   * There is no language-modelling loss and no JSON to learn.
4. **Temperature scaling** per question type (choice / score / yes-no), fitted on the validation set, so that
   confidences are calibrated.
5. **CPU serving** through llama.cpp (GGUF Q8_0 / Q4_K_M) or PyTorch fp32, with key-value-cache reuse.

**Three prompt layouts**, a trade-off between flexibility and CPU cost:

| Layout | Prefix (computed per record) | Per-question branch | Use |
|---|---|---|---|
| `branch` | system + record | full question + options + prefill (~80 tokens) | arbitrary ad-hoc questions; slowest |
| `catalogue` | static question catalogue (**KV-cached once per process**) + record | `{"<id>": "` (~5 tokens) | a fixed schema plus ad-hoc questions; ~3× faster than branch on the M4, ~6× on x86 |
| `nocat` | record only (no question list at all) | `{"<id>": "` | the 9 trained questions only; 6× less training compute (section 7) |

The catalogue layout matches cached and uncached logits to 1e-4 (`tests/test_catalogue_cache.py`). It needs dense
attention. Qwen3.5's Gated DeltaNet layers carry recurrent state, so neither tree packing nor prefix reuse is valid
there, and the engine falls back to `unpacked` mode.

### 3.2 Encoder model (Ettin-150M + heads): `tiny`

For a **fixed** set of questions the task is plain multi-head classification, and an encoder built for
classification is enough.

* **Architecture.** `train/train_encoder.py` puts one small linear head per question on Ettin-150M (a
  ModernBERT-style encoder). The heads score the question's options from the mean of the record's token states.
* **Training.** Every weight is trained, for 8 epochs, with cross-entropy plus the same ordinal penalty.
* **Export.** `scripts/export_encoder.py` traces encoder, pooling and heads into one ONNX graph and checks it
  against PyTorch (largest logit difference 3e-5). It then quantizes the weights.
* **Serving.** `aerojev/lite.py` runs it with ONNX Runtime + `tokenizers` + numpy. One forward pass answers all
  nine questions.

The full story, and every variant tried, is in [`docs/ENCODER.md`](docs/ENCODER.md).

### 3.3 Methods surveyed, and what was tested here

| Approach | Source | Idea | Tested here? |
|---|---|---|---|
| Zero-shot logit readout | Meanblock/JEV-CPU (SemIf) | pretrained model + letter options, softmax over option-letter logits | Yes: Qwen2.5-0.5B, Qwen3-0.6B, Qwen3.5-0.8B, Qwen3-1.7B |
| Prompt-policy search | featherless-ai/simple-jev | pick a prompt layout per model family without training | partly: branch vs catalogue |
| Decision training on label logits (RFDT) | featherless-ai/simple-jev `RFDT/` | CE / KL over allowed-label logits at the readout only | **Yes: the main decoder method** |
| Reserved option tokens, tree batching, per-type temperature | Oumi blog (Jev reverse-engineering) | score `<opt_i>` tokens; lay out the state once with a branch mask; calibrate per type | tree batching + per-type temperature: yes. Reserved tokens: no (letters and digits are single tokens in Qwen's vocabulary) |
| Catalogue-first prompt, cached prefix, trie readout | sting8k/qwen-jev-like | catalogue before the state, one pass, multi-token values via a trie | catalogue-first with a persistent KV cache: yes (labels are single tokens, so no trie) |
| Encoder + classification heads | standard | fine-tune a small encoder, one head per question | **Yes: the `tiny` model** (Ettin 32M/68M/150M, UltraBERT-150M) |
| Structured pruning | Minitron / ShortGPT-style | drop blocks, narrow FFNs, retrain | Yes: `small` (section 7) |
| Supervised classical classifiers | standard | TF-IDF / sentence embeddings + logistic regression per question | Yes (strong, fast baseline; cannot take new questions) |
| Generate JSON with a small LLM | standard | prompt for a JSON object, optionally grammar-constrained | Yes: free and schema-constrained (llama.cpp) |
| Hosted decision model | TypeSafe Jev | large closed decision model behind an API | Yes, as the external reference (`eval/jev.py`) |

## 4. The decisions

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
| `supplier_region`, `mentions_new_date`, `military_program` | held out | **never trained on**; they measure transfer to new questions |

The questions, their wording and their options live in `aerojev/schema.py`.

Labels come from a written **playbook** (`data/gen/policy.py`). For example, the shortfall is days late minus days
of stock, and severity, line-stop risk and the action follow from it. Several decisions therefore need arithmetic
across facts, such as "40 on hand, 10 per week" against "slip of 6 weeks".

## 5. The data

There is no labelled history of supply-chain records, so the training data is **built from facts**. The order
matters, because labels must never be guessed from text:

1. **World** (`data/gen/world.py`).
   * Fictional suppliers, each tied to the commodities it plausibly makes.
   * Parts, programs (some military), materials, ports and defect types.
   * Swap in your own supplier and part masters here.
2. **Facts** (`Scenario` in `data/gen/policy.py`): one record's ground truth.
   * Event and commodity.
   * Days late, days of stock, line down / AOG.
   * Sourcing and export marking.
   * Counterfeit signals and months to last-time buy.
3. **Playbook** (`derive_labels` in `data/gen/policy.py`): the team's rules, as code, turning facts into every
   answer. Writing them down is where a team agrees on the right answers.
4. **Sampling** (`sample_scenario`).
   * Facts are drawn per event.
   * Draws are steered toward a balanced mix of regimes: stock covers the delay, short shortfall, long shortfall,
     no recovery date.
   * Draws near any rule's threshold are rejected.
5. **Rendering** (`data/gen/render.py`).
   * **Phrase banks per fact:** "6 weeks late", "40 on hand, 10 a week", "sole source", "ITAR controlled".
   * **Situations per event,** in formal and informal registers.
   * **Seven formats:** supplier email, ERP exception, Teams/Slack message, quality notification, risk alert, portal
     notice, call notes.
   * **Noise:** about a third of records get distractor sentences, and a few percent get a prompt-injection line
     (labels unchanged).
6. **Splits.** Each phrase bank is split by a stable hash of the template index. About 25% of every bank's wording,
   and the whole `call_notes` format, appear only in test. `scripts/make_data.sh` builds every split
   deterministically.
7. **Leakage check:** the share of each evaluation set's 8-grams that also occur in training text.
8. **Gold sets** (`data/gold/`). Records written by hand in real-world styles, facts first, labelled by the same
   playbook.
   * `gold`: 80 records, used for error analysis.
   * `gold2`: 48 records, written blind before any data iteration.

| split | records | how it differs from train | 8-gram overlap with train |
|---|---|---|---|
| `train` / `val` (v1) | 6,000 / 500 | training templates, 6 formats | val 60% |
| `train` / `val` (v2) | 10,000 / 500 | adds early deliveries, equipment outages, supplier-caught defects, more injection styles | |
| `test` | 800 | held-out templates + the test-only `call_notes` format | 25% |
| `test_v2` | 800 | held-out templates of the v2 generator | |
| `injection` | 200 | a prompt-injection line in every record | |
| `gold` / `gold2` | 80 / 48 | hand-written, formats the generator never produces | 0.7% |

The processed splits are committed under `data/processed/` (v1) and `data/processed_v2/` (v2). Section 2 of the
LoRA notebook walks through every step with runnable code.

## 6. Results

Mean accuracy over the nine trained questions. Sampling error is about ±1.0 point on gold, ±1.3 on gold2 and ±0.3 on
test. Two training runs of the same recipe differ by up to about 1.5 points on gold, so smaller differences are noise.

| System | Gold (80) | Gold2 (48, blind) | Test (800) | M4 per record | x86 per record |
|---|---|---|---|---|---|
| **Qwen3-0.6B + LoRA, catalogue, data v2** (GGUF Q8_0, `best`) | **0.950** | 0.931 | 0.960 | 0.49 s | – |
| same, GGUF Q4_K_M (400 MB) | 0.940 | 0.921 | 0.951 | 0.51 s | – |
| Qwen3-1.7B + LoRA, catalogue, data v2 (GGUF Q8_0, `large`) | 0.943 | **0.949** | 0.965 | 1.17 s | – |
| UltraBERT-150M encoder + 9 heads (fp32, two-run mean; not shipped) | 0.938 | 0.927 | **0.982** | – | 0.27 sᵇ |
| **Ettin-150M encoder + 9 heads, ONNX int8 (`tiny`)** | 0.932 | 0.928 | 0.969 | **0.028 s**ᵃ | 0.32 s (4-bit file)ᵇ |
| Qwen2.5-0.5B + LoRA, catalogue (GGUF Q8_0, `fast`) | 0.931 | 0.938 | – | 0.32 s | 1.80 s |
| Qwen2.5-0.5B pruned, catalogue-free (GGUF Q8_0, `small`) | 0.915 | 0.914 | 0.951 | 0.083 sᵃ | – |
| Qwen3-0.6B + LoRA, branch layout (fp32) | 0.929 | 0.921 | 0.933 | 1.42 s | 12.2 s |
| Jev hosted API, best of 3 prompt variants (GPU + network) | 0.918 | – | 0.882 | 0.12 s (API) | – |
| Jev hosted API, questions as-is | 0.869 | – | 0.817 | 0.17 s (API) | – |
| TF-IDF + logistic regression | 0.797 | 0.806 | 0.852 | 0.0008 s | 0.009 s |
| Qwen3-0.6B untrained, logit readout | 0.632 | 0.646 | 0.570 | 1.49 s | 11.6 s |
| Qwen3-0.6B generating JSON (Q8_0) | 0.590 | – | – | 1.73 s | 6.9 s |

ᵃ Median over gold + test records (`results/latency_lite.jsonl`); the other M4 rows used 40 gold records.
ᵇ Kaggle CPU, Intel Xeon at 2.2 GHz, 4 threads (`results/int8diag/int8_x86.md`); the other x86 rows are a 4-vCPU
cloud VM.

**What the results say:**

* **Decision training is what makes a sub-1B model usable.** Untrained, the Qwen3-0.6B readout is at 0.63; trained,
  it is at 0.95. Asking the same model to *generate* the answers is slower and worse (0.59).
* **For a fixed set of questions, a small classifier is enough.** The Ettin-150M encoder matches the Qwen2.5-0.5B
  models on the hand-written sets. In int8 it is a 151 MB file that answers in about 30 ms and under 0.5 GB of RAM,
  without PyTorch. It cannot take ad-hoc questions.
* **The prompt layout is the main CPU lever for decoders.** Caching the question catalogue's keys and values makes the
  same model about 3× faster on the M4 and about 6× faster on the x86 VM, and it was also more accurate than the
  branch layout.
* **GGUF quantization is nearly free.** Q8_0 matches fp32, and Q4_K_M costs about a point. Both cut memory about 3×,
  but neither makes decisions faster than torch fp32 on these CPUs.
* **ONNX int8 depends on the CPU.** It is lossless and fastest on Apple Silicon. On x86 CPUs without VNNI it
  collapses, and weight-only 4-bit is the lossless replacement (section 9).
* **Training cost came down about 6×.** Dropping the question list from the prompt (`nocat`) needs 6× less training
  compute at the same accuracy (12,000 records). The encoder trains at 0.20 s per record on the M4. See
  [`docs/TRAIN_ON_CPU.md`](docs/TRAIN_ON_CPU.md).
* **Smaller decoders lose more than encoders.** With the same data, SmolLM2-135M and Gemma 3 270M were 3–8 points
  below the 0.5B reference. Ettin encoders of 68M and 150M were 1–2 points short after 2 epochs, and level with it
  after 8 (`tiny`).
* **The hosted Jev API is zero-shot; our models are trained on the playbook.** That is why they beat it on this
  domain. Jev needed no training data at all. Given the playbook as prompt text, it rose from 0.869 to 0.918 on
  gold.
* **Held-out questions** (never trained on) are the decoders' weak spot after training:
  * `supplier_region` 0.47–0.85, depending on the model;
  * `mentions_new_date` 0.57–0.90;
  * `military_program` 0.81–0.94.

  Use the trained questions for anything that matters (`results/REPORT.md`).

Full tables: [`results/REPORT.md`](results/REPORT.md): every system, per question, held-out questions,
calibration (ECE), option-order robustness, latency and training runs. Each experiment round has its own table under
`results/<round>/` (section 7 and [`results/README.md`](results/README.md)).

## 7. What was tried, in order

Each round ran as a private Kaggle kernel (section 12). The full log has the setup, the decision rule and the outcome
of each round: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

| Round (kernel) | Question | Answer | Details |
|---|---|---|---|
| `baselines` | How good is an untrained small LLM at logit readout? | 0.51–0.70 gold; Jev-CPU-style zero-shot is not enough | `results/REPORT.md` |
| `train`, `train-cat`, `gold2-v1` | Does LoRA decision training work, and which layout? | 0.90–0.94 gold; the catalogue layout is better and ~3–6× faster on CPU | `results/REPORT.md` |
| `train-v2`, `train-extra` | More varied data (v2), a bigger model, CPU-sized runs | data v2 → 0.950 gold (`best`); Qwen3-1.7B most robust; 4,800 records → 0.88 | `results/REPORT.md`, `docs/TRAIN_ON_CPU.md` |
| `cpu-bench`, `cpu-train-bench` | What do inference and training cost on a regular x86 CPU? | 1.3–2.8 s per record; 20 s per training record (~3 days for the full recipe) | `docs/RUN_ON_CPU.md`, `docs/TRAIN_ON_CPU.md` |
| `ladder`, `ladder-floor` | How far can CPU training cost come down? | shared-prefix training 3.6× less compute, exact; catalogue-free 6× less at the same accuracy | `results/ladder/ladder.md`, `docs/CPU_SPEEDUP_RESEARCH.md` |
| `prune` | Can the 0.5B model be cut down? | 16 blocks + FFN 2560, full fine-tune: 276M params; gold ~3 points lower (two-run means), gold2 and test level → `small` | `results/prune/prune.md` |
| `bakeoff` | Smallest model that is still accurate? | small decoders lose 3–8 points; Ettin encoders (68M, 150M) only 1–2 at 2 epochs | `results/bakeoff/bakeoff.md` |
| `encoder` | Close the encoder's gap to the reference | train longer: 8 epochs → 0.932 / 0.928 / 0.969 → `tiny` | `results/encoder/encoder.md`, `docs/ENCODER.md` |
| `ultra` | Is the newer UltraBERT-150M better? | about +1 point at fp32, mostly on synthetic test; not adopted | `results/ultra/ultra.md` |
| `int8diag`, `int8fix` | Why is int8 wrong on Kaggle's CPUs, and what fixes it? | no-VNNI saturation; weight-only 4-bit is lossless on x86 | `results/int8diag/int8_x86.md` |

## 8. Reproduce everything

Everything below runs locally. GPU steps need CUDA; section 12 runs them on Kaggle instead. All splits and seeds are
fixed, so reruns land within the run-to-run noise above.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. data: v1 + v2 splits and both gold sets (deterministic)
bash scripts/make_data.sh

# 2. baselines
python -m eval.classical                                                    # TF-IDF / MiniLM + logistic regression
python -m eval.evaluate --model Qwen/Qwen3-0.6B --sets gold,test --out results/zs-qwen3-0.6b   # untrained readout
python -m eval.generative --gguf models/base/Qwen3-0.6B-Q8_0.gguf --out results/gen-qwen3-0.6b   # generate JSON
python -m eval.jev --variant plain --out results/jev-plain          # hosted Jev; needs jev_api_key in .env;
                                                                    # every response is cached in the --out folder
```

**Decoder models (Qwen + LoRA):**

```bash
# 3. decision training (GPU: ~1.5 h for Qwen3-0.6B on a T4; CPU: docs/TRAIN_ON_CPU.md)
python -m train.train --model Qwen/Qwen3-0.6B --layout catalogue --data data/processed_v2 --epochs 2 \
    --out runs/qwen3-0.6b-cat-v2

# 4. evaluation; fits per-type temperatures on val
python -m eval.evaluate --model Qwen/Qwen3-0.6B --adapter runs/qwen3-0.6b-cat-v2/best --layout catalogue \
    --data data/processed_v2 --sets val,test,test_v2,gold,gold2,injection --out results/ft-qwen3-0.6b-cat-v2

# 5. export for CPU: merge LoRA, GGUF Q8_0 + Q4_K_M (builds llama.cpp's quantizer on first use)
python -m aerojev.export --model Qwen/Qwen3-0.6B --adapter runs/qwen3-0.6b-cat-v2/best --out models \
    --name aerojev-qwen3-0.6b-cat-v2
```

The catalogue-free recipe (`--layout nocat`), pruning (`scripts/prune.py`) and the shared-prefix training options
are explained in `notebooks/aerojev_qwen2.5-0.5b_nocat.md` and `docs/CPU_SPEEDUP_RESEARCH.md`. The exact command for
every run in this report is in `kaggle/make_kernel.py`.

**Encoder model (`tiny`):**

```bash
# 3e. training (GPU: ~30 min on a T4; CPU: add --device cpu, 4.5 h on an M4)
python -m train.train_encoder --model jhu-clsp/ettin-encoder-150m --epochs 8 \
    --out runs/ettin-150m --results results/ft-ettin-150m

# 4e. export: int8 ONNX (ARM) + 4-bit ONNX (x86) + tokenizer + decider.json with the fitted temperatures
python scripts/export_encoder.py --model runs/ettin-150m/model --out models/aerojev-ettin-150m \
    --temps results/ft-ettin-150m/metrics.json --int4

# 5e. accuracy and speed of the exported files, on this CPU
python -m eval.evaluate --onnx models/aerojev-ettin-150m --data data/processed_v2 \
    --sets val,test,gold,gold2 --out results/onnx-ettin-150m
python scripts/bench_lite.py --model models/aerojev-ettin-150m --threads 1 4
```

**Latency, reports, serving:**

```bash
python -m eval.latency_suite --out results/latency_local.jsonl     # every local system, same protocol (idle machine)
python scripts/report.py                                           # results/REPORT.md + results/summary.json
python scripts/ladder_report.py --levels encoder --root kaggle/outputs/encoder/out kaggle/outputs/bakeoff/out \
    kaggle/outputs/prune/out kaggle/outputs/ladder/out --out results/encoder   # one round's table
python scripts/export_dashboard.py                                 # CSVs for results/dashboard/index.html
python server.py --preset best
```

**Notebooks** (each has a Markdown copy next to it for reading on GitHub):

| Notebook | What it covers | Regenerate |
|---|---|---|
| `notebooks/aerojev_qwen2.5-0.5b_lora.ipynb` | the whole decoder pipeline for one base model: data step by step, prompt layout and readout, tree packing, baselines, training, evaluation, calibration, GGUF export, CPU latency, usage. An executed copy is in `notebooks/executed/` | `python scripts/build_notebook.py` |
| `notebooks/aerojev_qwen2.5-0.5b_nocat.ipynb` | the catalogue-free recipe: no question list in the prompt, all 9 questions per record, 8 records per shared prefix; ~9 min on a T4 | `python scripts/build_notebook_nocat.py` |
| `notebooks/aerojev_ettin-150m_encoder.ipynb` | the `tiny` model: train, evaluate, export to ONNX, serve without PyTorch | `python scripts/build_notebook_encoder.py` |

On a CUDA GPU each notebook runs the full recipe. On a CPU it switches to FAST mode: the same steps on a few dozen
records. Overrides:
* `AEROJEV_NB_MODEL`: another base model.
* `AEROJEV_NB_FAST=0|1`: force full or FAST mode.
* `AEROJEV_NB_EXPORT=0`: skip the llama.cpp build.
* `AEROJEV_ROOT`: the repo location, if Jupyter starts elsewhere.

## 9. CPU deployment notes

The full guide is [`docs/RUN_ON_CPU.md`](docs/RUN_ON_CPU.md). The rules that matter most:

* **Run one long-lived process.** Loading costs 0.3–3.5 s; a decision costs milliseconds to a few hundred.
* **Use as many threads as performance cores** (4 on an M4). More threads did not help.
* **Keep records short.** Time grows linearly with record length, and long quoted reply chains also cost accuracy.
  Strip footers and quoted replies.
* **Ask only the questions you need** (decoders). Most of the time goes into reading the record.
* **Use the confidence.** With the fitted temperatures, `best`'s decisions with confidence ≥ 0.9 were right 98.5% of
  the time. Send the rest to an analyst.
* **On Apple Silicon, build llama.cpp without Metal** (`requirements.txt` has the command) to get honest CPU-only
  numbers.

**The encoder's weight files, by CPU** (`aerojev.lite` picks automatically; override with `--weights`):

| CPU | File | Why | Measured |
|---|---|---|---|
| Apple Silicon / ARM | `model.int8.onnx` (dynamic int8, 151 MB) | lossless and fastest there | M4: 28 ms (4 threads), 0.40–0.48 GB |
| x86 without VNNI (Intel before Ice Lake / Cascade Lake, AMD before Zen 4) | `model.int4.onnx` (4-bit weights in blocks of 32, int8 compute per block, 109 MB) | int8's u8×s8 kernels saturate there: gold 0.52 | 2.2 GHz Xeon: 320 ms (4 threads), 0.33 GB, lossless |
| x86, memory not tight | `model.onnx` (fp32, `--keep-fp32`, 597 MB) | lossless, about 1.2× faster than 4-bit there | 258 ms, 0.96 GB |
| x86 with VNNI | int8 is likely fine (VNNI accumulates in int32, with no int16 step) | not measured | – |

Diagnosis and every recipe tried: [`results/int8diag/int8_x86.md`](results/int8diag/int8_x86.md).

## 10. Repository layout

```
aerojev/          the runtime package
  schema.py         the questions: ids, types, wording, options (9 trained + 3 held out)
  prompt.py         prompt layouts (branch / catalogue / nocat), tree packing, readout rendering
  engine.py         DecisionEngine (PyTorch), GGUFEngine (llama.cpp), EncoderEngine (PyTorch encoder checkpoint)
  lite.py           LiteEngine: the torch-free ONNX Runtime engine for encoder models; picks weights by CPU
  metrics.py        accuracy, macro-F1, balanced accuracy, ECE, temperature fitting, option-order consistency
  export.py         merge LoRA + GGUF export (Q8_0 / Q4_K_M / f16) via llama.cpp
  benchrec.py       one latency-record schema for every system
data/
  gen/              world.py (suppliers, parts, programs) · policy.py (facts + playbook) · render.py (7 formats,
                    split-aware phrase banks) · generate.py (splits)
  gold/             hand-written gold (80) and blind gold2 (48) records, and the script that labels them
  processed/        data v1 splits (jsonl) · processed_v2/: data v2 splits
train/            train.py (LoRA / full decision training for decoders) · train_encoder.py (encoder + heads)
eval/             evaluate.py (any model on any split, temperatures, per-question metrics) · classical.py ·
                  generative.py · jev.py (hosted Jev, cached) · bench_cpu.py · bench_length.py ·
                  latency_suite.py · analyze.py (error analysis)
scripts/          make_data.sh · report.py · ladder_report.py · export_dashboard.py · build_notebook*.py ·
                  prune.py (structured pruning) · probe.py (frozen-model probes) · bench_train_cpu.py ·
                  export_encoder.py (ONNX int8 / 4-bit) · int8_variants.py (quantization recipes) · bench_lite.py
general/          general (domain-agnostic) decision data: domain specs, renderer, counterfactual pairs, Jev-format
                  inputs, NIM client (general/README.md)
tests/            tree packing, catalogue cache, shared-prefix gradients, GGUF parity, general data
notebooks/        three end-to-end notebooks + Markdown copies; executed/ holds an executed run
docs/             guides and research notes (section 16)
results/          every metric, prediction and latency file behind the docs (results/README.md)
kaggle/           how the GPU and x86 runs were executed: make_kernel.py, runner.py, kernels/, outputs/
                  (kaggle/README.md)
ui/               index.html (test page) · examples.py (8 example records)
server.py         stdlib HTTP API + test page; presets tiny / small / fast / best / large
requirements.txt  full install · requirements-lite.txt: the torch-free runtime only
```

## 11. Weights and where they live

Model files are **not in git** (`.gitignore`: `models/`, `*.gguf`, `*.safetensors`). GGUF files are 0.3–1.8 GB and
LoRA adapters about 35–40 MB each. The server presets expect these in `models/`:

| Preset | Files | How to make them |
|---|---|---|
| `tiny` | `aerojev-ettin-150m/` (`model.int8.onnx`, `tokenizer.json`, `decider.json`; add `model.int4.onnx` for x86) | steps 3e–4e of section 8 |
| `small` | `aerojev-qwen2.5-0.5b-nocat-d16w-full-q8_0.gguf`, `temps-qwen2.5-0.5b-nocat-d16w-full.json` | the `prune` kernel (`kaggle/make_kernel.py`) |
| `fast` | `aerojev-qwen2.5-0.5b-cat-q8_0.gguf`, `temps-qwen2.5-0.5b-cat.json` | steps 3–5 with `Qwen/Qwen2.5-0.5B-Instruct` |
| `best` | `aerojev-qwen3-0.6b-cat-v2-q8_0.gguf`, `temps-qwen3-0.6b-cat-v2.json` | steps 3–5 as written |
| `large` | `aerojev-qwen3-1.7b-cat-v2-q8_0.gguf`, `temps-qwen3-1.7b-cat-v2.json` | steps 3–5 with `Qwen/Qwen3-1.7B --epochs 1` |

A temperature file is the `temperatures` entry that `eval.evaluate` writes to `metrics.json`, for example
`{"choice": 1.3, "score": 1.0, "yesno": 1.0}`. The encoder keeps its temperatures inside `decider.json`. Tokenizers
are fetched from Hugging Face on first use; after that, `HF_HUB_OFFLINE=1` keeps everything offline.

The maintainer's trained files live in the private Kaggle kernels' outputs (section 12). The shipped `tiny` model's
x86 file can be fetched without retraining:

```bash
kaggle kernels output sparshnagpal/aerojev-int8fix -p /tmp/x86 --file-pattern 'onnx/e150-ep8/v-mnb4-a4/.*'
cp /tmp/x86/onnx/e150-ep8/v-mnb4-a4/model.int8.onnx models/aerojev-ettin-150m/model.int4.onnx   # the variant's file name is int8
```

## 12. Running the heavy steps on Kaggle

The development machine has no CUDA GPU and little free disk, so every GPU run and every x86 measurement here ran as
a **private Kaggle kernel**. `kaggle/` holds that tooling. None of the project code depends on it.

* `kaggle/make_kernel.py bundle` packs code + data into a private dataset (`aerojev-bundle`).
* `kaggle/make_kernel.py kernel <name>` writes `kaggle/kernels/<name>/`. Each kernel is `kaggle/runner.py` with a
  job queue per GPU filled in.
* `kaggle kernels push -p kaggle/kernels/<name>` runs it. Model weights are downloaded inside Kaggle, so only
  metrics, predictions and logs come back (`kaggle/outputs/<name>/`).

[`kaggle/README.md`](kaggle/README.md) covers the job format, every kernel, how to download outputs selectively, and
the gotchas.

## 13. Tests

The tests are plain scripts that check exactness claims (each prints its largest difference and asserts a bound):

```bash
python tests/test_tree_pack.py           # tree-packed forward == each question run on its own
python tests/test_catalogue_cache.py     # cached catalogue KV + record == uncached packed forward
python tests/test_shared_prefix.py       # shared-prefix training gradients == per-record gradients
python tests/test_gguf_parity.py models/base/Qwen3-0.6B-Q8_0.gguf Qwen/Qwen3-0.6B   # llama.cpp == HF readout
python tests/test_general_data.py        # general data: valid specs, reproducible builds, labels follow the rules
```

`scripts/export_encoder.py` also checks the ONNX graph against PyTorch on real records of several lengths before it
quantizes anything (largest logit difference must be under 1e-3).

## 14. Next: general decision models

Everything above answers **nine fixed aerospace questions**. The next step is a **general** small decision model.
It should take any record, any question and any options at request time, as Jev does, and be benchmarked against
Jev on domains it never saw. The plan is in
[`docs/GENERAL_DECISION_PLAN.md`](docs/GENERAL_DECISION_PLAN.md). It covers:

* two read-outs piloted on Ettin-150M and UltraBERT-150M;
* generated "decision worlds" with rule-derived labels and counterfactual rule pairs;
* reuse of existing published Jev benchmarks, so that few new Jev calls are needed;
* Kaggle-only training;
* a go/no-go gate after the pilot.

**Phase 0 has started.** The data tooling lives in [`general/`](general/README.md): domain specs with rule
variants, a renderer, counterfactual pairs, the Jev-format input contract, build checks and a NIM client. Three of
the twenty pilot domains are written, and the whole pipeline dry-runs on a CPU in seconds.

## 15. Limitations

* **Synthetic training data.** Labels come from a written playbook and records from templates, so the models learn
  *that* playbook. The hand-written gold sets are the check on realism, and they are small (80 + 48 records).
* **Fixed questions for `tiny` and `small`.** They refuse any other question. Adding a question or option means
  retraining.
* **Held-out questions are weak** after training (section 6). Trained questions are reliable; ad-hoc ones are a
  convenience.
* **Long records.** The models were trained on 50–300-token records. Records over 2,048 tokens are trimmed, and long
  noisy ones lose accuracy.
* **Untested hardware:** the encoder's 4-bit file on Apple Silicon, and int8 on x86 CPUs with VNNI.
* **English only.**

## 16. Documentation index

| Document | What it covers |
|---|---|
| [`docs/RUN_ON_CPU.md`](docs/RUN_ON_CPU.md) | inference on a CPU: recommended setup, install, measurements, tuning, troubleshooting |
| [`docs/TRAIN_ON_CPU.md`](docs/TRAIN_ON_CPU.md) | training without a GPU: measured times, record budgets, what does not help |
| [`docs/ENCODER.md`](docs/ENCODER.md) | the `tiny` model: why an encoder, what closed the gap, quantization (ARM and x86), UltraBERT |
| [`docs/CPU_SPEEDUP_RESEARCH.md`](docs/CPU_SPEEDUP_RESEARCH.md) | where CPU training time goes and the ladder of speed-ups, with measured outcomes |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | every experiment round: question, setup, decision rule, outcome, files |
| [`docs/GENERAL_DECISION_PLAN.md`](docs/GENERAL_DECISION_PLAN.md) | research and plan for general (domain-agnostic) decision models |
| [`results/README.md`](results/README.md) | what every results file is and its format |
| [`kaggle/README.md`](kaggle/README.md) | the Kaggle tooling and every kernel |
| `results/REPORT.md` | the generated cross-system report |

## 17. Base models and licences

| Model | Used for | Licence |
|---|---|---|
| Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-0.8B | decoder models and baselines | Apache-2.0 |
| Ettin encoders (32M / 68M / 150M, `jhu-clsp`) | the `tiny` model | MIT |
| UltraBERT-150M (`dmis-lab`) | encoder comparison | Apache-2.0 |
| SmolLM2-135M | bake-off | Apache-2.0 |
| Gemma 3 270M | bake-off only (not shipped) | Gemma terms |
| all-MiniLM-L6-v2 | classical baseline | Apache-2.0 |

Check each model's card before redistributing trained weights. All data in this repository is synthetic or written
by hand, and every name in it is fictional.
