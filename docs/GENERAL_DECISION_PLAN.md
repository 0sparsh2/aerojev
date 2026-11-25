# General decision models: research and plan

Status (2026-11-24): **Phase 0 under way** (`general/`, see `general/README.md`).

**Built and dry-run on CPU:**
* the domain-spec format with validation;
* the renderer (7 formats, hash-split wording);
* counterfactual pairs and option variation;
* the Jev / arm A / arm B input contract;
* build checks (facts present in text, shortcut check, 8-gram leakage);
* the NIM client with a cache;
* all 22 pilot domains: 15 training, 3 dev-unseen and 4 test-unseen. Aerospace is the fifth test-unseen domain.

A 600-record build of every domain takes about 3 s and passes every check (`tests/test_general_data.py`).

**First generation on Kaggle (2026-11-21, 3 domains, Nemotron 3.5 Lightning):**
* **Data:** 736 new wordings, 701 fact-checked prose records, and 26,100 examples with 1,535 counterfactual pairs.
* **Checks:** every one passes.
* **Details:** `general/README.md`.

**Run 3 (2026-11-24, all 22 domains):**
* **Data:** 4,767 new wordings, 3,454 of 4,100 prose records kept (84%), 86,700 training examples with 5,312
  counterfactual pairs, plus 2,887 dev-unseen and 4,076 test-unseen examples.
* **Checks:** every one passes.
* **How it ran:** the NIM calls ran from the Mac with the `.env` key, the text went to a private Kaggle dataset, and
  the keyless build ran on Kaggle.
* **Where it is kept:** generated text stays in the private kernel output. The repository is public, and the trial
  terms forbid distributing generated content.

Nothing has been trained yet. Every hour, size and cost below is an estimate
unless it cites a measurement from this repo or a published source.

Decisions taken so far:

* **The general model must be commercially usable.** So every training input (data and generated text) must allow
  commercial use.
* **Generation uses the NVIDIA NIM API with an open model.** See section 6.1 for a licensing condition this places on
  the production dataset.

## 1. Summary

The AeroJev encoder models answer exactly nine aerospace questions, each through its own output layer. A general
model has to take **any question and any options at run time**, as Jev does, and be tested on domains and questions
it never saw.

* **Model.** The record, the question and its options go into one input, and the model scores each option. Two
  read-outs are piloted on Ettin-150M and UltraBERT-150M:
  * a marker token per option, scored by a small shared layer (GLiClass-style);
  * the encoder's masked-word head predicting the option letter (ModernBERT-Large-Instruct-style).
* **Data.** Generated "decision worlds" in many domains, with labels computed from rules. Prose and phrase banks come
  from an open model through NVIDIA NIM. On top of that, commercially licensed public datasets rewritten as questions.
* **Benchmark: most Jev numbers already exist** (section 4.2), so we don't call Jev for them:
  * our aerospace results;
  * independent benchmarks that publish Jev's per-example predictions;
  * the Jev Decision Index leaderboard, which also scores unseen domains privately.

  Our own Jev calls drop from ~5,300 to at most ~1,300, for the one thing nobody has measured: whether a model
  follows changed rules.
* **There is now a field of open decision models** (section 4.3), including encoders of 395-437M parameters (Laya,
  Verdict-MQ, JevK5-Lite). Our aim is the **smallest CPU model**: 150M parameters, int8, well under 0.5 GB. These
  models are baselines.
* **Expected outcome.** A 150M encoder should do well on read-and-classify decisions, and on any domain after a short
  fine-tune. The Verdict-MQ paper reports that Jev beat every fine-tuned encoder zero-shot in its tests. Expect to
  trail Jev on unseen rule-heavy questions; the pilot measures by how much.
* **Cost.**
  * Kaggle GPU time: about 10-15 GPU-hours for the pilot and 25-30 for the full run.
  * Generation runs on Kaggle CPU kernels calling NIM, so it uses no GPU quota.
  * Jev calls: about 2M tokens.

## 2. What "general" means here

**Input contract: the same as Jev's**, so every system sees the same thing.

| Field | Content |
|---|---|
| `state` | the record: free text, or JSON rendered as text |
| `questions` | a map of question id to question |
| question `type` | `choice` (criteria: option key to description), `score` (criteria: ordered level descriptions), `noul` (yes/no; instructions only) |
| question `instructions` | the question text, optionally followed by rules ("Use our playbook: ...") |

**Output:** a probability per option, per question.

**Out of scope:** images (Jev's `box` primitive) and free-text answers.

**Two modes, both benchmarked:**

* **Zero-shot:** a domain and questions the model never saw.
* **Adapted:** the general model fine-tuned on 100-2,000 labelled records of the target domain, as we did for
  aerospace, but starting from the general model.

## 3. Where we start

| Asset | State | Use in this plan |
|---|---|---|
| `train/train_encoder.py` | fixed heads, one per question | new input format and scoring; loop and loss carry over |
| Decoder path (`train/train.py`, `--layout branch`) | already answers any question via label-token logits | third arm: a general Qwen3-0.6B |
| `data/gen/` | latent scenario, then labels from playbook rules, then templated text with held-out phrase banks | the pattern for generated domains |
| `eval/jev.py` | Jev-format requests, every response cached, our metrics | small rule-following run only (section 7) |
| `aerojev/metrics.py`, `permutation_consistency` | accuracy, macro-F1, ECE, option-order consistency | reused unchanged |
| Specialised Ettin-150M / UltraBERT-150M | aerospace gold 0.932 / 0.938-0.944 (8 epochs, 2 runs each, full precision) | ceiling for the aerospace tier |

## 4. Research

### 4.1 Approaches for the model

* **Jev** (TypeSafe, September 2026) takes a state and typed questions (`choice`, `score`, `noul`, `box`) and returns
  calibrated probabilities. Per-domain adaptation is not documented.
* **GLiClass** (Knowledgator, 2025): candidate labels are prefixed with a `<<LABEL>>` token and encoded together with
  the text in one pass.
  * **Accuracy:** 0.719 average F1 zero-shot over 14 datasets, against 0.682 for the best NLI cross-encoder.
  * **Speed:** almost flat as the number of labels grows.
  * **DeBERTa-v3 clearly beat ModernBERT as the backbone** (base 0.676 against 0.558). Ettin is ModernBERT-style,
    hence the DeBERTa control in this plan.
* **ModernBERT-Large-Instruct** ("It's All in The [MASK]", 2025) answers multiple-choice questions with the encoder's
  masked-word head. It is strong zero-shot for its size, provided the encoder was pretrained on recent, diverse data;
  Ettin and UltraBERT both qualify.
* **Verdict-MQ** ("Encode Once, Decide Many", arXiv 2610.20778) is a 395M encoder that reads a case once and answers
  all its questions. A structured attention mask keeps the record's encoding independent of the questions, keeps
  questions from influencing each other, and makes option order irrelevant. This is our option F, already
  demonstrated. It beat Laya on agreement and calibration, but Jev beat every fine-tuned encoder zero-shot in its
  tests. Code: `LEC-AI/verdict`.
* **"Releasing the Inner Jev"** (arXiv 2610.17205): LoRA on 4-31B decoders, reading letter logits, trained on
  **DecisionBank** (261,561 decisions).
  * **Result:** fine-tuned Gemma 4 31B scores 89.0 on their **OpenDecisions** set (2,352 questions), against Jev's
    80.1.
  * **Released:** data, models and code. Licences aren't stated on the abstract page; check before using.
* **JEVal** (arXiv 2610.03935): 11,257 instances from 36 datasets in 10 domains. Decision models are most
  competitive on "evidence-resolvable" decisions, and weak on specialist knowledge and on probability estimates.

### 4.2 Jev benchmarks that already exist (reuse instead of calling Jev)

| Source | What | Jev version | Reusable how |
|---|---|---|---|
| **This repo** (`results/jev-*`) | aerospace val (500), test (800), gold (80), injection (200); 3 variants (plain, playbook, playbook + export definition); held-out questions; gold with shuffled options | jev-1.14.0 | directly; same inputs as our models |
| **`yanng981/system-one-benchmark`** (MIT code) | SST-2, TREC, Banking77 (300 each, seed 42); MASSIVE in 8 languages (150 each); also Kev-0.8B, Laya, Von, GLiNER2.5-Decide | 1.13 | per-example predictions in `results/*.jsonl`; we run our models on the same samples and wording |
| **`elcronos/jev-vs-open-decision-models`** | emotion (2,000), tweet_topic (1,693), fin_topic (4,117), daily_dialog (7,740); frozen protocol; also PrismNLI-0.4B, Laya | 1.13 (dated snapshot) | raw predictions (`results/raw_predictions.parquet`) and API caches; prompts in `PROTOCOL.md` / `datasets_registry.py` |
| `Gaurav-Gosain/jev-sec-bench` | prompt injection (662), vulnerable-code pairs (200) | not stated | raw results committed |
| **Jev Decision Index** (Hugging Face Space, edition 0.4.1, 12 Nov 2026) | 37 public benchmarks plus private same-skill and **new-domain** tests; 301 rows | Jev 1.14: Full 60.24, public 60.7, ECE 0.080 | **submit our model**; reported (second-hand) as a pull request with a dataset link and hardware. Its private new-domain score is a ready-made unseen-domain test |
| JevBench (open-weights board) | v1.4; Jev 1.13 scores 63.29 | 1.13 | submission; eligibility needs public weights, a licence and a source link |
| OpenDecisions (from "Releasing the Inner Jev") | 2,352 questions | Jev 80.1 (their Capability metric) | run our models; licence to check |
| Vals.ai LegalBench | Jev last of 12 | not stated | headline only |

**Caveats on reuse:**

* **Version:** most third-party runs used Jev 1.13; the current version is 1.14. On the Decision Index the two are
  about 1 point apart (Full 59.22 against 60.24).
* **Sample size:** with 300-example samples, differences under about 3 points are noise.
* **Identical inputs:** we must send our models exactly the published question text and options. The repos contain
  them; Phase 0 verifies this.

### 4.3 Open decision models (baselines; our niche is the size)

| Model | Size | Type | Licence | Notes |
|---|---|---|---|---|
| JevK5-Lite | 437M | DeBERTa-v3-large encoder | Apache-2.0 | CPU-only; 71 ms p50 at 16 threads (easy items) |
| Laya | 421M encoder | typed-decision encoder | not stated | about 20-option budget |
| Verdict-MQ | 395M | encoder, encode once | code public; weights not stated | |
| PrismNLI-0.4B | 0.4B | NLI cross-encoder (from deberta-v3-large-zeroshot-v2.0) | not stated | |
| GLiNER2.5-Decide | not stated | GLiNER family | not stated | Index Full 10.14 |
| Kev-0.8B | 0.8B | not stated | not stated | strong on SST-2 / TREC / Banking77 |
| JevK5 (4B / 9B) | 4-9B | Qwen3.5 + LoRA | Apache-2.0 | Index Full 36.31 (v0.2.2) |
| *Ours: Ettin-150M / UltraBERT-150M general* | 150M | encoder | MIT / Apache-2.0 backbones | target: int8, under 0.5 GB, CPU |

## 5. Model design

### 5.1 Options

| | Design | Passes per question | Reads rules | Verdict |
|---|---|---|---|---|
| A | option markers: options in the input, one marker token each, shared scorer | 1 | yes | **pilot arm** |
| B | masked-word answer: lettered options, masked-word head predicts the letter | 1 | yes | **pilot arm** |
| C | NLI cross-encoder, one pass per option | one per option | partly | baseline only |
| D | embeddings: compare record and option vectors | 1 | no | rejected: cannot apply rules |
| E | small decoder (Qwen3-0.6B + LoRA, our branch layout) | 1 per record | yes | third arm |
| F | all questions of a record in one pass with a structured attention mask (as Verdict-MQ) | 1 per record | yes | phase 2 speed-up |

### 5.2 Input format (A and B)

```
[CLS] Question: <instructions incl. rules> [SEP]
      [OPT] <key>: <description>  [OPT] <key>: <description>  ...  [SEP]
      Record: <state> [SEP]
```

* **A:** the hidden state at each `[OPT]` (a new learned token), optionally concatenated with the mean of the record
  tokens, goes through a two-layer scorer that outputs one logit per option.
* **B:** `A) ... B) ... Answer: [MASK]`, read off the letter logits.
  * **Limit:** 26 options; more need grouped passes, as JevK5 does above 16.
  * **Letter bias:** training shuffles option order; `permutation_consistency` measures what bias remains.
* **Types:** `choice` is a softmax over the options; `score` uses the given order plus our ordinal penalty; `noul` is
  two options, yes and no.
* **Length:** rules and up to 10 options add about 300 tokens. Neither backbone is limited by this: Ettin takes 8k
  tokens, UltraBERT 32k.

### 5.3 CPU speed, estimated

* **Today:** 47 ms per record on the M4 with 1 thread, one pass of about 140 tokens (measured).
* **Arms A and B:** one pass per question. I'd estimate 0.3-0.5 s for 9 questions; not measured.
* **Option F:** perhaps 80-150 ms for the same; not measured.

## 6. Data

### 6.1 Generated decision worlds (main source)

Generalise `data/gen/` from one domain to many. A **domain spec** (a Python module) declares:

* **Facts:** typed fields with distributions, sampled away from decision thresholds.
* **Questions:** type, options with descriptions, a **label function** over the facts, and optional **rule text**
  stating that function in plain language.
* **Phrase banks and formats:** how each fact can be stated, and which document types the domain uses (email,
  ticket, chat, log line, form or JSON, report, call notes).

Labels come from facts, never from text, and every fact a label needs must appear in the text.

**What makes the data teach generality:**

1. **Rule-conditioned (counterfactual) pairs.** The same facts with two versions of a rule, each with its own label.
   This forces the model to read the rules.
2. **Question and option variation.** Paraphrases; renamed, shuffled and subsetted options; options with or without
   descriptions.
3. **Held-out material.** Held-out phrase banks, formats and question wordings for test, plus whole held-out domains.

**Generation: NVIDIA NIM API, open model, called from Kaggle CPU kernels.**

* **Endpoint:** NIM is OpenAI-compatible (`https://integrate.api.nvidia.com/v1`). The key goes in as a Kaggle Secret.
* **Two routes:**
  * **Phrase banks:** a few hundred phrasings per fact; minutes per domain. Our renderer combines them, as
    `render.py` does today.
  * **Prose:** about 30% of records written whole from a fact sheet, then checked automatically that every
    label-relevant fact is present (numbers and dates by pattern, the rest by a second model call). Records that fail
    are dropped.
* **Throughput:** the free tier is reported (not officially) at about 40 requests a minute per key. That is about
  57k requests a day, enough for the pilot (about 2k bank requests and 3k prose requests) in a few hours.
* **Domain specs:** written by hand for the pilot (15 training + 7 held-out domains). For phase 2, drafted as code through NIM, then
  checked automatically: every option reachable, labels balanced, a fixed seed reproduces the labels, every fact
  rendered. A sample is reviewed by hand.

**Licensing: two separate layers.**

* **The model's licence must allow its outputs to train a commercial model.** Use Apache-2.0 or MIT models (for
  example gpt-oss, Qwen, Mistral Apache-2.0 releases, DeepSeek MIT releases) or NVIDIA's open model licence for
  Nemotron, after checking its terms.
  * **Avoid Llama:** a model trained on its outputs must carry "Llama" in its name.
  * **Avoid Gemma:** models distilled from its outputs fall under the Gemma terms.
  * **Check availability:** which of these are live on NIM changes often; check with your key in Phase 0.
* **The NIM service terms (official PDF, NVIDIA API Trial Terms of Service, file dated 2025-10-07):**
  * §1.2 gives access "for limited trial purposes only and without use of the API Service or Generated Content in
    production".
  * §1.4: "Unless you purchase a Subscription ... you may only use the API Service for internal testing and
    evaluation purposes, not in production".
  * §4.2 forbids distributing Generated Content.
  * §4.12 forbids using it "to develop or improve products or services that compete with the API Service".
  * §6.3: you own the outputs, subject to the model's licence.
  * §3.3(iv): NVIDIA may use prompts and outputs to improve its products.

  **My reading, which is not legal advice:** the free trial is fine for the **pilot**, which is internal evaluation.
  A **dataset that trains a commercial model** should be generated under a NIM **paid subscription** (whose terms
  then apply; check them), or with the same open model through another provider, or self-hosted (for example vLLM on
  Kaggle). The pilot's generator code is the same either way: only the endpoint changes. Never send confidential
  data, because of §3.3(iv).

### 6.2 Public datasets turned into questions

* **Commercial-friendly licences only** (for example CC BY, MIT, Apache-2.0). The licence is recorded per source, as
  the `-c` zero-shot models do.
* **Candidates:** DecisionBank (261,561 decisions) is the obvious one if its licence allows commercial use. If it was
  generated by a model whose terms restrict training competitors, it can't be used for training.
* **Balance:** a cap per dataset and temperature sampling.
* **Leakage:** no benchmark task in training; duplicate text across the two is removed.

### 6.3 Shortcut checks

For each generated question, train a bag-of-words logistic regression. If it is near-perfect, the label leaks
through wording, and that question's phrasing needs more variety.

### 6.4 Splits

* **Training domains:** train, val and test (test with held-out banks and formats).
* **Dev-unseen domains (3):** for all choices during the project.
* **Test-unseen domains (5, including aerospace):** scored once at the end.

Aerospace is kept entirely out of general training.

## 7. Benchmark

### 7.1 Tiers

| Tier | Content | New Jev calls | Measures |
|---|---|---|---|
| A | aerospace val, test, gold, injection, 3 rule variants | **0** (exists, jev-1.14.0) | zero-shot on a real domain; adapted mode against our specialised models |
| B | our test-unseen generated domains, with rules and counterfactual rule pairs | **up to ~1,000**, only for the rule-following rate (about 2M tokens; elcronos paid $0.028 for 2,000 calls) | unseen domains; does the model follow changed rules |
| C | reused public benchmarks: system-one-benchmark (4 tasks, 8 languages for MASSIVE) and elcronos (4 tasks), same samples and wording | **0** (Jev 1.13 predictions published); optional ~300-call spot check with 1.14 | real text, many options, multilingual |
| D | Jev Decision Index submission (37 public benchmarks plus private same-skill and new-domain); JevBench; OpenDecisions | **0** | independent, private unseen-domain scoring next to Jev and every open model |

Tier D depends on the Index accepting a 150M CPU model. Submission requirements need confirming on the Space.

### 7.2 Systems

* **Ours:** general Ettin-150M and UltraBERT-150M (zero-shot and adapted); a DeBERTa-v3-base control; a general
  Qwen3-0.6B (decoder arm).
* **Open baselines:** JevK5-Lite, Laya, GLiNER2.5-Decide, PrismNLI-0.4B, an NLI zero-shot `-c` model, GLiClass. Where
  a reused benchmark already has their scores, we reuse them.
* **Jev:** reused results (1.13 and 1.14), plus the small tier B run.
* **Our specialised aerospace encoders:** tier A only, as the ceiling.

### 7.3 Metrics

* **Accuracy and macro-F1:** per type and per tier.
* **ECE:** raw, and with per-tier temperatures fitted on dev.
* **Option-order consistency.**
* **Rule-following rate:** on counterfactual pairs.
* **CPU latency and memory:** int8 on ARM and 4-bit weight-only on x86, the files `aerojev/lite.py` picks (section
  10).
* **Cost** per 1,000 decisions.

### 7.4 Fairness rules

* **Identical inputs** to every system; for reused benchmarks, the published wording.
* **No tuning on test:** choices are made on dev-unseen domains only.
* **Two runs** per trained system; gaps under about 2 points (3 on 300-sample sets) are ties.
* **Jev version recorded** for every number.

## 8. Training recipe

* **Loss:** cross-entropy over a question's options, plus the ordinal penalty on `score` questions.
* **Batching:** one example per (record, question), with temperature-0.5 sampling across domains and datasets.
* **Learning rates:** Ettin 5e-5, UltraBERT 3e-5 for the encoder; 1e-3 for new layers. Confirmed on dev-unseen.
* **Length:** inputs capped at 1,024 tokens for training.
* **Checkpoint and resume** in `train_encoder.py`, needed for Kaggle's 12 h sessions.
* **Adapted mode:** 100 / 500 / 2,000 labels of a target domain, starting from the general model and from the raw
  backbone, to measure how many labels the general model saves.

## 9. Kaggle pipeline

All steps are private Kaggle kernels built by `kaggle/make_kernel.py`. Only scores and final int8 models return to
the Mac.

| Kernel | Machine | Does | Output | Estimate |
|---|---|---|---|---|
| `gen-banks` | CPU, NIM API (key as a Kaggle Secret) | phrase banks per domain spec | dataset `aerojev-general-banks` | about 1 h |
| `gen-prose` | CPU, NIM API | prose records plus fact check | `aerojev-general-data` | 2-4 h for about 10k records |
| `gen-compose` | CPU | composed records, counterfactual pairs, variation, splits, shortcut checks | `aerojev-general-data` | under 1 h |
| `public-mix` | CPU | download, licence filter, convert, dedupe | `aerojev-general-data` | under 1 h |
| `general-pilot` | 2× T4 | 2 backbones × 2 arms, plus DeBERTa control and Qwen3-0.6B | scores, weights | 8-12 h |
| `general-full` | 2× T4 | winning arm, both backbones, 2 seeds | scores, weights | 25-30 h over 2-3 sessions |
| `general-adapt` | 2× T4 | label-budget runs | scores | 3-5 h |
| `general-bench` | CPU | open baselines, reused-benchmark runs, int8 export, latency, tier B Jev calls (key as a Kaggle Secret) | scores | 2-4 h |

**Kaggle limits:**

* **GPU quota:** about 30 GPU-hours a week.
* **Session length:** 12 hours.
* **Output:** 20 GB per kernel.

## 10. Phases and decision points

| Phase | What | GPU h | Ends with |
|---|---|---|---|
| 0. Groundwork | spec format and renderer; 15 + 7 pilot specs; public-mix converter; reused-benchmark loaders (verify prompts and samples); NIM client with caching; checkpoint resume; confirm Decision Index submission rules | 0 | dry run on CPU |
| 1. Pilot | generate pilot data (NIM trial = evaluation); train both arms on both backbones plus controls; tiers A and C, plus B on dev-unseen | 10-15 | **Gate 1** |
| 2. Scale | regenerate under a production-licensed endpoint; 40-60 domains; packing (option F) if worthwhile; winning arm × 2 backbones × 2 seeds | 25-30 | tiers A-C final; Index submission (tier D) |
| 3. Adaptation | label-budget curves on aerospace and two other domains | 3-5 | labels needed to reach Jev and our specialised model |
| 4. Ship | int8 ONNX (x86-safe recipe), `--preset general`, docs, notebook | under 1 | release |

**Gate 1, fixed before the pilot.** Continue to phase 2 if the best encoder arm is within 5 points of Jev
(plain-question variant) on aerospace gold (zero-shot), **or** on the mean of the reused tier C tasks.

* **If it fails both:** pivot to adapted mode, "general base plus a short fine-tune", where encoders are strong. The
  decoder arm becomes the zero-shot candidate.
* **Arm and backbone** are chosen on dev-unseen only.

**Quantization.** Resolved for x86 (`results/int8diag/int8_x86.md`):

* **The problem:** dynamic int8 breaks on x86 CPUs without VNNI.
* **The fix:** weight-only 4-bit with per-block int8 compute (`export_encoder.py --int4`) is lossless there.
* **What ships:** the int8 file for ARM and the 4-bit file for x86, picked by `aerojev/lite.py`.
* **Still to measure:** CPU latency for the general models, on both.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Encoder ignores rules | counterfactual pairs in training; rule-following rate |
| ModernBERT-style backbones weaker zero-shot | DeBERTa control; UltraBERT; decoder arm |
| Jev beats fine-tuned encoders zero-shot (Verdict-MQ finding) | Gate 1; adapted mode as a first-class result; size and CPU as our differentiator |
| Generated-text shortcuts | shortcut checks, prose route, held-out banks, formats and domains |
| Licences of training data | commercial-friendly only, recorded per source; production data from a production-licensed endpoint |
| NIM free-tier limits change | cache every response; resumable generation; fall back to vLLM on Kaggle with the same model |
| Reused Jev results are 1.13 | stated with every comparison; optional 1.14 spot check |
| Public text seen in pretraining | conclusions rest on tiers A, B and D; tier C reported with this caveat |
| Kaggle 12 h limit | checkpoint and resume |

## 12. Decisions

**Taken:**

* Commercial use: yes.
* Generator: NVIDIA NIM API with an open model.

* **Generator** (2026-11-20): `nvidia/nemotron-3.5-lightning-30b-a3b`, under OpenMDW-1.1, which allows commercial
  use. gpt-oss is not offered to this key, and GLM-5.3-Flash (MIT) timed out on the free tier.
* **Keys:** Kaggle Secrets, attached per kernel. Both keys were verified on 2026-11-20: NIM lists 33 models, and Jev
  answers as jev-1.14.0.

**Still needed:**

1. **Production generation:** a NIM paid subscription, another provider for the same open model, or self-hosting on
   Kaggle. The free trial covers the pilot only, per the terms above.
2. **Tier B Jev run:** about 1,000 calls, for the rule-following test only. Or skip it and rely on tiers A, C and D.
3. **Domains:** any real use case you want in the held-out set.

## Sources

* Jev: [TypeSafe quick start](https://docs.typesafe.ai/introduction/quickstart),
  [MarkTechPost](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/)
* Jev benchmarks: [systemonemodels.org overview](https://systemonemodels.org/guides/jev-benchmarks/),
  [system-one-benchmark](https://github.com/yanng981/system-one-benchmark),
  [jev-vs-open-decision-models](https://github.com/elcronos/jev-vs-open-decision-models),
  [Jev Decision Index coverage](https://aiweekly.co/alerts/jev-decision-index-ranks-30-open-weight-decision-models),
  [Vals.ai on Jev](https://www.vals.ai/blogs/independent-evaluation-of-jev),
  [arXiv 2609.37647](https://arxiv.org/html/2609.37647v1)
* Open decision models and recipes: [Verdict-MQ, arXiv 2610.20778](https://arxiv.org/pdf/2610.20778),
  [Releasing the Inner Jev, arXiv 2610.17205](https://arxiv.org/pdf/2610.17205),
  [JEVal, arXiv 2610.03935](https://arxiv.org/pdf/2610.03935), [JevK5](https://github.com/allebee/jevk5),
  [awesome-decision-models](https://github.com/AnotiaWang/awesome-decision-models)
* GLiClass: [arXiv 2508.07662](https://arxiv.org/html/2508.07662v1);
  ModernBERT-Large-Instruct: [arXiv 2502.03793](https://arxiv.org/abs/2502.03793v2)
* NVIDIA NIM: [API Trial Terms of Service (PDF)](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf),
  [free-tier limits (unofficial)](https://mvalentsev.github.io/awesome-free-ai-coding/providers/nvidia-nim/)
* Benchmarks: [LegalBench](https://huggingface.co/datasets/nguha/legalbench), [RAFT](https://Huggingface.co/papers/2109.14076)
