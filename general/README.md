# `general/`: data and inputs for general decision models

Phase 0 of [`docs/GENERAL_DECISION_PLAN.md`](../docs/GENERAL_DECISION_PLAN.md). The aim is training and test data for
small decision models that take **any record, any question and any options**, with the same input contract as Jev.

| File | What it does |
|---|---|
| `spec.py` | the domain-spec format: `Fact` (sampled, steered off rule thresholds), `Question` (choice / score / noul), `Rule` (rule text + label function; several per question for counterfactuals), `Domain` (+ `validate()`) |
| `render.py` | facts → a record in one of 7 formats: email, ticket, chat, log, form (JSON), report, and test-only call notes. Hash-split phrase banks, as in `data/gen/` |
| `compose.py` | records → examples: rule text in the instructions, counterfactual pairs, option shuffling / opaque keys / keys only / subsets, paraphrased questions |
| `contract.py` | one example → the Jev question object, arm A input (`[OPT]` markers) and arm B input (lettered options + `[MASK]`) |
| `build.py` | CLI: validate specs, build train / val / test, then run the checks (needed numbers present in text, bag-of-words shortcut check, 8-gram leakage) |
| `nim.py` | NVIDIA NIM client (OpenAI-compatible) with a response cache, pacing and retries; keys via `secret()`: environment, Kaggle Secret, or the git-ignored `.env` |
| `gen_banks.py` | grows every domain's wording through NIM: generate → rule checks → a judge call per bank; `build.py --banks` uses the result |
| `gen_prose.py` | whole records written by the model from a fact sheet, kept only if every number appears verbatim, a judge reads every other fact back correctly, and no decision is stated; `build.py --prose` adds them |
| `domains/` | one module per domain; `DOMAINS` finds them all |

```bash
python -m general.build --check-only                         # validate every domain spec
python -m general.build --records 600 --out /tmp/general-dry # template-only build plus all checks (~3 s)
python tests/test_general_data.py                            # reproducibility, labels, pairs, contract, NIM cache
```

## Domains

22 domains, all fictional:
* **15 training domains** fill `train`, `val` and `test`.
* **3 dev-unseen domains** fill `dev_unseen`. They are for choices made during the project.
* **4 test-unseen domains** fill `test_unseen`. They are scored once, at the end.

Aerospace is the fifth test-unseen domain and is scored on its own data. Held-out domains (`Domain.heldout`) never
reach a training file. They use all of their wording and every format, call notes included.

| Domain | Split | Setting | Facts | Questions (type) | Rule variants |
|---|---|---|---|---|---|
| `ap_invoice` | training | Accounts payable invoice exceptions | 6 | action (choice), variance_band (score), fraud_review (noul), needs_approval (noul) | 4 |
| `appointment_booking` | training | Outpatient clinic appointment booking | 6 | booking (choice), slot_length (score), reminder_call (noul) | 2 |
| `cloud_cost` | training | Cloud spend anomaly alerts | 6 | action (choice), severity (score), finance_review (noul) | 3 |
| `content_moderation` | training | Community content moderation queue | 6 | action (choice), severity (score), notify_lead (noul) | 3 |
| `expense_report` | training | Employee expense claims | 6 | decision (choice), audit (noul) | 2 |
| `hr_leave` | training | Employee leave requests | 7 | decision (choice), second_signoff (noul), staffing_risk (score) | 3 |
| `insurance_claim` | training | Auto insurance claim intake and triage | 7 | route (choice), severity (score), siu_referral (noul) | 2 |
| `it_incident` | training | IT service desk incident triage | 7 | priority (score), route_to (choice), sla_breached (noul), notify_ciso (noul) | 3 |
| `kyc_onboarding` | training | Know-your-customer review at account opening | 7 | decision (choice), risk_rating (score), senior_signoff (noul) | 2 |
| `loan_application` | training | Personal loan application pre-screening | 7 | decision (choice), risk_tier (score), verify_income (noul) | 3 |
| `maintenance_request` | training | Building maintenance work orders | 6 | priority (score), trade (choice), root_cause_review (noul) | 2 |
| `purchase_request` | training | Internal purchase requests | 6 | approver (score), next_step (choice), competitive_bid (noul) | 3 |
| `retail_returns` | training | Retail return and refund requests | 6 | resolution (choice), abuse_flag (noul), handling_priority (score) | 2 |
| `shipment_exception` | training | Parcel delivery exceptions | 6 | action (choice), compensation (score), carrier_claim (noul) | 3 |
| `subscription_retention` | training | Subscription cancellation requests | 6 | offer (choice), churn_risk (score), csm_alert (noul) | 3 |
| `coursework_extension` | dev-unseen | University coursework extension requests | 6 | decision (choice), notify_tutor (noul) | 2 |
| `hotel_front_desk` | dev-unseen | Hotel check-in and room assignment | 6 | room_action (choice), compensation (score), manager_welcome (noul) | 2 |
| `power_outage` | dev-unseen | Electric utility outage dispatch | 7 | priority (score), dispatch (choice), regulator_report (noul) | 2 |
| `food_inspection` | test-unseen | Food safety inspections | 7 | outcome (choice), risk_band (score), warning_letter (noul) | 3 |
| `grant_screening` | test-unseen | Charitable grant application screening | 7 | decision (choice), review_level (score), site_visit (noul) | 3 |
| `pharmacy_refill` | test-unseen | Pharmacy refill requests | 6 | action (choice), counselling (noul) | 2 |
| `release_gate` | test-unseen | Software release go / no-go | 7 | decision (choice), blast_radius (score), notify_support (noul) | 2 |

**What the rules exercise:**
* **Threshold ladders:** approval limits, severity bands, priorities.
* **Ordered "otherwise" chains,** where the first matching condition wins.
* **Comparisons between two facts:** days requested against the balance; a grant request against the budget.
* **Lookups:** a trade by fault type; the earliest refill day by supply length.
* **Rules that read another question's answer:** dispatch uses the outage priority; compensation uses the room
  decision.

Each question reads only some of a record's facts, and three domains add facts no rule reads at all: a loan's
purpose, an after-hours flag and a venue type. A model has to pick out the facts that matter.

**Variants change the policy** in different ways: a threshold, an extra condition, a removed branch, or a level bump
for one group. Each variant changes 6–50% of labels.

## Generated wording (NIM)

**The generator is `nvidia/nemotron-3.5-lightning-30b-a3b`.**
* **Licence:** OpenMDW-1.1, which allows commercial use.
* **Speed:** about 1 s a call, with reasoning switched off.

Other models checked on 2026-11-20:
* **gpt-oss, Qwen and DeepSeek** are not offered to this key.
* **GLM-5.3-Flash** (MIT) timed out on the free tier.
* **Mistral Large 4** has a modified MIT licence with a hosting restriction.
* **Gemma and Meta models** are excluded by the plan.

**Each bank goes through three steps:**
1. **Generate:** n new items, given the spec's own items as examples.
2. **Rule checks:** allowed slots only, `{v}` exactly once where a value goes in, length, duplicates.
3. **Judge:** one call labels every candidate.

**The judge drops:**
* wordings that state the wrong value;
* any that state a decision instead of a fact (the first prompt version wrote "no receipt, so we have to deny
  this"; that leaks the label);
* situations that imply an outcome;
* paraphrases that change the question.

Generation runs on Kaggle (kernel `aerojev-general-gen`). Its key is the Kaggle Secret `NVIDIA_API_KEY`, attached
to that kernel, and every response is cached in the kernel's output.

**Never commit generated text (banks, prose).** This repository is public, and the NIM trial terms forbid
distributing generated content. It stays in the private kernel output, and later Kaggle jobs mount it from there
(`_kernel_sources`). `.gitignore` blocks `general/banks/`, `general/prose/` and NIM caches.

**Runs** (2026-11-20/21, 3 domains):

| | Run 1 | Run 2 (fixed judge and prompts) |
|---|---|---|
| New wordings kept | 445 | **736** |
| Empty banks | 22 | 2 |
| Prose records passing the fact check | 697 / 750 | 701 / 750 |
| Examples (train / val / test) | 20,857 / 2,635 / 2,594 | 20,866 / 2,613 / 2,623 |
| Counterfactual pairs (train) | 1,214 | 1,219 |
| Test 8-gram overlap with train | 0.048 | 0.051 (0.021 without banks or prose) |
| Shortcut flags / needed numbers missing | none / none | none / none |

**Run 1 found four bugs, all fixed:**
* **The judge read yes / no backwards.** It labelled "bank details unchanged" as true for *bank details changed*.
  Yes / no facts are now asked as plain claims.
* **Concrete values instead of the placeholder.** The model wrote real values in place of `{v}`, so every amount
  and tier template was rejected.
* **Answers wrapped in `{ }` were parsed as empty.**
* **New wording was unbalanced between yes and no.**

**Known weaknesses that remain:**
* **Loose paraphrases:** a few still drift, for example "logged for pattern analysis?" for the abuse question.
* **Echoed fact sheets:** prose sometimes repeats the fact sheet's wording.
* **Placeholders:** one email ended with "[Your Name]".

None of these changes a label, because labels come from the facts.

```bash
python -m general.gen_banks --domains all --n 12 --out banks/            # needs NVIDIA_API_KEY
python -m general.build --banks banks/ --records 2000 --out data/
```

## Adding a domain

Copy a module in `domains/` and keep the three-step order: **facts → rules → text**. Labels are never read from text.

1. **Facts.** Give each one a kind, a range or categories, and `avoid` bands around every rule threshold. Write at
   least 4 phrasings per value and register; a quarter of them become test-only.
2. **Questions.**
   * Each needs a `default` rule, written as code plus the same policy in plain words.
   * Add variants that change the policy (thresholds, extra conditions), so the model has to read the rule.
   * `needs` lists every fact the rules read.
   * Give 3–4 paraphrases.
3. **Validate.** Run `python -m general.build --check-only`. It fails a spec when:
   * an option is never chosen;
   * one option takes over 85% of labels;
   * a variant changes under 5% of labels;
   * sampling is not reproducible;
   * a phrase is missing.
4. **Build and look.** Build a few hundred records, then read a sample of each format. Check `stats.json`: no
   `facts_missing_from_text`, and no `shortcut` flags.
5. **Held-out domains** set `heldout="dev"` or `heldout="test"`. Write them for settings unlike any training domain,
   so they test transfer rather than recall.

All names in specs are fictional. Do not put confidential data in specs or prompts: generated wording goes through
NIM, whose trial terms let NVIDIA use prompts.

## Status

Built:
* the spec format, renderer, composer, input contract and build checks;
* the NIM client, and the NIM bank generator with rule checks and a judge;
* all 22 pilot domains (table above): 15 training, 3 dev-unseen and 4 test-unseen;
* tests.

Next, in plan order:
1. Generate wording and prose for all 22 domains on Kaggle. The 3 first domains are done (runs above).
2. Loaders for the reused Jev benchmarks: system-one-benchmark and elcronos.
3. A public-dataset converter.
4. Checkpoint and resume in `train/train_encoder.py`.
5. Confirm the Jev Decision Index submission rules.
