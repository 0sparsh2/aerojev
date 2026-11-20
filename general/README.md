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
python -m general.build --records 600 --out /tmp/general-dry # template-only build plus all checks (~2 s)
python tests/test_general_data.py                            # reproducibility, labels, pairs, contract, NIM cache
```

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

Generation runs on Kaggle (kernel `aerojev-general-gen`). Its key is the Kaggle Secret `NVIDIA_API_KEY`, attached to
that kernel, and every response is cached in the kernel's output.

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

All names in specs are fictional. Do not put confidential data in specs or prompts: generated wording goes through
NIM, whose trial terms let NVIDIA use prompts.

## Status

Built:
* the spec format, renderer, composer, input contract and build checks;
* the NIM client, and the NIM bank generator with rule checks and a judge;
* 3 of the pilot's 20 domains: `it_incident`, `ap_invoice`, `retail_returns`;
* tests.

Next, in plan order:
1. 12 more training domains and 5 held-out ones.
2. A pilot-size run of banks and prose on Kaggle (kernel `general-gen`), then a review of samples. In a 6-record
   local smoke test, 5 prose records passed the check and 1 was dropped for stating a decision.
3. Loaders for the reused Jev benchmarks: system-one-benchmark and elcronos.
4. A public-dataset converter.
5. Checkpoint and resume in `train/train_encoder.py`.
6. Confirm the Jev Decision Index submission rules.
