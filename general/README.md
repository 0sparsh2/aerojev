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
| `nim.py` | NVIDIA NIM client (OpenAI-compatible) with a response cache, pacing and retries; the key comes from `NVIDIA_API_KEY` |
| `domains/` | one module per domain; `DOMAINS` finds them all |

```bash
python -m general.build --check-only                         # validate every domain spec
python -m general.build --records 600 --out /tmp/general-dry # template-only build plus all checks (~2 s)
python tests/test_general_data.py                            # reproducibility, labels, pairs, contract, NIM cache
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
* the NIM client;
* 3 of the pilot's 20 domains: `it_incident`, `ap_invoice`, `retail_returns`;
* tests.

Next, in plan order:
1. 12 more training domains and 5 held-out ones.
2. NIM-written phrase banks and prose records, run on a Kaggle CPU kernel. This needs a NIM key as a Kaggle Secret,
   and a check that the generator is live.
3. Loaders for the reused Jev benchmarks: system-one-benchmark and elcronos.
4. A public-dataset converter.
5. Checkpoint and resume in `train/train_encoder.py`.
6. Confirm the Jev Decision Index submission rules.
