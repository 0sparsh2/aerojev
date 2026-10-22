#!/usr/bin/env bash
# Build every dataset used by the project, from the generator and the hand-written gold records.
#
#   bash scripts/make_data.sh
#
# data/processed/     data v1: train 6000 / val 500 / test 800 / injection 200, plus gold and gold2
# data/processed_v2/  data v2: train 10000 / val 500 / test_v2 800 (extra phrase banks, 6% injections),
#                     plus copies of the fixed evaluation files (test, injection, gold, gold2) so every
#                     model is scored on identical records
# Generation is deterministic (fixed seeds), so re-running reproduces the same files.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m data.gen.generate --out data/processed                  # v1 splits
python -m data.gold.build_gold                                     # gold.jsonl + gold2.jsonl (playbook labels)

python -m data.gen.generate --v2 --n-train 10000 --out data/processed_v2 --only train,val,test
mv data/processed_v2/test.jsonl data/processed_v2/test_v2.jsonl    # v2's own held-out-template test split
cp data/processed/{test,injection,gold,gold2}.jsonl data/processed_v2/

wc -l data/processed/*.jsonl data/processed_v2/*.jsonl
