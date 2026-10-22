"""Catalogue layout: cached static KV + record must equal the uncached packed forward."""

import json

import torch

from aerojev import schema
from aerojev.engine import DecisionEngine
from aerojev.prompt import materialize, pack

eng = DecisionEngine("Qwen/Qwen3-0.6B", layout="catalogue")
rows = [json.loads(l) for l in open("data/processed/gold.jsonl")][:3]
qds = [materialize(q)[0] for q in schema.CATALOGUE]
worst = 0.0
for r in rows:
    cached = eng.readout_logits(r["state"], qds)  # uses static KV cache, then crops it back
    p = pack(eng.tok, r["state"], qds, eng.think, layout="catalogue", catalogue=qds)
    with torch.no_grad():
        out = eng.model(input_ids=p["input_ids"][None], position_ids=p["position_ids"][None],
                        attention_mask=p["mask"][None, None], logits_to_keep=torch.tensor(p["readouts"])).logits[0]
    for i, (lab, c) in enumerate(zip(p["label_ids"], cached)):
        worst = max(worst, (out[i, lab] - c).abs().max().item())
print(f"static={eng.S} tokens, max logit diff cached vs uncached = {worst:.2e}")
assert worst < 1e-3
print("OK catalogue cache matches")
