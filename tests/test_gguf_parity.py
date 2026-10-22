"""GGUF (llama.cpp, CPU) readout must agree with the HF fp32 readout.

  python tests/test_gguf_parity.py models/base/Qwen3-0.6B-Q8_0.gguf Qwen/Qwen3-0.6B [adapter]
"""

import json
import sys
import time

import torch

from aerojev import schema
from aerojev.engine import DecisionEngine, GGUFEngine
from aerojev.prompt import materialize

gguf, base = sys.argv[1], sys.argv[2]
adapter = sys.argv[3] if len(sys.argv) > 3 else None
rows = [json.loads(l) for l in open("data/processed/gold.jsonl")][:8]
qds = [materialize(q)[0] for q in schema.CATALOGUE]

hf = DecisionEngine(base, adapter=adapter)
gg = GGUFEngine(gguf, base, n_threads=4)
agree, n, maxdiff = 0, 0, 0.0
t_hf = t_gg = 0.0
for r in rows:
    t0 = time.time(); a = hf.readout_logits(r["state"], qds); t_hf += time.time() - t0
    t0 = time.time(); b = gg.readout_logits(r["state"], qds); t_gg += time.time() - t0
    for x, y in zip(a, b):
        px, py = torch.softmax(x, -1), torch.softmax(y, -1)
        maxdiff = max(maxdiff, (px - py).abs().max().item())
        agree += int(px.argmax() == py.argmax())
        n += 1
print(f"argmax agreement {agree}/{n}  max prob diff {maxdiff:.3f}")
print(f"latency per record (12 questions): HF fp32 {t_hf / len(rows):.2f}s   GGUF {t_gg / len(rows):.2f}s")
