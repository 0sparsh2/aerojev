"""Tree-packed forward must equal running each (prefix + branch) on its own."""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aerojev import schema
from aerojev.prompt import materialize, pack, render_branch, render_prefix, think_block, to_4d

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
think = think_block(tok)

state = (
    "FW: Kestrel Forge - revised promise date\n"
    "Buyer note: Inconel 718 HPT disk forgings for NB-700 slipping 6 weeks. "
    "~2 weeks on hand. Kestrel is sole source."
)
qs = [materialize(q)[0] for q in [schema.EVENT_TYPE, schema.IMPACT_SEVERITY, schema.LINE_STOP_RISK]]
packed = pack(tok, state, qs, think)

with torch.no_grad():
    out = model(
        input_ids=packed["input_ids"][None],
        position_ids=packed["position_ids"][None],
        attention_mask=to_4d(packed["mask"], torch.float32),
    ).logits[0]

worst = 0.0
for q, r, lab in zip(qs, packed["readouts"], packed["label_ids"]):
    p_ids = tok.encode(render_prefix(state), add_special_tokens=False)
    b_ids = tok.encode(render_branch(q, think), add_special_tokens=False)
    ids = p_ids + b_ids
    natural = tok.encode(render_prefix(state) + render_branch(q, think), add_special_tokens=False)
    assert natural == ids, "prefix/branch boundary is not a natural token boundary"
    with torch.no_grad():
        ref = model(input_ids=torch.tensor([ids])).logits[0, -1]
    diff = (out[r, lab] - ref[lab]).abs().max().item()
    worst = max(worst, diff)
    p = torch.softmax(out[r, lab], -1)
    print(f"{q['id']:16s} maxdiff={diff:.2e} probs={[round(x, 3) for x in p.tolist()]}")
print("think block:", repr(think), "| packed len", len(packed["input_ids"]))
assert worst < 1e-3, worst
print("OK tree packing matches")
