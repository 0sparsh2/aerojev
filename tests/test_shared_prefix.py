"""--shared-catalogue training must give the gradients of training each record on its own.

  python tests/test_shared_prefix.py          # catalogue and nocat layouts, CPU, fp32

The shared path encodes the static prefix (catalogue, or the nocat system line) once, runs the records as a padded
batch in micro-batches on top of its K/V, and backpropagates through the prefix once at the end. The reference runs
each record as one sequence (prefix + record + readouts) with the same prefix. LoRA B matrices are randomized so
every adapter receives a gradient. Also checks that cached-catalogue validation scores like the uncached pass.
"""

import json
import random
import types

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from aerojev.prompt import think_block
from train.train import build_group, quick_eval, readout_loss, shared_step

torch.manual_seed(0)
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
think = think_block(tok)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa")
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                         "up_proj", "down_proj"]))
for n, p in model.named_parameters():
    if "lora_B" in n:
        torch.nn.init.normal_(p, std=0.02)
model.train()
params = {n: p for n, p in model.named_parameters() if p.requires_grad}
rows = [json.loads(l) for l in open("data/processed_v2/train.jsonl")][:3]
off = lambda: torch.autocast("cpu", enabled=False)
scaler = torch.amp.GradScaler("cpu", enabled=False)

for layout in ("catalogue", "nocat"):
    g = build_group(tok, think, rows, random.Random(1), 5, 9, layout)
    # micro_batch=2 over 3 records: one padded batch of 2, then a batch of 1, accumulated into the prefix K/V
    args = types.SimpleNamespace(micro_batch=2, grad_accum=len(rows), ordinal_weight=0.5, distill_weight=0.0,
                                 distill_temp=1.0)
    losses, _, _ = shared_step(model, g, "cpu", args, off, scaler)
    shared = {n: p.grad.clone() for n, p in params.items()}
    model.zero_grad(set_to_none=True)

    S, ref_losses = len(g["static_ids"]), []
    for p in g["records"]:
        n = len(p["input_ids"])
        mask = torch.zeros(S + n, S + n, dtype=torch.bool)
        mask[:S, :S] = torch.tril(torch.ones(S, S, dtype=torch.bool))
        mask[S:, :] = p["mask"]
        pos = torch.cat([torch.arange(S), p["position_ids"]])  # readout branches restart after the record
        out = model(input_ids=torch.tensor(g["static_ids"] + p["input_ids"].tolist())[None], position_ids=pos[None],
                    attention_mask=mask[None, None],
                    logits_to_keep=torch.tensor([S + r for r in p["readouts"]])).logits[0].float()
        loss, _ = readout_loss(out, p, 0.5)
        (loss / len(rows)).backward()
        ref_losses.append(loss.item())
    ref = {n: p.grad.clone() for n, p in params.items()}
    model.zero_grad(set_to_none=True)

    loss_diff = max(abs(a - b) for a, b in zip(losses, ref_losses))
    flat = lambda d: torch.cat([d[n].flatten() for n in params])
    total = ((flat(shared) - flat(ref)).norm() / flat(ref).norm()).item()   # whole gradient vector
    # per tensor, relative to its own norm (max-abs ratios blow up on tensors whose gradient is ~0)
    rel = {n: ((shared[n] - ref[n]).norm() / ref[n].norm().clamp_min(1e-12)).item() for n in params}
    worst = max(rel, key=rel.get)
    print(f"{layout:9s} prefix {S} tokens · max loss diff {loss_diff:.1e} · gradient rel. error {total:.1e} "
          f"(worst tensor {rel[worst]:.1e}, |grad| {ref[worst].norm():.1e}: {worst.split('model.')[-1]})")
    assert loss_diff < 1e-4 and total < 1e-4 and rel[worst] < 1e-2, \
        f"{layout}: shared-prefix gradients differ from per-record training"

val = [json.loads(l) for l in open("data/processed_v2/val.jsonl")][:6]
for layout in ("catalogue", "nocat"):
    a = quick_eval(model, tok, think, val, "cpu", torch.float32, layout, cached=False)
    b = quick_eval(model, tok, think, val, "cpu", torch.float32, layout, cached=True)
    print(f"{layout:9s} validation uncached {a:.4f} · cached {b:.4f}")
    assert a == b
print("OK shared-prefix training matches per-record training")
