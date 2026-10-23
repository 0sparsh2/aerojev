"""Build notebooks/aerojev_qwen2.5-0.5b_nocat.ipynb (and its Markdown copy): the catalogue-free recipe, explained.

  python scripts/build_notebook_nocat.py

This is level L4 of the CPU speed-up ladder (results/ladder/ladder.md): no question catalogue in the prompt,
all 9 questions per record, 8 records per batch on a shared prefix, 6,000 training records.
"""

import os
import textwrap

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "aerojev_qwen2.5-0.5b_nocat"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))


# =====================================================================================================
md("""
# AeroJev, catalogue-free recipe: Qwen2.5-0.5B + LoRA on 6,000 records

This notebook trains, checks, exports and serves the cheapest AeroJev recipe from the CPU speed-up ladder
(`results/ladder/ladder.md`, level **L4, no catalogue, 6k records**). It is a companion to the main
notebook, `aerojev_qwen2.5-0.5b_lora.ipynb`, which explains the project from scratch (question schema, how the
dataset is generated, tree packing). Read that one first if AeroJev is new to you; this one focuses on what is
different and on running it.

| | Main recipe | This recipe |
|---|---|---|
| Prompt | system + **question catalogue (~900 tokens)** + record + readouts | system line (~65 tokens) + record + readouts |
| Questions asked per training record | a random 5–9 of the 9 | all 9 |
| Training records | 12,000 | **6,000** |
| Records per forward pass | 1 | 8, as one batch on a shared prefix |
| Learning rate · LoRA dropout | 2e-4 · 0.05 | 5e-4 · 0 |
| Training compute per record | 1,864 GFLOP | **309 GFLOP** |
| Gold · gold2 · test accuracy (GPU ladder) | 0.922 · 0.919 · 0.948 | run 1: 0.921 · 0.912 · 0.953 <br> run 2: 0.885 · 0.882 · 0.921 |
| Training time on a T4 GPU (measured) | 62 min | **9 min** |
| Training time on a CPU (estimated): regular 2-core CPU · Apple M4 | ~70 h · ~15 h | **~6 h · ~1.2 h** |
| Serving on an M4 CPU, Q8_0 (measured): per record · start-up | 0.25 s · 2.3 s | **0.16 s · 0.85 s** |

The CPU training times are projections: the measured CPU speed of the main recipe (20.2 s and 4.35 s per record)
scaled by the compute each recipe needs, validation passes not included. Section 6 measures your own machine.

**Read this before relying on 6,000 records.** The two GPU runs of this exact recipe differ only in the random
order of the training records, yet the second scored 3 points lower on the 800-record test set, far more than
measurement noise (about ±0.3). Neither run had finished learning when its schedule ended. The difference sits in
the three questions that need arithmetic on the record (line-stop risk, impact severity, recommended action). With
**12,000 records** the same recipe gave 0.928 · 0.905 · 0.944, as good as the main recipe, for ~12 h on the regular
CPU or ~2.5 h on an M4. Set `AEROJEV_NB_RECORDS=12000` for that. With 6,000 records, check the per-question
accuracy in section 9 before using the model. Fewer records lose more: 4,800 gave 0.907 and 3,000 gave 0.902 on test.

**The trade-off.** The model never sees the questions or their options; it learns them, the way a classifier
learns its classes. It therefore answers only the 9 questions it was trained on, each with its options in the
order of `aerojev/schema.py`. Changing a question, adding one or reordering options means retraining (about 9
minutes on a GPU, a few hours on a CPU), and the model refuses anything else rather than guessing. If your
question list changes often, or you need ad-hoc questions, use the main recipe.

**How to run**

```bash
pip install -r requirements.txt
jupyter lab notebooks/aerojev_qwen2.5-0.5b_nocat.ipynb
```

| Environment variable | Default | Effect |
|---|---|---|
| `AEROJEV_NB_FAST` | `0` on a CUDA GPU, `1` otherwise | `1` = a quick end-to-end check on a few dozen records; `0` = the full recipe |
| `AEROJEV_NB_RECORDS` | `6000` | training records; `12000` is the reliable setting (see above) |
| `AEROJEV_NB_EXPORT` | `1` | `0` skips the GGUF export (it needs git, cmake and a C++ compiler) |

On a CUDA GPU the full recipe runs in about 15–20 minutes, evaluation and export included. On a CPU the notebook starts in FAST
mode; set `AEROJEV_NB_FAST=0` before starting Jupyter to train the full recipe on the CPU (section 6 tells you how
long it will take first). The same recipe without a notebook is in section 15.

**Contents**

1. What this recipe changes, and why it works
2. The data
3. The prompt without a question list
4. One forward pass for all 9 questions
5. Training setup: 8 records per batch on a shared prefix
6. How long will training take on this machine?
7. Training
8. Training curves
9. Evaluation
10. Calibration
11. Exporting for CPU (GGUF)
12. CPU latency
13. Using the model
14. Changing the questions
15. The same recipe from the command line
16. Summary
""")

md("## 0. Setup")
code('''
# Standard library first; the notebook only needs packages from requirements.txt.
import gc, json, math, os, random, sys, time, types
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_repo_root():
    """The notebook lives in <repo>/notebooks/. Walk up from the working directory until we find the repo
    (the folder that contains aerojev/engine.py), or use $AEROJEV_ROOT if it is set."""
    if os.environ.get("AEROJEV_ROOT"):
        return os.path.abspath(os.environ["AEROJEV_ROOT"])
    d = os.path.abspath(os.getcwd())
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "aerojev", "engine.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Start Jupyter inside the AeroJev repository, or set AEROJEV_ROOT.")


ROOT = find_repo_root()
os.chdir(ROOT)               # every path below is relative to the repository
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # makes the project's packages importable: aerojev, train, eval
pd.set_option("display.max_colwidth", 100)
print("repository:", ROOT)
''')
code('''
import torch
import transformers
import peft

# Device. CUDA runs the full recipe in minutes. Apple's MPS backend is deliberately not used: sequence lengths
# change every step and MPS recompiles per shape, which made it slower than the CPU in our tests.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAST = os.environ.get("AEROJEV_NB_FAST", "0" if DEVICE == "cuda" else "1") == "1"

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # the recipe was tuned and measured on this model only
LAYOUT = "nocat"                       # no question catalogue in the prompt (section 3)
DATA = os.path.join(ROOT, "data", "processed_v2")             # the data v2 splits shipped with the repository
DO_EXPORT = os.environ.get("AEROJEV_NB_EXPORT", "1") == "1"  # section 11 needs git, cmake and a C++ compiler
OUT = os.path.join(ROOT, "results", "notebook-nocat")        # everything this notebook writes goes here
os.makedirs(OUT, exist_ok=True)

CFG = dict(
    n_records=int(os.environ.get("AEROJEV_NB_RECORDS", "6000")),  # training records seen (6,000 = 0.6 passes over
                         # the 10,000 training records). 12,000 is the reliable setting; see the introduction
    lr=5e-4,             # tuned in the ladder for this recipe; 1e-3 made training diverge
    lora_r=16, lora_alpha=32,
    lora_dropout=0.0,    # no dropout; the shuffled records already vary enough, and dropout costs a copy per layer
    records_per_step=8,  # records per optimizer step; they share one prefix and run as one batch
    micro_batch=8,       # records per forward pass inside a step. Lower it (4, 2, 1) if memory runs short:
                         # the gradients are exactly the same, only the peak memory and speed change
    questions=9,         # every record asks all 9 trained questions
    ordinal_weight=0.5,  # weight of the distance penalty on score questions
    warmup=50,           # optimizer steps of linear warm-up before the cosine decay
    eval_every=150, val_limit=150,  # quick validation during training: every N steps, on this many records
    eval_limit=None,     # cap on records per evaluation set (None = all)
    bench_records=40,    # records timed in the CPU latency section
)
if FAST:  # a quick end-to-end check: same code path, a few dozen records
    CFG.update(n_records=64, warmup=2, eval_every=4, val_limit=12, eval_limit=24, bench_records=8)

# Mixed precision: fp16 autocast on CUDA (T4-class GPUs have no bf16), plain fp32 on CPU (bf16 is slow on most CPUs).
AMP = torch.float16 if DEVICE == "cuda" else torch.float32
autocast = lambda: torch.autocast(device_type=DEVICE, dtype=AMP, enabled=AMP != torch.float32)
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print(f"device={DEVICE}  FAST={FAST}  model={MODEL}  layout={LAYOUT}  training records={CFG['n_records']}")
print(f"torch {torch.__version__} · transformers {transformers.__version__} · peft {peft.__version__}")
''')

# =====================================================================================================
md("""
## 1. What this recipe changes, and why it works

In the main recipe, every training sequence is about 1,090 tokens. About 900 of them are the **question catalogue**:
the 9 questions with all their options, reshuffled for every record so the model learns to read options rather than
memorise positions. The model re-reads, and backpropagates through, that list 12,000 times. That is 83% of all
training compute, spent on the same text.

This recipe makes three changes:

1. **No catalogue in the prompt.** The prompt is a one-line instruction, the record, and one short readout per
   question (`{"impact_severity": "`). The model learns what each question asks and what each label letter means
   from the training labels themselves. The question ids in the readouts tell it which question is being asked.
2. **All 9 questions on every record.** Each question's readout is an independent branch that sees only the prompt
   and itself, so asking a random subset (as the main recipe does) never changes what any readout sees. It only
   throws labels away. Asking all 9 gives ~29% more labels for ~1% more compute.
3. **8 records per batch on a shared prefix.** The 8 records of one optimizer step run as a single padded batch on top
   of the system line, which is encoded once. The gradients are exactly those of running the 8 records one at a
   time. It matters little for compute here (the prefix is short), but it keeps a GPU busy and CPU matrix
   multiplications large.

Without the catalogue, the model no longer has to learn to read shuffled option lists, which is the hard part of
the main recipe. It learns faster per record as a result: at 6,000 records one run matched the main recipe and
another was 3 points lower, while the catalogue recipe lost 2–4 points at 6,000. At 12,000 records it matches the
main recipe at a sixth of the compute.
""")
code('''
# The measured ladder, if this repository has it (scripts/ladder_report.py writes it after the GPU run).
ladder_path = os.path.join(ROOT, "results", "ladder", "ladder.json")
LADDER = {r["run"]: r for r in json.load(open(ladder_path))} if os.path.exists(ladder_path) else {}
if LADDER:
    show = ["l0-baseline", "l1-shared", "l4-nocat-12k", "l4-nocat-6k", "l4-nocat-6k-seed1", "l4-nocat-4.8k", "l4-nocat-3k"]
    display(pd.DataFrame({r: {**{k: LADDER[r]["acc"][k] for k in ("gold", "gold2", "test")},
                              "T4 train min": LADDER[r]["gpu_train_min"], "GFLOP/record": LADDER[r]["gflop_per_record"],
                              "regular CPU h (est.)": LADDER[r]["cpu_hours"]["regular CPU"],
                              "M4 h (est.)": LADDER[r]["cpu_hours"]["M4"]} for r in show if r in LADDER}).T.round(3))
else:
    print("results/ladder/ladder.json not found; the numbers in the introduction come from that run")
''')

# =====================================================================================================
md("""
## 2. The data

The splits in `data/processed_v2/` are the project's data v2: synthetic supply-chain records whose labels come from a
written playbook, plus two hand-written gold sets (the main notebook's section 2 shows how they are built). Every
row is one record with an answer for every question:

```json
{"id": "...", "format": "email", "state": "<the record text>", "labels": {"event_type": "delivery_delay", ...}}
```

| set | records | what it tests |
|---|---|---|
| `train` | 10,000 | training (this recipe sees 6,000 of them) |
| `val` | 500 | same templates as training; used for checkpoints and calibration |
| `test` | 800 | held-out wording and a format only found in test |
| `test_v2` | 800 | held-out wording of the data v2 generator |
| `gold`, `gold2` | 80, 48 | written by hand; `gold2` was written blind, before any tuning |
| `injection` | 200 | a prompt-injection line in every record |

**Your own records** need the same fields. Each label must be one of the question's option keys in
`aerojev/schema.py` (`yes`/`no` for yes/no questions, `"0"`–`"4"` for scores).
""")
code('''
from aerojev import schema

SETS = ["train", "val", "test", "test_v2", "gold", "gold2", "injection"]
sets = {name: [json.loads(l) for l in open(os.path.join(DATA, f"{name}.jsonl"))] for name in SETS}
display(pd.Series({name: len(rows) for name, rows in sets.items()}, name="records"))

CORE = [q.id for q in schema.TRAIN_QUESTIONS]    # the 9 questions this model learns
ex = sets["gold"][0]
print(ex["state"], "\\n")
print({q: ex["labels"][q] for q in CORE})
''')
code('''
# A small data card: how the training answers spread for three questions.
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for ax, q in zip(axes, ["event_type", "impact_severity", "recommended_action"]):
    cnt = Counter(r["labels"][q] for r in sets["train"])
    keys = sorted(cnt) if q == "impact_severity" else [k for k, _ in cnt.most_common()]
    ax.barh(keys, [cnt[k] for k in keys], color="#4a6fa5"); ax.invert_yaxis()
    ax.set_title(q); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.show()
''')

# =====================================================================================================
md("""
## 3. The prompt without a question list

```
<|im_start|>system
You are AeroJev, a decision engine ... answer one question, by id, with exactly one label. ...<|im_end|>
<|im_start|>user
Record:
<the record><|im_end|>
<|im_start|>assistant
{"impact_severity": "▌        ← the next token is the answer; one readout like this per question
```

The answer is read from the next-token logits, restricted to the question's label tokens (`A`, `B`, … for choice
questions, `0`–`4` for scores, `yes`/`no`), and softmaxed. Nothing is generated.

Because the options are never shown, **the label letters are fixed**: `A` is always the first option in
`aerojev/schema.py`, `B` the second, and so on. This mapping is the model's contract. The cell saves it as
`label_map.json`, and section 14 checks it before the model is used.
""")
code('''
from transformers import AutoTokenizer
from aerojev.prompt import (labels_for, materialize, render_readout, render_record, render_static,
                            render_static_nocat, think_block)

tok = AutoTokenizer.from_pretrained(MODEL)
think = think_block(tok)   # "" for Qwen2.5 (no thinking block in its chat template)
core_q = [materialize(q)[0] for q in schema.TRAIN_QUESTIONS]   # canonical form: schema text and option order
n_tok = lambda t: len(tok.encode(t, add_special_tokens=False))

prompt = render_static_nocat() + render_record(ex["state"], think)
print(prompt + render_readout(core_q[2]) + "▌")
print(f"\\ntokens: system line {n_tok(render_static_nocat())} · this record {n_tok(render_record(ex['state'], think))} · "
      f"{n_tok(render_readout(core_q[0]))} per question readout")
print(f"(the main recipe puts a {n_tok(render_static(core_q))}-token question catalogue in front of every record)")

# The contract: question -> label -> option key, in schema order.
LABEL_MAP = {q["id"]: {"type": q["type"], "labels": dict(zip(labels_for(q), materialize(sq)[1]))}
             for q, sq in zip(core_q, schema.TRAIN_QUESTIONS)}
json.dump(LABEL_MAP, open(os.path.join(OUT, "label_map.json"), "w"), indent=1)
pd.DataFrame([{"question": qid, "type": m["type"], "label → option": ", ".join(f"{l}={k}" for l, k in m["labels"].items())}
              for qid, m in LABEL_MAP.items()])
''')

# =====================================================================================================
md("""
## 4. One forward pass for all 9 questions

The record is laid out once, and each question's readout follows it as its own **branch**. A 4D attention mask lets
every branch see the prompt and itself but not the other branches, and each branch's position ids restart at the
same offset. One forward pass then gives exactly the answers that 9 separate passes would. The cell checks that
on the untrained model.
""")
code('''
from transformers import AutoModelForCausalLM
from aerojev.prompt import pack

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE).eval()
p = pack(tok, ex["state"], core_q, think, layout=LAYOUT)
with torch.no_grad():
    # one pass: system line + record + 9 readout branches; logits only at the 9 readout positions
    packed = base(input_ids=p["input_ids"][None].to(DEVICE), position_ids=p["position_ids"][None].to(DEVICE),
                  attention_mask=p["mask"][None, None].to(DEVICE),
                  logits_to_keep=torch.tensor(p["readouts"], device=DEVICE)).logits[0].float().cpu()
    worst = 0.0
    for i, q in enumerate(core_q):  # reference: each question on its own, plain causal attention
        single = pack(tok, ex["state"], [q], think, layout=LAYOUT)
        ref = base(input_ids=single["input_ids"][None].to(DEVICE)).logits[0, -1].float().cpu()
        worst = max(worst, (packed[i, p["label_ids"][i]] - ref[single["label_ids"][0]]).abs().max().item())
print(f"{len(p['input_ids'])} tokens for 9 questions in one pass · max logit difference vs. 9 separate passes: {worst:.1e}")
del base; gc.collect()
''')

# =====================================================================================================
md("""
## 5. Training setup: 8 records per batch on a shared prefix

**Loss** (RFDT, as in the main recipe). For every question of every record: cross-entropy over that question's label
tokens at its readout. Score questions add an ordinal penalty, the expected distance from the gold level, so
predicting 3 for a gold 4 costs less than predicting 0. There is no loss anywhere else in the prompt.

**LoRA** of rank 16 on every linear layer of every block; about 1.7% of the parameters train.

**One optimizer step = 8 records**, laid out like this:

```
system line (~65 tokens)          encoded once per step; its keys/values are shared by the whole batch
├── record 1 + 9 readouts  ┐
├── record 2 + 9 readouts  │      one padded batch: each row attends to the system line and to itself only
│   ...                    │
└── record 8 + 9 readouts  ┘
```

`train.train.shared_step` runs this. The system line's keys/values are kept as separate tensors while the records
run (in micro-batches of `CFG["micro_batch"]`). Their gradients are summed, then sent back through the system line once.
This gives exactly the gradients of training each record on its own: `tests/test_shared_prefix.py` checks it
to fp32 rounding.
""")
code('''
from peft import LoraConfig, get_peft_model
from train.train import build_group, quick_eval, shared_step

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa")
model.config.use_cache = False          # the trainer asks for the prefix's K/V explicitly; nothing else is cached
model = get_peft_model(model, LoraConfig(
    r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"], bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
model.to(DEVICE).train()

# shared_step reads its settings from an argparse-style namespace (the same function train.py uses)
ARGS = types.SimpleNamespace(micro_batch=CFG["micro_batch"], grad_accum=CFG["records_per_step"],
                             ordinal_weight=CFG["ordinal_weight"], distill_weight=0.0, distill_temp=1.0)
scaler = torch.amp.GradScaler(DEVICE, enabled=AMP == torch.float16)   # loss scaling for fp16 on GPU

# What one step's batch looks like
g = build_group(tok, think, sets["train"][:CFG["records_per_step"]], random.Random(0),
                CFG["questions"], CFG["questions"], LAYOUT)
lens = [len(r["input_ids"]) for r in g["records"]]
print(f"shared prefix: {len(g['static_ids'])} tokens · {len(lens)} records of {min(lens)}–{max(lens)} tokens "
      f"(record + 9 readouts) · {len(g['records'][0]['readouts'])} questions each")
''')

# =====================================================================================================
md("""
## 6. How long will training take on this machine?

Before committing hours of CPU time, time a few real training steps (forward + backward, no weight update) and
project the full run. The first step is a warm-up and is not counted. Validation passes add roughly 10% on top.
On a CPU, use as many threads as performance cores (`OMP_NUM_THREADS`, or `torch.set_num_threads` below); extra
hyperthreads did not help in our tests.
""")
code('''
if DEVICE == "cpu":
    print(f"CPU threads: {torch.get_num_threads()} (set OMP_NUM_THREADS before starting Jupyter to change it)")
timing_rng = random.Random(123)                 # its own RNG, so timing does not change the training run
step_s = []
for i in range(4 if DEVICE == "cpu" else 6):
    rows = [timing_rng.choice(sets["train"]) for _ in range(CFG["records_per_step"])]
    g = build_group(tok, think, rows, timing_rng, CFG["questions"], CFG["questions"], LAYOUT)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    shared_step(model, g, DEVICE, ARGS, autocast, scaler)
    if DEVICE == "cuda": torch.cuda.synchronize()
    if i > 0:                                   # skip the warm-up step
        step_s.append(time.time() - t0)
model.zero_grad(set_to_none=True)               # throw the timing gradients away; no weights were updated
per_record = float(np.mean(step_s)) / CFG["records_per_step"]
full = 6000 * per_record / 60
print(f"{per_record:.2f} s per training record on {DEVICE} → the full recipe (6,000 records) takes about "
      f"{full:.0f} min ({full / 60:.1f} h) of training here")
''')

# =====================================================================================================
md("""
## 7. Training

The loop is short because `shared_step` does the batch work. Each step:

1. take the next 8 records (a fresh shuffle every pass over the data),
2. run them as one batch on the shared prefix, forward and backward,
3. clip the gradient norm to 1.0 and take one AdamW step, with linear warm-up and then cosine decay of the LR.

Every `eval_every` steps, accuracy on `val_limit` validation records is measured, and the adapter is saved when it
improves. If the run stops early, the best adapter so far is in `results/notebook-nocat/adapter`.
""")
code('''
rng = random.Random(0)
params = [q for q in model.parameters() if q.requires_grad]   # only the LoRA matrices
opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
total_steps = math.ceil(CFG["n_records"] / CFG["records_per_step"])
# linear warm-up, then cosine decay to zero over the run
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CFG["warmup"]) *
                                          0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
val_rows = sets["val"][: CFG["val_limit"]]
order = []


def next_record():
    """Records in a fresh random order every pass over the training set."""
    global order
    if not order:
        order = list(range(len(sets["train"]))); rng.shuffle(order)
    return sets["train"][order.pop()]


log, best, step, seen, t0, val_s = [], -1.0, 0, 0, time.time(), 0.0
t = time.time(); acc = quick_eval(model, tok, think, val_rows, DEVICE, AMP, LAYOUT, cached=True); val_s += time.time() - t
log.append({"step": 0, "val_acc": acc})
print(f"step 0  val acc {acc:.3f}  ({total_steps} optimizer steps planned)")
while step < total_steps:
    g = build_group(tok, think, [next_record() for _ in range(CFG["records_per_step"])], rng,
                    CFG["questions"], CFG["questions"], LAYOUT)
    losses, correct, _ = shared_step(model, g, DEVICE, ARGS, autocast, scaler)
    if losses is None:                          # a rare fp16 overflow: drop this step instead of corrupting the weights
        opt.zero_grad(set_to_none=True); continue
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
    step += 1; seen += len(losses)
    log.append({"step": step, "loss": float(np.mean(losses)), "train_acc": float(np.mean(correct))})
    if step % CFG["eval_every"] == 0 or step == total_steps:
        t = time.time(); acc = quick_eval(model, tok, think, val_rows, DEVICE, AMP, LAYOUT, cached=True); val_s += time.time() - t
        log.append({"step": step, "val_acc": acc})
        print(f"step {step:4d}/{total_steps}  val acc {acc:.3f}  {seen / (time.time() - t0 - val_s):.2f} records/s")
        if acc > best:                          # keep the best checkpoint by validation accuracy
            best = acc; model.save_pretrained(os.path.join(OUT, "adapter"))
train_min = (time.time() - t0 - val_s) / 60
print(f"done: {seen} records in {train_min:.1f} min of training (+{val_s / 60:.1f} min validation) · "
      f"best val acc {best:.3f} · adapter saved to {OUT}/adapter")
json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"))
''')

# =====================================================================================================
md("## 8. Training curves")
code('''
L = pd.DataFrame(log)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
tr = L.dropna(subset=["loss"])
ax[0].plot(tr["step"], tr["loss"].rolling(10, min_periods=1).mean(), color="#4a6fa5")
ax[0].set_title("training loss (rolling mean, 10 steps)"); ax[0].set_xlabel("optimizer step")
va = L.dropna(subset=["val_acc"])
ax[1].plot(va["step"], va["val_acc"], marker="o", color="#4a6fa5")
ax[1].set_title("validation accuracy, 9 questions"); ax[1].set_xlabel("optimizer step")
plt.tight_layout(); plt.show()
''')

# =====================================================================================================
md("""
## 9. Evaluation

The best adapter is merged into the base weights, so inference has no adapter overhead. The merged model is then
scored on every set; accuracy is the mean over the 9 questions. There is no shuffled-options test here (the main
notebook has one): this model never sees options, so there is nothing to shuffle.
""")
code('''
from peft import PeftModel
from aerojev.engine import DecisionEngine
from aerojev.metrics import fit_temperatures, softmax, summarize
from eval.evaluate import run_set

del model, opt
gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
ft = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa")
ft = PeftModel.from_pretrained(ft, os.path.join(OUT, "adapter")).merge_and_unload().to(DEVICE).eval()

EVAL_SETS = ["val", "test", "test_v2", "gold", "gold2", "injection"]
limit = lambda rows: rows[: CFG["eval_limit"]] if CFG["eval_limit"] else rows


def evaluate(engine, names=EVAL_SETS):
    """Ask the 9 trained questions of every record; return raw predictions and a metrics table."""
    preds, table = {}, {}
    for name in names:
        preds[name], sec = run_set(engine, limit(sets[name]), schema.TRAIN_QUESTIONS)
        m = summarize(preds[name], CORE)
        table[name] = {"accuracy": m["core_mean_accuracy"], "macro-F1": m["core_mean_macro_f1"],
                       "all 9 right": m["record_exact_match"], "ECE": m["core_ece"], "s/record": sec}
    return preds, pd.DataFrame(table).T


engine = DecisionEngine(tok=tok, model=ft, device=DEVICE, layout=LAYOUT)
preds, table = evaluate(engine)
display(table.round(3))
if LADDER:  # the same recipe and the main recipe, as measured in the GPU ladder
    cols = {"this notebook": table["accuracy"]}
    for run, label in [("l4-nocat-6k", "ladder: 6k, run 1"), ("l4-nocat-6k-seed1", "ladder: 6k, run 2"),
                       ("l4-nocat-12k", "ladder: 12k"), ("l0-baseline", "ladder: main recipe")]:
        if run in LADDER:
            cols[label] = pd.Series(LADDER[run]["acc"])
    ref = pd.DataFrame(cols)
    display(ref.round(3))
''')
code('''
# Accuracy per question on the hand-written gold set and the larger test set.
per_q = lambda name: {q: v["accuracy"] for q, v in summarize(preds[name], CORE)["per_question"].items()}
pd.DataFrame({"gold": per_q("gold"), "test": per_q("test")}).loc[CORE].round(3)
''')
code('''
def confusion(rows, qid):
    """Rows: gold answer; columns: predicted answer."""
    rows = [p for p in rows if p["qid"] == qid]
    keys = rows[0]["keys"]
    m = pd.DataFrame(0, index=keys, columns=keys)
    for p in rows:
        m.loc[p["gold"], p["keys"][int(np.argmax(p["logits"]))]] += 1
    m.index.name, m.columns.name = "gold", "predicted"
    return m.loc[(m.sum(axis=1) > 0) | (m.sum(axis=0) > 0)]

display(confusion(preds["gold"], "impact_severity"))
display(confusion(preds["gold"], "recommended_action"))
''')

# =====================================================================================================
md("""
## 10. Calibration

A probability is only useful if 80% confidence means right about 80% of the time. One **temperature** per question
type (choice, score, yes/no) is fitted on the validation set and applied everywhere. ECE is the average gap between
confidence and accuracy; lower is better. The temperatures are saved next to the model and passed to the server.
""")
code('''
temps = fit_temperatures([p for p in preds["val"] if p["qid"] in CORE])
ece = {name: {"raw": summarize(preds[name], CORE)["core_ece"],
              "calibrated": summarize(preds[name], CORE, temps)["core_ece"]} for name in ("test", "gold", "gold2")}
print("temperatures:", temps); display(pd.DataFrame(ece).T.round(4))
json.dump(temps, open(os.path.join(OUT, "temperatures.json"), "w"))

conf, correct = [], []
for name in ("test", "gold", "gold2"):
    for p in preds[name]:
        pr = softmax(p["logits"], temps.get(p["type"], 1.0))
        conf.append(pr.max()); correct.append(p["keys"][int(pr.argmax())] == p["gold"])
conf, correct = np.array(conf), np.array(correct, float)
mid, acc, cnt = [], [], []
for lo, hi in zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
    m = (conf > lo) & (conf <= hi)
    if m.any():
        mid.append(conf[m].mean()); acc.append(correct[m].mean()); cnt.append(m.sum())
plt.figure(figsize=(4.2, 4.2)); plt.plot([0, 1], [0, 1], color="#bbb", lw=1)
plt.scatter(mid, acc, s=[20 + 200 * c / max(cnt) for c in cnt], color="#4a6fa5")
plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title("reliability after temperature scaling\\n(dot size = decisions)")
plt.show()
''')

# =====================================================================================================
md("""
## 11. Exporting for CPU (GGUF)

`aerojev.export` saves the merged model, converts it to GGUF with llama.cpp's converter (Q8_0) and makes a Q4_K_M
copy with `llama-quantize`. The first run clones and builds llama.cpp's quantizer in `~/.cache/aerojev/llama.cpp`
(git, cmake and a C++ compiler needed); set `AEROJEV_NB_EXPORT=0` to skip it. The label map and the temperatures are
copied next to the GGUF files: the model is not usable without them.
""")
code('''
import shutil

GGUF = {}
if DO_EXPORT:
    from aerojev.export import export_gguf
    t0 = time.time()
    gdir = os.path.join(OUT, "gguf")
    GGUF = {q.upper(): path for q, path in export_gguf(ft.to("cpu"), tok, gdir, "aerojev-qwen2.5-0.5b-nocat").items()}
    for f in ("label_map.json", "temperatures.json"):
        shutil.copy(os.path.join(OUT, f), os.path.join(gdir, f))
    ft.to(DEVICE)
    print(f"exported in {time.time() - t0:.0f}s to {gdir}")
pd.Series({q: f"{os.path.getsize(p) / 1e6:.0f} MB" for q, p in GGUF.items()}, name="file size", dtype=object)
''')

# =====================================================================================================
md("""
## 12. CPU latency

Inference is cheaper too: each record is processed with a 65-token system line instead of next to a 900-token
catalogue. The cell times the 9 decisions for the same gold records, one record at a time, in a warm process, with
up to 4 threads:

* **torch fp32**: the merged model in PyTorch
* **llama.cpp Q8_0 / Q4_K_M**: the exported files (needs `llama-cpp-python`)
* **main recipe, llama.cpp Q8_0**: for comparison, if `models/aerojev-qwen2.5-0.5b-cat-q8_0.gguf` exists

Accuracy on the timed records is shown next to each row, to show what quantization costs.
""")
code('''
from aerojev.engine import GGUFEngine

THREADS = min(4, os.cpu_count() or 4)
torch.set_num_threads(THREADS)
bench_rows = sets["gold"][: CFG["bench_records"]]
ft_cpu = ft.to("cpu")


def bench(label, eng):
    """p50 seconds per record and ms per decision, plus accuracy on the same records."""
    eng.readout_logits(bench_rows[0]["state"], core_q)   # warm-up (caches, allocator)
    lat, hits = [], []
    for r in bench_rows:
        t0 = time.time(); logits = eng.readout_logits(r["state"], core_q); lat.append(time.time() - t0)
        for q, lg in zip(core_q, logits):
            keys = ["yes", "no"] if q["type"] == "yesno" else [o["key"] for o in q["options"]]
            hits.append(keys[int(torch.as_tensor(lg).argmax())] == r["labels"][q["id"]])
    lat.sort()
    return {"configuration": label, "p50 s/record": lat[len(lat) // 2],
            "ms/decision": 1000 * np.mean(lat) / len(core_q), "accuracy (these records)": float(np.mean(hits))}


rows = [bench("this recipe · torch fp32", DecisionEngine(tok=tok, model=ft_cpu, device="cpu", layout=LAYOUT))]
main_q8 = os.path.join(ROOT, "models", "aerojev-qwen2.5-0.5b-cat-q8_0.gguf")
try:
    for q, path in GGUF.items():
        rows.append(bench(f"this recipe · llama.cpp {q}", GGUFEngine(path, MODEL, n_threads=THREADS, layout=LAYOUT)))
    if os.path.exists(main_q8):
        rows.append(bench("main recipe · llama.cpp Q8_0", GGUFEngine(main_q8, MODEL, n_threads=THREADS, layout="catalogue")))
except ImportError:
    print("llama-cpp-python is not installed; skipping the GGUF rows (see requirements.txt)")
latency = pd.DataFrame(rows).set_index("configuration")
print(f"CPU: {os.cpu_count()} cores, {THREADS} threads"); latency.round(3)
''')

# =====================================================================================================
md("## 13. Using the model")
code('''
# decide() returns every answer with its calibrated probability distribution.
engine = DecisionEngine(tok=tok, model=ft_cpu, device="cpu", layout=LAYOUT, temperatures=temps)
note = """FW: Brennerwerk - servo valve PO 4523300117
Buyer note: Brennerwerk says the valves slip 5 weeks (their test stand is down). We have 12 days of valves.
They are the only qualified source. VTX-12 program, ECCN 9A610.x."""
answers = engine.decide(note, schema.TRAIN_QUESTIONS)
display(pd.DataFrame({q: {"answer": a["answer"], "confidence": round(a["confidence"], 3)} for q, a in answers.items()}).T)

# Questions it was not trained on are refused, not guessed (the model would never see their text).
try:
    engine.decide(note, [schema.MILITARY_PROGRAM])
except ValueError as e:
    print("refused:", e)
''')
md("""
**Serving it.** `server.py` runs the same engine behind a small HTTP API and a test page. Without a `questions`
field, a request gets all 9 answers:

```bash
python server.py --gguf results/notebook-nocat/gguf/aerojev-qwen2.5-0.5b-nocat-q8_0.gguf \\
                 --tokenizer Qwen/Qwen2.5-0.5B-Instruct --layout nocat \\
                 --temps results/notebook-nocat/gguf/temperatures.json
curl -s localhost:8080/v1/decide -d '{"state": "Kestrel: 718 disks slip 6 weeks. 2 weeks on hand. Sole source."}'
```

A request for a question outside the trained 9 gets HTTP 422 with an explanation.
""")

# =====================================================================================================
md("""
## 14. Changing the questions

The model's knowledge of its questions lives in its weights, keyed by question id and option order. So:

| Change in `aerojev/schema.py` | What to do |
|---|---|
| reword a question or an option description | nothing: the model never reads them (they still document what the labels mean) |
| reorder, add or remove an option | retrain |
| add or remove a question | retrain (and give every training record a label for the new question) |
| change the playbook rules behind the labels | regenerate the data, then retrain |

Retraining is the full recipe: about 9 minutes on a T4, or the time section 6 projected for your CPU. The cell below
is the guard to run before using a saved model. It compares the model's `label_map.json` with the current schema
and stops if they disagree, instead of letting a renamed or reordered option silently swap answers.
""")
code('''
def check_label_map(path):
    """Raise if the schema no longer matches the label map a model was trained with."""
    saved = json.load(open(path))
    now = {q.id: dict(zip(labels_for(materialize(q)[0]), materialize(q)[1])) for q in schema.TRAIN_QUESTIONS}
    problems = [qid for qid in set(saved) | set(now) if saved.get(qid, {}).get("labels") != now.get(qid)]
    if problems:
        raise RuntimeError(f"schema changed since training for {sorted(problems)}: retrain before using this model")
    print(f"label map OK: {len(saved)} questions match aerojev/schema.py")

check_label_map(os.path.join(OUT, "label_map.json"))
''')

# =====================================================================================================
md("""
## 15. The same recipe from the command line

Everything above, without the notebook. On a GPU:

```bash
python -m train.train --model Qwen/Qwen2.5-0.5B-Instruct --data data/processed_v2 \\
    --layout nocat --shared-catalogue --min-q 9 --lora-dropout 0 --cached-val \\
    --epochs 0.6 --lr 5e-4 --out runs/qwen2.5-0.5b-nocat-6k
python -m eval.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen2.5-0.5b-nocat-6k/best \\
    --layout nocat --data data/processed_v2 --sets val,test,test_v2,gold,gold2,injection \\
    --out results/ft-qwen2.5-0.5b-nocat-6k
python -m aerojev.export --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen2.5-0.5b-nocat-6k/best \\
    --out models --name aerojev-qwen2.5-0.5b-nocat
python server.py --gguf models/aerojev-qwen2.5-0.5b-nocat-q8_0.gguf --tokenizer Qwen/Qwen2.5-0.5B-Instruct \\
    --layout nocat --temps results/ft-qwen2.5-0.5b-nocat-6k/metrics.json
```

(`--epochs 0.6` over the 10,000 training records is the 6,000 records; use `--epochs 1.2` for the reliable 12,000.
`--shared-catalogue` is the 8-records-per-batch mode; the evaluation's `metrics.json` carries the fitted temperatures
the server reads.)

**On a CPU**, add the CPU settings and keep the machine awake:

```bash
OMP_NUM_THREADS=4 nohup caffeinate -i python -m train.train --model Qwen/Qwen2.5-0.5B-Instruct \\
    --data data/processed_v2 --layout nocat --shared-catalogue --min-q 9 --lora-dropout 0 --cached-val \\
    --epochs 0.6 --lr 5e-4 --device cpu --precision fp32 --micro-batch 8 --val-limit 60 \\
    --out runs/qwen2.5-0.5b-nocat-6k > runs/qwen2.5-0.5b-nocat-6k.log 2>&1 &
tail -f runs/qwen2.5-0.5b-nocat-6k.log
```

* `OMP_NUM_THREADS`: your number of performance (physical) cores. `caffeinate -i` is macOS only; on Linux drop it.
* `--micro-batch`: lower it (4, 2, 1) if the machine runs short of memory. The result is identical, only the
  peak memory and speed change.
* `--val-limit 60`: validates on 60 records instead of 150, which saves time; the final evaluation is unaffected.
* The best adapter is saved to `runs/qwen2.5-0.5b-nocat-6k/best` at every improvement, so an interrupted run
  still leaves a usable model.
""")

# =====================================================================================================
md("## 16. Summary")
code('''
summary = {
    "model": MODEL, "layout": LAYOUT, "fast_mode": FAST, "device": DEVICE, "config": CFG, "temperatures": temps,
    "train_minutes": train_min, "sec_per_training_record_measured": per_record,
    "accuracy": table["accuracy"].to_dict(), "latency_cpu": latency.reset_index().to_dict(orient="records"),
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(f"saved {OUT}/summary.json")
table[["accuracy"]].round(3).T
''')
md("""
* The adapter, label map, temperatures, GGUF files and this summary are in `results/notebook-nocat/`.
* To compare with the main recipe on the same machine, run `aerojev_qwen2.5-0.5b_lora.ipynb`. For every level of
  the speed-up ladder, see `results/ladder/ladder.md` and `docs/CPU_SPEEDUP_RESEARCH.md`.
* If the question list starts changing often, switch back to the main recipe, or to its shared-catalogue variant
  (`--layout catalogue --shared-catalogue`), which kept full accuracy at 3.6× less compute and still reads its
  questions.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
path = os.path.join(ROOT, "notebooks", f"{NAME}.ipynb")
nbf.write(nb, path)
print(f"wrote {path} ({len(cells)} cells)")

# A Markdown copy for reading on GitHub or in any editor: prose stays prose, code cells become ```python blocks.
from nbconvert import MarkdownExporter

body, _ = MarkdownExporter().from_notebook_node(nb)
note = (f"_Markdown copy of [`{NAME}.ipynb`]({NAME}.ipynb), generated by `scripts/build_notebook_nocat.py`. "
        "Edit the builder, not this file._\n\n")
md_path = path[: -len(".ipynb")] + ".md"
open(md_path, "w").write(note + body)
print(f"wrote {md_path}")
