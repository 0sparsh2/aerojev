"""Build notebooks/aerojev_qwen2.5-0.5b_lora.ipynb: the whole pipeline for one base model.

  python scripts/build_notebook.py
"""

import os
import textwrap

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))


md("""
# AeroJev end to end: a Jev-style decision model that runs on CPU

**Base model:** Qwen2.5-0.5B-Instruct + LoRA &nbsp;·&nbsp; **Use case:** the supply-chain analytics team at an aerospace manufacturer

AeroJev reads a messy supply-chain record (a forwarded supplier email, an ERP exception row, a Teams message,
a quality notification, a risk alert) and answers a fixed set of typed questions with a probability for every
allowed answer. It never generates text. Each answer is read from the next-token logits of the label tokens,
so one forward pass over the record answers every question, and a 0.5B model is fast enough for a laptop CPU.

The recipe follows TypeSafe Jev and its open re-creations (featherless-ai/simple-jev and its RFDT trainer,
Meanblock/JEV-CPU, sting8k/qwen-jev-like, and Oumi's write-up):

| Step | What happens here |
|---|---|
| 0 | Setup: find the project, install pinned packages on Kaggle, pick the device |
| 1 | The decision schema: 9 trained questions + 3 held-out questions |
| 2 | The playbook that turns facts into labels |
| 3 | Synthetic data: facts first, then text in 7 formats, with a template-disjoint test split |
| 4 | Hand-written gold sets (out-of-distribution wording) |
| 5 | Prompt layout and label readout |
| 6 | Tree packing: one forward pass per record, checked against separate passes |
| 7 | Baselines: the untrained model and TF-IDF + logistic regression |
| 8 | LoRA decision training on label logits only |
| 9 | Evaluation: per question, confusions, option-order robustness, held-out questions |
| 10 | Calibration with per-type temperature scaling |
| 11 | Export: merge LoRA, convert to GGUF, quantize to Q8_0 and Q4_K_M |
| 12 | CPU latency: torch fp32 vs llama.cpp, two prompt layouts |
| 13 | Using the model on a new record |
| 14 | Summary |

On a CPU-only machine the notebook runs in **FAST** mode (a few dozen records per step) as a smoke test.
On a GPU (a Kaggle T4) it runs the full recipe.
""")

md("## 0. Setup")
code('''
import glob, json, math, os, random, shutil, subprocess, sys, time

ON_KAGGLE = os.path.exists("/kaggle/input")


def find_project_root():
    """The repo when run locally; on Kaggle, a copy of the attached code+data bundle."""
    if ON_KAGGLE:
        hits = glob.glob("/kaggle/input/**/aerojev/engine.py", recursive=True)
        if not hits:
            raise RuntimeError("attach the aerojev-bundle dataset to this notebook")
        dst = "/tmp/aerojev_project"
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), dst, dirs_exist_ok=True)
        return dst
    d = os.path.abspath(os.getcwd())
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "aerojev", "engine.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("run this notebook from inside the AeroJev repository")


ROOT = find_project_root()
os.chdir(ROOT)
sys.path.insert(0, ROOT)

if ON_KAGGLE:
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run(pip + ["transformers==5.20.2", "peft==0.21.2", "accelerate", "scikit-learn"], check=True)
    # the image ships torchao 0.10, which makes peft >= 0.21 refuse to build LoRA layers
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
print("project root:", ROOT)
''')

code('''
import numpy as np
import pandas as pd
import torch
import transformers
import peft

# CUDA trains the full recipe. Apple MPS is skipped on purpose: packed sequences change length every
# step and MPS recompiles per shape, which made it slower than the CPU in our tests.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAST = os.environ.get("AEROJEV_NB_FAST", "0" if DEVICE == "cuda" else "1") == "1"
MODEL = os.environ.get("AEROJEV_NB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LAYOUT = "catalogue"  # question catalogue first, KV-cached once per process (see section 5)
DO_EXPORT = os.environ.get("AEROJEV_NB_EXPORT", "1") == "1"
OUT = "/kaggle/working/out" if ON_KAGGLE else os.path.join(ROOT, "results", "notebook")
os.makedirs(OUT, exist_ok=True)

CFG = dict(
    n_train=10000, n_val=500, n_test=800,  # data v2 sizes
    epochs=1.2, lr=2e-4, lora_r=16, lora_alpha=32, lora_dropout=0.05, grad_accum=8, warmup=50,
    min_q=5, max_q=9, ordinal_weight=0.5, eval_every=150, val_limit=150,
    eval_limit=None, bench_records=40,
)
if FAST:
    CFG.update(n_train=48, n_val=24, n_test=24, epochs=1.0, grad_accum=4, warmup=2, eval_every=6,
               val_limit=12, eval_limit=24, bench_records=8)
AMP = torch.float16 if DEVICE == "cuda" else torch.float32  # T4: fp16 autocast; CPU: fp32
random.seed(0); torch.manual_seed(0)
print(f"device={DEVICE}  FAST={FAST}  model={MODEL}  layout={LAYOUT}")
print(f"torch {torch.__version__} · transformers {transformers.__version__} · peft {peft.__version__}")
print(CFG)
''')

md("""
## 1. The decision schema

Three question types, as in Jev:

* **choice**: pick one of N described options; labels `A`, `B`, `C`, ...
* **score**: ordered levels; labels `0` .. `N-1`
* **yesno**: labels `yes` / `no`

Nine questions are trained. Three more are **held out**: never trained on, used to measure whether
decision training keeps the model able to answer questions it has not seen.
""")
code('''
from aerojev import schema
from aerojev.prompt import labels_for, materialize

pd.set_option("display.max_colwidth", 90)
pd.DataFrame([{
    "id": q.id, "type": q.type, "labels": " ".join(labels_for(materialize(q)[0])),
    "question": q.text, "trained": not q.holdout,
} for q in schema.CATALOGUE])
''')

md("""
## 2. The playbook: from facts to labels

Labels are never read off text. Each record starts as a set of **facts** (event, commodity, days late, days of
stock, sole source, export marking, counterfeit signals, ...). A written playbook turns those facts into the
labels, and only then is text rendered that states every fact the playbook needs.

The central quantity is the **shortfall** = days late − days of stock:

* severity 4: line already down / AOG, or a defect escaped onto delivered aircraft
* severity 3: shortfall > 14 days, sole-source supplier in financial distress, counterfeit suspect, export violation
* severity 2: shortfall 1–14 days, a contained quality issue, end-of-life with < 6 months to last-time buy
* severity 1: buffer stock covers it, a commercial change, end-of-life with ≥ 6 months
* line-stop risk: AOG now, or stock runs out before supply resumes **and** that happens within 30 days

Two worked examples:
""")
code('''
from data.gen.policy import INDEFINITE, Scenario, derive_labels

late_forgings = Scenario(
    event="delivery_delay", commodity="castings_forgings", part=("Inconel 718 HPT disk forging", "718 disks"),
    supplier=("Halvorsen Precision Forge AS", "Halvorsen", "Kongsberg", "Norway", "europe"),
    program=("WB-950", "twin-aisle commercial airliner", False),
    days_late=42, coverage_days=21, sole_source=True, sole_mention="sole")
fake_chips = Scenario(
    event="quality_escape", sub="counterfeit", commodity="electronics", part=("rad-tolerant FPGA", "FPGAs"),
    supplier=("GlobalChip Surplus", "GlobalChip", "", "", "unknown"), program=("RQ-30", "military ISR UAS", True),
    days_late=0, coverage_days=21, counterfeit=True, export_mention="ear600", region_stated=False)
pd.DataFrame({"6-week slip, 3 weeks of stock, sole source": derive_labels(late_forgings),
              "counterfeit FPGAs, 3 weeks of genuine stock": derive_labels(fake_chips)})
''')

md("""
## 3. Synthetic data

Facts are sampled first, labels follow from the playbook, then text is rendered in one of seven formats
(supplier email, ERP exception record, Teams/Slack message, quality notification, risk alert, supplier portal
notice, and call notes). Each phrase bank is split by a stable hash: about 25% of every bank's wording, and the
whole *call notes* format, appear only in the test split. Data v2 adds phrasings for early deliveries,
equipment that fails its own inspection, supplier-caught defects, and more prompt-injection lines.
""")
code('''
from data.gen import policy
from data.gen.generate import build

policy.FEATURES["v2"] = True
plan = {"train": (CFG["n_train"], 101, 0.06), "val": (CFG["n_val"], 102, 0.06), "test_v2": (CFG["n_test"], 103, 0.06)}
sets = {}
for name, (n, seed, inject_p) in plan.items():
    t0 = time.time()
    sets[name] = build(n, "test" if name == "test_v2" else name, seed, inject_p=inject_p)
    print(f"{name:8s} {len(sets[name]):6d} records  ({time.time() - t0:.1f}s)")
train_text = {r["state"] for r in sets["train"]}
for name in ("val", "test_v2"):
    sets[name] = [r for r in sets[name] if r["state"] not in train_text]

# fixed evaluation files shared by every model in the project
for name in ("test", "injection"):
    sets[name] = [json.loads(l) for l in open(os.path.join(ROOT, "data/processed", f"{name}.jsonl"))]
pd.Series({k: len(v) for k, v in sets.items()}, name="records")
''')
code('''
def show(r, keys=("event_type", "commodity", "impact_severity", "line_stop_risk", "recommended_action", "delay_bucket")):
    print(f"[{r['format']}]\\n{r['state']}\\n-> " + ", ".join(f"{k}={r['labels'][k]}" for k in keys) + "\\n")

seen = set()
for r in sets["train"]:
    if r["format"] in ("email", "erp", "chat") and r["format"] not in seen:
        seen.add(r["format"]); show(r)
''')
code('''
import matplotlib.pyplot as plt
from collections import Counter

fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, q in zip(axes, ["event_type", "impact_severity", "recommended_action"]):
    c = Counter(r["labels"][q] for r in sets["train"])
    keys = sorted(c) if q == "impact_severity" else [k for k, _ in c.most_common()]
    ax.barh(keys, [c[k] for k in keys], color="#4a6fa5")
    ax.invert_yaxis(); ax.set_title(q); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.show()
''')
md("A leakage check: how many 8-word sequences of each evaluation set also occur in the training text.")
code('''
import re

def grams(t, n=8):
    w = re.findall(r"[a-z]+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

G = set().union(*(grams(r["state"]) for r in sets["train"]))
overlap = {name: np.mean([len(grams(r["state"]) & G) / max(1, len(grams(r["state"]))) for r in rows])
           for name, rows in sets.items() if name != "train"}
pd.Series(overlap, name="share of 8-grams also in train").round(3)
''')

md("""
## 4. Hand-written gold sets

`gold` (80 records) and `gold2` (48 records, written blind before any data iteration) use wording and formats
the generator never produces: email threads, MRP pipe rows, tab-separated pastes, compliance hotline reports,
AOG desk notes. Their labels come from the same playbook, applied to the facts each record states.
""")
code('''
from data.gold.build_gold import GOLD, write
from data.gold.gold2_records import GOLD2

write(GOLD, os.path.join(OUT, "data", "gold.jsonl"), "gold")
write(GOLD2, os.path.join(OUT, "data", "gold2.jsonl"), "gold2")
for name in ("gold", "gold2"):
    sets[name] = [json.loads(l) for l in open(os.path.join(OUT, "data", f"{name}.jsonl"))]
show(sets["gold"][0])
''')

md("""
## 5. Prompt layout and label readout

Every question ends with a prefill such as `{"answer": "` (branch layout) or `{"event_type": "` (catalogue layout).
The model's next-token logits, restricted to that question's label tokens and softmaxed, are the answer.

* **branch layout**: system prompt + record, then one branch per question holding the full question and options.
* **catalogue layout** (used here): the question catalogue comes *before* the record. With a fixed schema its KV
  cache is computed once per process, so each record costs only its own tokens plus ~5 tokens per question.
""")
code('''
from transformers import AutoTokenizer
from aerojev.prompt import label_token_ids, render_readout, render_record, render_static, think_block

tok = AutoTokenizer.from_pretrained(MODEL)
think = think_block(tok)
qds = [materialize(q)[0] for q in schema.CATALOGUE]
static = render_static(qds)
record = render_record(sets["gold"][0]["state"], think)
n = lambda t: len(tok.encode(t, add_special_tokens=False))
print(static[:1200], "...\\n")
print(record + render_readout(qds[0]) + "▌")
print(f"\\ntokens: static catalogue {n(static)} (computed once) · this record {n(record)} · readout {n(render_readout(qds[0]))} per question")
for q in qds[:3] + qds[6:7]:
    print(f"{q['id']:18s} labels {labels_for(q)} -> token ids {label_token_ids(tok, labels_for(q))}")
''')

md("""
## 6. Tree packing: every question in one forward pass

The record is laid out once and each question branch follows it. A 4D boolean mask lets each branch attend to
the record and itself only, and position ids restart for every branch, so the packed pass equals running each
question separately. The check below runs both on the untrained model.
""")
code('''
from transformers import AutoModelForCausalLM
from aerojev.prompt import pack

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE).eval()
r = sets["gold"][0]
asked = qds[:4]
p = pack(tok, r["state"], asked, think, layout=LAYOUT, catalogue=qds)
with torch.no_grad():
    packed = base(input_ids=p["input_ids"][None].to(DEVICE), position_ids=p["position_ids"][None].to(DEVICE),
                  attention_mask=p["mask"][None, None].to(DEVICE),
                  logits_to_keep=torch.tensor(p["readouts"], device=DEVICE)).logits[0].float().cpu()
    worst = 0.0
    for i, q in enumerate(asked):
        single = pack(tok, r["state"], [q], think, layout=LAYOUT, catalogue=qds)
        out = base(input_ids=single["input_ids"][None].to(DEVICE)).logits[0, -1].float().cpu()
        worst = max(worst, (packed[i, p["label_ids"][i]] - out[single["label_ids"][0]]).abs().max().item())
print(f"packed sequence {len(p['input_ids'])} tokens for {len(asked)} questions; max logit difference vs separate passes: {worst:.1e}")

tail = p["mask"][-60:, -60:].numpy()
plt.figure(figsize=(4, 4)); plt.imshow(tail, cmap="Greys", interpolation="nearest")
plt.title("attention mask, last 60 tokens (dark = may attend)"); plt.xticks([]); plt.yticks([]); plt.show()
''')

md("""
## 7. Baselines

**Untrained model, logit readout.** The JEV-CPU / SemIf setting: the pretrained model with the same prompt.
**TF-IDF + logistic regression.** A classical classifier per question, trained on the same synthetic records.
It cannot answer a question it has no labels for.
""")
code('''
from aerojev.engine import DecisionEngine
from aerojev.metrics import fit_temperatures, summarize
from eval.evaluate import permutation_consistency, run_set

CORE = [q.id for q in schema.TRAIN_QUESTIONS]
HELD = [q.id for q in schema.HOLDOUT_QUESTIONS]
EVAL_SETS = ["val", "test", "test_v2", "gold", "gold2", "injection"]
lim = lambda rows: rows[: CFG["eval_limit"]] if CFG["eval_limit"] else rows


def evaluate(engine, names=EVAL_SETS):
    preds, out = {}, {}
    for name in names:
        preds[name], spr = run_set(engine, lim(sets[name]), schema.CATALOGUE)
        m = summarize(preds[name], CORE)
        out[name] = {"core acc": m["core_mean_accuracy"], "macro-F1": m["core_mean_macro_f1"],
                     "exact": m["record_exact_match"], "ECE": m["core_ece"], "s/record": spr}
    return preds, pd.DataFrame(out).T


zs_engine = DecisionEngine(tok=tok, model=base, device=DEVICE, layout=LAYOUT)
zs_preds, zs_table = evaluate(zs_engine, ["test", "gold", "gold2"])
zs_table.round(3)
''')
code('''
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300000)
texts = [r["state"] for r in sets["train"]]
Xtr = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
clfs = {q: LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, [r["labels"][q] for r in sets["train"]]) for q in CORE
        if len({r["labels"][q] for r in sets["train"]}) > 1}
tfidf = {}
for name in ("test", "gold", "gold2"):
    rows = lim(sets[name])
    X = hstack([word.transform([r["state"] for r in rows]), char.transform([r["state"] for r in rows])]).tocsr()
    acc = [np.mean([p == r["labels"][q] for p, r in zip(clf.predict(X), rows)]) for q, clf in clfs.items()]
    tfidf[name] = float(np.mean(acc))
pd.Series(tfidf, name="TF-IDF + LR core accuracy").round(3)
''')

md("""
## 8. LoRA decision training

The loss is the RFDT recipe: at each question's readout position, cross-entropy over **only that question's label
tokens**, plus an ordinal penalty (expected distance to the gold level) on score questions. There is no language-
modelling loss on the prompt and no JSON to learn. Each training record is one tree-packed sequence with a random
subset of 5–9 questions, shuffled option letters, paraphrased questions and occasionally fewer options, so the
model has to read the options rather than memorise positions.
""")
code('''
import gc
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from train.train import build_example, quick_eval

del zs_engine, base
gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa")
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"], bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
model.to(DEVICE).train()


def record_loss(model, p, ordinal_weight=CFG["ordinal_weight"]):
    """Mean over questions of CE on the label-token logits at each readout (+ ordinal term for scores)."""
    out = model(input_ids=p["input_ids"][None].to(DEVICE), position_ids=p["position_ids"][None].to(DEVICE),
                attention_mask=p["mask"][None, None].to(DEVICE),
                logits_to_keep=torch.tensor(p["readouts"], device=DEVICE)).logits[0].float()
    losses = []
    for i, (lab, g, t) in enumerate(zip(p["label_ids"], p["golds"], p["types"])):
        logp = F.log_softmax(out[i, lab], -1)
        loss = -logp[g]
        if t == "score":
            dist = (torch.arange(len(lab), device=logp.device) - g).abs().float() / (len(lab) - 1)
            loss = loss + ordinal_weight * (logp.exp() * dist).sum()
        losses.append(loss)
    return torch.stack(losses).mean()

example = build_example(tok, think, sets["train"][0], random.Random(0), CFG["min_q"], CFG["max_q"], LAYOUT)
print(f"one training record: {len(example['input_ids'])} tokens, {len(example['readouts'])} questions asked")
''')
code('''
rng = random.Random(0)
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
total_steps = math.ceil(len(sets["train"]) * CFG["epochs"] / CFG["grad_accum"])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CFG["warmup"]) *
                                          0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
scaler = torch.amp.GradScaler(DEVICE, enabled=AMP == torch.float16)
val_rows = sets["val"][: CFG["val_limit"]]
log, best, step, seen, run_loss, order, t0 = [], -1.0, 0, 0, 0.0, [], time.time()
log.append({"step": 0, "val_acc": quick_eval(model, tok, think, val_rows, DEVICE, AMP, LAYOUT)})
print(f"step 0  val acc {log[-1]['val_acc']:.3f}  ({total_steps} optimizer steps planned)")
while step < total_steps:
    if not order:
        order = list(range(len(sets["train"]))); rng.shuffle(order)
    p = build_example(tok, think, sets["train"][order.pop()], rng, CFG["min_q"], CFG["max_q"], LAYOUT)
    with torch.autocast(device_type=DEVICE, dtype=AMP, enabled=AMP != torch.float32):
        loss = record_loss(model, p)
    if not torch.isfinite(loss):
        continue
    scaler.scale(loss / CFG["grad_accum"]).backward()
    run_loss += loss.item(); seen += 1
    if seen % CFG["grad_accum"]:
        continue
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
    log.append({"step": step, "loss": run_loss / CFG["grad_accum"]}); run_loss = 0.0
    if step % CFG["eval_every"] == 0 or step == total_steps:
        acc = quick_eval(model, tok, think, val_rows, DEVICE, AMP, LAYOUT)
        log.append({"step": step, "val_acc": acc})
        print(f"step {step:5d}/{total_steps}  val acc {acc:.3f}  {seen / (time.time() - t0):.2f} records/s")
        if acc > best:
            best = acc; model.save_pretrained(os.path.join(OUT, "adapter"))
print(f"done in {(time.time() - t0) / 60:.1f} min, best val acc {best:.3f}")
json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"))
''')
code('''
L = pd.DataFrame(log)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
loss = L.dropna(subset=["loss"]) if "loss" in L else L.iloc[:0]
ax[0].plot(loss["step"], loss["loss"].rolling(10, min_periods=1).mean(), color="#4a6fa5")
ax[0].set_title("training loss (rolling mean of 10 steps)"); ax[0].set_xlabel("optimizer step")
va = L.dropna(subset=["val_acc"])
ax[1].plot(va["step"], va["val_acc"], marker="o", color="#4a6fa5")
ax[1].set_title("validation accuracy (core questions)"); ax[1].set_xlabel("optimizer step")
plt.tight_layout(); plt.show()
''')

md("""
## 9. Evaluation

The best adapter (by validation accuracy) is merged into the base weights and scored on every set:

* `val`: same templates as training (data v2)
* `test`: the project's fixed test set (held-out templates, test-only *call notes* format)
* `test_v2`: held-out templates of the v2 generator
* `gold`, `gold2`: hand-written, out-of-distribution wording
* `injection`: every record carries a prompt-injection line
""")
code('''
from peft import PeftModel

ft = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa")
ft = PeftModel.from_pretrained(ft, os.path.join(OUT, "adapter")).merge_and_unload().to(DEVICE).eval()
del model, opt
gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

ft_engine = DecisionEngine(tok=tok, model=ft, device=DEVICE, layout=LAYOUT)
ft_preds, ft_table = evaluate(ft_engine)
compare = pd.DataFrame({"untrained": zs_table["core acc"], "TF-IDF + LR": pd.Series(tfidf),
                        "LoRA decision-trained": ft_table["core acc"]})
display(ft_table.round(3)); display(compare.round(3))
''')
code('''
def per_question(preds):
    return {q: v["accuracy"] for q, v in summarize(preds, CORE)["per_question"].items()}

pq = pd.DataFrame({"untrained": per_question(zs_preds["gold"]), "LoRA": per_question(ft_preds["gold"])})
pq["change"] = pq["LoRA"] - pq["untrained"]
pq.loc[CORE + HELD].round(3)
''')
code('''
def confusion(preds, qid):
    rows = [p for p in preds if p["qid"] == qid]
    keys = rows[0]["keys"]
    m = pd.DataFrame(0, index=keys, columns=keys)
    for p in rows:
        m.loc[p["gold"], p["keys"][int(np.argmax(p["logits"]))]] += 1
    m.index.name, m.columns.name = "gold", "predicted"
    return m.loc[(m.sum(1) > 0) | (m.sum(0) > 0)]

display(confusion(ft_preds["gold"], "impact_severity"))
display(confusion(ft_preds["gold"], "recommended_action"))
''')
md("Option-order robustness: every choice question is asked again with its options shuffled.")
code('''
perm_preds, _ = run_set(ft_engine, lim(sets["gold"]), schema.CATALOGUE, shuffle_seed=1)
pd.Series(permutation_consistency(ft_preds["gold"], perm_preds)).round(3)
''')

md("""
## 10. Calibration

Raw confidences from a fine-tuned model are often over- or under-confident. One temperature per question type
(choice, score, yes/no) is fitted on the validation set by minimising negative log-likelihood, then applied to
every set.
""")
code('''
from aerojev.metrics import softmax

temps = fit_temperatures([p for p in ft_preds["val"] if p["qid"] in CORE])
ece = {name: {"raw": summarize(ft_preds[name], CORE)["core_ece"],
              "calibrated": summarize(ft_preds[name], CORE, temps)["core_ece"]} for name in ("test", "gold", "gold2")}
print("temperatures:", temps); display(pd.DataFrame(ece).T.round(4))

conf, corr = [], []
for name in ("test", "gold", "gold2"):
    for p in ft_preds[name]:
        if p["qid"] in CORE:
            pr = softmax(p["logits"], temps.get(p["type"], 1.0))
            conf.append(pr.max()); corr.append(p["keys"][int(pr.argmax())] == p["gold"])
conf, corr = np.array(conf), np.array(corr, float)
bins = np.linspace(0, 1, 11); mid, acc, cnt = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf > lo) & (conf <= hi)
    if m.any():
        mid.append(conf[m].mean()); acc.append(corr[m].mean()); cnt.append(m.sum())
plt.figure(figsize=(4.2, 4.2)); plt.plot([0, 1], [0, 1], color="#bbb", lw=1)
plt.scatter(mid, acc, s=[20 + 200 * c / max(cnt) for c in cnt], color="#4a6fa5")
plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title("reliability after temperature scaling"); plt.show()
json.dump(temps, open(os.path.join(OUT, "temperatures.json"), "w"))
''')

md("""
## 11. Export: merge, GGUF, quantize

llama.cpp's converter turns the merged weights into GGUF (f16 and Q8_0), and `llama-quantize` makes Q4_K_M.
These files run on any CPU through llama.cpp or `llama-cpp-python`.
""")
code('''
GGUF = {}
if DO_EXPORT:
    merged_dir, llama = "/tmp/aerojev_merged", "/tmp/llama.cpp"
    ft.save_pretrained(merged_dir); tok.save_pretrained(merged_dir)  # fp32; the converter writes f16 / q8_0
    if not os.path.exists(f"{llama}/build/bin/llama-quantize"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", llama], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{llama}/gguf-py"], check=True)
        subprocess.run(f"cmake -S {llama} -B {llama}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_METAL=OFF && "
                       f"cmake --build {llama}/build --target llama-quantize -j 4", shell=True, check=True,
                       stdout=subprocess.DEVNULL)
    gdir = os.path.join(OUT, "gguf"); os.makedirs(gdir, exist_ok=True)
    name = "aerojev-" + MODEL.split("/")[1].lower()
    f16 = f"/tmp/{name}-f16.gguf"
    for outtype, path in [("f16", f16), ("q8_0", f"{gdir}/{name}-q8_0.gguf")]:
        subprocess.run([sys.executable, f"{llama}/convert_hf_to_gguf.py", merged_dir, "--outtype", outtype,
                        "--outfile", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([f"{llama}/build/bin/llama-quantize", f16, f"{gdir}/{name}-q4_k_m.gguf", "Q4_K_M"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    GGUF = {"Q8_0": f"{gdir}/{name}-q8_0.gguf", "Q4_K_M": f"{gdir}/{name}-q4_k_m.gguf"}
    os.remove(f16); shutil.rmtree(merged_dir, ignore_errors=True)
pd.Series({k: f"{os.path.getsize(v) / 1e6:.0f} MB" for k, v in GGUF.items()}, name="file size", dtype=object)
''')

md("""
## 12. CPU latency

The point of the project: decisions on a CPU. Every configuration answers the same 9 core questions for the same
gold records, one record at a time, in a warm process, with up to 4 threads.

* torch fp32, **branch** layout: the record and every question's full text are processed per record
* torch fp32, **catalogue** layout: the question catalogue's KV cache is reused, so only the record is processed
* llama.cpp **Q8_0 / Q4_K_M**, catalogue layout

Accuracy on the same records is reported beside each, to show what quantization costs.
""")
code('''
if ON_KAGGLE and GGUF:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--prefer-binary", "llama-cpp-python==0.3.16",
                    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"], check=False)
from aerojev.engine import GGUFEngine

THREADS = min(4, os.cpu_count() or 4)
torch.set_num_threads(THREADS)
bench_rows = sets["gold"][: CFG["bench_records"]]
core_q = [materialize(q)[0] for q in schema.TRAIN_QUESTIONS]
ft_cpu = ft.to("cpu")


def bench(label, engine):
    engine.readout_logits(bench_rows[0]["state"], core_q)  # warm-up
    lat, hit = [], []
    for r in bench_rows:
        t0 = time.time(); lg = engine.readout_logits(r["state"], core_q); lat.append(time.time() - t0)
        for q, l in zip(core_q, lg):
            keys = ["yes", "no"] if q["type"] == "yesno" else [o["key"] for o in q["options"]]
            hit.append(keys[int(torch.as_tensor(l).argmax())] == r["labels"][q["id"]])
    lat.sort()
    return {"configuration": label, "p50 s/record": lat[len(lat) // 2], "ms/decision": 1000 * np.mean(lat) / len(core_q),
            "gold acc (these records)": float(np.mean(hit))}

rows = [bench("torch fp32 · branch", DecisionEngine(tok=tok, model=ft_cpu, device="cpu", layout="branch")),
        bench("torch fp32 · catalogue", DecisionEngine(tok=tok, model=ft_cpu, device="cpu", layout="catalogue"))]
for q, path in GGUF.items():
    try:
        rows.append(bench(f"llama.cpp {q} · catalogue", GGUFEngine(path, MODEL, n_threads=THREADS, layout="catalogue")))
    except Exception as e:  # llama-cpp-python missing or too old for this file
        print(f"skipping {q}: {e}")
latency = pd.DataFrame(rows).set_index("configuration")
print(f"CPU: {os.cpu_count()} cores, {THREADS} threads"); latency.round(3)
''')
code('''
ax = latency["ms/decision"].plot.barh(figsize=(7, 0.6 + 0.5 * len(latency)), color="#4a6fa5")
ax.invert_yaxis(); ax.set_xlabel("ms per decision (9 decisions per record)")
for i, v in enumerate(latency["ms/decision"]):
    ax.text(v, i, f" {v:.0f}", va="center")
plt.tight_layout(); plt.show()
''')

md("## 13. Using the model on a new record")
code('''
engine = DecisionEngine(tok=tok, model=ft_cpu, device="cpu", layout=LAYOUT, temperatures=temps)
note = """FW: Brennerwerk - servo valve PO 4523300117
Buyer note: Brennerwerk says the valves slip 5 weeks (their test stand is down). We have 12 days of valves.
They are the only qualified source. VTX-12 program, ECCN 9A610.x."""
answers = engine.decide(note, schema.TRAIN_QUESTIONS)
pd.DataFrame({q: {"answer": a["answer"], "confidence": round(a["confidence"], 3)} for q, a in answers.items()}).T
''')

md("## 14. Summary")
code('''
summary = {
    "model": MODEL, "layout": LAYOUT, "fast_mode": FAST, "config": CFG, "temperatures": temps,
    "accuracy": {k: {c: float(v) for c, v in row.items()} for k, row in compare.T.to_dict().items()},
    "latency_cpu": latency.reset_index().to_dict(orient="records"), "cpu_cores": os.cpu_count(),
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary["accuracy"], indent=1, default=float))
''')
md("""
**Where to go next**

* Swap `MODEL` for `Qwen/Qwen3-0.6B` (the project's deployment candidate) or `Qwen/Qwen3-1.7B` (more accurate, ~3× slower on CPU).
* Serve the GGUF with `python server.py --gguf <file> --tokenizer <model id> --layout catalogue`.
* Add questions to `aerojev/schema.py` and facts to `data/gen/policy.py`; the data, training and evaluation code
  pick them up unchanged.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
path = os.path.join(ROOT, "notebooks", "aerojev_qwen2.5-0.5b_lora.ipynb")
nbf.write(nb, path)
print(f"wrote {path} ({len(cells)} cells)")
