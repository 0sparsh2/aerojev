"""Metrics over saved predictions (rows of {qid, type, keys, logits, gold})."""

import math
from collections import Counter, defaultdict

import numpy as np


def softmax(x, T=1.0):
    x = np.asarray(x, dtype=np.float64) / T
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def macro_f1(pred, gold):
    classes = sorted(set(gold) | set(pred))
    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(pred, gold) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred, gold) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred, gold) if p != c and g == c)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def balanced_acc(pred, gold):
    by = defaultdict(list)
    for p, g in zip(pred, gold):
        by[g].append(p == g)
    return float(np.mean([np.mean(v) for v in by.values()]))


def question_metrics(rows, T=1.0):
    """rows for one question id."""
    qtype = rows[0]["type"]
    probs = [softmax(r["logits"], T) for r in rows]
    gold_idx = [r["keys"].index(r["gold"]) for r in rows]
    pred_idx = [int(np.argmax(p)) for p in probs]
    pred = [r["keys"][i] for r, i in zip(rows, pred_idx)]
    gold = [r["gold"] for r in rows]
    conf = [float(p.max()) for p in probs]
    correct = [p == g for p, g in zip(pred, gold)]
    maj = Counter(gold).most_common(1)[0][1] / len(gold)
    m = {
        "n": len(rows), "type": qtype,
        "accuracy": float(np.mean(correct)), "majority_rate": maj,
        "macro_f1": macro_f1(pred, gold), "balanced_acc": balanced_acc(pred, gold),
        "ece": ece(conf, correct),
        "nll": float(np.mean([-math.log(max(p[g], 1e-12)) for p, g in zip(probs, gold_idx)])),
    }
    if qtype == "score":
        exp = [float(np.dot(p, np.arange(len(p)))) for p in probs]
        m["mae_expected"] = float(np.mean([abs(e - g) for e, g in zip(exp, gold_idx)]))
        m["within_1"] = float(np.mean([abs(pi - g) <= 1 for pi, g in zip(pred_idx, gold_idx)]))
    if qtype == "yesno":
        py = [float(p[0]) for p in probs]  # keys = ["yes", "no"]
        y = [g == "yes" for g in gold]
        m["auroc"] = auroc(py, y)
        m["brier"] = float(np.mean([(p - float(t)) ** 2 for p, t in zip(py, y)]))
    return m


def summarize(pred_rows, train_qids, T_by_type=None):
    T_by_type = T_by_type or {}
    by_q = defaultdict(list)
    for r in pred_rows:
        by_q[r["qid"]].append(r)
    per_q = {q: question_metrics(rows, T_by_type.get(rows[0]["type"], 1.0)) for q, rows in by_q.items()}
    core = [per_q[q] for q in train_qids if q in per_q]
    # record exact match on core questions
    by_rec = defaultdict(list)
    for r in pred_rows:
        if r["qid"] in train_qids:
            by_rec[r["rid"]].append(r["keys"][int(np.argmax(r["logits"]))] == r["gold"])
    all_conf, all_corr = [], []
    for r in pred_rows:
        if r["qid"] in train_qids:
            p = softmax(r["logits"], T_by_type.get(r["type"], 1.0))
            all_conf.append(float(p.max()))
            all_corr.append(r["keys"][int(np.argmax(p))] == r["gold"])
    return {
        "per_question": per_q,
        "core_mean_accuracy": float(np.mean([m["accuracy"] for m in core])) if core else float("nan"),
        "core_mean_macro_f1": float(np.mean([m["macro_f1"] for m in core])) if core else float("nan"),
        "core_mean_balanced_acc": float(np.mean([m["balanced_acc"] for m in core])) if core else float("nan"),
        "core_ece": ece(all_conf, all_corr) if all_conf else float("nan"),
        "record_exact_match": float(np.mean([all(v) for v in by_rec.values()])) if by_rec else float("nan"),
        "n_records": len(by_rec),
    }


def fit_temperatures(pred_rows, grid=None):
    """Per question type, the temperature minimising NLL (fit on validation)."""
    grid = grid or [round(x, 2) for x in np.concatenate([np.linspace(0.3, 1.0, 15), np.linspace(1.1, 5.0, 40)])]
    by_t = defaultdict(list)
    for r in pred_rows:
        by_t[r["type"]].append(r)
    out = {}
    for t, rows in by_t.items():
        best = None
        for T in grid:
            nll = np.mean([-math.log(max(softmax(r["logits"], T)[r["keys"].index(r["gold"])], 1e-12)) for r in rows])
            if best is None or nll < best[1]:
                best = (T, nll)
        out[t] = float(best[0])
    return out
