"""Classical per-question classifiers trained on the same synthetic train split.

  python -m eval.classical --out results/classical

  tfidf  : word 1-2 gram + char 3-5 gram TF-IDF -> logistic regression (one per question)
  minilm : all-MiniLM-L6-v2 sentence embedding -> logistic regression

These need labeled examples for every question, so they cannot answer the
held-out questions at all; that gap is part of the comparison.
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from aerojev import schema
from aerojev.metrics import summarize
from aerojev.prompt import YESNO_LABELS

CORE = [q.id for q in schema.TRAIN_QUESTIONS]


def load(path):
    return [json.loads(l) for l in open(path)]


def keys_for(qid):
    q = schema.BY_ID[qid]
    return YESNO_LABELS if q.type == "yesno" else [o.key for o in q.options]


def bench(featurize, models, rows, name, out_path, threads):
    """Single-record latency: featurize one record and answer all 9 core questions."""
    from aerojev import benchrec
    featurize([rows[0]["state"]], fit=False)  # warm-up
    lat = []
    for r in rows:
        t0 = time.time()
        X = featurize([r["state"]], fit=False)
        for clf in models.values():
            clf.predict_proba(X)
        lat.append(time.time() - t0)
    benchrec.append(out_path, benchrec.record(f"{name} + logistic regression", "sklearn", lat, len(models), threads,
                                              acc_key=f"classical:{name}"))


def run(featurize, train, sets, name, out, bench_out=None, bench_rows=None):
    t0 = time.time()
    Xtr = featurize([r["state"] for r in train], fit=True)
    models = {}
    for qid in CORE:
        y = [r["labels"][qid] for r in train]
        clf = LogisticRegression(max_iter=2000, C=4.0)
        clf.fit(Xtr, y)
        models[qid] = clf
    fit_s = time.time() - t0
    res = {"fit_seconds": fit_s, "sets": {}}
    for sname, rows in sets.items():
        t1 = time.time()
        X = featurize([r["state"] for r in rows], fit=False)
        preds = []
        for qid, clf in models.items():
            keys = keys_for(qid)
            P = clf.predict_proba(X)
            cls = list(clf.classes_)
            for r, p in zip(rows, P):
                full = np.full(len(keys), 1e-6)
                for c, v in zip(cls, p):
                    full[keys.index(c)] = v
                preds.append({"rid": r["id"], "qid": qid, "type": schema.BY_ID[qid].type, "keys": keys,
                              "logits": np.log(full / full.sum()).round(4).tolist(), "gold": r["labels"][qid]})
        res["sets"][sname] = {"raw": summarize(preds, CORE), "sec_per_record": (time.time() - t1) / len(rows)}
        print(f"{name:7s} {sname:10s} core acc {res['sets'][sname]['raw']['core_mean_accuracy']:.3f}  "
              f"macroF1 {res['sets'][sname]['raw']['core_mean_macro_f1']:.3f}  "
              f"exact {res['sets'][sname]['raw']['record_exact_match']:.3f}")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"{name}.json"), "w"), indent=1)
    if bench_out:
        bench(featurize, models, bench_rows, name, bench_out, threads=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/classical")
    ap.add_argument("--skip-minilm", action="store_true")
    ap.add_argument("--only", choices=["tfidf", "minilm"])
    ap.add_argument("--bench-out", help="also write single-record CPU latency lines here")
    args = ap.parse_args()
    bench_rows = load(os.path.join(args.data, "gold.jsonl"))[:40]
    train = load(os.path.join(args.data, "train.jsonl"))
    sets = {s: load(os.path.join(args.data, f"{s}.jsonl")) for s in ["test", "gold", "gold2", "injection"]}

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300000)

    def tfidf(texts, fit):
        if fit:
            return hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
        return hstack([word.transform(texts), char.transform(texts)]).tocsr()

    if args.only in (None, "tfidf"):
        run(tfidf, train, sets, "tfidf", args.out, args.bench_out, bench_rows)

    if not args.skip_minilm and args.only in (None, "minilm"):
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        def emb(texts, fit):
            return st.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

        run(emb, train, sets, "minilm", args.out, args.bench_out, bench_rows)


if __name__ == "__main__":
    main()
