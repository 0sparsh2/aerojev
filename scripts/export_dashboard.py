"""Export results/summary.json into the CSV datasets the benchmark dashboard reads.

  python scripts/report.py && python scripts/export_dashboard.py
writes results/dashboard/{systems,per_question,training,reliability}.csv
"""

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/dashboard")
CORE = ["event_type", "commodity", "impact_severity", "line_stop_risk", "recommended_action",
        "export_controlled", "counterfeit_suspect", "sole_source", "delay_bucket"]


def family(system, engine):
    if "LoRA" in system or system.startswith("ft-"):
        return "Decision-trained (this project)"
    if engine == "sklearn" or "logistic regression" in system:
        return "Classical, supervised"
    if "generate JSON" in system:
        return "LLM generating JSON"
    if "hosted" in engine or "Jev" in system:
        return "Hosted Jev API"
    return "Zero-shot logit readout"


def host_key(h):
    return "mac" if h.startswith("mac") else ("x86" if h.startswith("x86") else "api")


def main():
    s = json.load(open(os.path.join(ROOT, "results/summary.json")))
    llm, cls, gen = s["llm"], s["classical"], s["generative"]
    os.makedirs(OUT, exist_ok=True)

    rows = {}
    for t in s["tradeoff"]:
        rid = f"{t['system']} | {t['engine']} | {t['layout']}"
        r = rows.setdefault(rid, {"id": rid, "system": t["system"], "family": family(t["system"], t["engine"]),
                                  "engine": t["engine"], "layout": t["layout"], "acc_source": t["acc_key"] or "",
                                  "acc_is_exact": int(bool(t["acc_exact"]))})
        for k in ("gold", "gold2", "test", "injection"):
            r[f"{k}_acc"] = t["acc"].get(k)
        for h, v in t["hosts"].items():
            hk = host_key(h)
            r[f"{hk}_p50_s"] = v["p50_s"]
            r[f"{hk}_ms_per_decision"] = v["ms_per_decision"]
        r["peak_rss_mb"] = t["peak_rss_mb"]
        r["file_mb"] = t["file_mb"]
        key = t["acc_key"] or ""
        perm = (llm.get(key) or {}).get("permutation") or {}
        r["option_shuffle_agreement"] = perm.get("choice_agreement_with_canonical")

    cols = ["id", "system", "family", "engine", "layout", "gold_acc", "gold2_acc", "test_acc", "injection_acc",
            "option_shuffle_agreement", "mac_p50_s", "mac_ms_per_decision", "x86_p50_s", "x86_ms_per_decision",
            "api_p50_s", "api_ms_per_decision", "peak_rss_mb", "file_mb", "acc_source", "acc_is_exact"]
    with open(os.path.join(OUT, "systems.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: -(r.get("gold_acc") or 0)):
            w.writerow({c: ("" if r.get(c) is None else (round(r[c], 4) if isinstance(r.get(c), float) else r[c]))
                        for c in cols})

    # per-question gold accuracy for the systems the story compares
    pick = {"zs-qwen3-0.6b": "Qwen3-0.6B zero-shot", "ft-qwen3-0.6b-cat": "Qwen3-0.6B decision-trained",
            "ft-qwen3-0.6b-cat-v2": "Qwen3-0.6B decision-trained (data v2)", "jev-playbook": "Jev API (+ playbook)"}
    with open(os.path.join(OUT, "per_question.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "system", "gold_acc"])
        for key, name in pick.items():
            g = (llm.get(key) or {}).get("sets", {}).get("gold")
            if g:
                for q in CORE:
                    w.writerow([q, name, round(g["per_q"][q], 4)])
        if "tfidf" in cls:
            for q in CORE:
                w.writerow([q, "TF-IDF + logistic regression", round(cls["tfidf"]["gold"]["per_q"][q], 4)])

    with open(os.path.join(OUT, "training.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "val_acc"])
        for run, t in sorted(s["training"].items()):
            for p in t["curve"]:
                if "val_acc" in p:
                    w.writerow([run, p["step"], round(p["val_acc"], 4)])

    rel_src = None
    for cand in ["ft-qwen3-0.6b-cat-v2", "ft-qwen3-0.6b-cat", "ft-qwen3-0.6b"]:
        for base in ["kaggle/outputs/train-v2", "kaggle/outputs/train-cat", "kaggle/outputs/train"]:
            p = os.path.join(ROOT, base, "out/results", cand, "analysis.json")
            if os.path.exists(p) and rel_src is None:
                rel_src = (cand, p)
    with open(os.path.join(OUT, "reliability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "bin", "n", "confidence", "accuracy"])
        if rel_src:
            for b in json.load(open(rel_src[1]))["reliability"]:
                w.writerow([rel_src[0], b["bin"], b["n"], b["confidence"], b["accuracy"]])
    for name in ["systems", "per_question", "training", "reliability"]:
        n = sum(1 for _ in open(os.path.join(OUT, f"{name}.csv"))) - 1
        print(f"{name}.csv: {n} rows")


if __name__ == "__main__":
    main()
