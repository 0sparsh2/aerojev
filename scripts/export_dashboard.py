"""Export results/summary.json into the CSV datasets the benchmark dashboard reads.

  python scripts/report.py && python scripts/export_dashboard.py
writes results/dashboard/{systems,per_question,training,reliability}.csv
"""

import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/dashboard")
CORE = ["event_type", "commodity", "impact_severity", "line_stop_risk", "recommended_action",
        "export_controlled", "counterfeit_suspect", "sole_source", "delay_bucket"]
BEST = "ft-qwen3-0.6b-cat-v2"  # the model the per-question and calibration views describe

RUN_LABELS = {  # training runs, in the words the dashboard uses
    "qwen3-0.6b": "Qwen3-0.6B · branch · data v1",
    "qwen3-0.6b-cat": "Qwen3-0.6B · catalogue · data v1",
    "qwen3-0.6b-cat-v2": "Qwen3-0.6B · catalogue · data v2",
    "qwen2.5-0.5b": "Qwen2.5-0.5B · branch · data v1",
    "qwen2.5-0.5b-cat": "Qwen2.5-0.5B · catalogue · data v1",
    "notebook": "Qwen2.5-0.5B · catalogue · data v2 (notebook)",
    "qwen2.5-0.5b-cat-v2-top8": "Qwen2.5-0.5B · LoRA top 8 blocks",
    "qwen2.5-0.5b-cat-v2-4800": "Qwen2.5-0.5B · 4,800 records",
    "qwen2.5-0.5b-cat-v2-top8-4800": "Qwen2.5-0.5B · top 8 blocks, 4,800 records",
    "qwen3-1.7b-cat-v2": "Qwen3-1.7B · catalogue · data v2",
}


def family(system, engine):
    if "LoRA" in system:
        return "Decision-trained (this project)"
    if engine == "sklearn":
        return "Classical, supervised"
    if "generate JSON" in system:
        return "LLM generating JSON"
    if "hosted" in engine:
        return "Hosted Jev API"
    return "Zero-shot logit readout"


def base_model(system):
    m = re.search(r"Qwen[\d.]+-[\d.]+B", system)
    return m.group(0) if m else ("Jev" if "Jev" in system else system.split(" +")[0])


def precision(system, engine):
    for q in ("Q8_0", "Q4_K_M"):
        if q in system:
            return q
    return "fp32" if engine == "torch fp32" else "–"


def host_key(h):
    return "mac" if h.startswith("mac") else ("x86" if h.startswith("x86") else "api")


def r4(x):
    return "" if x is None else (round(x, 4) if isinstance(x, float) else x)


def main():
    s = json.load(open(os.path.join(ROOT, "results/summary.json")))
    llm, cls = s["llm"], s["classical"]
    os.makedirs(OUT, exist_ok=True)

    # ---- systems: one row per system (label + engine + layout), latency per host, accuracy by acc_key
    rows = []
    for t in s["tradeoff"]:
        sysname = t["system"]
        r = {"id": f"{sysname} | {t['engine']} | {t['layout']}", "system": sysname,
             "family": family(sysname, t["engine"]), "base_model": base_model(sysname),
             "precision": precision(sysname, t["engine"]), "engine": t["engine"], "layout": t["layout"],
             "data_version": "v2" if " v2 " in sysname else ("v1" if "LoRA" in sysname else ""),
             "acc_source": t["acc_key"] or "", "acc_is_exact": int(bool(t["acc_exact"]))}
        for k in ("gold", "gold2", "test", "injection"):
            r[f"{k}_acc"] = t["acc"].get(k)
        for h, v in t["hosts"].items():
            r[f"{host_key(h)}_p50_s"] = v["p50_s"]
            r[f"{host_key(h)}_ms_per_decision"] = v["ms_per_decision"]
        r["peak_rss_mb"], r["file_mb"] = t["peak_rss_mb"], t["file_mb"]
        perm = (llm.get(t["acc_key"] or "") or {}).get("permutation") or {}
        r["option_shuffle_agreement"] = perm.get("choice_agreement_with_canonical")
        rows.append(r)
    cols = ["id", "system", "family", "base_model", "precision", "engine", "layout", "data_version",
            "gold_acc", "gold2_acc", "test_acc", "injection_acc", "option_shuffle_agreement",
            "mac_p50_s", "mac_ms_per_decision", "x86_p50_s", "x86_ms_per_decision", "api_p50_s", "api_ms_per_decision",
            "peak_rss_mb", "file_mb", "acc_source", "acc_is_exact"]
    with open(os.path.join(OUT, "systems.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: -(r.get("gold_acc") or 0)):
            w.writerow({c: r4(r.get(c)) for c in cols})

    # ---- per-question gold accuracy: untrained vs trained vs references
    pick = {"zs-qwen3-0.6b": "Untrained Qwen3-0.6B", BEST: "Decision-trained Qwen3-0.6B",
            "jev-playbook-export": "Jev API (best variant)"}
    with open(os.path.join(OUT, "per_question.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "system", "gold_acc"])
        for key, name in pick.items():
            g = (llm.get(key) or {}).get("sets", {}).get("gold")
            if g:
                for q in CORE:
                    w.writerow([q, name, round(g["per_q"][q], 4)])
        for q in CORE:
            w.writerow([q, "TF-IDF + logistic regression", round(cls["tfidf"]["gold"]["per_q"][q], 4)])

    # ---- training curves in records seen (optimizer steps x records per step)
    with open(os.path.join(OUT, "training.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "label", "records", "val_acc"])
        for sumf in sorted(glob.glob(os.path.join(ROOT, "kaggle/outputs/*/out/runs/*/summary.json"))):
            run = os.path.basename(os.path.dirname(sumf))
            accum = json.load(open(sumf))["args"]["grad_accum"]
            for line in open(os.path.join(os.path.dirname(sumf), "train_log.jsonl")):
                p = json.loads(line)
                if "val_acc" in p:
                    w.writerow([run, RUN_LABELS.get(run, run), p["step"] * accum, round(p["val_acc"], 4)])
        nb = os.path.join(ROOT, "kaggle/outputs/notebook-exec/out/notebook")
        if os.path.exists(os.path.join(nb, "train_log.json")):
            accum = json.load(open(os.path.join(nb, "summary.json")))["config"]["grad_accum"]
            for p in json.load(open(os.path.join(nb, "train_log.json"))):
                if "val_acc" in p and p["step"] > 0:
                    w.writerow(["notebook", RUN_LABELS["notebook"], p["step"] * accum, round(p["val_acc"], 4)])

    # ---- calibration of the best model (after per-type temperature scaling)
    with open(os.path.join(OUT, "reliability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "bin", "decisions", "confidence", "accuracy"])
        for p in glob.glob(os.path.join(ROOT, f"kaggle/outputs/*/out/results/{BEST}/analysis.json")):
            for b in json.load(open(p))["reliability"]:
                w.writerow([BEST, b["bin"], b["n"], b["confidence"], b["accuracy"]])

    for name in ["systems", "per_question", "training", "reliability"]:
        with open(os.path.join(OUT, f"{name}.csv")) as fh:
            n = sum(1 for _ in fh) - 1
        print(f"{name}.csv: {n} rows")


if __name__ == "__main__":
    main()
