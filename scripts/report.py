"""Aggregate every metrics file into results/summary.json and results/REPORT.md.

  python scripts/report.py
"""

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = ["event_type", "commodity", "impact_severity", "line_stop_risk", "recommended_action",
        "export_controlled", "counterfeit_suspect", "sole_source", "delay_bucket"]
HOLD = ["supplier_region", "mentions_new_date", "military_program"]

NAMES = {
    "zs-qwen2.5-0.5b": ("Qwen2.5-0.5B-Instruct", "zero-shot readout", "branch"),
    "zs-qwen3-0.6b": ("Qwen3-0.6B", "zero-shot readout", "branch"),
    "zs-qwen3.5-0.8b": ("Qwen3.5-0.8B", "zero-shot readout", "branch (unpacked)"),
    "zs-qwen3-1.7b": ("Qwen3-1.7B", "zero-shot readout", "branch"),
    "ft-qwen2.5-0.5b": ("Qwen2.5-0.5B-Instruct", "LoRA decision-trained", "branch"),
    "ft-qwen3-0.6b": ("Qwen3-0.6B", "LoRA decision-trained", "branch"),
    "ft-qwen2.5-0.5b-cat": ("Qwen2.5-0.5B-Instruct", "LoRA decision-trained", "catalogue"),
    "ft-qwen3-0.6b-cat": ("Qwen3-0.6B", "LoRA decision-trained", "catalogue"),
    "jev-plain": ("Jev (TypeSafe API)", "hosted, zero-shot", "questions as-is"),
    "jev-playbook": ("Jev (TypeSafe API)", "hosted, zero-shot", "+ playbook rules"),
    "jev-playbook-export": ("Jev (TypeSafe API)", "hosted, zero-shot", "+ playbook + export definition"),
}


def load_llm_results():
    out = {}
    # eval.evaluate / eval.jev runs: local results/<run>/ first (e.g. results/ft-qwen3-0.6b-cat-v2), then runs
    # downloaded from remote GPU jobs; separate gold2-only runs (<run>-gold2) are merged into their system's row
    local = [f for f in sorted(glob.glob(os.path.join(ROOT, "results/*/metrics.json")))
             if not os.path.basename(os.path.dirname(f)).startswith(("gen-", "classical", "notebook", "dashboard"))]
    files = sorted(glob.glob(os.path.join(ROOT, "kaggle/outputs/*/out/results/*/metrics.json"))) + local
    plain = [f for f in files if not f.endswith("-gold2/metrics.json")]
    gold2 = [f for f in files if f.endswith("-gold2/metrics.json")]  # merged last, once every row exists
    for f in plain + gold2:
        key = os.path.basename(os.path.dirname(f))
        m = json.load(open(f))
        if key.endswith("-gold2"):  # a separate gold2-only eval run: merge into the system's row
            key = key[: -len("-gold2")]
            if key in out:
                for s, r in m["sets"].items():
                    out[key]["sets"][s] = _set_row(r)
                continue
        model, method, layout = NAMES.get(key, (m.get("model"), "gguf" if m.get("gguf") else "?", m.get("layout")))
        row = {"key": key, "model": model, "method": method, "layout": layout,
               "temperatures": m.get("temperatures"), "permutation": m.get("gold_permutation"), "sets": {},
               "api": m.get("api")}
        for s, r in m["sets"].items():
            row["sets"][s] = _set_row(r)
        out[key] = row
    return out


def _set_row(r):
    raw, cal = r["raw"], r.get("calibrated", r["raw"])
    return {
        "acc": raw["core_mean_accuracy"], "macro_f1": raw["core_mean_macro_f1"],
        "bal_acc": raw["core_mean_balanced_acc"], "exact": raw["record_exact_match"],
        "ece_raw": raw["core_ece"], "ece_cal": cal["core_ece"], "sec_per_record_gpu": r["sec_per_record"],
        "per_q": {q: v["accuracy"] for q, v in raw["per_question"].items()},
        "per_q_f1": {q: v["macro_f1"] for q, v in raw["per_question"].items()},
        "aux": {q: {k: v[k] for k in ("auroc", "mae_expected", "within_1") if k in v}
                for q, v in raw["per_question"].items()},
    }


def load_latency():
    """Unified latency records (eval.latency_suite) from every host."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results/latency_*.jsonl")) +
                    glob.glob(os.path.join(ROOT, "kaggle/outputs/*/out/latency_*.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def resolve_accuracy(acc_key, llm, cls, gen):
    """acc_key: 'llm:<key>' | 'classical:<name>' | 'gen:<dir>:<variant>', '|' separates fallbacks.
    Returns ({set: acc}, key used, exact?)."""
    for i, k in enumerate((acc_key or "").split("|")):
        kind, _, rest = k.partition(":")
        if kind == "llm" and rest in llm:
            return {s: v["acc"] for s, v in llm[rest]["sets"].items()}, rest, i == 0
        if kind == "classical" and rest in cls:
            return {s: v["acc"] for s, v in cls[rest].items()}, rest, i == 0
        if kind == "gen":
            d, _, variant = rest.partition(":")
            if d in gen:
                return {"gold": gen[d][variant]["core_mean_accuracy"]}, rest, i == 0
    return {}, None, False


def load_classical():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results/classical/*.json")):
        name = os.path.splitext(os.path.basename(f))[0]
        m = json.load(open(f))
        out[name] = {s: {"acc": r["raw"]["core_mean_accuracy"], "macro_f1": r["raw"]["core_mean_macro_f1"],
                         "exact": r["raw"]["record_exact_match"], "ece_raw": r["raw"]["core_ece"],
                         "per_q": {q: v["accuracy"] for q, v in r["raw"]["per_question"].items()}}
                     for s, r in m["sets"].items()}
    return out


def load_generative():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results/gen-*/metrics.json")):
        out[os.path.basename(os.path.dirname(f))] = json.load(open(f))
    return out


def load_bench():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results/bench_cpu*.jsonl")) + \
            glob.glob(os.path.join(ROOT, "kaggle/outputs/*/out/bench_cpu*.jsonl")):
        host = "kaggle-x86" if "kaggle/outputs" in f else "mac-m4"
        for l in open(f):
            r = json.loads(l)
            r["host"] = r.get("host", host)
            rows.append(r)
    return rows


def load_training():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "kaggle/outputs/*/out/runs/*/summary.json")):
        run = os.path.basename(os.path.dirname(f))
        s = json.load(open(f))
        logf = os.path.join(os.path.dirname(f), "train_log.jsonl")
        curve = [json.loads(l) for l in open(logf)] if os.path.exists(logf) else []
        out[run] = {"best_val_acc": s["best_val_acc"], "steps": s["steps"], "records": s["records"],
                    "minutes": s["elapsed_s"] / 60, "curve": curve}
    return out


def fmt(x):
    return "–" if x is None else f"{x:.3f}"


def build_tradeoff(latency, llm, cls, gen):
    """One row per system (system label + engine + layout), latency per host, accuracy joined by acc_key."""
    rows = {}
    for r in latency:
        if r.get("questions") == "one" or r.get("threads", 4) not in (1, 4) or \
                (r.get("threads") == 1 and r.get("engine") != "sklearn"):
            continue  # thread-scaling / single-question rows are reported separately
        key = (r["system"], r["engine"], r.get("layout", "–"))
        acc, used, exact = resolve_accuracy(r.get("acc_key"), llm, cls, gen)
        row = rows.setdefault(key, {"system": r["system"], "engine": r["engine"], "layout": r.get("layout", "–"),
                                    "acc": acc, "acc_key": used, "acc_exact": exact, "hosts": {},
                                    "peak_rss_mb": None, "file_mb": r.get("file_mb")})
        row["hosts"][r["host"].split(" (")[0]] = {"p50_s": r["p50_s"], "p95_s": r["p95_s"],
                                                  "ms_per_decision": r["ms_per_decision"], "host": r["host"]}
        if r["host"].startswith("mac"):
            row["peak_rss_mb"] = r["peak_rss_mb"]
    for key, m in llm.items():  # hosted APIs report their own client-side p50
        if m.get("api") and m["api"].get("latency_p50_s"):
            p50 = m["api"]["latency_p50_s"]
            rows[(m["model"], "hosted API", m["layout"])] = {
                "system": f"{m['model']} ({m['api'].get('served_by', ['?'])[0]})", "engine": "hosted API (GPU, network)",
                "layout": m["layout"], "acc": {s: v["acc"] for s, v in m["sets"].items()}, "acc_key": key,
                "acc_exact": True, "hosts": {"api": {"p50_s": p50, "ms_per_decision": round(1000 * p50 / 12, 1)}},
                "peak_rss_mb": None, "file_mb": None}
    return sorted(rows.values(), key=lambda t: -(t["acc"].get("gold") or 0))


def main():
    llm, cls, gen, bench, train = load_llm_results(), load_classical(), load_generative(), load_bench(), load_training()
    latency = load_latency()
    tradeoff = build_tradeoff(latency, llm, cls, gen)
    summary = {"llm": llm, "classical": cls, "generative": gen, "bench_cpu": bench, "training": train,
               "latency": latency, "tradeoff": tradeoff}
    json.dump(summary, open(os.path.join(ROOT, "results/summary.json"), "w"), indent=1)

    L = ["# AeroJev results", "", "Core = the 9 trained questions; accuracy is the mean over questions.", ""]
    if tradeoff:
        hosts = sorted({h for t in tradeoff for h in t["hosts"]})
        L += ["## Accuracy vs latency", "",
              "Same protocol everywhere: 40 gold records, one at a time, warm process, 9 core decisions per record, "
              "4 threads (classical models single-threaded). Hosted API latency is the client-measured p50 round trip. "
              "`*` = accuracy of the fp32 model (quantized accuracy not measured separately).", "",
              "| System | Engine | Layout | Gold acc | Gold2 acc | Test acc | " +
              " | ".join(f"{h} p50 s/record | {h} ms/decision" for h in hosts) + " | Peak RSS MB | File MB |",
              "|---|---|---|---|---|---|" + "---|---|" * len(hosts) + "---|---|"]
        for t in tradeoff:
            star = "" if t["acc_exact"] else "*"
            cells = []
            for h in hosts:
                hh = t["hosts"].get(h)
                cells += [f"{hh['p50_s']:.3f}" if hh else "–", f"{hh['ms_per_decision']}" if hh else "–"]
            L.append(f"| {t['system']} | {t['engine']} | {t['layout']} | {fmt(t['acc'].get('gold'))}{star} | "
                     f"{fmt(t['acc'].get('gold2'))}{star if t['acc'].get('gold2') is not None else ''} | "
                     f"{fmt(t['acc'].get('test'))}{star if t['acc'].get('test') is not None else ''} | "
                     + " | ".join(cells) + f" | {t['peak_rss_mb'] or '–'} | {t['file_mb'] or '–'} |")
        L.append("")
    L += ["## Accuracy (core questions)", "",
          "| System | Method | Layout | Gold acc | Gold macro-F1 | Test acc | Injection acc | Gold exact | ECE raw→cal (gold) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, r in llm.items():
        g, t, i = r["sets"].get("gold", {}), r["sets"].get("test", {}), r["sets"].get("injection", {})
        L.append(f"| {r['model']} | {r['method']} | {r['layout']} | {fmt(g.get('acc'))} | {fmt(g.get('macro_f1'))} | "
                 f"{fmt(t.get('acc'))} | {fmt(i.get('acc'))} | {fmt(g.get('exact'))} | "
                 f"{fmt(g.get('ece_raw'))}→{fmt(g.get('ece_cal'))} |")
    for k, r in cls.items():
        L.append(f"| {k} + logistic regression | supervised classical | – | {fmt(r['gold']['acc'])} | "
                 f"{fmt(r['gold']['macro_f1'])} | {fmt(r['test']['acc'])} | {fmt(r['injection']['acc'])} | "
                 f"{fmt(r['gold']['exact'])} | {fmt(r['gold']['ece_raw'])} |")
    for k, r in gen.items():
        for v in ("free", "schema"):
            L.append(f"| {k} | generate JSON ({v}) | – | {fmt(r[v]['core_mean_accuracy'])} | "
                     f"{fmt(r[v]['core_mean_macro_f1'])} | – | – | {fmt(r[v]['record_exact_match'])} | – |")
    L += ["", "## Gold accuracy per question", "", "| System | " + " | ".join(CORE) + " |",
          "|---|" + "---|" * len(CORE)]
    for k, r in llm.items():
        g = r["sets"].get("gold")
        if g:
            L.append(f"| {r['model']} ({r['method']}, {r['layout']}) | " + " | ".join(f"{g['per_q'].get(q, 0):.2f}" for q in CORE) + " |")
    for k, r in cls.items():
        L.append(f"| {k} | " + " | ".join(f"{r['gold']['per_q'].get(q, 0):.2f}" for q in CORE) + " |")
    L += ["", "## Held-out questions (never trained on) — gold accuracy", "",
          "| System | " + " | ".join(HOLD) + " |", "|---|" + "---|" * len(HOLD)]
    for k, r in llm.items():
        g = r["sets"].get("gold")
        if g:
            L.append(f"| {r['model']} ({r['method']}, {r['layout']}) | " + " | ".join(f"{g['per_q'].get(q, 0):.2f}" for q in HOLD) + " |")
    L += ["", "## Option-order robustness (gold, choice questions shuffled)", "",
          "| System | agreement with canonical order | accuracy shuffled |", "|---|---|---|"]
    for k, r in llm.items():
        if r.get("permutation"):
            L.append(f"| {r['model']} ({r['method']}, {r['layout']}) | {r['permutation']['choice_agreement_with_canonical']:.3f} | "
                     f"{r['permutation']['choice_accuracy_shuffled']:.3f} |")
    if bench:
        L += ["", "## CPU latency (12 decisions per record unless noted)", "",
              "| Host | Label | Engine | Layout | Threads | Questions | s/record p50 | ms/decision | tokens/record | Peak RSS MB | File MB |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in bench:
            L.append(f"| {b['host']} | {b['label']} | {b['engine']} | {b.get('layout', 'branch')} | {b['threads']} | {b['n_questions']} | "
                     f"{b['p50_s']:.3f} | {b['ms_per_decision']} | {b['mean_tokens']} | {b['peak_rss_mb']} | {b.get('file_mb') or '–'} |")
    if train:
        L += ["", "## Training runs (Kaggle T4, LoRA r=16)", "", "| Run | best val acc | steps | records | minutes |",
              "|---|---|---|---|---|"]
        for k, t in train.items():
            L.append(f"| {k} | {t['best_val_acc']:.3f} | {t['steps']} | {t['records']} | {t['minutes']:.0f} |")
    open(os.path.join(ROOT, "results/REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
