"""Evaluate TypeSafe's hosted Jev on the same sets, questions and metrics as eval.evaluate.

  python -m eval.jev --variant plain --out results/jev-plain
  python -m eval.jev --variant playbook --out results/jev-playbook
  python -m eval.jev --variant playbook_export --out results/jev-playbook-export

plain            canonical question text and options, exactly what the local models see
playbook         plain + the written playbook rules (data/gen/policy.py) in the instructions
                 of the four questions whose labels depend on them
playbook_export  playbook + the playbook's definition of export-controlled (ITAR or EAR
                 600-series only; EAR99 and unmarked items are not)

The API key is read from .env (jev_api_key = ...). Every successful response is
cached in <out>/cache.jsonl, so a re-run only pays for requests that have not
succeeded yet. Jev's per-option probabilities are stored as log-probabilities in
the "logits" field, so aerojev.metrics treats them exactly like label logits.
"""

import argparse
import hashlib
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from aerojev import schema
from aerojev.metrics import fit_temperatures, summarize
from aerojev.prompt import materialize
from eval.evaluate import ALLQ, CORE, load, permutation_consistency

URL = "https://api.typesafe.ai/v1/systemone"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLAYBOOK = {
    "impact_severity": (
        "Use our playbook. Shortfall = days late minus days of stock on hand (no shortfall if not late). "
        "4 if the line is already down or an aircraft is on ground (AOG), or a defect escaped onto delivered aircraft. "
        "0 if there is no issue. 1 for a purely commercial change. End-of-life: 1 if 6 or more months remain "
        "before the last-time-buy deadline, otherwise 2. Every other event starts from the shortfall: "
        "1 if stock covers the delay, 2 for a shortfall of 1-14 days, 3 for more than 14 days; then raise it to "
        "at least 2 for any quality escape (3 if parts are counterfeit or suspect), at least 2 for a lapsed "
        "certification (3 for an export control or sanctions violation), and at least 3 when a sole-source "
        "supplier is in financial distress."
    ),
    "line_stop_risk": (
        "Use our playbook. Yes if the line is already down or an aircraft is on ground (AOG). Also yes if the "
        "delivery is late, stock on hand runs out before the late delivery arrives (days of stock < days late), "
        "and stock runs out within 30 days (days of stock <= 30). Otherwise no."
    ),
    "recommended_action": (
        "Use our playbook and apply the first rule that matches: "
        "1) line down, AOG or escape onto delivered aircraft -> escalate_leadership; "
        "2) counterfeit or suspect parts, or nonconforming stock caught at our own receiving inspection -> "
        "quarantine_inventory; "
        "3) export control or sanctions issue -> involve_trade_compliance; "
        "4) supplier financial distress -> activate_alternate_source; "
        "5) force majeure with more than 90 days or indefinite recovery -> activate_alternate_source; "
        "6) end-of-life / obsolescence -> last_time_buy; "
        "7) price or commercial term change -> renegotiate_commercial; "
        "8) quality problem caught at the supplier, lapsed certification, or capacity / raw material problem "
        "with a shortfall -> request_recovery_plan; "
        "9) delivery delay, logistics disruption or shorter force majeure with a shortfall -> expedite; "
        "10) otherwise -> monitor. "
        "Shortfall = days late minus days of stock on hand."
    ),
    "delay_bucket": (
        "Measure days late against the need date (1 week = 7 days). On time, early or no delivery affected is 0; "
        "unknown or indefinite recovery is 4."
    ),
}
PLAYBOOK_EXPORT = {
    **PLAYBOOK,
    "export_controlled": (
        "Use our playbook. Yes only if the record marks the items or technical data as ITAR / USML, or as "
        "EAR 600-series (an ECCN such as 9A610 or 3A611). EAR99 items are not export-controlled, and if no "
        "export classification is stated the answer is no: do not assume aerospace parts are controlled."
    ),
}
RULES = {"plain": {}, "playbook": PLAYBOOK, "playbook_export": PLAYBOOK_EXPORT}


def api_key():
    for line in open(os.path.join(ROOT, ".env")):
        k, _, v = line.partition("=")
        if k.strip().lower() == "jev_api_key":
            return v.strip().strip("\"'")
    raise SystemExit("jev_api_key not found in .env")


def to_jev(qd, variant):
    text = qd["text"]
    if qd["id"] in RULES[variant]:
        text = f"{text}\n{RULES[variant][qd['id']]}"
    if qd["type"] == "yesno":
        return {"type": "noul", "instructions": text}
    if qd["type"] == "score":
        return {"type": "score", "instructions": text, "criteria": [o["description"] for o in qd["options"]]}
    return {"type": "choice", "instructions": text, "criteria": {o["key"]: o["description"] for o in qd["options"]}}


def to_logprobs(ans, qd, keys):
    if qd["type"] == "yesno":
        probs = [ans["noul"], 1.0 - ans["noul"]]
    elif qd["type"] == "score":
        p = ans["probabilities"]
        probs = [p.get(str(i), p.get(i, 0.0)) for i in range(len(keys))]
    else:
        probs = [ans["probabilities"].get(k, 0.0) for k in keys]
    return [round(math.log(max(x, 1e-6)), 4) for x in probs]


class Client:
    def __init__(self, model, cache_path, workers):
        self.model, self.workers = model, workers
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key()}"
        self.cache, self.lock = {}, threading.Lock()
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            for l in open(cache_path):
                c = json.loads(l)
                self.cache[c["key"]] = c
        self.input_tokens, self.latencies, self.retries = 0, [], 0

    def ask(self, state, questions):
        body = {"model": self.model, "state": state, "questions": questions}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]["resp"]
        err = ""
        for attempt in range(8):
            wait, t0 = 0.0, time.time()
            try:
                r = self.session.post(URL, json=body, timeout=120)
            except requests.RequestException as e:
                err = repr(e)
            else:
                if r.status_code == 200:
                    resp = r.json()
                    with self.lock:
                        self.latencies.append(time.time() - t0)
                        self.input_tokens += resp.get("usage", {}).get("input_tokens", 0)
                        self.cache[key] = {"key": key, "resp": resp}
                        with open(self.cache_path, "a") as f:
                            f.write(json.dumps(self.cache[key]) + "\n")
                    return resp
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                err = f"HTTP {r.status_code}"
                try:
                    wait = float(r.headers.get("retry-after") or 0)
                except ValueError:
                    pass
            with self.lock:
                self.retries += 1
            time.sleep(max(wait, min(60, 2 ** attempt)))
        raise RuntimeError(f"gave up after retries: {err}")


def run_set(client, rows, variant, shuffle_seed=None):
    def one(r):
        rng = random.Random(f"{shuffle_seed}-{r['id']}") if shuffle_seed is not None else None
        qs = [materialize(q, rng=rng, shuffle=shuffle_seed is not None) for q in ALLQ]
        resp = client.ask(r["state"], {qd["id"]: to_jev(qd, variant) for qd, _ in qs})
        return [{"rid": r["id"], "qid": qd["id"], "type": qd["type"], "keys": keys,
                 "logits": to_logprobs(resp["answers"][qd["id"]], qd, keys), "gold": r["labels"][qd["id"]],
                 "format": r.get("format"), "jev_model": resp.get("model")} for qd, keys in qs]

    preds, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(client.workers) as ex:
        for rec in ex.map(one, rows):
            preds += rec
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(rows)}  {time.time() - t0:.0f}s  retries={client.retries}", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(RULES), required=True)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--sets", default="val,test,gold,injection")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-perm", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    client = Client(args.model, os.path.join(args.out, "cache.jsonl"), args.workers)

    results = {"model": args.model, "variant": args.variant, "layout": f"hosted API ({args.variant})", "sets": {}}
    all_preds = {}
    for name in args.sets.split(","):
        rows = load(os.path.join(args.data, f"{name}.jsonl"), args.limit)
        print(f"[{name}] {len(rows)} records", flush=True)
        preds = run_set(client, rows, args.variant)
        all_preds[name] = preds
        with open(os.path.join(args.out, f"preds_{name}.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        results["sets"][name] = {}

    temps = fit_temperatures([p for p in all_preds["val"] if p["qid"] in CORE]) if "val" in all_preds else {}
    results["temperatures"] = temps
    for name, preds in all_preds.items():
        results["sets"][name]["raw"] = summarize(preds, CORE)
        if temps:
            results["sets"][name]["calibrated"] = summarize(preds, CORE, temps)
        holdout = [p for p in preds if p["qid"] not in CORE]
        results["sets"][name]["holdout_questions"] = summarize(holdout, [q.id for q in schema.HOLDOUT_QUESTIONS])["per_question"]

    if "gold" in all_preds and not args.no_perm:
        rows = load(os.path.join(args.data, "gold.jsonl"), args.limit)
        perm = run_set(client, rows, args.variant, shuffle_seed=1)
        results["gold_permutation"] = permutation_consistency(all_preds["gold"], perm)

    lat = sorted(client.latencies)
    results["api"] = {"served_by": sorted({p["jev_model"] for ps in all_preds.values() for p in ps}),
                      "billed_input_tokens_this_run": client.input_tokens, "retries": client.retries,
                      "latency_p50_s": lat[len(lat) // 2] if lat else None, "workers": args.workers}
    for name in results["sets"]:
        results["sets"][name]["sec_per_record"] = results["api"]["latency_p50_s"] or 0.0
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    for name, r in results["sets"].items():
        raw = r["raw"]
        print(f"== {name}: core acc {raw['core_mean_accuracy']:.3f}  macroF1 {raw['core_mean_macro_f1']:.3f}  "
              f"exact {raw['record_exact_match']:.3f}  ECE {raw['core_ece']:.3f}")
        for q, m in raw["per_question"].items():
            print(f"   {q:20s} acc={m['accuracy']:.3f} (maj {m['majority_rate']:.2f}) f1={m['macro_f1']:.3f}")
        for q, m in r["holdout_questions"].items():
            print(f"   {q:20s} acc={m['accuracy']:.3f} (held out)")
    if "gold_permutation" in results:
        print("permutation:", results["gold_permutation"])
    print("api:", results["api"])


if __name__ == "__main__":
    main()
