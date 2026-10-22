"""Turn hand-written gold records into labeled JSONL via the shared playbook.

  python -m data.gold.build_gold
"""

import json
import os

from data.gen.policy import Scenario, derive_labels
from data.gold.gold_records import GOLD

REGION_SUPPLIER = {"north_america": "north_america", "europe": "europe", "asia_pacific": "asia_pacific"}


def to_scenario(g):
    reg = g.get("reg", "unknown")
    s = Scenario(
        event=g["ev"], commodity=g["com"], part=("", ""), supplier=("", "", "", "", reg),
        program=("", "", g.get("mil", False)), sub=g.get("sub", ""),
        days_late=g.get("d", 0), coverage_days=g.get("c", -1), aog=g.get("aog", False),
        sole_source=g.get("sole", False), sole_mention="sole" if g.get("sole") else "none",
        export_mention=g.get("exp", "none"), counterfeit=g.get("cf", False), ltb_months=g.get("ltb", 0),
        region_stated=reg != "unknown", new_date_given=g.get("nd", False),
    )
    return s


def write(records, out, split):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for g in records:
            labels = derive_labels(to_scenario(g))
            f.write(json.dumps({"id": g["id"], "split": split, "format": "handwritten", "state": g["text"],
                                "labels": labels, "meta": {k: v for k, v in g.items() if k != "text"}}) + "\n")
    print(f"wrote {len(records)} {split} records -> {out}")


def main():
    from data.gold.gold2_records import GOLD2
    write(GOLD, "data/processed/gold.jsonl", "gold")
    write(GOLD2, "data/processed/gold2.jsonl", "gold2")


if __name__ == "__main__":
    main()
