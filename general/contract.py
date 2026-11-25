"""One example (general/compose.py) -> what each system reads.

  to_jev(example)       the question object TypeSafe's Jev API takes (same mapping as eval/jev.py)
  markers(example)      arm A input: every option behind an [OPT] marker; a shared scorer reads the marker states
  letters(example)      arm B input: lettered options and a [MASK] answer slot read by the masked-word head

Keeping these in one place guarantees that every system sees the same question text, options and order.
"""

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OPT = "[OPT]"


def to_jev(example):
    q = example["question"]
    if q["type"] == "noul":
        return {"type": "noul", "instructions": q["instructions"]}
    if q["type"] == "score":
        return {"type": "score", "instructions": q["instructions"], "criteria": [o["description"] for o in q["options"]]}
    return {"type": "choice", "instructions": q["instructions"],
            "criteria": {o["key"]: o["description"] for o in q["options"]}}


def jev_request(examples):
    """Several questions about the same record -> one Jev request body (state + questions by id)."""
    states = {e["state"] for e in examples}
    assert len(states) == 1, "a Jev request carries one record"
    return {"state": examples[0]["state"],
            "questions": {e["question"]["id"] + (f"#{e['rule']}" if e["pair"] else ""): to_jev(e) for e in examples}}


def _opt_text(o, qtype=None):
    """How an option is shown: score levels by description (their keys are just 0..k-1), keys-only options and
    options without a description by key, everything else as key: description."""
    if qtype == "score":
        return o["description"]
    if not o["description"] or o["description"] == o["key"].replace("_", " "):
        return o["key"]
    return f"{o['key']}: {o['description']}"


def markers(example, mask_token=None):
    q = example["question"]
    opts = " ".join(f"{OPT} {_opt_text(o, q['type'])}" for o in q["options"])
    return f"Question: {q['instructions']}\nOptions: {opts}\nRecord: {example['state']}"


def letters(example, mask_token="[MASK]"):
    q = example["question"]
    if len(q["options"]) > len(LETTERS):
        raise ValueError("more than 26 options: split into grouped passes")
    opts = "\n".join(f"{LETTERS[i]}) {_opt_text(o, q['type'])}" for i, o in enumerate(q["options"]))
    return f"Question: {q['instructions']}\n{opts}\nRecord: {example['state']}\nAnswer: {mask_token}"


def label_index(example):
    return [o["key"] for o in example["question"]["options"]].index(example["label"])
