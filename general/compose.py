"""Turn domain specs into examples in Jev's input contract (docs/GENERAL_DECISION_PLAN.md, sections 2 and 6.1).

One example is one (record, question) pair:

  {"id", "domain", "split", "rid", "state", "format", "injected",
   "question": {"id", "type", "instructions", "options": [{"key", "description"}, ...]},
   "label", "rule", "rule_in_text", "pair", "facts"}

Score questions use keys "0".."k-1" (lowest first), like the aerospace schema; noul (yes/no) has options yes / no.

What teaches generality:
  rule text             with probability p_rules the instructions state the default rule; a variant rule is always
                        stated (otherwise its label could not be known)
  counterfactual pairs  the same record asked under two rules that give different labels; both go to the same split
                        and share a "pair" id, so the rule-following rate can be measured on them
  option variation      choice options are shuffled; sometimes keys are opaque ("option_3", the description carries
                        the meaning), sometimes descriptions are dropped, sometimes wrong options are left out
  wording               paraphrased questions and hash-split phrase banks (general/render.py)
"""

import random

from . import render


def _options(rng, q, keep, p_opaque=0.15, p_keys_only=0.15, p_subset=0.15):
    """-> (options as shown, {original key: shown key}). keep: original keys that must stay (the labels).
    Score and noul options keep their order."""
    if q.type == "noul":
        return [{"key": "yes", "description": "yes"}, {"key": "no", "description": "no"}], {"yes": "yes", "no": "no"}
    if q.type == "score":
        return ([{"key": str(i), "description": o.description} for i, o in enumerate(q.options)],
                {k: str(i) for i, k in enumerate(q.keys())})
    opts = [{"key": o.key, "description": o.description} for o in q.options]
    if len(opts) >= 5 and rng.random() < p_subset:
        others = [o for o in opts if o["key"] not in keep]
        drop = {o["key"] for o in rng.sample(others, min(len(others), rng.choice([1, 2])))}
        opts = [o for o in opts if o["key"] not in drop]
    rng.shuffle(opts)
    keymap = {o["key"]: o["key"] for o in opts}
    r = rng.random()
    if r < p_opaque:
        keymap = {o["key"]: f"option_{i + 1}" for i, o in enumerate(opts)}
        opts = [{"key": keymap[o["key"]], "description": o["description"]} for o in opts]
    elif r < p_opaque + p_keys_only:
        opts = [{"key": o["key"], "description": o["key"].replace("_", " ")} for o in opts]
    return opts, keymap


def _examples(rng, q, facts, rule_names, meta, split, with_rule):
    """One example per rule name, all with the same wording and options (a pair differs only in its rule)."""
    wording = render.ask(rng, q, split)
    labels = {r: q.rules[r].label(facts) for r in rule_names}
    options, keymap = _options(rng, q, set(labels.values()))
    out = []
    for r in rule_names:
        instructions = wording + ("\nRules: " + q.rules[r].text if with_rule else "")
        out.append({**meta, "question": {"id": q.id, "type": q.type, "instructions": instructions,
                                         "options": options},
                    "label": keymap[labels[r]], "rule": r, "rule_in_text": with_rule})
    return out


def record_examples(rng, dom, split, rid, p_rules=0.5, p_pair=0.5, facts=None, written=None):
    """All examples for one sampled record: one per question, plus counterfactual partners.
    facts / written: a record written elsewhere (the NIM prose route): its facts and (text, format). By default the
    facts are sampled here and the text comes from general/render.py."""
    facts = facts or dom.sample(rng)
    needed = {n for q in dom.questions for n in q.needs}
    if written:
        (state, fmt), injected = written, False
    else:
        state, fmt, stated, injected = render.render(rng, dom, facts, needed, split)
    meta = {"domain": dom.name, "split": split, "rid": rid, "state": state, "format": fmt, "injected": injected,
            "facts": {k: v for k, v in facts.items() if not k.startswith("_")}}
    out = []
    for q in dom.questions:
        variants = [v for v in q.rules if v != "default"]
        if variants and rng.random() < p_pair:
            v = rng.choice(variants)
            if q.rules[v].label(facts) != q.rules["default"].label(facts):
                out += [{**e, "pair": f"{rid}:{q.id}"} for e in _examples(rng, q, facts, ["default", v], meta, split,
                                                                           True)]
            else:  # the variant agrees here: ask it alone, still with its rule stated
                out += [{**e, "pair": None} for e in _examples(rng, q, facts, [v], meta, split, True)]
            continue
        out += [{**e, "pair": None} for e in _examples(rng, q, facts, ["default"], meta, split,
                                                       rng.random() < p_rules)]
    for i, e in enumerate(out):
        e["id"] = f"{rid}:{i}"
    return out


def build_domain(dom, n_records, split, seed, **kw):
    rng = random.Random(f"{dom.name}|{split}|{seed}")
    out = []
    for i in range(n_records):
        out += record_examples(rng, dom, split, f"{dom.name}-{split}-{i:05d}", **kw)
    return out
