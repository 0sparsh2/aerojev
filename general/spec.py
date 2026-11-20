"""Domain specs for general decision data: facts -> labels (rules written as code) -> text.

A domain is a Python module in general/domains/ that defines DOMAIN = Domain(...). The same three-step order as the
aerospace generator (data/gen/) holds for every domain:

  1. sample facts       Fact.sample, steered away from rule thresholds (Fact.avoid)
  2. derive labels      Rule.label(facts) -> option key; a Question can carry several rules (counterfactual variants
                        of the same policy), each with its own plain-language text
  3. render text        general/render.py; every fact a label needs is stated in the text

Labels never come from text. Domain.validate() checks a spec before any data is built: every option is reachable
under every rule, no option dominates, sampling is reproducible from a seed, every needed fact renders, and each
counterfactual variant disagrees with the default often enough to be useful.
"""

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

TYPES = ("choice", "score", "noul")  # Jev's question types (noul = yes/no); box (images) is out of scope


@dataclass
class Fact:
    """One latent fact of a record.

    kind:    "int" | "float" | "bool" | "cat"
    values:  cat: the categories; int / float: (low, high)
    p:       bool: P(True); cat: optional weights (same length as values)
    log:     int / float: sample log-uniformly (counts and amounts)
    avoid:   ((threshold, half_width), ...): resample values inside a band around a rule threshold, so labels are
             never decided by a hair
    phrases: how the fact is stated. int / float / cat: templates with {v}; bool: {True: [...], False: [...]}.
             Formal and informal registers can be split with keys "formal" / "informal" holding those structures.
    fmt:     value -> string for {v} (default: thousands separators for numbers, plain for categories)
    """
    name: str
    kind: str
    values: tuple = ()
    p: object = 0.5
    log: bool = False
    avoid: tuple = ()
    phrases: object = ()
    fmt: Callable = None
    about: str = ""  # what the fact means, in a few words; used in generation prompts (general/gen_banks.py)

    def sample(self, rng):
        for _ in range(100):
            v = self._draw(rng)
            near = self.kind in ("int", "float") and any(abs(v - t) <= w for t, w in self.avoid)
            if not near:
                return v
        return v

    def _draw(self, rng):
        if self.kind == "bool":
            return rng.random() < self.p
        if self.kind == "cat":
            return rng.choices(self.values, weights=self.p if isinstance(self.p, (list, tuple)) else None)[0]
        lo, hi = self.values
        if self.log:
            x = math.exp(rng.uniform(math.log(max(lo, 1e-9)), math.log(hi)))
        else:
            x = rng.uniform(lo, hi)
        return int(round(x)) if self.kind == "int" else round(x, 2)

    def text(self, v):
        if self.fmt:
            return self.fmt(v)
        if self.kind == "int":
            return f"{v:,}"
        if self.kind == "float":
            return f"{v:,.2f}"
        if self.kind == "bool":
            return "yes" if v else "no"
        return str(v).replace("_", " ")

    def templates(self, v, informal=False):
        """The phrase templates that state value v (one register)."""
        ph = self.phrases
        if isinstance(ph, dict) and ("formal" in ph or "informal" in ph):
            ph = ph.get("informal" if informal else "formal") or ph.get("formal") or ph.get("informal")
        if self.kind == "bool":
            return list(ph[bool(v)])
        if isinstance(ph, dict):  # cat: per-value templates
            return list(ph[v])
        return list(ph)


@dataclass
class Option:
    key: str
    description: str


@dataclass
class Rule:
    """A policy for one question. text is stated in the question's instructions; label computes the answer."""
    text: str
    label: Callable[[dict], str]


@dataclass
class Question:
    """One decision. type "score" options are ordered from lowest to highest; "noul" has no options (yes / no).

    rules:       {"default": Rule, "<variant>": Rule, ...}. Variants keep the facts and change the policy, so a model
                 has to read the rule text to answer them (counterfactual pairs).
    needs:       the facts the labels depend on; they are always rendered.
    paraphrases: other ways to ask the same question (training uses them; held-out ones go to test).
    """
    id: str
    type: str
    text: str
    rules: dict
    needs: tuple
    options: list = field(default_factory=list)
    paraphrases: tuple = ()

    def keys(self):
        return ["yes", "no"] if self.type == "noul" else [o.key for o in self.options]


@dataclass
class Domain:
    """name: module id; title: human-readable; situations: opening sentences with {entity} slots (and {fact} slots
    for facts that are also stated there); entities: name pools ({org}, {person}, {item}, ...), all fictional;
    distractors: irrelevant sentences in the domain's vocabulary. Question paraphrases and every phrase list are
    split by a stable hash, so about a quarter of each list's wording appears only in test (general/render.py)."""
    name: str
    title: str
    facts: list
    questions: list
    situations: list
    entities: dict = field(default_factory=dict)
    distractors: list = field(default_factory=list)
    formats: tuple = ()       # document formats this domain uses (general/render.py); empty = all of them
    derived: Callable = None  # facts -> extra derived values available to templates (e.g. a computed date)

    def fact(self, name):
        return next(f for f in self.facts if f.name == name)

    def sample(self, rng):
        facts = {f.name: f.sample(rng) for f in self.facts}
        facts["_entities"] = {k: rng.choice(v) for k, v in self.entities.items()}
        return facts

    def labels(self, facts, variant="default"):
        return {q.id: (q.rules.get(variant) or q.rules["default"]).label(facts) for q in self.questions}

    def validate(self, n=3000, seed=0, max_share=0.85, min_flip=0.05):
        """Check the spec; returns {"ok": bool, "problems": [...], "stats": {...}}."""
        problems, stats = [], {}
        names = {f.name for f in self.facts}
        for q in self.questions:
            if q.type not in TYPES:
                problems.append(f"{q.id}: type {q.type!r} not in {TYPES}")
            if q.type != "noul" and len(q.options) < 2:
                problems.append(f"{q.id}: needs at least 2 options")
            if "default" not in q.rules:
                problems.append(f"{q.id}: no 'default' rule")
            missing = set(q.needs) - names
            if missing:
                problems.append(f"{q.id}: needs unknown facts {sorted(missing)}")
        for f in self.facts:
            if not f.phrases:
                problems.append(f"fact {f.name}: no phrases")
        if problems:
            return {"ok": False, "problems": problems, "stats": stats}

        a = [self.sample(random.Random(seed + i)) for i in range(50)]
        b = [self.sample(random.Random(seed + i)) for i in range(50)]
        if a != b:
            problems.append("sampling is not reproducible from a seed")

        rng = random.Random(seed)
        draws = [self.sample(rng) for _ in range(n)]
        for q in self.questions:
            seen = Counter()
            for var, rule in q.rules.items():
                counts = Counter(rule.label(f) for f in draws)
                seen.update(counts)
                bad = set(counts) - set(q.keys())
                if bad:
                    problems.append(f"{q.id}/{var}: labels outside the options: {sorted(bad)}")
                top = counts.most_common(1)[0][1] / n
                if top > max_share:
                    problems.append(f"{q.id}/{var}: one option is {top:.0%} of labels")
                stats[f"{q.id}/{var}"] = {k: round(counts[k] / n, 3) for k in q.keys()}
                if var != "default":
                    flip = sum(rule.label(f) != q.rules["default"].label(f) for f in draws) / n
                    stats[f"{q.id}/{var}"]["_flip_vs_default"] = round(flip, 3)
                    if flip < min_flip:
                        problems.append(f"{q.id}/{var}: changes the label on only {flip:.1%} of records")
            unreached = [k for k in q.keys() if seen[k] == 0]
            if unreached:  # an option only some rules use is fine; one no rule ever picks is a spec bug
                problems.append(f"{q.id}: options no rule ever chooses: {unreached}")
        for f in self.facts:  # every value must render in both registers
            for v in ([True, False] if f.kind == "bool" else f.values if f.kind == "cat" else [f.values[0]]):
                for informal in (False, True):
                    if not f.templates(v, informal):
                        problems.append(f"fact {f.name}: no {'informal' if informal else 'formal'} phrase for {v!r}")
        return {"ok": not problems, "problems": problems, "stats": stats}
