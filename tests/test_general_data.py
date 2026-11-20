"""General decision data (general/): specs are valid, builds are reproducible, labels follow the rules.

  python tests/test_general_data.py

Checks, for every domain in general/domains/:
  * Domain.validate() passes (reachable options, balance, reproducible sampling, useful rule variants)
  * the same seed gives byte-identical examples; another seed gives different ones
  * every label is one of the shown option keys, and recomputing it from the facts under the example's rule agrees
  * counterfactual partners share record, wording and options, state different rules and have different labels
  * score options keep their order; the Jev request and both encoder inputs are well formed
  * the NIM client serves cached responses without a key and refuses uncached ones
"""

import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general import compose, contract  # noqa: E402
from general.domains import DOMAINS  # noqa: E402
from general.nim import NIMClient  # noqa: E402

assert DOMAINS, "no domains found"
for name, dom in DOMAINS.items():
    v = dom.validate()
    assert v["ok"], (name, v["problems"])

    a = compose.build_domain(dom, 40, "train", seed=1)
    b = compose.build_domain(dom, 40, "train", seed=1)
    c = compose.build_domain(dom, 40, "train", seed=2)
    assert json.dumps(a) == json.dumps(b), f"{name}: build not reproducible"
    assert json.dumps(a) != json.dumps(c), f"{name}: seed has no effect"

    rows = a + compose.build_domain(dom, 40, "test", seed=1)
    for e in rows:
        q = next(x for x in dom.questions if x.id == e["question"]["id"])
        keys = [o["key"] for o in e["question"]["options"]]
        assert e["label"] in keys, (name, e["id"])
        truth = q.rules[e["rule"]].label(dict(e["facts"]))
        if q.type == "score":
            assert e["label"] == str(q.keys().index(truth)), (name, e["id"])
            assert [o["description"] for o in e["question"]["options"]] == [o.description for o in q.options]
        elif q.type == "noul":
            assert e["label"] == truth and keys == ["yes", "no"], (name, e["id"])
        else:
            shown = {o["key"]: o["description"] for o in e["question"]["options"]}
            desc = {o.key: o.description for o in q.options}
            assert shown[e["label"]] in (desc[truth], truth.replace("_", " ")), (name, e["id"])
        if not e["rule_in_text"]:
            assert e["rule"] == "default", "a variant rule must be stated"
        jev = contract.to_jev(e)
        assert jev["type"] == q.type and jev["instructions"] == e["question"]["instructions"]
        assert "[OPT]" in contract.markers(e) and contract.letters(e).endswith("[MASK]")
        assert 0 <= contract.label_index(e) < len(keys)

    pairs = {}
    for e in rows:
        if e["pair"]:
            pairs.setdefault(e["pair"], []).append(e)
    for pid, (x, y) in pairs.items():
        assert x["state"] == y["state"] and x["question"]["options"] == y["question"]["options"], pid
        wx, wy = (e["question"]["instructions"].split("\nRules: ")[0] for e in (x, y))
        assert wx == wy and x["rule"] != y["rule"] and x["label"] != y["label"], pid
    print(f"{name:16s} ok: {len(rows)} examples, {len(pairs)} counterfactual pairs")

# key="" turns off every key source (environment, Kaggle Secret, .env), so nothing here can reach the network
with tempfile.TemporaryDirectory() as d:
    cache = os.path.join(d, "cache.jsonl")
    try:
        NIMClient("some/model", cache=cache, key="").chat([{"role": "user", "content": "hi"}])
        raise AssertionError("an uncached request ran without a key")
    except RuntimeError:
        pass
    body = {"model": "some/model", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.7,
            "top_p": 0.95, "max_tokens": 1024}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    open(cache, "w").write(json.dumps({"key": h, "model": "some/model", "content": "cached"}) + "\n")
    assert NIMClient("some/model", cache=cache, key="").chat([{"role": "user", "content": "hi"}]) == "cached"
print("OK general data")
