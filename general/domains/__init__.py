"""Domain specs (general/spec.py). Each module defines DOMAIN; this registry finds them all.

The pilot needs 15 training domains and 5 held-out ones (docs/GENERAL_DECISION_PLAN.md, section 6.4). The aerospace
domain is never used for training: it is a held-out test domain through its own data (data/processed_v2).
"""

import importlib
import pkgutil

DOMAINS = {}
for _m in pkgutil.iter_modules(__path__):
    _mod = importlib.import_module(f"{__name__}.{_m.name}")
    if hasattr(_mod, "DOMAIN"):
        DOMAINS[_mod.DOMAIN.name] = _mod.DOMAIN
