"""Domain specs (general/spec.py). Each module defines DOMAIN; this registry finds them all.

The pilot has 15 training domains, 3 dev-unseen and 4 test-unseen ones (Domain.heldout; docs/GENERAL_DECISION_PLAN.md,
section 6.4). Aerospace is the fifth test-unseen domain, scored on its own data (data/processed_v2) and never trained on.
"""

import importlib
import pkgutil

DOMAINS = {}
for _m in pkgutil.iter_modules(__path__):
    _mod = importlib.import_module(f"{__name__}.{_m.name}")
    if hasattr(_mod, "DOMAIN"):
        DOMAINS[_mod.DOMAIN.name] = _mod.DOMAIN
