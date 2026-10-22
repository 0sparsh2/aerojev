"""Latent scenario sampling and the analytics team's decision playbook.

Labels are never read off rendered text: a Scenario is sampled first, labels
are derived from it by `derive_labels`, and only then is text rendered. Every
fact the playbook uses is guaranteed to be stated in the rendered record.

Playbook (shortfall = days late - days of stock coverage):
  impact_severity
    4  line already down / AOG, or escape onto delivered aircraft
    3  shortfall > 14 d, sole-source supplier in financial distress,
       counterfeit suspect, export / sanctions violation
    2  shortfall 1-14 d, contained quality issue, EOL with < 6 months to LTB
    1  buffer covers it, commercial change, EOL with >= 6 months to LTB
    0  no issue
  line_stop_risk = AOG / line down, or (stock runs out before supply resumes
                   and that happens within 30 days)
  recommended_action (first rule that fires)
    severity 4                         -> escalate_leadership
    counterfeit or suspect stock held  -> quarantine_inventory
    export / sanctions issue           -> involve_trade_compliance
    financial distress                 -> activate_alternate_source
    force majeure > 90 d / indefinite  -> activate_alternate_source
    obsolescence                       -> last_time_buy
    commercial change                  -> renegotiate_commercial
    supplier-caught quality, cert lapse, capacity, raw material w/ shortfall
                                       -> request_recovery_plan
    delivery delay, logistics, short force majeure w/ shortfall -> expedite
    otherwise                          -> monitor
"""

import random
from dataclasses import dataclass, field

from . import world

EVENTS = [
    "delivery_delay", "quality_escape", "capacity_constraint", "raw_material_shortage",
    "logistics_disruption", "financial_distress", "regulatory_compliance", "force_majeure",
    "obsolescence", "commercial_change", "no_issue",
]
EVENT_WEIGHTS = [16, 14, 9, 9, 8, 7, 8, 7, 6, 6, 10]
INDEFINITE = 999

# Data v2 (set by generate.py --v2): extra sub-kinds whose wording lives in new phrase banks,
# so v1 banks keep their train/test assignment and v1 test files stay uncontaminated.
FEATURES = {"v2": False}


@dataclass
class Scenario:
    event: str
    commodity: str
    part: tuple
    supplier: tuple
    program: tuple
    sub: str = ""                    # event sub-kind (e.g. quality: receiving / supplier / fielded / counterfeit)
    days_late: int = 0               # 0 = not late, INDEFINITE = unknown recovery
    coverage_days: int = -1          # -1 = not stated (only allowed when it cannot matter)
    aog: bool = False                # line already down / aircraft on ground
    sole_source: bool = False
    sole_mention: str = "none"       # "sole" | "dual" | "none"
    export_mention: str = "none"     # "itar" | "ear600" | "ear99" | "none"
    counterfeit: bool = False
    ltb_months: int = 0
    region_stated: bool = True
    new_date_given: bool = False
    extra: dict = field(default_factory=dict)


def _bucket(days):
    if days <= 0:
        return 0
    if days <= 7:
        return 1
    if days <= 30:
        return 2
    if days <= 90:
        return 3
    return 4


def _sample_delay_and_cover(rng, allow_indefinite=True, allow_zero_delay=False):
    """Draw (days_late, coverage_days) for a target shortfall regime, away from decision edges.

    covered: stock outlasts the delay (gap <= -4)      -> severity 1
    short  : shortfall of 4..11 days                   -> severity 2
    long   : shortfall of 18+ days                     -> severity 3
    """
    bad_d = set(range(6, 10)) | set(range(29, 33)) | set(range(89, 93))
    while True:
        if allow_zero_delay and rng.random() < 0.2:
            return 0, rng.choice([0, rng.randint(5, 60)])
        if allow_indefinite and rng.random() < 0.15:
            c = rng.choice([0, rng.randint(3, 25), rng.randint(36, 120)])
            return INDEFINITE, c
        regime = rng.choice(["covered", "short", "long", "long"])
        if regime == "covered":
            d = rng.choice([rng.randint(2, 5), rng.randint(10, 28), rng.randint(35, 84), rng.randint(100, 160)])
            c = d + rng.randint(4, 60)
        elif regime == "short":
            d = rng.choice([rng.randint(10, 28), rng.randint(35, 84), rng.randint(100, 160)])
            c = d - rng.randint(4, 11)
        else:
            c = rng.choice([0, 0, rng.randint(3, 12), rng.randint(14, 25), rng.randint(36, 70)])
            d = c + rng.randint(18, 130)
        if d in bad_d or c < 0:
            continue
        if c < d and 26 <= c <= 34:  # 30-day stock-out edge
            continue
        return d, c


def sample_scenario(rng: random.Random) -> Scenario:
    event = rng.choices(EVENTS, EVENT_WEIGHTS)[0]
    commodity = rng.choice(list(world.PARTS))
    s = Scenario(
        event=event,
        commodity=commodity,
        part=rng.choice(world.PARTS[commodity]),
        supplier=rng.choice(world.suppliers_for(commodity)),
        program=rng.choice(world.PROGRAMS),
    )
    military = s.program[2]

    # Export marking: military programs always carry an explicit ITAR / 600-series marking.
    if military:
        s.export_mention = rng.choice(["itar", "itar", "ear600"])
    else:
        s.export_mention = rng.choice(["none", "none", "ear99", "ear99", "ear600"])

    s.sole_mention = rng.choice(["sole", "dual", "none", "none"])
    s.sole_source = s.sole_mention == "sole"
    s.region_stated = rng.random() < 0.75

    if event == "quality_escape":
        if rng.random() < 0.3:
            s.commodity = rng.choice(["electronics", "electronics", "fasteners", "mechanical_assemblies",
                                      "harnesses_cables"])
            s.part = rng.choice(world.PARTS[s.commodity])
            s.supplier = rng.choice(world.suppliers_for(s.commodity))
            s.sub = "counterfeit"
        else:
            s.sub = rng.choice(["receiving", "receiving", "supplier", "fielded", "doc_error_receiving"])
        if s.sub == "doc_error_receiving":
            s.sub = "receiving"  # documentation slip mentioned as a distractor alongside a real defect
            s.extra["doc_note"] = True
        s.counterfeit = s.sub == "counterfeit"
        if s.sub == "fielded":
            s.aog = True
            s.days_late, s.coverage_days = 0, -1
        else:
            s.days_late, s.coverage_days = _sample_delay_and_cover(rng, allow_indefinite=False)
    elif event in ("obsolescence", "commercial_change", "no_issue"):
        s.days_late = 0
        s.coverage_days = rng.choice([-1, -1, rng.randint(14, 90)])
        if event == "obsolescence":
            s.ltb_months = rng.choice([2, 3, 4, 8, 9, 12, 18])
            s.sub = "chip_eol" if commodity == "electronics" else "part_eol"
        if event == "commercial_change":
            s.sub = rng.choice(["price", "moq", "leadtime", "tariff", "lta"])
        if event == "no_issue":
            s.sub = rng.choice(["on_track", "shipped", "qbr", "audit_pass", "admin", "released"])
            if FEATURES["v2"] and rng.random() < 0.3:
                s.sub = "early"  # pulled-in / ahead-of-schedule deliveries
    else:
        if event == "regulatory_compliance":
            s.sub = rng.choice(["export", "export", "cert"])
            if s.sub == "export":
                s.export_mention = rng.choice(["itar", "ear600"])
                s.extra["kind"] = rng.choice(["entity_list", "unlicensed", "license_expired", "foreign_national"])
            else:
                s.extra["kind"] = rng.choice(["nadcap", "as9100", "faa"])
        if event == "financial_distress":
            s.sub = rng.choice(["chapter11", "payroll", "credit", "score", "layoffs"])
        if event == "force_majeure":
            s.sub = rng.choice(["fire", "flood", "earthquake", "typhoon", "explosion", "power"])
        if event == "logistics_disruption":
            s.sub = rng.choice(["customs", "port", "air", "truck", "rail", "lane"])
        if event == "raw_material_shortage":
            s.extra["material"] = rng.choice(world.MATERIALS[commodity])
        if event == "capacity_constraint" and FEATURES["v2"] and rng.random() < 0.35:
            s.sub = "equipment"  # supplier equipment fails its own inspection: capacity, not part quality
        allow_zero = event == "financial_distress"
        s.days_late, s.coverage_days = _sample_delay_and_cover(rng, allow_zero_delay=allow_zero)
        if s.coverage_days == 0 and s.days_late > 0 and rng.random() < 0.3:
            s.aog = True

    if s.days_late not in (0,) and s.days_late != INDEFINITE:
        s.new_date_given = rng.random() < 0.45
    return s


def derive_labels(s: Scenario) -> dict:
    d, c = s.days_late, s.coverage_days
    late = d > 0
    gap = (d - max(c, 0)) if late else -1
    stockout_30 = late and c >= 0 and c < d and c <= 30
    if late and c < 0:
        raise ValueError("coverage must be stated whenever there is a delay")

    # ---- severity
    if s.aog:
        sev = 4
    elif s.event == "no_issue":
        sev = 0
    elif s.event == "commercial_change":
        sev = 1
    elif s.event == "obsolescence":
        sev = 1 if s.ltb_months >= 6 else 2
    else:
        sev = 1 if gap <= 0 else (2 if gap <= 14 else 3)
        if s.event == "quality_escape":
            sev = max(sev, 3 if s.counterfeit else 2)
        if s.event == "regulatory_compliance":
            sev = max(sev, 3 if s.sub == "export" else 2)
        if s.event == "financial_distress" and s.sole_source:
            sev = max(sev, 3)

    line_stop = bool(s.aog or stockout_30)

    # ---- action
    if sev == 4:
        action = "escalate_leadership"
    elif s.counterfeit or (s.event == "quality_escape" and s.sub == "receiving"):
        action = "quarantine_inventory"
    elif s.event == "regulatory_compliance" and s.sub == "export":
        action = "involve_trade_compliance"
    elif s.event == "financial_distress":
        action = "activate_alternate_source"
    elif s.event == "force_majeure" and d > 90:
        action = "activate_alternate_source"
    elif s.event == "obsolescence":
        action = "last_time_buy"
    elif s.event == "commercial_change":
        action = "renegotiate_commercial"
    elif s.event == "quality_escape" and s.sub == "supplier":
        action = "request_recovery_plan"
    elif s.event == "regulatory_compliance" and s.sub == "cert":
        action = "request_recovery_plan"
    elif s.event in ("capacity_constraint", "raw_material_shortage") and gap > 0:
        action = "request_recovery_plan"
    elif s.event in ("delivery_delay", "logistics_disruption", "force_majeure") and gap > 0:
        action = "expedite"
    else:
        action = "monitor"

    labels = {
        "event_type": s.event,
        "commodity": s.commodity,
        "impact_severity": str(sev),
        "line_stop_risk": "yes" if line_stop else "no",
        "recommended_action": action,
        "export_controlled": "yes" if s.export_mention in ("itar", "ear600") else "no",
        "counterfeit_suspect": "yes" if s.counterfeit else "no",
        "sole_source": "yes" if s.sole_source else "no",
        "delay_bucket": str(_bucket(d)),
        # held-out questions
        "supplier_region": s.supplier[4] if s.region_stated else "unknown",
        "mentions_new_date": "yes" if s.new_date_given else "no",
        "military_program": "yes" if s.program[2] else "no",
    }
    return labels
