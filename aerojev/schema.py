"""Question catalogue for the aerospace supply-chain decision model.

Three question types, mirroring Jev / simple-jev:
  choice : pick one of N described options, read out as letters A, B, C, ...
  score  : ordered rubric levels, read out as digits 0..N-1
  yesno  : binary, read out as the tokens "yes" / "no"

Each question has a canonical phrasing (used for eval) plus paraphrases
(sampled during training so the model reads the question instead of
memorising one string).
"""

from dataclasses import dataclass, field


@dataclass
class Option:
    key: str
    description: str
    alt_descriptions: list = field(default_factory=list)


@dataclass
class Question:
    id: str
    type: str  # "choice" | "score" | "yesno"
    text: str
    paraphrases: list = field(default_factory=list)
    options: list = field(default_factory=list)  # list[Option]; empty for yesno
    holdout: bool = False  # never seen in training; used to test transfer


EVENT_TYPE = Question(
    id="event_type",
    type="choice",
    text="What kind of supply chain event does this record describe?",
    paraphrases=[
        "Classify the primary supply chain event in this record.",
        "Which event category best fits what is happening here?",
        "What is the main type of supply issue reported?",
    ],
    options=[
        Option("delivery_delay", "Supplier delivery slip or late shipment of ordered parts",
               ["Late delivery / pushed-out promise date"]),
        Option("quality_escape", "Nonconformance, failed inspection, defective or suspect parts",
               ["Quality problem with delivered or in-process parts"]),
        Option("capacity_constraint", "Supplier lacks labor, machine time or capacity to meet demand",
               ["Supplier capacity, staffing or equipment downtime limits output"]),
        Option("raw_material_shortage", "Shortage of input material (titanium, aluminum, nickel alloy, resin, chips)",
               ["Upstream material or component supply is short"]),
        Option("logistics_disruption", "Freight, carrier, customs or port problem moving goods",
               ["Transportation / shipping / customs disruption"]),
        Option("financial_distress", "Supplier insolvency, bankruptcy, cash-flow crisis or distressed sale",
               ["Supplier financial health is at risk"]),
        Option("regulatory_compliance", "Export control, sanctions, or lapsed certification (AS9100, NADCAP, FAA)",
               ["Compliance, trade-control or certification problem"]),
        Option("force_majeure", "Natural disaster, fire, flood, war or other force majeure at a supplier site",
               ["Catastrophic external event at a supplier site"]),
        Option("obsolescence", "End-of-life, discontinuation or product change notice (DMSMS)",
               ["Part is being discontinued or changed by the manufacturer"]),
        Option("commercial_change", "Price increase, MOQ, lead-time policy or contract term change",
               ["Commercial terms or pricing change from supplier"]),
        Option("no_issue", "Routine confirmation or informational update with no problem",
               ["Nothing wrong: on-time confirmation or FYI"]),
    ],
)

COMMODITY = Question(
    id="commodity",
    type="choice",
    text="Which commodity family is affected?",
    paraphrases=[
        "What commodity group do the affected parts belong to?",
        "Assign the affected item to a commodity category.",
    ],
    options=[
        Option("castings_forgings", "Castings and forgings (disks, trunnions, ribs, housings as-forged/cast)"),
        Option("fasteners", "Fasteners and hardware (bolts, rivets, nuts, collars, Hi-Loks)"),
        Option("electronics", "Electronics, avionics and EEE components (ICs, FPGAs, connectors, boards)"),
        Option("composites", "Composites and resins (prepreg, honeycomb, laminates, radomes)"),
        Option("machined_parts", "Precision machined parts (fittings, brackets, frames, housings)"),
        Option("raw_material", "Metallic raw material (bar, plate, sheet, extrusion, billet)"),
        Option("harnesses_cables", "Wire harnesses and cable assemblies"),
        Option("chemicals", "Chemicals, sealants, adhesives, primers and coatings"),
        Option("mechanical_assemblies", "Hydraulic and mechanical assemblies (actuators, valves, bearings, pumps, seals)"),
        Option("other", "Other or not identifiable (interiors, services, unclear)"),
    ],
)

IMPACT_SEVERITY = Question(
    id="impact_severity",
    type="score",
    text="How severe is the expected impact on production and delivery?",
    paraphrases=[
        "Rate the production / program impact of this event.",
        "Score the severity of the impact on our build schedule.",
    ],
    options=[
        Option("0", "None: no problem, nothing to absorb"),
        Option("1", "Minor: covered by buffer stock or purely commercial/administrative"),
        Option("2", "Moderate: short shortfall (up to 2 weeks) or contained quality issue, recoverable"),
        Option("3", "Major: shortfall over 2 weeks, sole-source supplier failure, or suspect parts"),
        Option("4", "Critical: line stop or AOG now, or escape onto delivered aircraft / safety of flight"),
    ],
)

LINE_STOP_RISK = Question(
    id="line_stop_risk",
    type="yesno",
    text="Will we run out of stock (line stop or AOG) within the next 30 days if nothing changes?",
    paraphrases=[
        "Is there a stock-out, line-stop or AOG risk in the next 30 days?",
        "Does this put the line at risk of stopping within a month?",
    ],
)

RECOMMENDED_ACTION = Question(
    id="recommended_action",
    type="choice",
    text="What is the recommended next action for the supply chain team?",
    paraphrases=[
        "Which response should the analyst take next?",
        "Pick the most appropriate action per our playbook.",
    ],
    options=[
        Option("monitor", "Monitor only: no shortfall, track at next review"),
        Option("expedite", "Expedite: push supplier / freight to recover a short delay"),
        Option("request_recovery_plan", "Request a formal recovery plan or SCAR/CAPA from the supplier"),
        Option("activate_alternate_source", "Shift demand to or qualify an alternate source"),
        Option("quarantine_inventory", "Quarantine and stop use of suspect inventory"),
        Option("escalate_leadership", "Escalate to program leadership / customer: line stop or AOG"),
        Option("involve_trade_compliance", "Route to trade compliance / legal (export control, sanctions)"),
        Option("renegotiate_commercial", "Negotiate price or commercial terms with the supplier"),
        Option("last_time_buy", "Place a last-time buy or start a redesign for an end-of-life part"),
    ],
)

EXPORT_CONTROLLED = Question(
    id="export_controlled",
    type="yesno",
    text="Does this record involve ITAR or EAR export-controlled items or technical data?",
    paraphrases=[
        "Are export-controlled (ITAR / EAR 600-series) parts or data involved?",
        "Is any of this subject to ITAR or export control restrictions?",
    ],
)

COUNTERFEIT_SUSPECT = Question(
    id="counterfeit_suspect",
    type="yesno",
    text="Are there indicators of counterfeit or suspect unapproved parts?",
    paraphrases=[
        "Could these be counterfeit or suspect unapproved parts (SUP)?",
        "Do any signs point to counterfeit or untraceable parts?",
    ],
)

SOLE_SOURCE = Question(
    id="sole_source",
    type="yesno",
    text="Does the record state that the supplier is a sole or single qualified source?",
    paraphrases=[
        "Is this a sole-source / single-source dependency?",
        "Is there no qualified alternate supplier for this item?",
    ],
)

DELAY_BUCKET = Question(
    id="delay_bucket",
    type="score",
    text="How late is supply versus the need date?",
    paraphrases=[
        "Bucket the delivery delay relative to the need date.",
        "How big is the schedule slip?",
    ],
    options=[
        Option("0", "Not late: on time, early, or no delivery affected"),
        Option("1", "1 to 7 days late"),
        Option("2", "8 to 30 days late"),
        Option("3", "31 to 90 days late"),
        Option("4", "More than 90 days late, or indefinite / unknown recovery"),
    ],
)

# --- Held-out questions: never trained on, used to measure transfer -------

SUPPLIER_REGION = Question(
    id="supplier_region",
    type="choice",
    text="Where is the supplier located?",
    options=[
        Option("north_america", "North America (USA, Canada, Mexico)"),
        Option("europe", "Europe (including UK)"),
        Option("asia_pacific", "Asia-Pacific (China, Japan, Korea, India, Southeast Asia, Australia)"),
        Option("unknown", "Not stated in the record"),
    ],
    holdout=True,
)

MENTIONS_NEW_DATE = Question(
    id="mentions_new_date",
    type="yesno",
    text="Does the record give a specific new or committed delivery date?",
    holdout=True,
)

MILITARY_PROGRAM = Question(
    id="military_program",
    type="yesno",
    text="Is the affected program a defense / military program?",
    holdout=True,
)

CATALOGUE = [
    EVENT_TYPE, COMMODITY, IMPACT_SEVERITY, LINE_STOP_RISK, RECOMMENDED_ACTION,
    EXPORT_CONTROLLED, COUNTERFEIT_SUSPECT, SOLE_SOURCE, DELAY_BUCKET,
    SUPPLIER_REGION, MENTIONS_NEW_DATE, MILITARY_PROGRAM,
]
BY_ID = {q.id: q for q in CATALOGUE}
TRAIN_QUESTIONS = [q for q in CATALOGUE if not q.holdout]
HOLDOUT_QUESTIONS = [q for q in CATALOGUE if q.holdout]
