"""Local authority food safety: inspection outcomes. Held out of training (test-unseen). All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def outcome(f, close_at=2):
    crit = f["critical_violations"]
    if f["pest_evidence"] or crit >= close_at:
        return "close_immediately"
    if crit == 1 or f["fridge_temp_c"] > 8 or f["minor_violations"] >= 5:
        return "reinspect_in_14_days"
    if f["minor_violations"] > 0 or f["fridge_temp_c"] > 5:
        return "pass_with_advice"
    return "pass"


OUTCOME_RULE = ("Close immediately for evidence of pests or {c}. Otherwise reinspect in 14 days for one critical "
                "violation, a fridge above 8 °C, or 5 or more minor violations. Otherwise pass with advice for any "
                "minor violation or a fridge above 5 °C. Otherwise pass.")


def risk(f, edges=(12, 24)):
    if f["previous_closure"]:
        return "very_high"
    m = f["months_since_inspection"]
    return "high" if m > edges[1] else "medium" if m > edges[0] else "low"


DOMAIN = Domain(
    name="food_inspection",
    title="Food safety inspections",
    heldout="test",
    entities={
        "venue": ["Rosie's Bakery", "Golden Lantern Takeaway", "The Old Mill Café", "Sabor Street Food",
                  "Northgate Diner", "Bluefin Sushi Bar"],
        "person": ["Inspector Hughes", "Inspector Banerjee", "Inspector Kowal", "Inspector Mensah"],
    },
    situations=[
        "{person} inspected {venue}.",
        "Inspection report for {venue}.",
        "Routine food hygiene visit to {venue}.",
        "{person} filed findings from {venue}.",
        "Follow-up needed on the inspection at {venue}.",
        "Food safety visit completed at {venue}.",
    ],
    formats=("email", "ticket", "form", "report", "log"),
    distractors=[
        "The licensing office moves to the second floor next month.",
        "New hygiene rating stickers have been printed.",
        "The inspection van is due for its service.",
        "The team meeting is on Tuesday afternoon.",
    ],
    facts=[
        Fact("fridge_temp_c", "float", about="the highest fridge temperature measured, in degrees Celsius",
             values=(1, 10), avoid=((5, 0.3), (8, 0.3)), fmt=lambda v: f"{v:.1f} °C",
             phrases={"formal": ["The warmest fridge read {v}.", "Highest chilled storage temperature: {v}.",
                                 "Fridge probe reading: {v}.", "The walk-in fridge measured {v}."],
                      "informal": ["fridge at {v}", "chiller reading {v}", "fridge temp {v}", "probe said {v}"]}),
        Fact("critical_violations", "int", about="how many critical violations were found", values=(0, 3),
             p=(0.6, 0.25, 0.1, 0.05),
             phrases={"formal": ["{v} critical violations were found.", "Critical violations: {v}.",
                                 "The inspector recorded {v} critical breaches.", "Number of critical findings: {v}."],
                      "informal": ["{v} critical violations", "{v} criticals", "crit findings: {v}",
                                   "{v} serious breaches"]}),
        Fact("minor_violations", "int", about="how many minor violations were found", values=(0, 9),
             p=(0.3, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03),
             phrases={"formal": ["{v} minor violations were found.", "Minor violations: {v}.",
                                 "The inspector noted {v} minor issues.", "Number of minor findings: {v}."],
                      "informal": ["{v} minor issues", "{v} minors", "minor findings: {v}", "{v} small breaches"]}),
        Fact("pest_evidence", "bool", about="whether there is evidence of pests", p=0.12,
             phrases={"formal": {True: ["Rodent droppings were found in the store room.",
                                        "There is evidence of a cockroach infestation.",
                                        "Signs of pests were found in the kitchen.", "Pest activity was observed."],
                                 False: ["No evidence of pests was found.", "The premises is free of pest activity.",
                                         "Pest control records are clean and no signs were seen.",
                                         "There were no signs of pests."]},
                      "informal": {True: ["droppings in the store room", "roaches seen", "pest signs found",
                                          "evidence of mice"],
                                   False: ["no pests", "no pest signs", "pest-free", "nothing pest-wise"]}}),
        Fact("months_since_inspection", "int", about="how many months ago the previous inspection was",
             values=(3, 36), avoid=((12, 0), (18, 0), (24, 0)),
             phrases={"formal": ["The previous inspection was {v} months ago.",
                                 "Months since the last inspection: {v}.", "It was last inspected {v} months ago.",
                                 "The prior visit took place {v} months ago."],
                      "informal": ["last inspected {v} months ago", "{v} months since last visit",
                                   "previous visit {v} mo ago", "{v} months since we were there"]}),
        Fact("previous_closure", "bool", about="whether the premises was closed by inspectors in the last two years",
             p=0.12,
             phrases={"formal": {True: ["The premises was closed by inspectors last year.",
                                        "An emergency closure was ordered here within the last two years.",
                                        "It has a closure on record from eighteen months ago.",
                                        "Inspectors shut it down within the past two years."],
                                 False: ["The premises has never been closed by inspectors.",
                                         "There is no closure on record.", "No enforcement closure in two years.",
                                         "It has not been shut down before."]},
                      "informal": {True: ["closed by us last year", "had a closure recently", "shut down before",
                                          "closure on record"],
                                   False: ["never closed", "no closures", "no closure history",
                                           "never been shut down"]}}),
        Fact("venue_type", "cat", about="the kind of food business",
             values=("bakery", "takeaway", "café", "restaurant", "food truck"),
             phrases={"formal": ["The business is a {v}.", "Premises type: {v}.", "It operates as a {v}.",
                                 "Business category: {v}."],
                      "informal": ["it's a {v}", "{v}", "type: {v}", "a {v}"]}),
    ],
    questions=[
        Question(
            id="outcome", type="choice", text="What should the inspection outcome be?",
            paraphrases=("Decide the result of this inspection.", "How should this inspection be closed out?",
                         "What is the enforcement outcome?"),
            options=[Option("pass", "Pass"), Option("pass_with_advice", "Pass with written advice"),
                     Option("reinspect_in_14_days", "Reinspect within 14 days"),
                     Option("close_immediately", "Close the premises immediately")],
            rules={"default": Rule(OUTCOME_RULE.format(c="2 or more critical violations"), outcome),
                   "zero_tolerance": Rule(OUTCOME_RULE.format(c="any critical violation"),
                                          lambda f: outcome(f, close_at=1))},
            needs=("pest_evidence", "critical_violations", "fridge_temp_c", "minor_violations")),
        Question(
            id="risk_band", type="score", text="What risk band should this premises be in?",
            paraphrases=("Rate the premises' inspection risk.", "Which risk band applies?",
                         "How risky is this food business?"),
            options=[Option("low", "low"), Option("medium", "medium"), Option("high", "high"),
                     Option("very_high", "very high")],
            rules={"default": Rule("Very high if the premises was closed by inspectors in the last two years. "
                                   "Otherwise high if the previous inspection was over 24 months ago, medium if over "
                                   "12 months ago, and low otherwise.", risk),
                   "shorter": Rule("Very high if the premises was closed by inspectors in the last two years. "
                                   "Otherwise high if the previous inspection was over 18 months ago, medium if over "
                                   "12 months ago, and low otherwise.", lambda f: risk(f, (12, 18)))},
            needs=("previous_closure", "months_since_inspection")),
        Question(
            id="warning_letter", type="noul", text="Should a formal warning letter be sent?",
            paraphrases=("Does this premises get a warning letter?", "Issue a written warning?",
                         "Is a formal warning required?"),
            rules={"default": Rule("Send a warning letter for any critical violation, and for 5 or more minor "
                                   "violations.",
                                   lambda f: "yes" if f["critical_violations"] > 0 or f["minor_violations"] >= 5
                                   else "no"),
                   "minor_3": Rule("Send a warning letter for any critical violation, and for 3 or more minor "
                                   "violations.",
                                   lambda f: "yes" if f["critical_violations"] > 0 or f["minor_violations"] >= 3
                                   else "no")},
            needs=("critical_violations", "minor_violations")),
    ],
)
