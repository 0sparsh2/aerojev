"""Auto insurance: first notice of loss triage. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def money(v):
    return f"${v:,.2f}"


def siu(f, late_days=30):
    return ((f["incident_type"] == "theft" and not f["police_report"]) or f["prior_claims"] >= 3
            or f["days_to_report"] > late_days)


def route(f, fast_limit=2500):
    if not f["policy_active"]:
        return "deny_coverage"
    if siu(f):
        return "special_investigations"
    if f["injury"] or f["claim_amount"] > 50000:
        return "senior_adjuster"
    if f["claim_amount"] <= fast_limit or f["incident_type"] == "glass damage":
        return "fast_track"
    return "field_adjuster"


ROUTE_RULE = ("Deny coverage if the policy was not active on the date of the incident. Otherwise send to special "
              "investigations if it is a theft without a police report, the policyholder has 3 or more prior claims, "
              "or the claim was reported more than 30 days after the incident. Otherwise injuries and claims over "
              "$50,000 go to a senior adjuster. Otherwise fast-track claims up to {lim} and all glass damage. "
              "Everything else goes to a field adjuster.")

SIU_RULE = ("Refer thefts without a police report, policyholders with 3 or more prior claims, and claims reported "
            "more than {d} days after the incident.")

DOMAIN = Domain(
    name="insurance_claim",
    title="Auto insurance claim intake and triage",
    entities={
        "person": ["Marta Silva", "Kevin Obi", "Julia Novak", "Arjun Pillai", "Rosa Lindqvist", "Derek Huang"],
        "org": ["Harborline Mutual", "Crestview Insurance", "Alder Auto Cover"],
        "car": ["2019 hatchback", "2022 pickup", "2016 sedan", "2021 SUV", "2014 minivan"],
    },
    situations=[
        "{person} reported a claim on their {car}.",
        "New first notice of loss from {person}.",
        "Claim intake for a {car} insured with {org}.",
        "{person} called {org} to open a claim.",
        "Loss report received for policyholder {person}.",
        "A claim was filed online for a {car}.",
    ],
    formats=("email", "ticket", "chat", "form", "report"),
    distractors=[
        "The policyholder also asked about adding a second driver.",
        "Renewal notices go out at the end of the month.",
        "The claims portal will be down for maintenance on Sunday night.",
        "Body shop partner lists were updated last week.",
    ],
    facts=[
        Fact("claim_amount", "float", about="the estimated cost of the claim in dollars", values=(300, 80000),
             log=True, avoid=((2500, 200), (5000, 400), (10000, 800), (20000, 1500), (50000, 4000)), fmt=money,
             phrases={"formal": ["The repair estimate is {v}.", "Estimated loss: {v}.", "The claim is for {v}.",
                                 "The body shop quoted {v}."],
                      "informal": ["estimate is {v}", "{v} damage", "shop says {v}", "claim's about {v}"]}),
        Fact("incident_type", "cat", about="what kind of incident caused the loss",
             values=("collision", "theft", "storm damage", "vandalism", "glass damage"), p=(0.4, 0.15, 0.15, 0.1, 0.2),
             phrases={"formal": ["The incident type is {v}.", "Cause of loss: {v}.", "The loss was caused by {v}.",
                                 "This is a {v} claim."],
                      "informal": ["{v}", "it's {v}", "{v} claim", "cause: {v}"]}),
        Fact("policy_active", "bool", about="whether the policy was active on the date of the incident", p=0.9,
             phrases={"formal": {True: ["The policy was in force on the date of loss.",
                                        "Coverage was active when it happened.",
                                        "Premiums were paid up and the policy was current.",
                                        "The policy was active at the time of the incident."],
                                 False: ["The policy had lapsed before the incident.",
                                         "Coverage was cancelled for non-payment before the date of loss.",
                                         "The policy was not in force when it happened.",
                                         "The policy expired a week before the incident."]},
                      "informal": {True: ["policy was active", "coverage was in force", "policy current",
                                          "they were covered at the time"],
                                   False: ["policy had lapsed", "not covered at the time", "policy expired before",
                                           "coverage was cancelled"]}}),
        Fact("days_to_report", "int", about="how many days after the incident the claim was reported",
             values=(1, 45), log=True, avoid=((14, 1), (30, 1)),
             phrases={"formal": ["The claim was reported {v} days after the incident.",
                                 "Days between the incident and the report: {v}.",
                                 "It was reported to us {v} days after it happened.",
                                 "Notice was given {v} days after the loss."],
                      "informal": ["reported {v} days after", "took {v} days to call it in", "{v} days late reporting",
                                   "{v} days between incident and claim"]}),
        Fact("injury", "bool", about="whether anyone was injured", p=0.2,
             phrases={"formal": {True: ["The other driver was taken to hospital.", "A passenger was injured.",
                                        "There are bodily injuries.", "Someone was hurt in the incident."],
                                 False: ["No one was injured.", "There are no injuries.",
                                         "It was property damage only.", "Nobody was hurt."]},
                      "informal": {True: ["someone got hurt", "injury involved", "passenger hurt",
                                          "other driver went to ER"],
                                   False: ["no injuries", "nobody hurt", "property damage only", "everyone's fine"]}}),
        Fact("police_report", "bool", about="whether a police report was filed", p=0.6,
             phrases={"formal": {True: ["A police report was filed.", "The police report number is on file.",
                                        "Officers attended and filed a report.", "There is a police report."],
                                 False: ["No police report was filed.", "The police were not called.",
                                         "There is no police report.", "No report was made to the police."]},
                      "informal": {True: ["police report filed", "has a police report #", "cops came, report filed",
                                          "report on file"],
                                   False: ["no police report", "didn't call the police", "no report filed",
                                           "police not involved"]}}),
        Fact("prior_claims", "int", about="how many claims the policyholder made in the past 3 years",
             values=(0, 6), p=(0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01),
             phrases={"formal": ["The policyholder made {v} claims in the past 3 years.",
                                 "Prior claims in 3 years: {v}.", "Claims history: {v} in the last 3 years.",
                                 "They have filed {v} earlier claims in the last 3 years."],
                      "informal": ["{v} claims in 3 yrs", "{v} prior claims", "claims history: {v}",
                                   "filed {v} claims before"]}),
    ],
    questions=[
        Question(
            id="route", type="choice", text="Where should this claim be routed?",
            paraphrases=("Who should handle this claim?", "Route this claim.", "Which queue does this claim go to?"),
            options=[Option("fast_track", "Fast-track settlement"),
                     Option("field_adjuster", "Field adjuster inspection"),
                     Option("senior_adjuster", "Senior adjuster"),
                     Option("special_investigations", "Special investigations unit"),
                     Option("deny_coverage", "Deny: no coverage")],
            rules={"default": Rule(ROUTE_RULE.format(lim="$2,500"), route),
                   "fast_10k": Rule(ROUTE_RULE.format(lim="$10,000"), lambda f: route(f, fast_limit=10000))},
            needs=("policy_active", "incident_type", "police_report", "prior_claims", "days_to_report", "injury",
                   "claim_amount")),
        Question(
            id="severity", type="score", text="How severe is this claim?",
            paraphrases=("Rate the severity of this loss.", "What severity level fits this claim?",
                         "Classify the claim's severity."),
            options=[Option("minor", "minor"), Option("moderate", "moderate"), Option("major", "major"),
                     Option("catastrophic", "catastrophic")],
            rules={"default": Rule("Catastrophic if the estimate is over $50,000, or someone was injured and the "
                                   "estimate is over $20,000. Otherwise major if someone was injured or the estimate "
                                   "is over $5,000. Otherwise moderate over $2,500. Otherwise minor.",
                                   lambda f: "catastrophic" if f["claim_amount"] > 50000 or
                                   (f["injury"] and f["claim_amount"] > 20000) else
                                   "major" if f["injury"] or f["claim_amount"] > 5000 else
                                   "moderate" if f["claim_amount"] > 2500 else "minor")},
            needs=("injury", "claim_amount")),
        Question(
            id="siu_referral", type="noul", text="Should this claim be referred to special investigations?",
            paraphrases=("Does this need a fraud referral?", "Send this claim to the SIU?",
                         "Is an investigation referral required?"),
            rules={"default": Rule(SIU_RULE.format(d=30), lambda f: "yes" if siu(f) else "no"),
                   "late_14": Rule(SIU_RULE.format(d=14), lambda f: "yes" if siu(f, 14) else "no")},
            needs=("incident_type", "police_report", "prior_claims", "days_to_report")),
    ],
)
