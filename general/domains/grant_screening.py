"""Charitable foundation: grant application screening. Held out of training (test-unseen). All names are
fictional."""

from ..spec import Domain, Fact, Option, Question, Rule

PRIORITIES = ("education", "health", "the environment")


def usd(v):
    return f"${v:,}"


def decision(f, overhead=25):
    if not f["registered_charity"] or f["focus"] not in PRIORITIES:
        return "decline"
    if f["late_report"]:
        return "request_information"
    if f["overhead_pct"] > overhead or f["years_operating"] < 3 or f["amount_requested"] > f["annual_budget"]:
        return "fund_partially"
    return "fund_in_full"


DECISION_RULE = ("Decline if the applicant is not a registered charity or its focus is not education, health or the "
                 "environment. Otherwise ask for more information if its last grant report to us was late. Otherwise "
                 "fund partially if overhead is above {o}% of spending, it has operated for under 3 years, or it asks "
                 "for more than its whole annual budget. Otherwise fund in full.")


def level(f, limits=(25000, 100000)):
    a = f["amount_requested"]
    return "board" if a > limits[1] else "committee" if a > limits[0] else "staff"


DOMAIN = Domain(
    name="grant_screening",
    title="Charitable grant application screening",
    heldout="test",
    entities={
        "org": ["Bright Futures Trust", "River Valley Food Bank", "Open Canvas Arts", "Greenway Restoration",
                "Hearth Housing Collective", "Clearwater Health Outreach"],
        "person": ["Programme Officer Lane", "Programme Officer Osei", "Programme Officer Brandt"],
    },
    situations=[
        "{org} submitted a grant application.",
        "Grant application received from {org}.",
        "{person} is screening an application from {org}.",
        "New proposal in the funding round from {org}.",
        "{org} applied to this year's open call.",
        "Screening needed for {org}'s proposal.",
    ],
    formats=("email", "ticket", "form", "report", "chat"),
    distractors=[
        "The annual report design is out for review.",
        "Trustee elections take place in March.",
        "The office will be closed between the holidays.",
        "The website's grantee stories page was refreshed.",
    ],
    facts=[
        Fact("registered_charity", "bool", about="whether the applicant is a registered charity", p=0.88,
             phrases={"formal": {True: ["The applicant is a registered charity.", "Its charity number checks out.",
                                        "It is listed on the charity register.", "Charitable status is confirmed."],
                                 False: ["The applicant is not a registered charity.",
                                         "No charity registration could be found.",
                                         "It is a private company, not a charity.",
                                         "Charitable status could not be confirmed."]},
                      "informal": {True: ["registered charity", "charity # valid", "on the register",
                                          "charity status ok"],
                                   False: ["not a registered charity", "no charity number", "not on the register",
                                           "it's a company, not a charity"]}}),
        Fact("focus", "cat", about="the applicant's main area of work",
             values=("education", "health", "the environment", "the arts", "housing"),
             p=(0.3, 0.25, 0.25, 0.1, 0.1),
             phrases={"formal": ["The organisation works on {v}.", "Main focus: {v}.",
                                 "Its programmes are in {v}.", "Area of work: {v}."],
                      "informal": ["works on {v}", "focus is {v}", "{v} org", "they do {v}"]}),
        Fact("late_report", "bool", about="whether the applicant's last grant report to the foundation was late",
             p=0.15,
             phrases={"formal": {True: ["Their last grant report to us arrived late.",
                                        "The previous grant report was overdue.",
                                        "They missed the deadline for their last report.",
                                        "Their last end-of-grant report came in months late."],
                                 False: ["Their last grant report arrived on time.",
                                         "Previous reporting was on schedule.",
                                         "They met the deadline for their last report.",
                                         "There are no overdue reports."]},
                      "informal": {True: ["last report was late", "overdue report last time", "missed report deadline",
                                          "reporting was late"],
                                   False: ["reports on time", "last report on time", "no overdue reports",
                                           "reporting fine"]}}),
        Fact("overhead_pct", "int", about="the share of spending that goes to overheads, in percent",
             values=(5, 50), avoid=((25, 1), (35, 1)), fmt=lambda v: f"{v}%",
             phrases={"formal": ["Overheads are {v} of spending.", "Overhead ratio: {v}.",
                                 "Administration takes {v} of the budget.", "{v} of spending goes to overheads."],
                      "informal": ["overhead {v}", "{v} admin costs", "overheads at {v}", "{v} overhead"]}),
        Fact("years_operating", "int", about="how many years the organisation has operated", values=(1, 40),
             log=True,
             phrases={"formal": ["The organisation has operated for {v} years.", "Years in operation: {v}.",
                                 "It was founded {v} years ago.", "It has been running for {v} years."],
                      "informal": ["{v} yrs old", "running {v} years", "founded {v} years ago", "{v} years going"]}),
        Fact("amount_requested", "int", about="the grant amount requested in dollars", values=(2000, 400000),
             log=True, avoid=((10000, 800), (25000, 2000), (50000, 4000), (100000, 8000)), fmt=usd,
             phrases={"formal": ["They request {v}.", "Amount requested: {v}.", "The proposal asks for {v}.",
                                 "The grant sought is {v}."],
                      "informal": ["asking {v}", "want {v}", "{v} ask", "request is {v}"]}),
        Fact("annual_budget", "int", about="the organisation's total annual budget in dollars",
             values=(15000, 3000000), log=True, fmt=usd,
             phrases={"formal": ["Their annual budget is {v}.", "Total yearly budget: {v}.",
                                 "The organisation spends {v} a year.", "Annual turnover: {v}."],
                      "informal": ["budget {v}/yr", "{v} a year", "annual budget {v}", "they run on {v}"]}),
    ],
    questions=[
        Question(
            id="decision", type="choice", text="What is the screening decision on this application?",
            paraphrases=("How should this proposal proceed?", "Decide on this grant application.",
                         "What should we do with this application?"),
            options=[Option("fund_in_full", "Recommend funding in full"),
                     Option("fund_partially", "Recommend partial funding"),
                     Option("request_information", "Ask for more information"),
                     Option("decline", "Decline")],
            rules={"default": Rule(DECISION_RULE.format(o=25), decision),
                   "overhead_35": Rule(DECISION_RULE.format(o=35), lambda f: decision(f, overhead=35))},
            needs=("registered_charity", "focus", "late_report", "overhead_pct", "years_operating",
                   "amount_requested", "annual_budget")),
        Question(
            id="review_level", type="score", text="At what level must this application be reviewed?",
            paraphrases=("Who reviews this application?", "Which review tier applies?",
                         "What approval level does this grant need?"),
            options=[Option("staff", "programme staff"), Option("committee", "grants committee"),
                     Option("board", "board of trustees")],
            rules={"default": Rule("The board reviews requests over $100,000, the grants committee requests over "
                                   "$25,000, and staff the rest.", level),
                   "lower": Rule("The board reviews requests over $50,000, the grants committee requests over "
                                 "$10,000, and staff the rest.", lambda f: level(f, (10000, 50000)))},
            needs=("amount_requested",)),
        Question(
            id="site_visit", type="noul", text="Is a site visit needed before a decision?",
            paraphrases=("Should we visit the applicant?", "Does this need a site visit?",
                         "Must a programme officer visit them first?"),
            rules={"default": Rule("Visit when the request is over $100,000 or the organisation has operated for "
                                   "under 3 years.",
                                   lambda f: "yes" if f["amount_requested"] > 100000 or f["years_operating"] < 3
                                   else "no"),
                   "over_25k": Rule("Visit when the request is over $25,000 or the organisation has operated for "
                                    "under 3 years.",
                                    lambda f: "yes" if f["amount_requested"] > 25000 or f["years_operating"] < 3
                                    else "no")},
            needs=("amount_requested", "years_operating")),
    ],
)
