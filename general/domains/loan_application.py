"""Consumer lending: personal loan applications at pre-screen. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def usd(v):
    return f"${v:,}"


def decision(f, decline_dti=50):
    if f["prior_default"] or f["credit_score"] < 580 or f["dti_pct"] > decline_dti:
        return "decline"
    if f["credit_score"] < 640 or f["years_employed"] < 2:
        return "manual_review"
    if f["loan_amount"] > 25000 or f["dti_pct"] > 36:
        return "approve_lower_amount"
    return "approve"


DECISION_RULE = ("Decline if the applicant defaulted on a loan in the past 7 years, the credit score is below 580, or "
                 "debt-to-income is above {d}%. Otherwise send to manual review if the credit score is below 640 or "
                 "the applicant has been in their current job for less than 2 years. Otherwise offer a lower amount "
                 "if the request is over $25,000 or debt-to-income is above 36%. Otherwise approve.")


def risk(f, score_only=False):
    s, d = f["credit_score"], f["dti_pct"]
    if score_only:
        return "very_high" if s < 580 else "high" if s < 640 else "medium" if s < 700 else "low"
    if f["prior_default"]:
        return "very_high"
    if s < 620 or d > 43:
        return "high"
    if s < 700 or d > 36:
        return "medium"
    return "low"


DOMAIN = Domain(
    name="loan_application",
    title="Personal loan application pre-screening",
    entities={
        "person": ["Nadia Kowalski", "Brian Achebe", "Lena Fischer", "Carlos Mendoza", "Aiko Mori", "Sean Doyle"],
        "org": ["Ridgeway Credit Union", "Northfield Savings", "Bluegate Lending"],
    },
    situations=[
        "{person} applied for a personal loan with {org}.",
        "New personal loan application from {person}.",
        "Pre-screen request for {person}'s loan application.",
        "Loan application received online from {person}.",
        "{org} needs a pre-screen decision on {person}'s application.",
        "Underwriting queue item: personal loan, applicant {person}.",
    ],
    formats=("email", "ticket", "chat", "form", "report"),
    distractors=[
        "Branch opening hours change next month.",
        "The new mobile banking app update is rolling out.",
        "Interest rate sheets for next quarter are not out yet.",
        "The applicant also asked about opening a savings account.",
    ],
    facts=[
        Fact("credit_score", "int", about="the applicant's credit score", values=(520, 850),
             avoid=((580, 4), (620, 4), (640, 4), (700, 4)),
             phrases={"formal": ["The applicant's credit score is {v}.", "Credit bureau score: {v}.",
                                 "The bureau report shows a score of {v}.", "Their credit score comes back at {v}."],
                      "informal": ["score is {v}", "credit {v}", "bureau says {v}", "pulled credit, {v}"]}),
        Fact("dti_pct", "int", about="the applicant's debt-to-income ratio, in percent", values=(8, 58),
             avoid=((36, 1), (43, 1), (50, 1)), fmt=lambda v: f"{v}%",
             phrases={"formal": ["Debt-to-income ratio: {v}.", "Monthly debt payments are {v} of gross income.",
                                 "The applicant's DTI is {v}.", "Existing debts take up {v} of their income."],
                      "informal": ["DTI {v}", "debt to income is {v}", "{v} DTI", "debts eat {v} of income"]}),
        Fact("loan_amount", "int", about="the loan amount requested, in dollars", values=(1000, 60000), log=True,
             avoid=((10000, 600), (25000, 1500)), fmt=usd,
             phrases={"formal": ["The requested amount is {v}.", "They are asking to borrow {v}.",
                                 "Loan amount requested: {v}.", "The application is for {v}."],
                      "informal": ["wants {v}", "asking for {v}", "{v} loan", "request is {v}"]}),
        Fact("years_employed", "int", about="how many years the applicant has been in their current job",
             values=(0, 20),
             phrases={"formal": ["The applicant has been with their current employer for {v} years.",
                                 "Time in current job: {v} years.", "They have held their present job for {v} years.",
                                 "Employment history shows {v} years at the current employer."],
                      "informal": ["{v} yrs at current job", "been at their job {v} years", "same employer {v} years",
                                   "{v} years in the job"]}),
        Fact("prior_default", "bool", about="whether the applicant defaulted on a loan in the past 7 years", p=0.12,
             phrases={"formal": {True: ["The bureau report shows a loan default within the last 7 years.",
                                        "The applicant defaulted on a car loan four years ago.",
                                        "A past default appears on the credit file.",
                                        "There is a charged-off loan in the past 7 years."],
                                 False: ["There are no defaults on the credit file.",
                                         "The applicant has never defaulted on a loan.",
                                         "The credit file shows no charge-offs or defaults.",
                                         "No missed-loan defaults are recorded."]},
                      "informal": {True: ["had a default a few yrs back", "default on file", "charged-off loan in 2023",
                                          "they defaulted before"],
                                   False: ["no defaults", "clean file, never defaulted", "no charge-offs",
                                           "zero defaults"]}}),
        Fact("existing_customer", "bool", about="whether the applicant already banks with the lender", p=0.4,
             phrases={"formal": {True: ["The applicant already holds a checking account with us.",
                                        "They are an existing customer.", "They have banked with us for some years.",
                                        "The applicant is a current member."],
                                 False: ["The applicant is new to the bank.", "They have no accounts with us.",
                                         "This is a first-time customer.", "They are not yet a member."]},
                      "informal": {True: ["existing customer", "already banks with us", "current member",
                                          "has a checking acct here"],
                                   False: ["new customer", "no accounts with us", "not a member yet",
                                           "first time with us"]}}),
        Fact("purpose", "cat", about="what the loan is for",
             values=("debt consolidation", "home improvement", "a used car", "a wedding", "medical bills"),
             phrases={"formal": ["The stated purpose is {v}.", "Loan purpose: {v}.", "The funds are for {v}.",
                                 "They want the money for {v}."],
                      "informal": ["it's for {v}", "purpose: {v}", "needs it for {v}", "{v}"]}),
    ],
    questions=[
        Question(
            id="decision", type="choice", text="What should the pre-screen decision be?",
            paraphrases=("Decide on this loan application.", "How should this application be handled?",
                         "What is the outcome of the pre-screen?"),
            options=[Option("approve", "Approve the requested amount"),
                     Option("approve_lower_amount", "Approve a lower amount than requested"),
                     Option("manual_review", "Send to a human underwriter"),
                     Option("decline", "Decline the application")],
            rules={"default": Rule(DECISION_RULE.format(d=50), decision),
                   "dti_43": Rule(DECISION_RULE.format(d=43), lambda f: decision(f, decline_dti=43))},
            needs=("prior_default", "credit_score", "dti_pct", "years_employed", "loan_amount")),
        Question(
            id="risk_tier", type="score", text="What risk tier is this applicant?",
            paraphrases=("Rate the applicant's credit risk.", "Which risk band does this application fall in?",
                         "How risky is this borrower?"),
            options=[Option("low", "low risk"), Option("medium", "medium risk"), Option("high", "high risk"),
                     Option("very_high", "very high risk")],
            rules={"default": Rule("Very high if there is a default in the past 7 years. Otherwise high if the credit "
                                   "score is below 620 or debt-to-income is above 43%. Otherwise medium if the score "
                                   "is below 700 or debt-to-income is above 36%. Otherwise low.", risk),
                   "score_only": Rule("Use the credit score only: very high below 580, high below 640, medium below "
                                      "700, low from 700.", lambda f: risk(f, score_only=True))},
            needs=("prior_default", "credit_score", "dti_pct")),
        Question(
            id="verify_income", type="noul", text="Do we need income documents before deciding?",
            paraphrases=("Is income verification required?", "Should we ask for pay stubs or tax returns?",
                         "Must the applicant prove their income?"),
            rules={"default": Rule("Verify income for loans over $10,000 and for applicants in their current job "
                                   "for less than 2 years.",
                                   lambda f: "yes" if f["loan_amount"] > 10000 or f["years_employed"] < 2 else "no"),
                   "new_customers": Rule("Verify income for loans over $10,000, for applicants in their current job "
                                         "for less than 2 years, and for anyone who is not already a customer.",
                                         lambda f: "yes" if f["loan_amount"] > 10000 or f["years_employed"] < 2
                                         or not f["existing_customer"] else "no")},
            needs=("loan_amount", "years_employed", "existing_customer")),
    ],
)
