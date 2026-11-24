"""Finance: employee expense claims. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule

CAPS = {"meals": 100, "taxi": 80, "hotel": 400, "airfare": 1500, "client entertainment": 500}


def money(v):
    return f"${v:,.2f}"


def decision(f, late=60):
    if f["days_since_expense"] > late:
        return "reject_late"
    if f["amount"] > 25 and not f["receipt"]:
        return "request_receipt"
    if f["amount"] > CAPS[f["expense_type"]] and not f["preapproved"]:
        return "approve_up_to_cap"
    return "approve"


DECISION_RULE = ("Reject expenses submitted more than {d} days after they were incurred. Otherwise ask for a receipt "
                 "for anything over $25 without one. Otherwise, if the amount is over the cap for its type (meals "
                 "$100, taxi $80, hotel $400, airfare $1,500, client entertainment $500) and was not approved "
                 "in advance, approve up to the cap. Otherwise approve in full.")

def audit(f, abroad_limit=500):
    a = f["amount"]
    return ((f["expense_type"] == "client entertainment" and a > 100) or a > 1000
            or (f["international"] and a > abroad_limit))


DOMAIN = Domain(
    name="expense_report",
    title="Employee expense claims",
    entities={
        "person": ["Leah Fontaine", "Andre Booker", "Sunita Rao", "Mateo Vargas", "Chloe Ashford", "Erik Dahl"],
        "city": ["Lisbon", "Denver", "Osaka", "Toronto", "Nairobi", "Lyon"],
    },
    situations=[
        "{person} submitted an expense claim from a trip to {city}.",
        "Expense line for review: {person}.",
        "New expense report item from {person}.",
        "{person}'s claim from {city} is waiting for approval.",
        "Expense audit queue: an item from {person}.",
        "Reimbursement request from {person}.",
    ],
    formats=("email", "ticket", "chat", "form", "report"),
    distractors=[
        "The travel booking tool is changing next quarter.",
        "Corporate card statements close on the 25th.",
        "Per diem rates are reviewed every January.",
        "The finance team is hiring an analyst.",
    ],
    facts=[
        Fact("expense_type", "cat", about="the type of expense",
             values=("meals", "taxi", "hotel", "airfare", "client entertainment"), p=(0.3, 0.2, 0.2, 0.15, 0.15),
             phrases={"formal": ["Expense type: {v}.", "The claim is for {v}.", "This line is a {v} expense.",
                                 "Category: {v}."],
                      "informal": ["{v}", "it's {v}", "{v} expense", "for {v}"]}),
        Fact("amount", "float", about="the amount claimed in dollars", values=(6, 3000), log=True,
             avoid=((25, 2), (80, 6), (100, 8), (400, 30), (500, 40), (1000, 80), (1500, 120)),
             fmt=money,
             phrases={"formal": ["The amount claimed is {v}.", "Claimed: {v}.", "The receipt total is {v}.",
                                 "They are asking to be reimbursed {v}."],
                      "informal": ["{v}", "claiming {v}", "came to {v}", "spent {v}"]}),
        Fact("receipt", "bool", about="whether a receipt is attached", p=0.8,
             phrases={"formal": {True: ["A receipt is attached.", "The itemised receipt was uploaded.",
                                        "Proof of payment is included.", "The receipt is on file."],
                                 False: ["No receipt is attached.", "The receipt is missing.",
                                         "There is no proof of payment.", "They did not upload a receipt."]},
                      "informal": {True: ["receipt attached", "has the receipt", "receipt's uploaded",
                                          "receipt on file"],
                                   False: ["no receipt", "receipt missing", "lost the receipt",
                                           "didn't attach a receipt"]}}),
        Fact("preapproved", "bool", about="whether the expense was approved in advance", p=0.3,
             phrases={"formal": {True: ["The expense was approved in advance by the manager.",
                                        "Pre-approval is on file.", "It was signed off before the trip.",
                                        "Advance approval was granted."],
                                 False: ["It was not approved in advance.", "There is no pre-approval.",
                                         "Nobody signed off on it beforehand.", "No advance approval exists."]},
                      "informal": {True: ["pre-approved", "manager ok'd it beforehand", "approved in advance",
                                          "had sign-off before"],
                                   False: ["not pre-approved", "no sign-off beforehand", "no pre-approval",
                                           "didn't get approval first"]}}),
        Fact("days_since_expense", "int", about="how many days ago the expense was incurred", values=(1, 120),
             log=True, avoid=((60, 3), (90, 4)),
             phrases={"formal": ["The expense was incurred {v} days ago.", "Days since the expense: {v}.",
                                 "It dates from {v} days ago.", "The spend happened {v} days before submission."],
                      "informal": ["{v} days ago", "from {v} days back", "spent {v} days ago",
                                   "{v} days old"]}),
        Fact("international", "bool", about="whether the expense was incurred abroad", p=0.3,
             phrases={"formal": {True: ["The expense was incurred abroad.", "It was paid in a foreign currency.",
                                        "This was on an international trip.", "The spend happened overseas."],
                                 False: ["The expense was incurred domestically.", "It was paid in local currency.",
                                         "This was on a domestic trip.", "The spend happened at home."]},
                      "informal": {True: ["abroad", "international trip", "foreign currency", "overseas"],
                                   False: ["domestic", "local trip", "in-country", "not abroad"]}}),
    ],
    questions=[
        Question(
            id="decision", type="choice", text="How should this expense claim be processed?",
            paraphrases=("Decide on this expense line.", "What happens to this claim?",
                         "Approve this expense or not?"),
            options=[Option("approve", "Approve in full"), Option("approve_up_to_cap", "Approve up to the policy cap"),
                     Option("request_receipt", "Ask for a receipt"),
                     Option("reject_late", "Reject: submitted too late")],
            rules={"default": Rule(DECISION_RULE.format(d=60), decision),
                   "late_90": Rule(DECISION_RULE.format(d=90), lambda f: decision(f, late=90))},
            needs=("days_since_expense", "amount", "receipt", "expense_type", "preapproved")),
        Question(
            id="audit", type="noul", text="Should this claim go to the expense audit team?",
            paraphrases=("Does this need an audit check?", "Flag this claim for audit?",
                         "Is an expense audit required here?"),
            rules={"default": Rule("Audit client entertainment over $100, any expense over $1,000, and any "
                                   "expense incurred abroad over $500.",
                                   lambda f: "yes" if audit(f) else "no"),
                   "all_abroad": Rule("Audit client entertainment over $100, any expense over $1,000, and every "
                                      "expense incurred abroad.",
                                      lambda f: "yes" if audit(f, abroad_limit=0) else "no")},
            needs=("expense_type", "amount", "international")),
    ],
)
