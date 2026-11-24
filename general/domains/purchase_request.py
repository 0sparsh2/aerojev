"""Procurement: internal purchase requests. All supplier and staff names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def money(v):
    return f"${v:,.2f}"


def approver(f, limits=(1000, 10000, 100000)):
    a = f["amount"]
    return ["team_lead", "manager", "director", "cfo"][sum(a > x for x in limits)]


def next_step(f, quote_limit=10000):
    if not f["within_budget"]:
        return "reject_no_budget"
    if f["category"] == "software" and not f["security_reviewed"]:
        return "security_review"
    if f["amount"] > quote_limit and not f["preferred_supplier"] and f["quotes"] < 3:
        return "request_more_quotes"
    return "approve"


STEP_RULE = ("Reject if the cost center does not have enough budget left. Otherwise software from a vendor IT security "
             "has not reviewed goes to security review. Otherwise {scope} from a non-preferred supplier need at least "
             "3 quotes: request more if there are fewer. Otherwise approve.")

DOMAIN = Domain(
    name="purchase_request",
    title="Internal purchase requests",
    entities={
        "person": ["Clara Benson", "Yusuf Demir", "Megan Hollis", "Rahul Iyer", "Sienna Moreau", "Gabe Lindgren"],
        "supplier": ["Vantor Systems", "Kestrel Office Co", "Bramble Consulting", "Orvane Hardware",
                     "Lumio Software"],
        "dept": ["Marketing", "Engineering", "Facilities", "Finance", "Customer Success"],
    },
    situations=[
        "{person} in {dept} raised a purchase request.",
        "New requisition from {dept} for {supplier}.",
        "Purchase request: {supplier}, requested by {person}.",
        "{dept} wants to buy from {supplier}.",
        "Requisition waiting in the procurement queue from {person}.",
        "{person} asked procurement to place an order with {supplier}.",
    ],
    distractors=[
        "The procurement portal gets a new look next month.",
        "Year-end purchasing deadlines were announced.",
        "Supplier diversity training is on Thursday.",
        "Office supplies are now ordered through the new catalog.",
    ],
    facts=[
        Fact("amount", "float", about="the total cost of the purchase in dollars", values=(80, 400000), log=True,
             avoid=((500, 40), (1000, 80), (5000, 400), (10000, 800), (50000, 4000), (100000, 8000)), fmt=money,
             phrases={"formal": ["The total is {v}.", "Requested spend: {v}.", "The order comes to {v}.",
                                 "The purchase would cost {v}."],
                      "informal": ["{v} total", "costs {v}", "it's {v}", "about {v} all in"]}),
        Fact("category", "cat", about="what is being bought",
             values=("software", "hardware", "consulting", "office supplies", "travel"),
             p=(0.3, 0.2, 0.2, 0.15, 0.15),
             phrases={"formal": ["The request is for {v}.", "Category: {v}.", "This is a {v} purchase.",
                                 "Spend category: {v}."],
                      "informal": ["{v}", "buying {v}", "it's {v}", "category {v}"]}),
        Fact("preferred_supplier", "bool", about="whether the supplier is on the preferred supplier list", p=0.55,
             phrases={"formal": {True: ["The supplier is on the preferred list.", "It is a preferred supplier.",
                                        "We have a framework agreement with the supplier.",
                                        "The vendor is contracted as preferred."],
                                 False: ["The supplier is not on the preferred list.", "It is a new supplier to us.",
                                         "There is no framework agreement with them.",
                                         "The vendor is not a preferred supplier."]},
                      "informal": {True: ["preferred supplier", "on the preferred list", "we have a contract w them",
                                          "framework vendor"],
                                   False: ["not preferred", "not on the list", "new vendor for us",
                                           "no framework deal"]}}),
        Fact("quotes", "int", about="how many competing quotes were collected", values=(0, 4),
             p=(0.25, 0.3, 0.2, 0.15, 0.1),
             phrases={"formal": ["{v} quotes were collected.", "Competing quotes attached: {v}.",
                                 "The requester obtained {v} quotes.", "Number of quotes: {v}."],
                      "informal": ["{v} quotes", "got {v} quotes", "quotes: {v}", "{v} bids in"]}),
        Fact("within_budget", "bool", about="whether the cost center has enough budget left for it", p=0.82,
             phrases={"formal": {True: ["The cost center has enough budget left.", "Budget is available.",
                                        "The spend fits within the remaining budget.", "Funds are available."],
                                 False: ["The cost center has no budget left for this.", "Budget is exhausted.",
                                         "The spend exceeds the remaining budget.", "Funds are not available."]},
                      "informal": {True: ["budget's there", "we have budget", "fits the budget", "funds ok"],
                                   False: ["out of budget", "no budget left", "over budget", "no funds"]}}),
        Fact("security_reviewed", "bool", about="whether IT security has already reviewed the vendor", p=0.5,
             phrases={"formal": {True: ["IT security has reviewed the vendor.", "The vendor passed a security review.",
                                        "A security assessment is on file.", "Security cleared the vendor."],
                                 False: ["IT security has not reviewed the vendor.", "No security review is on file.",
                                         "The vendor has not been assessed by security.",
                                         "Security has not looked at the vendor."]},
                      "informal": {True: ["security reviewed them", "sec review done", "cleared by security",
                                          "security signed off on them"],
                                   False: ["no security review", "security hasn't seen them", "not sec reviewed",
                                           "security review missing"]}}),
    ],
    questions=[
        Question(
            id="approver", type="score", text="What approval level does this purchase need?",
            paraphrases=("Who must approve this request?", "Which approval tier applies?",
                         "What is the required sign-off level?"),
            options=[Option("team_lead", "team lead"), Option("manager", "manager"), Option("director", "director"),
                     Option("cfo", "CFO")],
            rules={"default": Rule("Team lead up to $1,000, manager up to $10,000, director up to $100,000, CFO "
                                   "above that.", approver),
                   "tight": Rule("Team lead up to $500, manager up to $5,000, director up to $50,000, CFO above "
                                 "that.", lambda f: approver(f, (500, 5000, 50000)))},
            needs=("amount",)),
        Question(
            id="next_step", type="choice", text="What should procurement do with this request?",
            paraphrases=("What is the next step for this requisition?", "How should this purchase request proceed?",
                         "Decide on this purchase request."),
            options=[Option("approve", "Approve and place the order"),
                     Option("request_more_quotes", "Ask for more quotes"),
                     Option("security_review", "Send to IT security review"),
                     Option("reject_no_budget", "Reject: no budget")],
            rules={"default": Rule(STEP_RULE.format(scope="purchases over $10,000"), next_step),
                   "quotes_all": Rule(STEP_RULE.format(scope="purchases of any amount"),
                                      lambda f: next_step(f, quote_limit=0))},
            needs=("within_budget", "category", "security_reviewed", "amount", "preferred_supplier", "quotes")),
        Question(
            id="competitive_bid", type="noul", text="Must this purchase go out to competitive bid?",
            paraphrases=("Is a formal tender required?", "Does this need a competitive bidding process?",
                         "Must we run an RFP for this?"),
            rules={"default": Rule("A competitive bid is required over $50,000, and over $10,000 for consulting.",
                                   lambda f: "yes" if f["amount"] > 50000 or
                                   (f["category"] == "consulting" and f["amount"] > 10000) else "no"),
                   "all_10k": Rule("A competitive bid is required for any purchase over $10,000.",
                                   lambda f: "yes" if f["amount"] > 10000 else "no")},
            needs=("amount", "category")),
    ],
)
