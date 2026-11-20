"""Accounts payable: invoice exceptions before payment. All vendor names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def money(v):
    return f"${v:,.2f}"


def fraud_check(f, new_vendor_limit=10000):
    return f["bank_changed"] or (f["new_vendor"] and f["amount"] > new_vendor_limit)


def action(f, tolerance=5):
    if fraud_check(f):
        return "verify_vendor"
    if not f["has_po"] and f["amount"] > 5000:
        return "reject"
    if f["received_pct"] != "100%":
        return "hold_for_receipt"
    if f["variance_pct"] > tolerance:
        return "hold_for_price_review"
    return "pay"


def band(v, edges):
    return str(sum(v > e for e in edges))


ACTION_RULE = ("First, verify the vendor if its bank details changed, or if it is a new vendor billing over $10,000. "
               "Otherwise reject invoices without a purchase order above $5,000. Otherwise hold for receipt until "
               "100% of the goods are received. Otherwise hold for price review if the price is more than {t}% above "
               "the PO. Otherwise pay.")

DOMAIN = Domain(
    name="ap_invoice",
    title="Accounts payable invoice exceptions",
    entities={
        "org": ["Halvorsen Fixtures", "Brightwell Packaging", "Corvane Logistics", "Tessaro Components",
                "Mirefield Labs", "Ostrander Paper"],
        "person": ["Ines Duarte", "Marcus Hale", "Anya Petrov", "Joel Kim", "Fatima Rahman", "Pete Morrow"],
    },
    situations=[
        "Invoice received from {org}.",
        "{org} sent an invoice that needs a decision.",
        "Payment run check for an invoice from {org}.",
        "AP exception on a {org} invoice.",
        "{person} flagged an invoice from {org} for review.",
        "Three-way match result for {org}.",
    ],
    formats=("email", "ticket", "chat", "form", "report"),
    distractors=[
        "The month-end close calendar was shared on Monday.",
        "Remember that expense reports are due on the 5th.",
        "The ERP upgrade test window is next weekend.",
        "Vendor statements for last quarter are filed.",
    ],
    facts=[
        Fact("amount", "float", about="the invoice's total amount in dollars", values=(200, 250000), log=True,
             avoid=((1000, 80), (5000, 400), (10000, 800), (25000, 2000), (50000, 4000)), fmt=money,
             phrases={"formal": ["The invoice totals {v}.", "Invoice amount: {v}.", "They billed {v}.",
                                 "The total due is {v}.", "Amount payable: {v}."],
                      "informal": ["it's for {v}", "{v} total", "they want {v}", "bill is {v}"]}),
        Fact("variance_pct", "float", about="how far the invoiced price is above the purchase-order price, in percent", values=(0, 20), avoid=((2, 0.4), (3, 0.4), (5, 0.4), (8, 0.5), (10, 0.5),
                                                              (15, 0.6)), fmt=lambda v: f"{v:.1f}%",
             phrases={"formal": ["The invoiced price is {v} above the purchase order.",
                                 "Price variance against the PO: {v}.", "Unit prices exceed the PO by {v}.",
                                 "The price check shows the invoice {v} over the PO."],
                      "informal": ["price is {v} over the PO", "{v} above what we agreed", "they're {v} higher",
                                   "variance {v}"]}),
        Fact("received_pct", "cat", about="what share of the invoiced goods has been received", values=("100%", "90%", "75%", "50%", "20%"), p=(0.6, 0.1, 0.1, 0.1, 0.1),
             phrases={"formal": ["Receiving shows {v} of the invoiced quantity has arrived.",
                                 "Goods receipt covers {v} of the invoice lines.",
                                 "{v} of the goods are booked as received.",
                                 "The warehouse has received {v} of what was billed."],
                      "informal": ["we got {v} of the stuff", "{v} received so far", "warehouse says {v} in",
                                   "only {v} of it showed up"]}),
        Fact("bank_changed", "bool", about="whether the vendor's bank or payment details changed recently", p=0.15,
             phrases={"formal": {True: ["The vendor asked to change its bank details last week.",
                                        "The remittance account differs from the vendor master.",
                                        "New bank details were submitted with this invoice.",
                                        "The payee account was updated by email."],
                                 False: ["Bank details match the vendor master.", "The remittance account is unchanged.",
                                         "No change to payment details was requested.",
                                         "The payee account is the one on file."]},
                      "informal": {True: ["they changed bank details", "new account number on this one",
                                          "bank info is different now", "they emailed new bank details"],
                                   False: ["bank details same as always", "account unchanged", "no bank changes",
                                           "same payee account"]}}),
        Fact("new_vendor", "bool", about="whether this is a new vendor with no payment history", p=0.3,
             phrases={"formal": {True: ["This is the vendor's first invoice.", "The vendor was onboarded this month.",
                                        "It is a newly registered supplier.", "There is no payment history with them."],
                                 False: ["The vendor is long established.", "We have paid them for years.",
                                         "They are an existing supplier.", "There is a long payment history."]},
                      "informal": {True: ["new vendor", "first time we pay them", "just onboarded them",
                                          "never paid them before"],
                                   False: ["old vendor", "we pay them all the time", "regular supplier",
                                           "known vendor"]}}),
        Fact("has_po", "bool", about="whether the invoice has a valid purchase order", p=0.8,
             phrases={"formal": {True: ["It references a valid purchase order.", "A matching PO exists.",
                                        "The PO number checks out.", "It is a PO-backed invoice."],
                                 False: ["There is no purchase order for it.", "It is a non-PO invoice.",
                                         "No PO number is quoted and none can be found.",
                                         "Nobody raised a PO for this."]},
                      "informal": {True: ["PO is fine", "has a PO", "PO matches", "PO's there"],
                                   False: ["no PO", "no purchase order at all", "can't find a PO", "non-PO"]}}),
    ],
    questions=[
        Question(
            id="action", type="choice", text="What should AP do with this invoice?",
            paraphrases=("Decide the next step for this invoice.", "How should this invoice be processed?",
                         "What is the right action on this invoice?"),
            options=[Option("pay", "Release for payment"),
                     Option("hold_for_receipt", "Hold until the goods are fully received"),
                     Option("hold_for_price_review", "Hold for a price review with purchasing"),
                     Option("verify_vendor", "Verify the vendor before any payment"),
                     Option("reject", "Reject the invoice")],
            rules={"default": Rule(ACTION_RULE.format(t=5), action),
                   "tolerance_10": Rule(ACTION_RULE.format(t=10), lambda f: action(f, tolerance=10))},
            needs=("bank_changed", "new_vendor", "amount", "has_po", "received_pct", "variance_pct")),
        Question(
            id="variance_band", type="score", text="How large is the price variance?",
            paraphrases=("Classify the price variance.", "Which variance band does this invoice fall in?",
                         "Rate the price deviation from the PO."),
            options=[Option("0", "within tolerance"), Option("1", "small"), Option("2", "material"),
                     Option("3", "large")],
            rules={"default": Rule("Within tolerance up to 2% above the PO; small up to 5%; material up to 10%; "
                                   "large above 10%.", lambda f: band(f["variance_pct"], (2, 5, 10))),
                   "wide": Rule("Within tolerance up to 3% above the PO; small up to 8%; material up to 15%; large "
                                "above 15%.", lambda f: band(f["variance_pct"], (3, 8, 15)))},
            needs=("variance_pct",)),
        Question(
            id="fraud_review", type="noul", text="Does this invoice need a fraud review?",
            paraphrases=("Should fraud prevention check this invoice?", "Is a payment-fraud review required?",
                         "Send this to the fraud team?"),
            rules={"default": Rule("Review when the bank details changed, or when a new vendor bills over $10,000.",
                                   lambda f: "yes" if fraud_check(f) else "no"),
                   "low_limit": Rule("Review when the bank details changed, or when a new vendor bills over $1,000.",
                                     lambda f: "yes" if fraud_check(f, 1000) else "no")},
            needs=("bank_changed", "new_vendor", "amount")),
        Question(
            id="needs_approval", type="noul", text="Does this invoice need finance director approval?",
            paraphrases=("Is a director sign-off required?", "Must this go to the finance director?",
                         "Does the approval limit apply here?"),
            rules={"default": Rule("Invoices over $50,000 need finance director approval.",
                                   lambda f: "yes" if f["amount"] > 50000 else "no"),
                   "limit_25k": Rule("Invoices over $25,000 need finance director approval.",
                                     lambda f: "yes" if f["amount"] > 25000 else "no")},
            needs=("amount",)),
    ],
)
