"""Retail customer service: return and refund requests. All store and customer names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def resolution(f, window=30):
    d, c = f["days_since_delivery"], f["condition"]
    if c in ("defective", "damaged in transit"):
        return "full_refund" if d <= window else "replacement"
    if c == "unopened":
        if d <= window and f["has_receipt"]:
            return "full_refund"
        return "store_credit" if d <= 60 else "deny"
    if c == "opened":
        if d <= window and f["has_receipt"]:
            return "partial_refund"
        return "store_credit" if d <= 60 else "deny"
    return "store_credit" if f["tier"] == "premium" and d <= 14 else "deny"  # used


RESOLUTION_RULE = ("Defective items, or items damaged in transit: full refund within {w} days of delivery, "
                   "otherwise a replacement. Unopened items: full refund within {w} days with a receipt, otherwise "
                   "store credit up to 60 days, then deny. Opened items: partial refund within {w} days with a "
                   "receipt, otherwise store credit up to 60 days, then deny. Used items: deny, except store credit "
                   "for premium members within 14 days.")


def abuse(f, limit=5):
    return f["prior_returns"] >= limit or (f["order_value"] > 500 and not f["has_receipt"])


DOMAIN = Domain(
    name="retail_returns",
    title="Retail return and refund requests",
    entities={
        "person": ["Grace Liu", "Omar Haddad", "Beth Carver", "Diego Ramos", "Hana Sato", "Will Osei"],
        "item": ["blender", "desk lamp", "pair of running shoes", "headphones", "coffee grinder", "rain jacket",
                 "board game", "air purifier"],
        "org": ["Larkspur Outfitters", "Kettlebrook Home", "Pinecrest Electronics"],
    },
    situations=[
        "{person} wants to return a {item} bought from {org}.",
        "Return request for a {item}.",
        "Customer {person} is asking for a refund on a {item}.",
        "{person} contacted {org} support about a {item} they want to send back.",
        "A return was opened for a {item}.",
        "Refund inquiry: {item}, customer {person}.",
    ],
    formats=("email", "ticket", "chat", "form", "report"),
    distractors=[
        "The customer also asked about store opening hours.",
        "Our summer sale starts next Monday.",
        "They mentioned they like the new loyalty app.",
        "Shipping times are slightly longer this week.",
    ],
    facts=[
        Fact("days_since_delivery", "int", about="how many days ago the order was delivered", values=(0, 90), avoid=((14, 1), (30, 1), (60, 2)),
             phrases={"formal": ["The order was delivered {v} days ago.", "Delivery was {v} days ago.",
                                 "It has been {v} days since delivery.", "Days since delivery: {v}."],
                      "informal": ["got it {v} days ago", "delivered {v} days back", "had it {v} days",
                                   "{v} days since it arrived"]}),
        Fact("condition", "cat", about="the condition of the item being returned", values=("unopened", "opened", "used", "defective", "damaged in transit"),
             p=(0.25, 0.25, 0.15, 0.2, 0.15),
             phrases={"unopened": ["The item is still sealed in its box.", "It has not been opened.",
                                   "The packaging is unopened.", "It is still in the original sealed packaging."],
                      "opened": ["The box was opened but the item looks unused.",
                                 "It was taken out of the box and tried once.",
                                 "The packaging is open; the item is like new.",
                                 "They opened it but did not really use it."],
                      "used": ["The item has clearly been used for a while.", "It shows wear from regular use.",
                               "They used it for several weeks.", "It is visibly worn."],
                      "defective": ["The item does not work properly.", "It stopped working after a day.",
                                    "There is a manufacturing fault.", "It is faulty out of the box."],
                      "damaged in transit": ["It arrived broken.", "The carrier damaged it on the way.",
                                             "It was cracked when it arrived.", "The parcel arrived crushed."]}),
        Fact("order_value", "float", about="the price the customer paid for the order", values=(5, 2000), log=True, avoid=((100, 8), (500, 30)),
             fmt=lambda v: f"${v:,.2f}",
             phrases={"formal": ["The order value was {v}.", "They paid {v}.", "Purchase price: {v}.",
                                 "The item cost {v}."],
                      "informal": ["paid {v}", "it was {v}", "cost them {v}", "{v} order"]}),
        Fact("has_receipt", "bool", about="whether the customer has a receipt or proof of purchase", p=0.75,
             phrases={"formal": {True: ["They have the receipt.", "Proof of purchase was attached.",
                                        "The order number is confirmed.", "A valid receipt is on file."],
                                 False: ["They have no receipt.", "No proof of purchase was provided.",
                                         "The order cannot be found without a receipt.",
                                         "They lost the receipt."]},
                      "informal": {True: ["has the receipt", "sent proof of purchase", "order # checks out",
                                          "receipt attached"],
                                   False: ["no receipt", "lost the receipt", "can't prove purchase",
                                           "no order number"]}}),
        Fact("prior_returns", "int", about="how many returns the customer made in the past 12 months", values=(0, 9),
             phrases={"formal": ["The customer made {v} returns in the past 12 months.",
                                 "Return history: {v} in the last year.", "They have returned {v} items this year.",
                                 "Prior returns in 12 months: {v}."],
                      "informal": ["{v} returns this year already", "they've sent back {v} things this year",
                                   "{v} past returns", "returned {v} items in the last 12 months"]}),
        Fact("tier", "cat", about="the customer's loyalty membership tier", values=("standard", "plus", "premium"), p=(0.6, 0.25, 0.15),
             phrases={"formal": ["The customer is a {v} member.", "Membership tier: {v}.",
                                 "They hold a {v} membership.", "Loyalty status: {v}."],
                      "informal": ["{v} member", "they're {v} tier", "{v} customer", "on the {v} plan"]}),
    ],
    questions=[
        Question(
            id="resolution", type="choice", text="How should this return be resolved?",
            paraphrases=("What should we offer the customer?", "Decide the outcome of this return request.",
                         "Which resolution applies to this return?"),
            options=[Option("full_refund", "Full refund to the original payment method"),
                     Option("partial_refund", "Partial refund (restocking fee applies)"),
                     Option("replacement", "Send a replacement item"),
                     Option("store_credit", "Store credit for the item's value"),
                     Option("deny", "Decline the return")],
            rules={"default": Rule(RESOLUTION_RULE.format(w=30), resolution),
                   "window_14": Rule(RESOLUTION_RULE.format(w=14), lambda f: resolution(f, window=14))},
            needs=("days_since_delivery", "condition", "has_receipt", "tier")),
        Question(
            id="abuse_flag", type="noul", text="Should this return be flagged for possible abuse?",
            paraphrases=("Is this a potential return-abuse case?", "Flag this customer for review?",
                         "Does this look like serial returning?"),
            rules={"default": Rule("Flag customers with 5 or more returns in 12 months, and any return over $500 "
                                   "without a receipt.", lambda f: "yes" if abuse(f) else "no"),
                   "strict": Rule("Flag customers with 3 or more returns in 12 months, and any return over $500 "
                                  "without a receipt.", lambda f: "yes" if abuse(f, 3) else "no")},
            needs=("prior_returns", "order_value", "has_receipt")),
        Question(
            id="handling_priority", type="score", text="How quickly should this request be handled?",
            paraphrases=("What handling priority does this return get?", "How urgent is this return?",
                         "Set the queue priority for this request."),
            options=[Option("low", "standard queue"), Option("normal", "priority queue"),
                     Option("expedite", "same-day handling")],
            rules={"default": Rule(
                "Same-day handling for premium members and for orders over $500. Priority queue for plus members "
                "and for defective or transit-damaged items. Everything else goes to the standard queue.",
                lambda f: "expedite" if f["tier"] == "premium" or f["order_value"] > 500 else
                "normal" if f["tier"] == "plus" or f["condition"] in ("defective", "damaged in transit") else "low")},
            needs=("tier", "order_value", "condition")),
    ],
)
