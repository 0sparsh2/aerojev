"""Parcel logistics: delivery exceptions at a customer service desk. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def money(v):
    return f"${v:,.2f}"


def action(f, late=5):
    s = f["status"]
    if s == "held at customs":
        return "contact_customs_broker"
    if s in ("lost", "delivered damaged"):
        if f["perishable"] or f["customer_tier"] == "key account":
            return "reship_now"
        return "refund" if f["declared_value"] < 100 else "reship_now"
    limit = 2 if f["perishable"] else late
    return "reship_now" if f["days_late"] > limit else "wait_and_monitor"


ACTION_RULE = ("Held at customs: contact the customs broker. Lost or delivered damaged: reship now if the goods are "
               "perishable or the customer is a key account; otherwise refund if the declared value is under $100, "
               "and reship if not. Still in transit or returned to sender: reship now if more than {d} days late "
               "(more than 2 days for perishable goods); otherwise wait and monitor.")

COMP = ["none", "small_voucher", "large_voucher", "shipping_refund"]


def compensation(f, key_bump=False):
    d = f["days_late"]
    lvl = 0 if d <= 2 else 1 if d <= 5 else 2 if d <= 10 else 3
    if key_bump and f["customer_tier"] == "key account":
        lvl = min(3, lvl + 1)
    return COMP[lvl]


COMP_RULE = ("No compensation up to 2 days late, a small voucher for 3 to 5 days, a large voucher for 6 to 10 days, "
             "and a refund of the shipping charge for more than 10 days late.")

DOMAIN = Domain(
    name="shipment_exception",
    title="Parcel delivery exceptions",
    entities={
        "person": ["Mira Jensen", "Tariq Aziz", "Ellen Park", "Bruno Costa", "Ngozi Eze", "Felix Ward"],
        "org": ["Quillon Home Goods", "Brightfarm Produce", "Corbel Office Supply", "Tidewater Pharmacy Supply"],
        "carrier": ["Swiftlane", "Parcelon", "NorthRoute Freight"],
    },
    situations=[
        "Customer {person} is asking about an order shipped with {carrier}.",
        "Delivery exception on an order for {org}.",
        "{org} contacted us about a late shipment.",
        "Shipment query from {person}: tracking shows a problem.",
        "{carrier} flagged an exception on a parcel for {org}.",
        "Order status complaint from {person}.",
    ],
    distractors=[
        "Our warehouse moves to the new site next month.",
        "The holiday shipping cut-off dates were published.",
        "The customer also asked to update their billing address.",
        "Carrier rate cards are being renegotiated.",
    ],
    facts=[
        Fact("status", "cat", about="the shipment's current tracking status",
             values=("in transit", "held at customs", "lost", "delivered damaged", "returned to sender"),
             p=(0.4, 0.15, 0.15, 0.15, 0.15),
             phrases={"formal": ["Tracking status: {v}.", "The parcel is currently {v}.",
                                 "The carrier lists the shipment as {v}.", "Latest scan shows the parcel as {v}."],
                      "informal": ["status says {v}", "it's {v}", "tracking: {v}", "showing {v}"]}),
        Fact("days_late", "int", about="how many days past the promised delivery date the shipment is",
             values=(0, 16),
             phrases={"formal": ["The shipment is {v} days past the promised date.", "Days late: {v}.",
                                 "It is {v} days behind the delivery promise.",
                                 "The promised date passed {v} days ago."],
                      "informal": ["{v} days late", "running {v} days behind", "{v} days past due",
                                   "late by {v} days"]}),
        Fact("perishable", "bool", about="whether the goods are perishable", p=0.25,
             phrases={"formal": {True: ["The contents are perishable.", "The order contains chilled food.",
                                        "The goods have a short shelf life.", "It is a temperature-sensitive order."],
                                 False: ["The contents are not perishable.", "The goods are shelf-stable.",
                                         "Nothing in the order spoils.", "It contains durable goods only."]},
                      "informal": {True: ["perishable stuff", "it's fresh food", "chilled goods", "will spoil"],
                                   False: ["not perishable", "shelf-stable", "nothing that spoils",
                                           "durable goods"]}}),
        Fact("declared_value", "float", about="the declared value of the shipment's contents in dollars",
             values=(15, 4000), log=True, avoid=((100, 8), (500, 40)), fmt=money,
             phrases={"formal": ["The declared value is {v}.", "Contents are valued at {v}.",
                                 "Declared value of the parcel: {v}.", "The order is worth {v}."],
                      "informal": ["worth {v}", "{v} order", "declared {v}", "value {v}"]}),
        Fact("customer_tier", "cat", about="the customer's account tier",
             values=("standard", "business", "key account"), p=(0.55, 0.3, 0.15),
             phrases={"formal": ["The customer is on the {v} tier.", "Account tier: {v}.",
                                 "This is a {v} customer.", "Their account is classed as {v}."],
                      "informal": ["{v} customer", "tier: {v}", "{v} acct", "they're {v}"]}),
        Fact("carrier_fault", "bool", about="whether the carrier is responsible for the problem", p=0.5,
             phrases={"formal": {True: ["The carrier has admitted fault.", "The delay is the carrier's responsibility.",
                                        "The carrier mis-sorted the parcel.", "The carrier accepts liability."],
                                 False: ["The carrier is not at fault.", "The problem was caused on our side.",
                                         "The delay was outside the carrier's control.",
                                         "The carrier is not responsible here."]},
                      "informal": {True: ["carrier's fault", "they messed up", "carrier admits it",
                                          "carrier screwed up"],
                                   False: ["not the carrier's fault", "our side caused it", "carrier not to blame",
                                           "outside their control"]}}),
    ],
    questions=[
        Question(
            id="action", type="choice", text="What should we do about this shipment?",
            paraphrases=("What is the next step for this order?", "How should this delivery exception be handled?",
                         "Choose the action for this shipment."),
            options=[Option("wait_and_monitor", "Wait and keep monitoring"),
                     Option("reship_now", "Send a replacement now"),
                     Option("refund", "Refund the order"),
                     Option("contact_customs_broker", "Contact the customs broker")],
            rules={"default": Rule(ACTION_RULE.format(d=5), action),
                   "late_10": Rule(ACTION_RULE.format(d=10), lambda f: action(f, late=10))},
            needs=("status", "perishable", "customer_tier", "declared_value", "days_late")),
        Question(
            id="compensation", type="score", text="How much compensation should the customer get for the delay?",
            paraphrases=("What goodwill should we offer for the delay?", "Which compensation level applies?",
                         "Pick the delay compensation."),
            options=[Option("none", "no compensation"), Option("small_voucher", "small voucher"),
                     Option("large_voucher", "large voucher"), Option("shipping_refund", "refund of shipping")],
            rules={"default": Rule(COMP_RULE, compensation),
                   "key_bump": Rule(COMP_RULE + " Key accounts get one level more (at most a shipping refund).",
                                    lambda f: compensation(f, key_bump=True))},
            needs=("days_late", "customer_tier")),
        Question(
            id="carrier_claim", type="noul", text="Should we file a claim with the carrier?",
            paraphrases=("Open a carrier claim for this one?", "Is a claim against the carrier warranted?",
                         "Do we claim this back from the carrier?"),
            rules={"default": Rule("File a claim when the carrier is at fault and the declared value is $100 or more.",
                                   lambda f: "yes" if f["carrier_fault"] and f["declared_value"] >= 100 else "no"),
                   "high_value": Rule("File a claim when the carrier is at fault and the declared value is $500 or "
                                      "more.",
                                      lambda f: "yes" if f["carrier_fault"] and f["declared_value"] >= 500 else
                                      "no")},
            needs=("carrier_fault", "declared_value")),
    ],
)
