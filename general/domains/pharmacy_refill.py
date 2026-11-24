"""Retail pharmacy: refill request handling (administrative rules only, no clinical advice). Held out of training
(test-unseen). All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule

EARLIEST = {"30-day": 23, "90-day": 68}


def action(f, review_90=False):
    if f["expired"] or f["refills_left"] == 0:
        return "contact_prescriber"
    if f["days_since_fill"] < EARLIEST[f["supply"]]:
        return "too_early"
    if f["insurance_rejected"]:
        return "insurance_follow_up"
    review = f["controlled"] or (review_90 and f["supply"] == "90-day")
    return "pharmacist_review" if review else "fill_now"


ACTION_RULE = ("Contact the prescriber if the prescription has expired or has no refills left. Otherwise it is too "
               "early before day 23 of a 30-day supply or day 68 of a 90-day supply. Otherwise follow up with the "
               "insurer if it rejected the claim. Otherwise {r} go to pharmacist review, and everything else is "
               "filled now.")

DOMAIN = Domain(
    name="pharmacy_refill",
    title="Pharmacy refill requests",
    heldout="test",
    entities={
        "person": ["Harold Finch", "Imani Brooks", "Rosa Delgado", "Stefan Novak", "Mei-Ling Chou", "Patrick Byrne"],
        "store": ["the Main Street pharmacy", "the Westgate branch", "the drive-through location"],
    },
    situations=[
        "{person} requested a refill at {store}.",
        "Refill request from {person}.",
        "Automated refill queue: patient {person}.",
        "{person} called {store} about a refill.",
        "Online refill order for {person}.",
        "Refill request waiting at {store}.",
    ],
    formats=("email", "ticket", "chat", "form", "report", "log"),
    distractors=[
        "The flu shot clinic runs every Saturday this month.",
        "The store closes early on the public holiday.",
        "A new pharmacy technician starts on Monday.",
        "The loyalty card scheme now covers vitamins.",
    ],
    facts=[
        Fact("supply", "cat", about="how many days the last fill was meant to last", values=("30-day", "90-day"),
             p=(0.6, 0.4),
             phrases={"formal": ["The last fill was a {v} supply.", "Supply length: {v}.",
                                 "It was dispensed as a {v} supply.", "The prescription is filled as {v} supplies."],
                      "informal": ["{v} supply", "{v} fill", "gets a {v} supply", "on {v} fills"]}),
        Fact("days_since_fill", "int", about="how many days ago the last fill was dispensed", values=(10, 100),
             avoid=((23, 1), (68, 2)),
             phrases={"formal": ["The last fill was {v} days ago.", "Days since last dispensing: {v}.",
                                 "It was last dispensed {v} days ago.", "{v} days have passed since the last fill."],
                      "informal": ["last filled {v} days ago", "{v} days since last fill", "filled {v}d ago",
                                   "{v} days since pickup"]}),
        Fact("refills_left", "int", about="how many refills remain on the prescription", values=(0, 5),
             p=(0.15, 0.25, 0.2, 0.15, 0.13, 0.12),
             phrases={"formal": ["{v} refills remain on the prescription.", "Refills remaining: {v}.",
                                 "The prescription has {v} refills left.", "Remaining refills: {v}."],
                      "informal": ["{v} refills left", "refills: {v}", "{v} left on the script", "{v} refills remain"]}),
        Fact("expired", "bool", about="whether the prescription has expired", p=0.12,
             phrases={"formal": {True: ["The prescription has expired.", "The prescription is past its expiry date.",
                                        "It expired last month.", "The prescription is no longer valid."],
                                 False: ["The prescription is still valid.", "It has not expired.",
                                         "The prescription is current.", "It is within its validity period."]},
                      "informal": {True: ["script expired", "Rx is expired", "past expiry", "no longer valid"],
                                   False: ["script still valid", "not expired", "Rx current", "valid"]}}),
        Fact("controlled", "bool", about="whether the medicine is a controlled substance", p=0.2,
             phrases={"formal": {True: ["The medicine is a controlled substance.", "It is a scheduled drug.",
                                        "This is a controlled medication.", "The item is on the controlled list."],
                                 False: ["The medicine is not a controlled substance.", "It is not a scheduled drug.",
                                         "This is a standard prescription item.",
                                         "The item is not on the controlled list."]},
                      "informal": {True: ["controlled", "it's a scheduled drug", "CII-V item", "controlled med"],
                                   False: ["not controlled", "regular Rx", "non-controlled", "standard item"]}}),
        Fact("insurance_rejected", "bool", about="whether the insurer rejected the claim for this fill", p=0.12,
             phrases={"formal": {True: ["The insurer rejected the claim.", "The claim came back denied.",
                                        "Insurance returned a rejection code.", "Coverage was refused for this fill."],
                                 False: ["The insurer accepted the claim.", "The claim went through.",
                                         "Insurance approved the fill.", "Coverage was confirmed for this fill."]},
                      "informal": {True: ["insurance rejected", "claim denied", "got a reject code",
                                          "ins won't cover"],
                                   False: ["claim went through", "insurance ok", "approved by ins",
                                           "covered"]}}),
    ],
    questions=[
        Question(
            id="action", type="choice", text="What should the pharmacy do with this refill request?",
            paraphrases=("How should this refill be handled?", "What is the next step for this refill?",
                         "Decide on this refill request."),
            options=[Option("fill_now", "Fill the prescription now"), Option("too_early", "Too early to refill"),
                     Option("contact_prescriber", "Contact the prescriber"),
                     Option("insurance_follow_up", "Follow up with the insurer"),
                     Option("pharmacist_review", "Pharmacist review before filling")],
            rules={"default": Rule(ACTION_RULE.format(r="controlled substances"), action),
                   "review_90": Rule(ACTION_RULE.format(r="controlled substances and all 90-day supplies"),
                                     lambda f: action(f, review_90=True))},
            needs=("expired", "refills_left", "supply", "days_since_fill", "controlled", "insurance_rejected")),
        Question(
            id="counselling", type="noul", text="Should the pharmacist offer counselling with this fill?",
            paraphrases=("Does this patient get a counselling offer?", "Offer a consultation with the pharmacist?",
                         "Is pharmacist counselling needed for this pickup?"),
            rules={"default": Rule("Offer counselling for controlled substances, and when this is the last refill "
                                   "(exactly 1 refill remains before this fill).",
                                   lambda f: "yes" if f["controlled"] or f["refills_left"] == 1 else "no"),
                   "low_refills": Rule("Offer counselling for controlled substances, and when 2 or fewer refills "
                                       "remain (but more than none).",
                                       lambda f: "yes" if f["controlled"] or 1 <= f["refills_left"] <= 2 else "no")},
            needs=("controlled", "refills_left")),
    ],
)
