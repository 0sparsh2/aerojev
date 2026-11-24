"""Online community: user reports in a content moderation queue. All names and handles are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def action(f, strikes=2):
    if not f["confirmed"]:
        return "no_action"
    if f["verified"]:
        return "escalate_policy"
    if f["category"] == "misinformation":
        return "add_label"
    new_spam = f["account_age_days"] < 30 and f["category"] in ("spam", "impersonation")
    if f["prior_strikes"] >= strikes or new_spam:
        return "suspend_account"
    return "remove_content"


ACTION_RULE = ("If a reviewer has not confirmed a policy violation, take no action. Confirmed violations by verified "
               "accounts go to the policy team. Otherwise misinformation gets a warning label. Otherwise suspend the "
               "account if it has {s} or more prior strikes, or if it is under 30 days old and the violation is spam "
               "or impersonation. Otherwise remove the content.")

BASE = {"spam": 0, "impersonation": 1, "misinformation": 1, "harassment": 2, "hate speech": 2}


def severity(f, reports=50):
    return ["low", "medium", "high"][min(2, BASE[f["category"]] + (f["report_count"] >= reports))]


SEV_RULE = ("Spam is low, impersonation and misinformation are medium, harassment and hate speech are high. Raise "
            "the level by one (at most high) when {r} or more users reported it.")

DOMAIN = Domain(
    name="content_moderation",
    title="Community content moderation queue",
    entities={
        "handle": ["@sunny_trails", "@datadave", "@moonlit_kat", "@gearhead77", "@quietfox", "@the_real_tom"],
        "place": ["the gardening forum", "the gaming community", "the local news group", "the marketplace",
                  "the comments section"],
    },
    situations=[
        "A post by {handle} in {place} was reported.",
        "Moderation queue item: content from {handle}.",
        "User reports came in about {handle}.",
        "Review needed for a comment by {handle} in {place}.",
        "{handle}'s post in {place} reached the review queue.",
        "Flagged content from {handle}.",
    ],
    distractors=[
        "The community guidelines page was redesigned.",
        "Moderator training sessions resume next month.",
        "The forum will be read-only during the upgrade on Sunday.",
        "A new moderator joined the evening shift.",
    ],
    facts=[
        Fact("category", "cat", about="what policy the content was reported under",
             values=("spam", "impersonation", "misinformation", "harassment", "hate speech"),
             p=(0.3, 0.15, 0.2, 0.2, 0.15),
             phrases={"formal": ["It was reported as {v}.", "Report category: {v}.",
                                 "Users flagged it for {v}.", "The reports allege {v}."],
                      "informal": ["reported for {v}", "flagged as {v}", "{v} report", "people say it's {v}"]}),
        Fact("confirmed", "bool", about="whether a reviewer confirmed that the content breaks the policy", p=0.65,
             phrases={"formal": {True: ["A reviewer confirmed the content breaks the policy.",
                                        "The first-line review found a violation.",
                                        "The violation was verified by a moderator.",
                                        "The content was found to break the rules."],
                                 False: ["A reviewer found no policy violation.",
                                         "The first-line review did not confirm a violation.",
                                         "The content appears to be within the rules.",
                                         "No violation was found on review."]},
                      "informal": {True: ["reviewer confirmed it breaks the rules", "violation confirmed",
                                          "yep it's a violation", "mod confirmed"],
                                   False: ["reviewer says it's fine", "no violation found", "looks within the rules",
                                           "not a violation per review"]}}),
        Fact("report_count", "int", about="how many users reported it", values=(1, 300), log=True,
             avoid=((10, 1), (50, 4)),
             phrases={"formal": ["{v} users reported it.", "Number of reports: {v}.",
                                 "It has been reported {v} times.", "Reports received: {v}."],
                      "informal": ["{v} reports", "reported {v} times", "{v} ppl flagged it", "{v} flags"]}),
        Fact("prior_strikes", "int", about="how many earlier strikes the account has", values=(0, 4),
             p=(0.4, 0.2, 0.2, 0.12, 0.08),
             phrases={"formal": ["The account has {v} prior strikes.", "Previous strikes: {v}.",
                                 "Enforcement history shows {v} strikes.", "Strikes on record: {v}."],
                      "informal": ["{v} strikes already", "{v} prior strikes", "strikes: {v}", "has {v} strikes"]}),
        Fact("account_age_days", "int", about="how many days old the account is", values=(1, 3000), log=True,
             avoid=((30, 3),),
             phrases={"formal": ["The account was created {v} days ago.", "Account age: {v} days.",
                                 "It is a {v}-day-old account.", "The account has existed for {v} days."],
                      "informal": ["acct is {v} days old", "{v} days old", "made the account {v} days ago",
                                   "account age {v}d"]}),
        Fact("verified", "bool", about="whether the account is verified", p=0.15,
             phrases={"formal": {True: ["It is a verified account.", "The account carries a verification badge.",
                                        "The account holder is verified.", "The profile is verified."],
                                 False: ["The account is not verified.", "There is no verification badge.",
                                         "It is an unverified account.", "The profile is unverified."]},
                      "informal": {True: ["verified acct", "has the badge", "blue check", "verified"],
                                   False: ["not verified", "no badge", "unverified", "regular account"]}}),
    ],
    questions=[
        Question(
            id="action", type="choice", text="What moderation action should be taken?",
            paraphrases=("How should this report be resolved?", "Decide the enforcement action.",
                         "What do we do with this content?"),
            options=[Option("no_action", "No action"), Option("add_label", "Add a warning label"),
                     Option("remove_content", "Remove the content"),
                     Option("suspend_account", "Suspend the account"),
                     Option("escalate_policy", "Escalate to the policy team")],
            rules={"default": Rule(ACTION_RULE.format(s=2), action),
                   "three_strikes": Rule(ACTION_RULE.format(s=3), lambda f: action(f, strikes=3))},
            needs=("confirmed", "verified", "category", "prior_strikes", "account_age_days")),
        Question(
            id="severity", type="score", text="How severe is this report?",
            paraphrases=("Rate the severity of this report.", "What severity level does this case get?",
                         "How serious is this violation report?"),
            options=[Option("low", "low"), Option("medium", "medium"), Option("high", "high")],
            rules={"default": Rule(SEV_RULE.format(r=50), severity),
                   "reports_10": Rule(SEV_RULE.format(r=10), lambda f: severity(f, reports=10))},
            needs=("category", "report_count")),
        Question(
            id="notify_lead", type="noul", text="Should the trust and safety lead be notified?",
            paraphrases=("Does the T&S lead need to know about this?", "Notify the trust and safety lead?",
                         "Is a heads-up to the safety lead required?"),
            rules={"default": Rule("Notify the lead when 50 or more users reported it, or the account is verified.",
                                   lambda f: "yes" if f["report_count"] >= 50 or f["verified"] else "no"),
                   "reports_10": Rule("Notify the lead when 10 or more users reported it, or the account is "
                                      "verified.",
                                      lambda f: "yes" if f["report_count"] >= 10 or f["verified"] else "no")},
            needs=("report_count", "verified")),
    ],
)
