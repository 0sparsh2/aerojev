"""Software subscriptions: cancellation requests at the retention desk. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def money(v):
    return f"${v:,.2f}"


def offer(f, tenure=6):
    r = f["reason"]
    if r == "business closed":
        return "no_offer"
    if f["plan"] == "enterprise" or f["monthly_spend"] > 1000:
        return "account_manager_call"
    if r == "price":
        if f["tenure_months"] >= tenure:
            return "discount_20"
        return "downgrade" if f["plan"] == "pro" else "no_offer"
    if r == "poor support":
        return "free_month"
    return "free_month" if f["active_last_30_days"] else "no_offer"


OFFER_RULE = ("Business closed: no offer. Otherwise enterprise customers, and anyone spending over $1,000 a month, get "
              "a call from their account manager. Otherwise, for price complaints, offer 20% off to customers of "
              "{t} months or more, a downgrade to pro customers below that, and nothing to basic customers below "
              "that. Poor support: a free month. Missing features or switching to a competitor: a free month if they "
              "used the product in the last 30 days, otherwise no offer.")


def churn_risk(f, tickets=3):
    if not f["active_last_30_days"] or f["reason"] == "switching to a competitor":
        return "high"
    if f["support_tickets"] >= tickets or f["reason"] == "price":
        return "medium"
    return "low"


RISK_RULE = ("High if they have not used the product in the last 30 days or are switching to a competitor. Otherwise "
             "medium if they opened {k} or more support tickets in the last 90 days or the reason is price. "
             "Otherwise low.")

DOMAIN = Domain(
    name="subscription_retention",
    title="Subscription cancellation requests",
    entities={
        "person": ["Tessa Grant", "Malik Johnson", "Ingrid Larsen", "Hiro Watanabe", "Paula Reyes", "Dev Kapoor"],
        "org": ["Brookline Dental", "Pixelhaus Studio", "Granite Ridge Builders", "Fernwood Academy",
                "Lantern Coffee"],
    },
    situations=[
        "{person} from {org} wants to cancel their subscription.",
        "Cancellation request from {org}.",
        "{org} asked to close their account.",
        "Retention desk: {person} is cancelling.",
        "{person} clicked cancel on the {org} workspace.",
        "Churn request received from {org}.",
    ],
    distractors=[
        "The product roadmap webinar is next Tuesday.",
        "Our status page shows all systems normal.",
        "The customer also asked how to export invoices.",
        "A new integrations marketplace launches next month.",
    ],
    facts=[
        Fact("reason", "cat", about="the customer's main reason for cancelling",
             values=("price", "missing features", "poor support", "switching to a competitor", "business closed"),
             p=(0.35, 0.2, 0.15, 0.2, 0.1),
             phrases={"formal": ["The stated reason is {v}.", "Cancellation reason: {v}.",
                                 "They cite {v} as the reason.", "Main reason given: {v}."],
                      "informal": ["reason: {v}", "says it's {v}", "{v}", "because of {v}"]}),
        Fact("plan", "cat", about="the customer's subscription plan", values=("basic", "pro", "enterprise"),
             p=(0.45, 0.4, 0.15),
             phrases={"formal": ["They are on the {v} plan.", "Plan: {v}.", "Current subscription: {v}.",
                                 "The account is on {v}."],
                      "informal": ["{v} plan", "on {v}", "plan's {v}", "{v} tier"]}),
        Fact("tenure_months", "int", about="how many months they have been a customer", values=(1, 60), log=True,
             avoid=((6, 0), (24, 2)),
             phrases={"formal": ["They have been a customer for {v} months.", "Tenure: {v} months.",
                                 "The account was opened {v} months ago.", "Customer for {v} months."],
                      "informal": ["{v} months with us", "customer {v} mo", "been with us {v} months",
                                   "{v} months tenure"]}),
        Fact("monthly_spend", "float", about="how much the customer pays per month in dollars", values=(9, 4000),
             log=True, avoid=((100, 8), (1000, 80)), fmt=money,
             phrases={"formal": ["They pay {v} per month.", "Monthly spend: {v}.", "The account bills {v} a month.",
                                 "Their monthly invoice is {v}."],
                      "informal": ["{v}/mo", "pays {v} a month", "{v} monthly", "bill is {v}/month"]}),
        Fact("support_tickets", "int", about="how many support tickets they opened in the last 90 days",
             values=(0, 8), p=(0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.04, 0.03, 0.03),
             phrases={"formal": ["They opened {v} support tickets in the last 90 days.",
                                 "Support tickets in 90 days: {v}.", "{v} tickets were logged this quarter.",
                                 "Recent support contacts: {v}."],
                      "informal": ["{v} tickets lately", "opened {v} tickets in 90d", "{v} support tickets",
                                   "{v} tickets this quarter"]}),
        Fact("active_last_30_days", "bool", about="whether the customer used the product in the last 30 days",
             p=0.6,
             phrases={"formal": {True: ["They used the product this month.", "Usage logs show activity last week.",
                                        "The account was active in the last 30 days.",
                                        "Their team logged in recently."],
                                 False: ["They have not used the product in the last 30 days.",
                                         "There has been no login for over a month.",
                                         "The account has been dormant for weeks.",
                                         "Usage logs show no activity this month."]},
                      "informal": {True: ["still using it", "active this month", "logged in last week",
                                          "they use it"],
                                   False: ["no logins in a month", "dormant", "hasn't used it in 30+ days",
                                           "zero usage lately"]}}),
    ],
    questions=[
        Question(
            id="offer", type="choice", text="What retention offer should we make?",
            paraphrases=("Which save offer fits this customer?", "What should we offer to keep them?",
                         "Pick the retention offer."),
            options=[Option("no_offer", "No offer; process the cancellation"),
                     Option("discount_20", "20% discount for 6 months"),
                     Option("downgrade", "Offer a downgrade to the basic plan"),
                     Option("free_month", "One month free"),
                     Option("account_manager_call", "Call from the account manager")],
            rules={"default": Rule(OFFER_RULE.format(t=6), offer),
                   "tenure_24": Rule(OFFER_RULE.format(t=24), lambda f: offer(f, tenure=24))},
            needs=("reason", "plan", "monthly_spend", "tenure_months", "active_last_30_days")),
        Question(
            id="churn_risk", type="score", text="How likely is this customer to leave anyway?",
            paraphrases=("Rate the churn risk.", "What is this account's churn risk?",
                         "How hard will this customer be to keep?"),
            options=[Option("low", "low"), Option("medium", "medium"), Option("high", "high")],
            rules={"default": Rule(RISK_RULE.format(k=3), churn_risk),
                   "tickets_1": Rule(RISK_RULE.format(k=1), lambda f: churn_risk(f, tickets=1))},
            needs=("active_last_30_days", "reason", "support_tickets")),
        Question(
            id="csm_alert", type="noul", text="Should the customer success manager be alerted?",
            paraphrases=("Does the CSM need to hear about this?", "Alert customer success?",
                         "Is a CSM alert required for this account?"),
            rules={"default": Rule("Alert the CSM when monthly spend is over $100 and the customer opened 3 or more "
                                   "support tickets in the last 90 days.",
                                   lambda f: "yes" if f["monthly_spend"] > 100 and f["support_tickets"] >= 3
                                   else "no"),
                   "any_ticket": Rule("Alert the CSM when monthly spend is over $100 and the customer opened any "
                                      "support ticket in the last 90 days.",
                                      lambda f: "yes" if f["monthly_spend"] > 100 and f["support_tickets"] >= 1
                                      else "no")},
            needs=("monthly_spend", "support_tickets")),
    ],
)
