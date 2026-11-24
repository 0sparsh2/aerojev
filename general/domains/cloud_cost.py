"""Cloud cost management (FinOps): spend anomaly alerts. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def money(v):
    return f"${v:,.2f}"


def action(f, idle=14):
    prod = f["environment"] == "production"
    if not f["owner_tagged"] and f["monthly_cost"] > 1000:
        return "escalate_finops"
    if not prod and f["idle_days"] >= idle:
        return "schedule_shutdown"
    if prod and f["idle_days"] >= 30:
        return "rightsize"
    if f["spend_increase_pct"] >= 25:
        return "notify_owner"
    return "no_action"


ACTION_RULE = ("Untagged resources costing over $1,000 a month: escalate to FinOps. Otherwise schedule a shutdown for "
               "non-production resources idle for {d} days or more, and rightsize production resources idle for 30 "
               "days or more. Otherwise notify the owner if spend rose 25% or more. Otherwise no action.")

LEVELS = ["low", "medium", "high", "critical"]


def severity(f, prod_bump=False):
    b = f["budget_used_pct"]
    lvl = 3 if b > 120 else 2 if b > 100 else 1 if b > 80 else 0
    if prod_bump and f["environment"] == "production":
        lvl = min(3, lvl + 1)
    return LEVELS[lvl]


SEV_RULE = ("By budget used this month: critical above 120%, high above 100%, medium above 80%, low otherwise.")

DOMAIN = Domain(
    name="cloud_cost",
    title="Cloud spend anomaly alerts",
    entities={
        "service": ["the analytics cluster", "the image-resize workers", "the search index", "the build runners",
                    "the reporting database", "the ML training pool", "the event queue"],
        "team": ["Data Platform", "Mobile", "Payments", "Growth", "Internal Tools"],
        "account": ["acct-prod-eu", "acct-shared-us", "acct-dev-01", "acct-research"],
    },
    situations=[
        "Cost anomaly alert for {service} in {account}.",
        "The spend monitor flagged {service}, owned by {team}.",
        "Budget alert: {service} in {account}.",
        "Weekly cost review item: {service}.",
        "{team} received a cost alert for {service}.",
        "Anomaly detection fired on {service}.",
    ],
    distractors=[
        "Reserved instance renewals are due next quarter.",
        "The tagging policy document was updated last week.",
        "Remember to submit next year's capacity forecast.",
        "The cloud provider announced a new region.",
    ],
    facts=[
        Fact("environment", "cat", about="which environment the resource runs in",
             values=("production", "staging", "development", "sandbox"), p=(0.4, 0.2, 0.25, 0.15),
             phrases={"formal": ["The resource runs in {v}.", "Environment: {v}.", "It is a {v} resource.",
                                 "This sits in the {v} environment."],
                      "informal": ["it's {v}", "env={v}", "{v} env", "in {v}"]}),
        Fact("spend_increase_pct", "int", about="how much the resource's spend rose compared with last month, in "
             "percent", values=(1, 400), log=True, avoid=((25, 2), (100, 6)), fmt=lambda v: f"{v}%",
             phrases={"formal": ["Spend is up {v} on last month.", "Month-over-month increase: {v}.",
                                 "Costs rose {v} compared with the previous month.",
                                 "The bill for it grew by {v} since last month."],
                      "informal": ["up {v} vs last month", "spend +{v}", "{v} jump in cost", "grew {v} MoM"]}),
        Fact("budget_used_pct", "int", about="how much of the team's monthly budget is already used, in percent",
             values=(40, 160), avoid=((80, 2), (100, 2), (120, 2)), fmt=lambda v: f"{v}%",
             phrases={"formal": ["The team has used {v} of its monthly budget.", "Budget consumed this month: {v}.",
                                 "Month-to-date spend is {v} of budget.", "{v} of the budget is already used."],
                      "informal": ["{v} of budget gone", "budget at {v}", "burned {v} of budget", "{v} budget used"]}),
        Fact("idle_days", "int", about="for how many days the resource has had almost no usage", values=(0, 40),
             phrases={"formal": ["The resource has been idle for {v} days.", "Days with near-zero utilisation: {v}.",
                                 "Usage has been flat at almost nothing for {v} days.",
                                 "No meaningful traffic for {v} days."],
                      "informal": ["idle {v} days", "{v} days doing nothing", "no traffic for {v} days",
                                   "{v} days idle"]}),
        Fact("owner_tagged", "bool", about="whether the resource has an owner tag", p=0.7,
             phrases={"formal": {True: ["The resource carries an owner tag.", "An owning team is tagged.",
                                        "Ownership tags are present.", "The owner tag is set."],
                                 False: ["The resource has no owner tag.", "No owning team is tagged.",
                                         "Ownership tags are missing.", "Nobody is listed as the owner."]},
                      "informal": {True: ["owner tag set", "tagged with owner", "has an owner tag",
                                          "owner's tagged"],
                                   False: ["no owner tag", "untagged", "owner unknown", "missing owner tag"]}}),
        Fact("monthly_cost", "float", about="the resource's monthly cost in dollars", values=(50, 60000), log=True,
             avoid=((1000, 80), (10000, 800)), fmt=money,
             phrases={"formal": ["It costs {v} a month.", "Monthly cost: {v}.", "The run rate is {v} per month.",
                                 "Current monthly spend on it is {v}."],
                      "informal": ["{v}/mo", "costs {v} a month", "{v} monthly", "running {v} a month"]}),
    ],
    questions=[
        Question(
            id="action", type="choice", text="What action should be taken on this cost alert?",
            paraphrases=("How should we respond to this anomaly?", "What is the next step on this resource?",
                         "Pick the action for this cost alert."),
            options=[Option("no_action", "No action"), Option("notify_owner", "Notify the owner"),
                     Option("rightsize", "Rightsize the resource"),
                     Option("schedule_shutdown", "Schedule a shutdown"),
                     Option("escalate_finops", "Escalate to the FinOps team")],
            rules={"default": Rule(ACTION_RULE.format(d=14), action),
                   "idle_7": Rule(ACTION_RULE.format(d=7), lambda f: action(f, idle=7))},
            needs=("owner_tagged", "monthly_cost", "environment", "idle_days", "spend_increase_pct")),
        Question(
            id="severity", type="score", text="How severe is this budget situation?",
            paraphrases=("Rate the budget alert's severity.", "What severity should this alert carry?",
                         "How bad is the budget overrun?"),
            options=[Option("low", "low"), Option("medium", "medium"), Option("high", "high"),
                     Option("critical", "critical")],
            rules={"default": Rule(SEV_RULE, severity),
                   "prod_bump": Rule(SEV_RULE + " Production resources are one level higher (at most critical).",
                                     lambda f: severity(f, prod_bump=True))},
            needs=("budget_used_pct", "environment")),
        Question(
            id="finance_review", type="noul", text="Does this anomaly need a finance review?",
            paraphrases=("Should finance look at this?", "Is a finance review required for this spend?",
                         "Does this go to the finance partner?"),
            rules={"default": Rule("Finance reviews resources costing over $10,000 a month whose spend rose 100% "
                                   "or more, and any team above 120% of its budget.",
                                   lambda f: "yes" if (f["monthly_cost"] > 10000 and f["spend_increase_pct"] >= 100)
                                   or f["budget_used_pct"] > 120 else "no"),
                   "budget_100": Rule("Finance reviews resources costing over $10,000 a month whose spend rose "
                                      "100% or more, and any team above 100% of its budget.",
                                      lambda f: "yes" if (f["monthly_cost"] > 10000 and f["spend_increase_pct"] >= 100)
                                      or f["budget_used_pct"] > 100 else "no")},
            needs=("monthly_cost", "spend_increase_pct", "budget_used_pct")),
    ],
)
