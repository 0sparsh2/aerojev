"""HR: employee leave requests. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def decision(f, notice=14):
    t = f["leave_type"]
    if t in ("sick", "bereavement"):
        return "approve"
    if t in ("parental", "unpaid"):
        return "escalate_to_hr"
    if f["blackout"] or f["on_probation"]:
        return "deny"
    if f["days_requested"] > 3 and f["notice_days"] < notice:
        return "deny"
    if f["days_requested"] > f["balance_days"]:
        return "approve_partial"
    return "approve"


DECISION_RULE = ("Sick and bereavement leave: approve. Parental and unpaid leave: escalate to HR. Vacation: deny during "
                 "a blackout period or while the employee is on probation, and deny requests for more than 3 days "
                 "made with less than {n} days' notice. Otherwise approve only part of it if more days are requested "
                 "than the employee has left, and approve in full if not.")

DOMAIN = Domain(
    name="hr_leave",
    title="Employee leave requests",
    entities={
        "person": ["Hannah Weiss", "Kwame Asante", "Isabel Cruz", "Pavel Sokolov", "Leila Nasser", "Owen Gallagher"],
        "team": ["Accounts Receivable", "Customer Support", "Platform Engineering", "Field Sales", "Logistics"],
    },
    situations=[
        "{person} from {team} submitted a leave request.",
        "Leave request: {person}, {team}.",
        "{person} is asking for time off.",
        "New absence request in the {team} team from {person}.",
        "{person}'s manager forwarded a leave request.",
        "Time-off request logged for {person}.",
    ],
    formats=("email", "ticket", "chat", "form", "report"),
    distractors=[
        "The team offsite is planned for next quarter.",
        "Benefits enrollment closes at the end of the month.",
        "The office coffee machine was replaced.",
        "Payroll runs a day early this month because of the holiday.",
    ],
    facts=[
        Fact("leave_type", "cat", about="the type of leave requested",
             values=("vacation", "sick", "parental", "unpaid", "bereavement"), p=(0.55, 0.15, 0.1, 0.1, 0.1),
             phrases={"formal": ["The request is for {v} leave.", "Leave type: {v}.", "They are requesting {v} leave.",
                                 "This is a {v} leave request."],
                      "informal": ["{v} leave", "wants {v} leave", "type is {v}", "it's {v}"]}),
        Fact("days_requested", "int", about="how many working days of leave are requested", values=(1, 25),
             log=True,
             phrases={"formal": ["The request covers {v} working days.", "Days requested: {v}.",
                                 "They want {v} days off.", "The absence would last {v} working days."],
                      "informal": ["{v} days off", "asking for {v} days", "{v} days", "wants {v} days"]}),
        Fact("balance_days", "int", about="how many days of vacation the employee has left this year",
             values=(0, 25),
             phrases={"formal": ["The employee has {v} vacation days left this year.", "Remaining balance: {v} days.",
                                 "Their leave balance stands at {v} days.", "Vacation days remaining: {v}."],
                      "informal": ["{v} days left in balance", "has {v} days left", "balance {v}",
                                   "{v} vacation days remaining"]}),
        Fact("notice_days", "int", about="how many days before the start of the leave the request was made",
             values=(0, 45), avoid=((14, 1), (30, 1)),
             phrases={"formal": ["The request was made {v} days before the leave starts.",
                                 "Notice given: {v} days.", "The leave starts {v} days from the request date.",
                                 "They gave {v} days' notice."],
                      "informal": ["starts in {v} days", "{v} days notice", "gave us {v} days",
                                   "{v} days before it starts"]}),
        Fact("blackout", "bool", about="whether the dates fall in the team's blackout period", p=0.2,
             phrases={"formal": {True: ["The dates fall within the year-end blackout period.",
                                        "The requested dates overlap the team's blackout window.",
                                        "Those weeks are a blackout period for the team.",
                                        "The leave would fall during the quarterly close blackout."],
                                 False: ["The dates are outside any blackout period.",
                                         "No blackout period applies to these dates.",
                                         "The dates do not overlap a blackout window.",
                                         "Those weeks are not restricted for the team."]},
                      "informal": {True: ["dates are in blackout", "that's blackout weeks", "overlaps the freeze",
                                          "blackout period, fyi"],
                                   False: ["no blackout then", "outside blackout", "dates are fine re blackout",
                                           "not a blackout week"]}}),
        Fact("on_probation", "bool", about="whether the employee is still in their probation period", p=0.12,
             phrases={"formal": {True: ["The employee is still on probation.", "They joined two months ago and are "
                                        "in their probation period.", "Their probation period has not ended.",
                                        "They are a new starter on probation."],
                                 False: ["The employee has passed probation.", "They are a permanent employee.",
                                         "Their probation ended last year.", "They are not on probation."]},
                      "informal": {True: ["still on probation", "new starter, probation", "hasn't passed probation",
                                          "probation still"],
                                   False: ["past probation", "permanent staff", "done with probation",
                                           "not on probation"]}}),
        Fact("coverage", "bool", about="whether a colleague has agreed to cover their work", p=0.6,
             phrases={"formal": {True: ["A colleague has agreed to cover their work.", "Cover is arranged.",
                                        "Their duties will be covered by a teammate.", "Handover cover is in place."],
                                 False: ["No one has agreed to cover their work yet.", "Cover is not arranged.",
                                         "There is no backup for their duties.", "No teammate is available to cover."]},
                      "informal": {True: ["cover sorted", "teammate will cover", "someone's covering",
                                          "backup arranged"],
                                   False: ["no cover yet", "nobody to cover", "no backup", "cover not sorted"]}}),
    ],
    questions=[
        Question(
            id="decision", type="choice", text="How should this leave request be handled?",
            paraphrases=("Decide on this time-off request.", "What should happen with this leave request?",
                         "Approve this leave request or not?"),
            options=[Option("approve", "Approve the request"),
                     Option("approve_partial", "Approve only the days the balance covers"),
                     Option("escalate_to_hr", "Escalate to HR"),
                     Option("deny", "Deny the request")],
            rules={"default": Rule(DECISION_RULE.format(n=14), decision),
                   "notice_30": Rule(DECISION_RULE.format(n=30), lambda f: decision(f, notice=30))},
            needs=("leave_type", "blackout", "on_probation", "days_requested", "notice_days", "balance_days")),
        Question(
            id="second_signoff", type="noul", text="Does this request need a second-level manager sign-off?",
            paraphrases=("Must a senior manager also approve this?", "Is a skip-level approval required?",
                         "Does this need sign-off above the direct manager?"),
            rules={"default": Rule("A second-level sign-off is needed when more than 10 days are requested, or when "
                                   "no colleague is covering the work.",
                                   lambda f: "yes" if f["days_requested"] > 10 or not f["coverage"] else "no"),
                   "five_days": Rule("A second-level sign-off is needed when more than 5 days are requested, or when "
                                     "no colleague is covering the work.",
                                     lambda f: "yes" if f["days_requested"] > 5 or not f["coverage"] else "no")},
            needs=("days_requested", "coverage")),
        Question(
            id="staffing_risk", type="score", text="How much staffing risk does this absence create?",
            paraphrases=("Rate the impact of this absence on the team.", "How risky is this absence for coverage?",
                         "What is the staffing risk level?"),
            options=[Option("low", "low"), Option("medium", "medium"), Option("high", "high")],
            rules={"default": Rule("High if the dates are in a blackout period and no one is covering. Medium if "
                                   "either of those is true. Low otherwise.",
                                   lambda f: "high" if f["blackout"] and not f["coverage"] else
                                   "medium" if f["blackout"] or not f["coverage"] else "low"),
                   "long_leave": Rule("High if the dates are in a blackout period and no one is covering. Medium if "
                                      "either of those is true, or if more than 10 days are requested. Low otherwise.",
                                      lambda f: "high" if f["blackout"] and not f["coverage"] else
                                      "medium" if f["blackout"] or not f["coverage"] or f["days_requested"] > 10
                                      else "low")},
            needs=("blackout", "coverage", "days_requested")),
    ],
)
