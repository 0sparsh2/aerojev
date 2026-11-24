"""University administration: coursework extension requests. Held out of training (dev-unseen).
All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def decision(f, work_personal=True):
    d, r = f["days_requested"], f["reason"]
    if f["support_plan"]:
        return "grant_in_full" if d <= 14 else "refer_to_board"
    if r in ("illness", "bereavement") and f["documented"]:
        return "grant_in_full" if d <= 7 else "refer_to_board"
    if r in ("illness", "bereavement", "technical problems"):
        return "grant_in_full" if d <= 3 else "grant_3_days"
    if work_personal and f["prior_extensions"] == 0:
        return "grant_in_full" if d <= 3 else "grant_3_days"
    return "decline"


DECISION_RULE = ("Students with a learning support plan: grant in full up to 14 days, and refer longer requests to the "
                 "extenuating circumstances board. Otherwise documented illness or bereavement: grant in full up to "
                 "7 days, and refer longer requests to the board. Undocumented illness or bereavement, and technical "
                 "problems: grant up to 3 days. Work or personal reasons: {wp}.")
WORK_PERSONAL = "grant up to 3 days for a student's first extension this year, and decline otherwise"

DOMAIN = Domain(
    name="coursework_extension",
    title="University coursework extension requests",
    heldout="dev",
    entities={
        "person": ["Freya Coleman", "Jamal Wright", "Ana Petrova", "Liam O'Connell", "Zara Siddiqui", "Ben Holloway"],
        "module": ["Statistics II", "Organic Chemistry", "Modern History", "Software Design", "Microeconomics"],
    },
    situations=[
        "{person} asked for an extension on their {module} coursework.",
        "Extension request: {module}, student {person}.",
        "{person} emailed the {module} office about a deadline.",
        "New extension form from {person}.",
        "The {module} module leader forwarded a request from {person}.",
        "Deadline extension request for {module}.",
    ],
    formats=("email", "ticket", "chat", "form", "report"),
    distractors=[
        "The library extends its opening hours during exams.",
        "Module feedback surveys close on Friday.",
        "The graduation ceremony dates were confirmed.",
        "Room 2.14 is closed for refurbishment.",
    ],
    facts=[
        Fact("reason", "cat", about="the reason the student gives for the extension",
             values=("illness", "bereavement", "technical problems", "work commitments", "personal reasons"),
             p=(0.3, 0.1, 0.15, 0.2, 0.25),
             phrases={"formal": ["The reason given is {v}.", "Grounds for the request: {v}.",
                                 "The student cites {v}.", "Reason stated: {v}."],
                      "informal": ["because of {v}", "says {v}", "reason: {v}", "{v}"]}),
        Fact("documented", "bool", about="whether the student provided supporting documents", p=0.55,
             phrases={"formal": {True: ["Supporting evidence was attached.", "A doctor's note was provided.",
                                        "The student submitted documentation.", "Evidence is on file."],
                                 False: ["No supporting evidence was provided.", "There is no documentation.",
                                         "The student attached no evidence.", "Nothing supports the claim yet."]},
                      "informal": {True: ["sent evidence", "has a note", "docs attached", "evidence provided"],
                                   False: ["no evidence", "no docs", "nothing attached", "no note"]}}),
        Fact("days_requested", "int", about="how many days of extension the student asks for", values=(1, 21),
             log=True,
             phrases={"formal": ["The student asks for {v} extra days.", "Extension requested: {v} days.",
                                 "They want the deadline moved by {v} days.", "Requested extension length: {v} days."],
                      "informal": ["wants {v} more days", "{v} days extra", "asking for {v} days",
                                   "{v} day extension"]}),
        Fact("prior_extensions", "int", about="how many extensions the student already had this year",
             values=(0, 3), p=(0.55, 0.25, 0.12, 0.08),
             phrases={"formal": ["The student has had {v} extensions this year.", "Previous extensions: {v}.",
                                 "Extensions already granted this year: {v}.", "They had {v} earlier extensions."],
                      "informal": ["{v} extensions so far", "had {v} before", "{v} prior extensions",
                                   "extensions this year: {v}"]}),
        Fact("weight_pct", "int", about="how much the assessment counts toward the module mark, in percent",
             values=(5, 60), avoid=((25, 1), (40, 1)), fmt=lambda v: f"{v}%",
             phrases={"formal": ["The assessment is worth {v} of the module.", "Weighting: {v}.",
                                 "It counts for {v} of the final mark.", "Assessment weight: {v}."],
                      "informal": ["worth {v}", "{v} of the grade", "counts {v}", "{v} weighting"]}),
        Fact("support_plan", "bool", about="whether the student has a learning support plan that covers "
             "extensions", p=0.15,
             phrases={"formal": {True: ["The student has a learning support plan covering extensions.",
                                        "A support plan allowing extensions is on file.",
                                        "Disability services issued a plan that covers deadlines.",
                                        "Their support plan includes extensions."],
                                 False: ["The student has no learning support plan.", "No support plan is on file.",
                                         "There is no plan covering extensions.",
                                         "They are not registered with disability services."]},
                      "informal": {True: ["has a support plan", "support plan covers it", "SP on file",
                                          "registered w disability services"],
                                   False: ["no support plan", "no SP", "not registered for support",
                                           "no plan on file"]}}),
    ],
    questions=[
        Question(
            id="decision", type="choice", text="What should be decided on this extension request?",
            paraphrases=("How should this request be answered?", "Grant this extension?",
                         "What is the outcome for this extension request?"),
            options=[Option("grant_in_full", "Grant the extension as requested"),
                     Option("grant_3_days", "Grant 3 days only"),
                     Option("refer_to_board", "Refer to the extenuating circumstances board"),
                     Option("decline", "Decline the request")],
            rules={"default": Rule(DECISION_RULE.format(wp=WORK_PERSONAL), decision),
                   "no_work_personal": Rule(DECISION_RULE.format(wp="decline"),
                                            lambda f: decision(f, work_personal=False))},
            needs=("support_plan", "reason", "documented", "days_requested", "prior_extensions")),
        Question(
            id="notify_tutor", type="noul", text="Should the student's personal tutor be notified?",
            paraphrases=("Does the tutor need to know about this?", "Inform the personal tutor?",
                         "Is a tutor notification required?"),
            rules={"default": Rule("Notify the tutor when the student already had an extension this year, or the "
                                   "assessment is worth 40% or more.",
                                   lambda f: "yes" if f["prior_extensions"] >= 1 or f["weight_pct"] >= 40 else "no"),
                   "weight_25": Rule("Notify the tutor when the student already had an extension this year, or the "
                                     "assessment is worth 25% or more.",
                                     lambda f: "yes" if f["prior_extensions"] >= 1 or f["weight_pct"] >= 25 else
                                     "no")},
            needs=("prior_extensions", "weight_pct")),
    ],
)
