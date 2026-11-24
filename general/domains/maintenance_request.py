"""Facilities management: building maintenance work orders. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule

LEVELS = ["routine", "soon", "urgent", "emergency"]


def priority(f, lab_urgent=False):
    critical_dc = f["area"] == "data center" and f["issue"] in ("HVAC", "electrical")
    if f["hazard"] or critical_dc:
        return "emergency"
    if f["issue"] == "elevator" or f["people_affected"] >= 100 or (lab_urgent and f["area"] == "laboratory"):
        return "urgent"
    if f["people_affected"] >= 20 or f["area"] == "laboratory":
        return "soon"
    return "routine"


PRIORITY_RULE = ("Emergency if there is a safety hazard, or an HVAC or electrical fault in the data center. Urgent "
                 "for elevator faults or when 100 or more people are affected. Soon when 20 or more people are "
                 "affected or the problem is in a laboratory. Routine otherwise.")

TRADE = {"HVAC": "hvac_technician", "plumbing": "plumber", "electrical": "electrician", "lighting": "electrician",
         "elevator": "elevator_contractor", "cleaning": "janitorial"}

DOMAIN = Domain(
    name="maintenance_request",
    title="Building maintenance work orders",
    entities={
        "person": ["Greta Holm", "Samir Baig", "Lucy Tran", "Victor Neumann", "Abena Mensah", "Ian Rourke"],
        "building": ["the Elm Street office", "Building C", "the north annex", "the research tower",
                     "the downtown hub"],
    },
    situations=[
        "{person} reported a problem in {building}.",
        "New work order for {building}.",
        "Facilities request from {person}.",
        "Help desk logged a maintenance issue at {building}.",
        "{person} called the facilities line about {building}.",
        "Maintenance request raised for {building}.",
    ],
    distractors=[
        "The fire drill is scheduled for next Wednesday.",
        "Parking permits renew at the end of the month.",
        "The cafeteria menu changes next week.",
        "Window cleaning on the east side is booked for Friday.",
    ],
    facts=[
        Fact("issue", "cat", about="what kind of fault it is",
             values=("HVAC", "plumbing", "electrical", "lighting", "elevator", "cleaning"),
             p=(0.25, 0.2, 0.15, 0.15, 0.1, 0.15),
             phrases={"HVAC": ["The heating and cooling is not working.", "The air conditioning has failed.",
                               "Temperature control is broken.", "The HVAC unit is faulty."],
                      "plumbing": ["There is a leaking pipe.", "A toilet is blocked and overflowing.",
                                   "Water is coming through the ceiling from a pipe.", "A drain is backed up."],
                      "electrical": ["Several sockets have no power.", "A circuit breaker keeps tripping.",
                                     "There is a fault in the electrical panel.", "Power outlets are dead."],
                      "lighting": ["The ceiling lights are out.", "Several light fittings are flickering.",
                                   "The corridor lights do not come on.", "Lamps in the room have failed."],
                      "elevator": ["The elevator is stuck between floors.", "The lift is out of service.",
                                   "The elevator doors will not open.", "The lift makes a grinding noise and stops."],
                      "cleaning": ["There is a large spill that needs cleaning.", "The area needs a deep clean.",
                                   "Rubbish has not been collected.", "The floor is covered in debris."]}),
        Fact("area", "cat", about="the type of area where the fault is",
             values=("office floor", "laboratory", "public lobby", "data center"), p=(0.45, 0.2, 0.2, 0.15),
             phrases={"formal": ["The fault is in the {v}.", "Location type: {v}.", "It affects the {v}.",
                                 "The problem is located in the {v}."],
                      "informal": ["in the {v}", "{v} area", "it's the {v}", "location: {v}"]}),
        Fact("hazard", "bool", about="whether there is a safety hazard such as exposed wiring, flooding near "
             "power or a trapped person", p=0.2,
             phrases={"formal": {True: ["There is exposed wiring near the fault.", "Water is pooling near power points.",
                                        "A person may be trapped.", "Staff report a burning smell."],
                                 False: ["There is no safety hazard.", "Nothing about it is dangerous.",
                                         "No one is at risk.", "The area is safe to use."]},
                      "informal": {True: ["wires exposed!", "water near the sockets", "someone might be stuck",
                                          "smells like burning"],
                                   False: ["nothing dangerous", "no safety issue", "safe, just annoying",
                                           "nobody at risk"]}}),
        Fact("people_affected", "int", about="how many people are affected", values=(1, 400), log=True,
             avoid=((20, 2), (100, 8)),
             phrases={"formal": ["{v} people are affected.", "The issue affects {v} occupants.",
                                 "About {v} staff work in the affected space.", "Occupants affected: {v}."],
                      "informal": ["{v} people affected", "hits {v} ppl", "{v} folks impacted",
                                   "about {v} people"]}),
        Fact("recurring", "bool", about="whether the same fault was reported in the last 30 days", p=0.25,
             phrases={"formal": {True: ["The same fault was reported three weeks ago.", "This is a repeat issue.",
                                        "It recurred after last month's repair.", "This fault has come back."],
                                 False: ["This is the first report of the fault.", "It has not happened before.",
                                         "There is no earlier report of it.", "It is a new fault."]},
                      "informal": {True: ["happened again", "repeat issue", "this came back", "same as last month"],
                                   False: ["first time", "never had this before", "new issue", "not a repeat"]}}),
        Fact("after_hours", "bool", about="whether it was reported outside working hours", p=0.3,
             phrases={"formal": {True: ["It was reported at 10 pm.", "The report came in over the weekend.",
                                        "It was logged outside working hours.", "The call came in at night."],
                                 False: ["It was reported during working hours.", "The report came in mid-morning.",
                                         "It was logged during the business day.", "The call came in at 2 pm."]},
                      "informal": {True: ["after hours", "reported at night", "weekend call", "came in late"],
                                   False: ["during the day", "office hours", "reported this morning",
                                           "came in at lunch"]}}),
    ],
    questions=[
        Question(
            id="priority", type="score", text="What priority should this work order get?",
            paraphrases=("How urgent is this maintenance request?", "Set the priority for this work order.",
                         "Rate the urgency of this job."),
            options=[Option("routine", "routine"), Option("soon", "soon"), Option("urgent", "urgent"),
                     Option("emergency", "emergency")],
            rules={"default": Rule(PRIORITY_RULE, priority),
                   "lab_urgent": Rule(PRIORITY_RULE + " Laboratories are urgent rather than soon.",
                                      lambda f: priority(f, lab_urgent=True))},
            needs=("hazard", "area", "issue", "people_affected")),
        Question(
            id="trade", type="choice", text="Who should be sent to this job?",
            paraphrases=("Which trade should handle this?", "Assign this work order to the right trade.",
                         "Who do we dispatch?"),
            options=[Option("emergency_response", "Emergency response contractor"),
                     Option("hvac_technician", "HVAC technician"), Option("plumber", "Plumber"),
                     Option("electrician", "Electrician (power and lighting)"),
                     Option("elevator_contractor", "Elevator contractor"),
                     Option("janitorial", "Janitorial team")],
            rules={"default": Rule("Any safety hazard goes to the emergency response contractor. Otherwise by fault: "
                                   "HVAC technician, plumber, electrician for power and lighting, elevator contractor, "
                                   "janitorial team for cleaning.",
                                   lambda f: "emergency_response" if f["hazard"] else TRADE[f["issue"]])},
            needs=("hazard", "issue")),
        Question(
            id="root_cause_review", type="noul", text="Should a root-cause review be opened?",
            paraphrases=("Does this need a root-cause investigation?", "Open a root-cause review for this fault?",
                         "Is a root-cause analysis warranted?"),
            rules={"default": Rule("Open a root-cause review when the same fault was reported in the last 30 days, "
                                   "or when 100 or more people were affected.",
                                   lambda f: "yes" if f["recurring"] or f["people_affected"] >= 100 else "no"),
                   "recurring_only": Rule("Open a root-cause review only when the same fault was reported in the "
                                          "last 30 days.", lambda f: "yes" if f["recurring"] else "no")},
            needs=("recurring", "people_affected")),
    ],
)
