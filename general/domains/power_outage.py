"""Electric utility: outage response dispatch. Held out of training (dev-unseen). All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def priority(f, life_p1=False):
    if f["downed_wires"] or f["critical_facility"] or (life_p1 and f["life_support"] > 0):
        return "P1"
    if f["customers_out"] >= 1000 or f["life_support"] > 0:
        return "P2"
    if f["customers_out"] >= 100:
        return "P3"
    return "P4"


PRIORITY_RULE = ("P1 if there are downed live wires or a hospital, care home or water plant is without power. P2 if "
                 "1,000 or more customers are out or any registered life-support customer is out. P3 if 100 or more "
                 "customers are out. P4 otherwise.")


def dispatch(f):
    if f["downed_wires"]:
        return "send_crew_now"
    if f["weather"] in ("storm", "ice"):
        return "wait_for_weather"
    high = priority(f) in ("P1", "P2")
    if f["crews_available"] == 0:
        return "request_mutual_aid" if high else "add_to_queue"
    return "send_crew_now" if high else "add_to_queue"


def report(f, customers=1000):
    return (f["customers_out"] >= customers and f["hours_out"] >= 12) or f["critical_facility"]


DOMAIN = Domain(
    name="power_outage",
    title="Electric utility outage dispatch",
    heldout="dev",
    entities={
        "area": ["Riverside feeder 12", "the Oakmont substation area", "Hillcrest", "the east industrial park",
                 "Lakeshore circuit 4"],
        "person": ["Operator Kim", "Dispatcher Russo", "Supervisor Ahmed", "Operator Fields"],
    },
    situations=[
        "Outage reported in {area}.",
        "{person} opened an outage ticket for {area}.",
        "SCADA alarm: loss of supply in {area}.",
        "Customers in {area} are calling about a power cut.",
        "Outage management system event for {area}.",
        "Control room update on {area}.",
    ],
    distractors=[
        "Tree-trimming crews finish the north route this week.",
        "The new outage map goes live next month.",
        "Smart meter installs continue in the west district.",
        "Annual safety refresher training is due.",
    ],
    facts=[
        Fact("customers_out", "int", about="how many customers are without power", values=(1, 20000), log=True,
             avoid=((100, 8), (1000, 80), (5000, 400)),
             phrases={"formal": ["{v} customers are without power.", "Customers affected: {v}.",
                                 "The outage affects {v} meters.", "Supply is lost to {v} customers."],
                      "informal": ["{v} out", "{v} customers dark", "{v} meters off", "about {v} without power"]}),
        Fact("critical_facility", "bool", about="whether a hospital, care home or water treatment plant is without "
             "power", p=0.15,
             phrases={"formal": {True: ["A hospital in the area is on backup generators.",
                                        "A care home has lost supply.", "The water treatment plant is without power.",
                                        "A critical facility is affected."],
                                 False: ["No critical facilities are affected.",
                                         "No hospitals or care homes are in the outage area.",
                                         "Critical sites still have power.", "Only homes and shops are affected."]},
                      "informal": {True: ["hospital's on generators", "care home lost power", "water plant is down",
                                          "critical site hit"],
                                   False: ["no critical sites", "no hospitals affected", "critical sites fine",
                                           "just homes and shops"]}}),
        Fact("life_support", "int", about="how many registered life-support customers are without power",
             values=(0, 4), p=(0.65, 0.15, 0.1, 0.06, 0.04),
             phrases={"formal": ["{v} registered life-support customers are affected.",
                                 "Life-support customers without power: {v}.",
                                 "The register lists {v} medical-equipment customers in the area.",
                                 "Affected life-support customers: {v}."],
                      "informal": ["{v} life-support customers out", "LS customers: {v}", "{v} on the medical register",
                                   "{v} life support affected"]}),
        Fact("downed_wires", "bool", about="whether there are downed live wires", p=0.15,
             phrases={"formal": {True: ["A caller reports a live wire down on the road.", "There are downed lines.",
                                        "Wires are down across a driveway.", "A conductor is down and may be live."],
                                 False: ["No downed wires have been reported.", "All lines appear to be up.",
                                         "There are no reports of wires down.", "Patrol found no downed lines."]},
                      "informal": {True: ["wire down!", "lines down on the road", "live wire reported",
                                          "conductor down"],
                                   False: ["no wires down", "lines all up", "no downed lines", "nothing down"]}}),
        Fact("weather", "cat", about="the current weather in the area", values=("clear", "rain", "storm", "ice"),
             p=(0.45, 0.3, 0.15, 0.1),
             phrases={"clear": ["The weather is clear.", "Conditions are calm and dry.", "Skies are clear.",
                                "There is no weather concern."],
                      "rain": ["It is raining steadily.", "Light rain is falling.", "Conditions are wet but calm.",
                               "There are rain showers."],
                      "storm": ["A thunderstorm is passing over.", "High winds and lightning are ongoing.",
                                "Storm conditions are in force.", "There is a severe storm warning."],
                      "ice": ["Freezing rain is coating the lines.", "Roads and lines are iced over.",
                              "An ice storm is ongoing.", "There is heavy icing."]}),
        Fact("hours_out", "int", about="how many hours the outage has lasted", values=(1, 48), log=True,
             avoid=((12, 1),),
             phrases={"formal": ["The outage has lasted {v} hours.", "Duration so far: {v} hours.",
                                 "Supply was lost {v} hours ago.", "Customers have been out for {v} hours."],
                      "informal": ["out {v} hrs", "{v} hours so far", "dark for {v} hours", "{v}h outage"]}),
        Fact("crews_available", "int", about="how many repair crews are available right now", values=(0, 4),
             p=(0.25, 0.3, 0.2, 0.15, 0.1),
             phrases={"formal": ["{v} crews are available.", "Crews free right now: {v}.",
                                 "There are {v} crews on standby.", "Available crews: {v}."],
                      "informal": ["{v} crews free", "{v} crews available", "got {v} crews", "crews: {v}"]}),
    ],
    questions=[
        Question(
            id="priority", type="score", text="What priority should this outage get?",
            paraphrases=("How urgent is this outage?", "Assign an outage priority.", "Rate this outage's priority."),
            options=[Option("P4", "P4: small, no special customers"), Option("P3", "P3: medium-sized outage"),
                     Option("P2", "P2: large outage or life-support customers"),
                     Option("P1", "P1: public danger or critical facility")],
            rules={"default": Rule(PRIORITY_RULE, priority),
                   "life_support_p1": Rule(PRIORITY_RULE + " Any life-support customer out makes it P1.",
                                           lambda f: priority(f, life_p1=True))},
            needs=("downed_wires", "critical_facility", "life_support", "customers_out")),
        Question(
            id="dispatch", type="choice", text="What should dispatch do now?",
            paraphrases=("How should we respond to this outage?", "Choose the dispatch action.",
                         "What is the next dispatch step?"),
            options=[Option("send_crew_now", "Send a crew now"), Option("wait_for_weather", "Wait for the weather"),
                     Option("request_mutual_aid", "Request mutual aid from a neighbouring utility"),
                     Option("add_to_queue", "Add to the work queue")],
            rules={"default": Rule("Downed live wires: send a crew now in any weather. Otherwise, during a storm or "
                                   "ice, wait for the weather. Otherwise, with no crew available, request mutual aid "
                                   "for P1 and P2 outages and queue the rest; with a crew available, send it now for "
                                   "P1 and P2 and queue P3 and P4. " + PRIORITY_RULE, dispatch)},
            needs=("downed_wires", "weather", "crews_available", "critical_facility", "life_support",
                   "customers_out")),
        Question(
            id="regulator_report", type="noul", text="Must this outage be reported to the regulator?",
            paraphrases=("Is a regulatory report required?", "Does this outage need to be reported?",
                         "Report this to the energy regulator?"),
            rules={"default": Rule("Report when 1,000 or more customers have been out for 12 hours or more, or when "
                                   "a hospital, care home or water plant has lost power.",
                                   lambda f: "yes" if report(f) else "no"),
                   "any_size": Rule("Report every outage that has lasted 12 hours or more, whatever its size, and "
                                    "any outage where a hospital, care home or water plant has lost power.",
                                    lambda f: "yes" if report(f, 1) else "no")},
            needs=("customers_out", "hours_out", "critical_facility")),
    ],
)
