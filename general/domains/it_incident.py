"""IT service desk: incident triage. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule

SLA = {"P1": 4, "P2": 8, "P3": 24, "P4": 72}            # resolution SLA in hours
SLA_STRICT = {k: v / 2 for k, v in SLA.items()}


def priority(f, vip_bump=False):
    if f["security"] or (f["outage"] and f["users_affected"] >= 500):
        p = 1
    elif f["outage"] or f["users_affected"] >= 100:
        p = 2
    elif not f["workaround"]:
        p = 3
    else:
        p = 4
    if vip_bump and f["vip"]:
        p = max(1, p - 1)
    return f"P{p}"


PRIORITY_RULE = ("P1 if there are signs of a security breach, or the service is fully down for 500 or more users. "
                 "P2 if it is fully down, or 100 or more users are affected. Otherwise P3 if there is no workaround, "
                 "and P4 if there is one.")

DOMAIN = Domain(
    name="it_incident",
    title="IT service desk incident triage",
    entities={
        "person": ["Dana Okafor", "Lukas Brandt", "Mei Tanaka", "Ravi Menon", "Sofia Alvarez", "Tom Weller"],
        "org": ["Finance", "Customer Care", "Warehouse Ops", "Sales East", "Legal", "R&D"],
    },
    situations=[
        "{org} reported a problem with {service}.",
        "New incident from {org}: {service} is misbehaving.",
        "We have a {service} issue affecting {org}.",
        "Escalation from {org} about {service}.",
        "{person} from {org} opened a ticket about {service}.",
        "Monitoring raised an alert on {service} for {org}.",
    ],
    distractors=[
        "The change freeze starts on Friday.",
        "Patch Tuesday updates were installed last week without issues.",
        "The printer on floor 3 was fixed yesterday.",
        "Remember to rotate the on-call schedule.",
    ],
    facts=[
        Fact("service", "cat", about="which IT service has the problem", values=("email", "VPN", "payroll", "CRM", "website", "Wi-Fi"),
             phrases={"formal": ["The affected service is {v}.", "This concerns the {v} service.",
                                 "The issue is with {v}.", "System affected: {v}."],
                      "informal": ["it's {v} again", "{v} is acting up", "problem is {v}", "{v} broke"]}),
        Fact("users_affected", "int", about="how many users are affected by the incident", values=(1, 3000), log=True, avoid=((100, 12), (500, 50)),
             phrases={"formal": ["{v} users are affected.", "Roughly {v} people cannot work normally.",
                                 "Impact: {v} users.", "About {v} staff reported the problem.",
                                 "The number of affected users is {v}."],
                      "informal": ["like {v} people are stuck", "{v} users pinging us", "{v} ppl affected",
                                   "about {v} folks can't work"]}),
        Fact("outage", "bool", about="whether the service is completely down (a full outage) rather than degraded", p=0.4,
             phrases={"formal": {True: ["The service is completely down.", "Nobody can reach it at all.",
                                        "It is a full outage.", "The service is entirely unavailable."],
                                 False: ["The service is up but degraded.", "It is slow but still working.",
                                         "It has not gone down; only some functions fail.",
                                         "It responds intermittently but is not down."]},
                      "informal": {True: ["it's totally down", "dead, nothing loads", "full outage",
                                          "completely down for everyone"],
                                   False: ["it's slow but up", "works, just laggy", "not down, just flaky",
                                           "still up, but glitchy"]}}),
        Fact("workaround", "bool", about="whether users have a workaround while it is broken", p=0.5,
             phrases={"formal": {True: ["A workaround is available.", "Users can follow the backup process for now.",
                                        "There is a documented workaround.",
                                        "Staff can work around it through the web portal."],
                                 False: ["There is no workaround.", "No alternative process exists.",
                                         "Users have no way around it.", "Nothing can be done until it is fixed."]},
                      "informal": {True: ["there's a workaround tho", "folks can use the old portal for now",
                                          "people can get by with the backup flow", "workaround exists"],
                                   False: ["no workaround", "no way around it", "they're totally blocked",
                                           "nothing they can do meanwhile"]}}),
        Fact("security", "bool", about="whether there are signs of a security breach or unauthorized access", p=0.2,
             phrases={"formal": {True: ["There are signs of unauthorized access.",
                                        "Suspicious logins were seen from an unknown location.",
                                        "The intrusion detection system raised the alert.",
                                        "Credentials may have been compromised."],
                                 False: ["There is no sign of a security issue.", "Security has ruled out an attack.",
                                         "It looks like a plain technical fault.", "No suspicious activity was found."]},
                      "informal": {True: ["might be a hack, weird logins", "IDS flagged it", "creds possibly leaked",
                                          "looks like someone got in"],
                                   False: ["no security angle", "just a normal bug it seems",
                                           "security says it's not an attack", "nothing shady going on"]}}),
        Fact("vip", "bool", about="whether an executive or leadership user is affected", p=0.2,
             phrases={"formal": {True: ["The CFO's office is among those affected.", "An executive raised this.",
                                        "It affects the leadership team.", "A board member cannot work."],
                                 False: ["No executives are affected.", "It was raised by regular staff.",
                                         "Only general users are involved.", "No leadership users are impacted."]},
                      "informal": {True: ["the CFO is affected btw", "an exec is complaining", "leadership is hit",
                                          "a VP is stuck too"],
                                   False: ["no execs involved", "just regular staff", "no VIPs", "normal users only"]}}),
        Fact("hours_open", "int", about="how many hours the incident ticket has been open", values=(1, 96), log=True,
             avoid=((2, 0.5), (4, 0.5), (8, 1), (12, 1), (24, 2), (36, 2), (72, 4)),
             phrases={"formal": ["The ticket has been open for {v} hours.", "It was reported {v} hours ago.",
                                 "Open for {v} hours so far.", "{v} hours have passed since the first report."],
                      "informal": ["{v}h and counting", "been {v} hours already", "open {v} hrs now",
                                   "{v} hours since it started"]}),
    ],
    questions=[
        Question(
            id="priority", type="score", text="What priority should this incident get?",
            paraphrases=("How urgent is this incident?", "Assign a priority level to this ticket.",
                         "Which priority fits this incident?", "Rate the incident's priority."),
            options=[Option("P4", "P4: low, a workaround exists and few users are affected"),
                     Option("P3", "P3: moderate, users are blocked without a workaround"),
                     Option("P2", "P2: high, an outage or many users affected"),
                     Option("P1", "P1: critical, a security breach or a large outage")],
            rules={"default": Rule(PRIORITY_RULE, priority),
                   "vip_bump": Rule(PRIORITY_RULE + " Then raise the priority one level (at most P1) if an "
                                    "executive is affected.", lambda f: priority(f, vip_bump=True))},
            needs=("security", "outage", "users_affected", "workaround", "vip")),
        Question(
            id="route_to", type="choice", text="Which team should handle this incident?",
            paraphrases=("Who should this ticket be assigned to?", "Route this incident to the right team.",
                         "Which resolver group owns this?"),
            options=[Option("security_team", "Security operations"),
                     Option("network_team", "Network engineering (VPN, Wi-Fi)"),
                     Option("messaging_team", "Messaging and collaboration (email)"),
                     Option("hr_systems", "HR systems (payroll)"),
                     Option("app_support", "Business application support (CRM, website)")],
            rules={"default": Rule(
                "Anything with signs of a security breach goes to security operations. Otherwise route by "
                "service: VPN and Wi-Fi to network engineering, email to messaging, payroll to HR systems, CRM and "
                "website to application support.",
                lambda f: "security_team" if f["security"] else {
                    "VPN": "network_team", "Wi-Fi": "network_team", "email": "messaging_team",
                    "payroll": "hr_systems"}.get(f["service"], "app_support"))},
            needs=("security", "service")),
        Question(
            id="sla_breached", type="noul", text="Has this incident breached its resolution SLA?",
            paraphrases=("Is this ticket past its SLA?", "Is the resolution deadline already missed?",
                         "Has the SLA clock run out on this incident?"),
            rules={"default": Rule(
                "The resolution SLA is 4 hours for P1, 8 for P2, 24 for P3 and 72 for P4. " + PRIORITY_RULE,
                lambda f: "yes" if f["hours_open"] > SLA[priority(f)] else "no"),
                   "strict_sla": Rule(
                "The resolution SLA is 2 hours for P1, 4 for P2, 12 for P3 and 36 for P4. " + PRIORITY_RULE,
                lambda f: "yes" if f["hours_open"] > SLA_STRICT[priority(f)] else "no")},
            needs=("hours_open", "security", "outage", "users_affected", "workaround")),
        Question(
            id="notify_ciso", type="noul", text="Should the CISO be notified now?",
            paraphrases=("Does this need to go to the CISO?", "Notify the chief information security officer?",
                         "Is a CISO notification required?"),
            rules={"default": Rule("Notify the CISO when there are signs of a security breach.",
                                   lambda f: "yes" if f["security"] else "no"),
                   "exec_outages": Rule("Notify the CISO when there are signs of a security breach, and also when "
                                        "an executive is affected by a full outage.",
                                        lambda f: "yes" if f["security"] or (f["vip"] and f["outage"]) else "no")},
            needs=("security", "vip", "outage")),
    ],
)
