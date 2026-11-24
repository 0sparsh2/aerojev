"""Software engineering: release go / no-go checks. Held out of training (test-unseen). All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def decision(f, min_pass=95):
    if f["open_critical_bugs"] > 0 or f["test_pass_pct"] < min_pass or not f["on_call"]:
        return "hold"
    if f["change_freeze"]:
        return "needs_cab_approval"
    if f["change_risk"] == "high" or not f["rollback_tested"]:
        return "canary_first"
    return "ship"


DECISION_RULE = ("Hold if any critical bug is open, fewer than {p}% of tests pass, or no engineer is on call. "
                 "Otherwise, during a change freeze, the release needs change advisory board approval. Otherwise "
                 "high-risk changes and changes without a tested rollback go to a canary first. Otherwise ship.")


def radius(f):
    s, r = f["services_affected"], f["change_risk"]
    if s >= 10 or r == "high":
        return "high"
    if s >= 3 or r == "medium":
        return "medium"
    return "low"


DOMAIN = Domain(
    name="release_gate",
    title="Software release go / no-go",
    heldout="test",
    entities={
        "service": ["checkout-api", "search-indexer", "billing-worker", "mobile-gateway", "auth-service",
                    "reporting-ui"],
        "person": ["Nora Haddad", "Felipe Duarte", "Ruth Adeyemi", "Sven Olsen", "Kavya Reddy", "Jon Matsuda"],
    },
    situations=[
        "Release candidate for {service} is ready for the go / no-go check.",
        "{person} asked to deploy a new version of {service}.",
        "Deployment request: {service}.",
        "The release train for {service} reached the gate.",
        "{person} opened a change request for {service}.",
        "Pre-release review for {service}.",
    ],
    distractors=[
        "The engineering all-hands moved to Thursday.",
        "The staging cluster gets new nodes next week.",
        "The style guide for commit messages was updated.",
        "Remember to fill in the quarterly tooling survey.",
    ],
    facts=[
        Fact("test_pass_pct", "float", about="the share of automated tests that pass, in percent",
             values=(94, 100), avoid=((95, 0.2), (97, 0.2)), fmt=lambda v: f"{v:.2f}%",
             phrases={"formal": ["{v} of automated tests pass.", "Test pass rate: {v}.",
                                 "The CI suite reports a {v} pass rate.", "Passing tests: {v}."],
                      "informal": ["{v} green", "tests at {v}", "{v} passing", "CI: {v} pass"]}),
        Fact("open_critical_bugs", "int", about="how many critical bugs are open against the release",
             values=(0, 3), p=(0.75, 0.12, 0.08, 0.05),
             phrases={"formal": ["{v} critical bugs are open.", "Open critical defects: {v}.",
                                 "The tracker shows {v} critical issues against this release.",
                                 "Critical bugs outstanding: {v}."],
                      "informal": ["{v} crit bugs open", "{v} P0s open", "crit bugs: {v}", "{v} criticals"]}),
        Fact("change_freeze", "bool", about="whether a change freeze is in effect", p=0.15,
             phrases={"formal": {True: ["A change freeze is in effect.", "We are inside the holiday change freeze.",
                                        "The freeze window is active.", "Production changes are frozen this week."],
                                 False: ["No change freeze is in effect.", "We are outside any freeze window.",
                                         "Normal change rules apply.", "There is no freeze this week."]},
                      "informal": {True: ["freeze is on", "we're in freeze", "change freeze active",
                                          "prod frozen"],
                                   False: ["no freeze", "not in freeze", "freeze is over", "normal week"]}}),
        Fact("rollback_tested", "bool", about="whether the rollback plan has been tested", p=0.7,
             phrases={"formal": {True: ["The rollback was rehearsed in staging.", "The rollback plan is tested.",
                                        "A tested rollback procedure exists.", "Rollback was verified."],
                                 False: ["The rollback has not been tested.", "There is no tested rollback plan.",
                                         "Rollback was never rehearsed.", "The rollback procedure is unverified."]},
                      "informal": {True: ["rollback tested", "rollback rehearsed", "rollback works",
                                          "tested the rollback"],
                                   False: ["rollback untested", "never tried the rollback", "no tested rollback",
                                           "rollback unverified"]}}),
        Fact("change_risk", "cat", about="the change's risk rating", values=("low", "medium", "high"),
             p=(0.45, 0.35, 0.2),
             phrases={"formal": ["The change is rated {v} risk.", "Risk rating: {v}.", "This is a {v}-risk change.",
                                 "The change carries {v} risk."],
                      "informal": ["{v} risk", "risk: {v}", "{v}-risk change", "rated {v}"]}),
        Fact("on_call", "bool", about="whether an engineer is on call for the release window", p=0.9,
             phrases={"formal": {True: ["An engineer is on call for the release window.", "On-call cover is in place.",
                                        "The on-call rota covers the deployment.", "Someone is paged in if needed."],
                                 False: ["No engineer is on call for the release window.", "On-call cover is missing.",
                                         "The rota has a gap during the deployment.", "Nobody is on call."]},
                      "informal": {True: ["on-call covered", "someone's on call", "pager covered", "on-call set"],
                                   False: ["no on-call", "nobody on call", "rota gap", "pager uncovered"]}}),
        Fact("services_affected", "int", about="how many services the release touches", values=(1, 30), log=True,
             phrases={"formal": ["The release touches {v} services.", "Services affected: {v}.",
                                 "{v} services are part of this deployment.", "It changes {v} services."],
                      "informal": ["{v} services", "touches {v} svcs", "{v} services in scope",
                                   "deploys {v} services"]}),
    ],
    questions=[
        Question(
            id="decision", type="choice", text="Is this release a go?",
            paraphrases=("What is the go / no-go decision?", "Should this release go out?",
                         "Decide on this deployment."),
            options=[Option("ship", "Ship to production"), Option("canary_first", "Ship to a canary first"),
                     Option("needs_cab_approval", "Needs change advisory board approval"),
                     Option("hold", "Hold the release")],
            rules={"default": Rule(DECISION_RULE.format(p=95), decision),
                   "pass_97": Rule(DECISION_RULE.format(p=97), lambda f: decision(f, min_pass=97))},
            needs=("open_critical_bugs", "test_pass_pct", "on_call", "change_freeze", "change_risk",
                   "rollback_tested")),
        Question(
            id="blast_radius", type="score", text="How large is this release's blast radius?",
            paraphrases=("Rate the potential impact of this release.", "How wide could a failure spread?",
                         "What is the blast radius?"),
            options=[Option("low", "low"), Option("medium", "medium"), Option("high", "high")],
            rules={"default": Rule("High if it touches 10 or more services or the change is high risk. Medium if it "
                                   "touches 3 or more services or the change is medium risk. Low otherwise.", radius)},
            needs=("services_affected", "change_risk")),
        Question(
            id="notify_support", type="noul", text="Should customer support be notified before the release?",
            paraphrases=("Does support need a heads-up?", "Warn customer support about this release?",
                         "Is a support notification required?"),
            rules={"default": Rule("Notify support before releases that touch 5 or more services, and before any "
                                   "high-risk change.",
                                   lambda f: "yes" if f["services_affected"] >= 5 or f["change_risk"] == "high"
                                   else "no"),
                   "services_10": Rule("Notify support before releases that touch 10 or more services, and before "
                                       "any high-risk change.",
                                       lambda f: "yes" if f["services_affected"] >= 10 or f["change_risk"] == "high"
                                       else "no")},
            needs=("services_affected", "change_risk")),
    ],
)
