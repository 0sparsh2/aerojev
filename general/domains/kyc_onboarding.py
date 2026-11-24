"""Banking compliance: know-your-customer review at account opening. All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def money(v):
    return f"${v:,}"


def decision(f, edd_medium=False):
    if f["sanctions_match"]:
        return "reject"
    if not f["id_verified"]:
        return "request_documents"
    risky_country = f["country_risk"] == "high" or (edd_medium and f["country_risk"] == "medium")
    if f["pep"] or risky_country or f["adverse_media"] or f["monthly_volume"] > 250000:
        return "enhanced_due_diligence"
    return "approve"


DECISION_RULE = ("Reject on a sanctions list match. Otherwise ask for documents if identity is not verified. Otherwise "
                 "apply enhanced due diligence to politically exposed persons, applicants from {c}, applicants with "
                 "adverse media, and anyone expecting over $250,000 a month. Otherwise approve.")


def rating(f):
    if f["pep"] or f["sanctions_match"] or f["country_risk"] == "high":
        return "high"
    if f["country_risk"] == "medium" or f["adverse_media"] or f["account_type"] == "company" \
            or f["monthly_volume"] > 10000:
        return "medium"
    return "low"


DOMAIN = Domain(
    name="kyc_onboarding",
    title="Know-your-customer review at account opening",
    entities={
        "person": ["Viktor Hale", "Amara Obi", "Selin Yilmaz", "Jonas Berg", "Lucia Ferraro", "Kenji Arai"],
        "org": ["Tallis Trading Ltd", "Morrow & Pike LLP", "Ostra Imports", "Bellweather Studio"],
    },
    situations=[
        "{person} applied to open an account.",
        "New account application on behalf of {org}.",
        "KYC review needed for {person}.",
        "Onboarding file for {org} is ready for review.",
        "{person} submitted an online account application.",
        "Compliance queue: new customer {person}.",
    ],
    formats=("email", "ticket", "form", "report", "log"),
    distractors=[
        "The compliance training deadline is next Friday.",
        "Branch staff rotas for December are out.",
        "The online banking login page has a new design.",
        "Quarterly regulatory reporting starts next week.",
    ],
    facts=[
        Fact("sanctions_match", "bool", about="whether the applicant matched a sanctions list", p=0.07,
             phrases={"formal": {True: ["Screening returned a confirmed sanctions list match.",
                                        "The applicant matches an entry on a sanctions list.",
                                        "A sanctions hit was confirmed by the analyst.",
                                        "The name is a true match on the sanctions list."],
                                 False: ["Sanctions screening is clear.", "No sanctions match was found.",
                                         "The applicant does not appear on any sanctions list.",
                                         "Sanctions screening returned no hits."]},
                      "informal": {True: ["sanctions hit, confirmed", "matches the sanctions list",
                                          "true match on sanctions", "sanctions match"],
                                   False: ["sanctions clear", "no sanctions hits", "screening clean",
                                           "not on any sanctions list"]}}),
        Fact("id_verified", "bool", about="whether the applicant's identity has been verified", p=0.85,
             phrases={"formal": {True: ["Identity documents were verified.", "ID verification passed.",
                                        "The passport check succeeded.", "Identity is confirmed."],
                                 False: ["Identity could not be verified.", "The ID document was unreadable.",
                                         "ID verification is incomplete.", "Identity is not yet confirmed."]},
                      "informal": {True: ["ID verified", "ID check passed", "passport ok", "identity confirmed"],
                                   False: ["ID not verified", "ID check failed", "couldn't verify identity",
                                           "ID still pending"]}}),
        Fact("pep", "bool", about="whether the applicant is a politically exposed person", p=0.1,
             phrases={"formal": {True: ["The applicant is a politically exposed person.",
                                        "They hold a senior public office.", "Screening identified them as a PEP.",
                                        "A close family member holds high public office."],
                                 False: ["The applicant is not a politically exposed person.",
                                         "There is no PEP status.", "They hold no public office.",
                                         "PEP screening is negative."]},
                      "informal": {True: ["PEP", "politically exposed", "holds public office", "PEP flag"],
                                   False: ["not a PEP", "no PEP flag", "no public office", "PEP screening clear"]}}),
        Fact("country_risk", "cat", about="the risk rating of the applicant's country of residence",
             values=("low", "medium", "high"), p=(0.6, 0.25, 0.15),
             phrases={"formal": ["The country of residence is rated {v} risk.", "Country risk: {v}.",
                                 "They reside in a {v}-risk jurisdiction.", "Jurisdiction risk rating: {v}."],
                      "informal": ["country risk {v}", "{v}-risk country", "lives in a {v} risk country",
                                   "jurisdiction: {v}"]}),
        Fact("adverse_media", "bool", about="whether there is negative news coverage about the applicant", p=0.12,
             phrases={"formal": {True: ["Negative news coverage about the applicant was found.",
                                        "Adverse media screening returned relevant articles.",
                                        "Press reports link them to a fraud investigation.",
                                        "There are adverse media hits."],
                                 False: ["No adverse media was found.", "News screening is clean.",
                                         "There is no negative press about them.",
                                         "Adverse media screening returned nothing relevant."]},
                      "informal": {True: ["bad press found", "adverse media hits", "negative news on them",
                                          "media screening flagged stuff"],
                                   False: ["no adverse media", "news check clean", "no bad press",
                                           "media screening clear"]}}),
        Fact("monthly_volume", "int", about="the expected monthly transaction volume in dollars",
             values=(500, 2000000), log=True, avoid=((10000, 800), (100000, 8000), (250000, 20000)), fmt=money,
             phrases={"formal": ["Expected monthly volume is {v}.", "They expect to move {v} a month.",
                                 "Projected monthly turnover: {v}.", "Anticipated activity: {v} per month."],
                      "informal": ["~{v}/month", "{v} a month expected", "volume {v} monthly",
                                   "expects {v} per month"]}),
        Fact("account_type", "cat", about="the type of account being opened",
             values=("personal", "sole trader", "company"), p=(0.5, 0.2, 0.3),
             phrases={"formal": ["This is a {v} account application.", "Account type: {v}.",
                                 "They want to open a {v} account.", "Applicant type: {v}."],
                      "informal": ["{v} account", "type: {v}", "{v}", "opening a {v} acct"]}),
    ],
    questions=[
        Question(
            id="decision", type="choice", text="What is the KYC decision for this application?",
            paraphrases=("How should this onboarding be handled?", "Decide on this account application.",
                         "What should compliance do with this applicant?"),
            options=[Option("approve", "Approve and open the account"),
                     Option("request_documents", "Ask for identity documents"),
                     Option("enhanced_due_diligence", "Apply enhanced due diligence"),
                     Option("reject", "Reject the application")],
            rules={"default": Rule(DECISION_RULE.format(c="high-risk countries"), decision),
                   "edd_medium": Rule(DECISION_RULE.format(c="high-risk or medium-risk countries"),
                                      lambda f: decision(f, edd_medium=True))},
            needs=("sanctions_match", "id_verified", "pep", "country_risk", "adverse_media", "monthly_volume")),
        Question(
            id="risk_rating", type="score", text="What customer risk rating should be assigned?",
            paraphrases=("Rate this customer's risk.", "Which risk rating fits this applicant?",
                         "Assign a KYC risk rating."),
            options=[Option("low", "low"), Option("medium", "medium"), Option("high", "high")],
            rules={"default": Rule("High for politically exposed persons, sanctions matches and high-risk countries. "
                                   "Otherwise medium for medium-risk countries, adverse media, company accounts, or "
                                   "expected volume over $10,000 a month. Otherwise low.", rating)},
            needs=("pep", "sanctions_match", "country_risk", "adverse_media", "account_type", "monthly_volume")),
        Question(
            id="senior_signoff", type="noul", text="Does this application need senior compliance sign-off?",
            paraphrases=("Must a senior compliance officer approve this?", "Is senior sign-off required?",
                         "Does this go to the head of compliance?"),
            rules={"default": Rule("Senior sign-off is needed for politically exposed persons, and for companies "
                                   "expecting over $100,000 a month.",
                                   lambda f: "yes" if f["pep"] or (f["account_type"] == "company" and
                                                                   f["monthly_volume"] > 100000) else "no"),
                   "low_bar": Rule("Senior sign-off is needed for politically exposed persons, and for companies "
                                   "expecting over $10,000 a month.",
                                   lambda f: "yes" if f["pep"] or (f["account_type"] == "company" and
                                                                   f["monthly_volume"] > 10000) else "no")},
            needs=("pep", "account_type", "monthly_volume")),
    ],
)
