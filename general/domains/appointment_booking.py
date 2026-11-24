"""Outpatient clinic front desk: booking referred appointments (administrative rules only, no clinical advice).
All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def booking(f, refer=60):
    u = f["referral_urgency"]
    if u == "urgent":
        return "book_urgent_slot"
    if not f["insurance_verified"]:
        return "verify_insurance_first"
    limit = refer // 2 if u == "soon" else refer
    if f["wait_days"] > limit:
        return "refer_to_partner_clinic"
    if f["missed_appointments"] >= 2:
        return "add_to_waitlist"
    return "book_next_slot"


BOOKING_RULE = ("Urgent referrals: book an urgent slot. Otherwise verify insurance first if it is not verified. "
                "Otherwise refer to a partner clinic if the wait for the next routine slot is over {d} days (over "
                "{h} days for referrals marked soon). Otherwise put patients with 2 or more missed appointments in "
                "the past year on the waitlist, and book everyone else into the next slot.")


def length(f):
    n = f["new_patient"] + f["interpreter_needed"]
    return ["15_min", "30_min", "45_min"][n]


DOMAIN = Domain(
    name="appointment_booking",
    title="Outpatient clinic appointment booking",
    entities={
        "person": ["Agnes Moore", "Rashid Khan", "Elsa Bergström", "Tomás Ruiz", "Mbali Dlamini", "George Pratt"],
        "clinic": ["dermatology", "cardiology", "orthopaedics", "ENT", "gastroenterology"],
    },
    situations=[
        "A referral to {clinic} arrived for {person}.",
        "{person} needs an appointment in {clinic}.",
        "Booking request: {clinic}, patient {person}.",
        "The {clinic} front desk received a referral for {person}.",
        "{person} called to book their {clinic} appointment.",
        "New referral in the {clinic} inbox.",
    ],
    formats=("email", "ticket", "chat", "form", "report"),
    distractors=[
        "The car park barrier is being repaired this week.",
        "Flu vaccination clinics start next month.",
        "The waiting room TV was replaced.",
        "Staff photos for the new badges are on Monday.",
    ],
    facts=[
        Fact("referral_urgency", "cat", about="the urgency the referring clinician marked on the referral",
             values=("routine", "soon", "urgent"), p=(0.55, 0.3, 0.15),
             phrases={"formal": ["The referral is marked {v}.", "Referral urgency: {v}.",
                                 "The referring doctor marked it as {v}.", "Priority on the referral: {v}."],
                      "informal": ["marked {v}", "{v} referral", "urgency {v}", "GP put {v}"]}),
        Fact("insurance_verified", "bool", about="whether the patient's insurance has been verified", p=0.75,
             phrases={"formal": {True: ["Insurance eligibility was verified.", "Coverage is confirmed.",
                                        "The insurer confirmed the patient's cover.", "Insurance is verified."],
                                 False: ["Insurance has not been verified yet.", "Coverage is unconfirmed.",
                                         "The insurer has not confirmed cover.", "Eligibility check is pending."]},
                      "informal": {True: ["insurance verified", "cover confirmed", "eligibility ok",
                                          "insurer confirmed"],
                                   False: ["insurance not verified", "cover unconfirmed", "eligibility pending",
                                           "waiting on insurer"]}}),
        Fact("wait_days", "int", about="how many days away the clinic's next routine slot is", values=(1, 120),
             log=True, avoid=((15, 1), (30, 1), (60, 2)),
             phrases={"formal": ["The next routine slot is {v} days away.", "Current wait: {v} days.",
                                 "The first available routine appointment is in {v} days.",
                                 "Routine waiting time is {v} days."],
                      "informal": ["next slot in {v} days", "{v} day wait", "wait's {v} days",
                                   "earliest is {v} days out"]}),
        Fact("missed_appointments", "int", about="how many appointments the patient missed in the past year",
             values=(0, 4), p=(0.55, 0.2, 0.12, 0.08, 0.05),
             phrases={"formal": ["The patient missed {v} appointments in the past year.",
                                 "Missed appointments in 12 months: {v}.", "Did-not-attend count this year: {v}.",
                                 "They failed to attend {v} times in the last year."],
                      "informal": ["{v} no-shows this year", "missed {v} appts", "DNA count {v}",
                                   "{v} missed appointments"]}),
        Fact("new_patient", "bool", about="whether the patient is new to the clinic", p=0.5,
             phrases={"formal": {True: ["The patient is new to the clinic.", "This is their first visit here.",
                                        "They have not been seen by this clinic before.",
                                        "There is no previous record at this clinic."],
                                 False: ["The patient is already known to the clinic.", "They were seen here before.",
                                         "They are an existing patient.", "There are earlier visits on file."]},
                      "informal": {True: ["new patient", "first visit", "never seen here", "new to us"],
                                   False: ["existing patient", "seen before", "known to clinic",
                                           "been here before"]}}),
        Fact("interpreter_needed", "bool", about="whether the patient needs an interpreter", p=0.15,
             phrases={"formal": {True: ["The patient needs an interpreter.", "A Portuguese interpreter is required.",
                                        "Interpreter support was requested.", "They need language support."],
                                 False: ["No interpreter is needed.", "The patient does not need language support.",
                                         "No interpreter was requested.", "They are comfortable in English."]},
                      "informal": {True: ["needs interpreter", "interpreter required", "language support needed",
                                          "book an interpreter too"],
                                   False: ["no interpreter needed", "english fine", "no language support needed",
                                           "no interpreter"]}}),
    ],
    questions=[
        Question(
            id="booking", type="choice", text="How should this referral be booked?",
            paraphrases=("What should the front desk do with this referral?", "Decide how to book this patient.",
                         "What is the booking action?"),
            options=[Option("book_next_slot", "Book the next routine slot"),
                     Option("book_urgent_slot", "Book an urgent slot"),
                     Option("add_to_waitlist", "Add to the waitlist"),
                     Option("verify_insurance_first", "Verify insurance before booking"),
                     Option("refer_to_partner_clinic", "Refer to a partner clinic")],
            rules={"default": Rule(BOOKING_RULE.format(d=60, h=30), booking),
                   "refer_30": Rule(BOOKING_RULE.format(d=30, h=15), lambda f: booking(f, refer=30))},
            needs=("referral_urgency", "insurance_verified", "wait_days", "missed_appointments")),
        Question(
            id="slot_length", type="score", text="How long should the appointment be?",
            paraphrases=("What appointment length should be booked?", "Pick the slot length.",
                         "How much time should this visit get?"),
            options=[Option("15_min", "15 minutes"), Option("30_min", "30 minutes"), Option("45_min", "45 minutes")],
            rules={"default": Rule("15 minutes, plus 15 for a new patient and 15 more if an interpreter is needed.",
                                   length)},
            needs=("new_patient", "interpreter_needed")),
        Question(
            id="reminder_call", type="noul", text="Should the patient get a reminder phone call?",
            paraphrases=("Does this patient need a reminder call?", "Phone the patient before the visit?",
                         "Is a reminder call required?"),
            rules={"default": Rule("Call patients with 2 or more missed appointments in the past year, and patients "
                                   "who need an interpreter.",
                                   lambda f: "yes" if f["missed_appointments"] >= 2 or f["interpreter_needed"]
                                   else "no"),
                   "any_missed": Rule("Call patients with any missed appointment in the past year, and patients who "
                                      "need an interpreter.",
                                      lambda f: "yes" if f["missed_appointments"] >= 1 or f["interpreter_needed"]
                                      else "no")},
            needs=("missed_appointments", "interpreter_needed")),
    ],
)
