"""Hotel front desk: room assignment at check-in. Held out of training (dev-unseen). All names are fictional."""

from ..spec import Domain, Fact, Option, Question, Rule


def room(f, gold_below=90):
    t, occ = f["loyalty_tier"], f["occupancy_pct"]
    if occ > 100 and t == "none" and not f["booked_direct"]:
        return "relocate_to_partner_hotel"
    if t == "platinum" or (t == "gold" and occ < gold_below):
        return "upgrade"
    return "assign_booked_room" if f["room_ready"] else "lounge_until_ready"


ROOM_RULE = ("If occupancy is over 100%, guests with no loyalty status who booked through an agency are relocated to "
             "a partner hotel. Otherwise platinum members get an upgrade, and gold members {g}. Otherwise give the "
             "booked room if it is ready, and offer the lounge until it is if not.")
GOLD = "get one when occupancy is under 90%"


def compensation(f):
    a = room(f)
    if a == "relocate_to_partner_hotel":
        return "free_night"
    if a == "lounge_until_ready":
        return "free_breakfast" if f["prior_complaint"] else "drink_voucher"
    return "none"


DOMAIN = Domain(
    name="hotel_front_desk",
    title="Hotel check-in and room assignment",
    heldout="dev",
    entities={
        "person": ["Mr. Albright", "Ms. Okonkwo", "Dr. Varga", "Mrs. Lindahl", "Mr. Esposito", "Ms. Chandra"],
        "hotel": ["the Harbour View", "the Maple Court", "the Sandpiper Inn"],
    },
    situations=[
        "{person} is checking in at {hotel}.",
        "Arrival at the front desk: {person}.",
        "{person} arrived for their reservation.",
        "Check-in request for {person} at {hotel}.",
        "The night manager is handling {person}'s arrival.",
        "Guest {person} is at reception.",
    ],
    formats=("email", "chat", "form", "report", "ticket"),
    distractors=[
        "The rooftop bar closes early tonight for a private event.",
        "Housekeeping is short one person on the evening shift.",
        "The conference group checks out tomorrow morning.",
        "The pool heater was serviced this afternoon.",
    ],
    facts=[
        Fact("occupancy_pct", "int", about="tonight's occupancy as a share of available rooms, in percent",
             values=(72, 112), avoid=((90, 1), (100, 1)), fmt=lambda v: f"{v}%",
             phrases={"formal": ["Tonight's occupancy stands at {v}.", "Occupancy: {v}.",
                                 "The hotel is at {v} of capacity tonight.", "Rooms sold tonight: {v} of inventory."],
                      "informal": ["we're at {v}", "occ {v} tonight", "{v} full", "house is {v}"]}),
        Fact("loyalty_tier", "cat", about="the guest's loyalty programme tier",
             values=("none", "silver", "gold", "platinum"), p=(0.45, 0.25, 0.18, 0.12),
             phrases={"none": ["The guest is not a loyalty member.", "They have no loyalty status.",
                               "No loyalty membership is on the profile.", "The guest holds no status."],
                      "silver": ["The guest is a silver member.", "Loyalty status: silver.",
                                 "They hold silver status.", "The profile shows silver tier."],
                      "gold": ["The guest is a gold member.", "Loyalty status: gold.", "They hold gold status.",
                               "The profile shows gold tier."],
                      "platinum": ["The guest is a platinum member.", "Loyalty status: platinum.",
                                   "They hold platinum status.", "The profile shows platinum tier."]}),
        Fact("booked_direct", "bool", about="whether the guest booked directly with the hotel rather than through "
             "an agency", p=0.5,
             phrases={"formal": {True: ["The reservation was made directly with the hotel.",
                                        "They booked on our own website.", "It is a direct booking.",
                                        "The booking came through our reservations line."],
                                 False: ["The reservation came through an online travel agency.",
                                         "It was booked via a third-party agency.", "It is an agency booking.",
                                         "They booked through a travel site."]},
                      "informal": {True: ["booked direct", "direct booking", "booked on our site",
                                          "came through our res line"],
                                   False: ["agency booking", "booked via a travel site", "OTA booking",
                                           "third-party booking"]}}),
        Fact("nights", "int", about="how many nights the guest is staying", values=(1, 14), log=True,
             phrases={"formal": ["The stay is {v} nights.", "Length of stay: {v} nights.",
                                 "They are booked for {v} nights.", "The reservation covers {v} nights."],
                      "informal": ["{v} nights", "staying {v} nights", "here for {v} nights", "{v}-night stay"]}),
        Fact("room_ready", "bool", about="whether the booked room is clean and ready", p=0.7,
             phrases={"formal": {True: ["The booked room is clean and ready.", "Housekeeping released the room.",
                                        "The room is inspected and available.", "Their room is ready."],
                                 False: ["The booked room is not ready yet.", "Housekeeping is still cleaning it.",
                                         "The room has not been released.", "Their room is still being prepared."]},
                      "informal": {True: ["room's ready", "room released", "room clean", "good to go on the room"],
                                   False: ["room not ready", "still being cleaned", "room not released yet",
                                           "HK still on it"]}}),
        Fact("prior_complaint", "bool", about="whether the guest complained during a previous stay", p=0.2,
             phrases={"formal": {True: ["The guest complained during a previous stay.",
                                        "The profile notes a complaint from their last visit.",
                                        "They raised a complaint last time.", "There is a past complaint on file."],
                                 False: ["There are no past complaints on the profile.",
                                         "They have never complained.", "No complaint history.",
                                         "Previous stays went without complaint."]},
                      "informal": {True: ["complained last time", "past complaint on file", "unhappy last stay",
                                          "has a complaint history"],
                                   False: ["no complaints before", "no complaint history", "clean history",
                                           "never complained"]}}),
    ],
    questions=[
        Question(
            id="room_action", type="choice", text="How should this guest's room be handled at check-in?",
            paraphrases=("What should reception do with this arrival?", "Decide the room assignment.",
                         "Which check-in action applies?"),
            options=[Option("assign_booked_room", "Give the booked room"), Option("upgrade", "Upgrade the room"),
                     Option("lounge_until_ready", "Offer the lounge until the room is ready"),
                     Option("relocate_to_partner_hotel", "Relocate to a partner hotel")],
            rules={"default": Rule(ROOM_RULE.format(g=GOLD), room),
                   "gold_always": Rule(ROOM_RULE.format(g="always get one"), lambda f: room(f, gold_below=1000))},
            needs=("occupancy_pct", "loyalty_tier", "booked_direct", "room_ready")),
        Question(
            id="compensation", type="score", text="What goodwill gesture should the guest get?",
            paraphrases=("What compensation fits this arrival?", "Which amenity do we offer?",
                         "Pick the goodwill gesture."),
            options=[Option("none", "nothing"), Option("drink_voucher", "a drink voucher"),
                     Option("free_breakfast", "free breakfast"), Option("free_night", "a free night")],
            rules={"default": Rule(ROOM_RULE.format(g=GOLD) + " Relocated guests get a free night. Guests waiting for "
                                   "their room get free breakfast if they complained on a previous stay, and a drink "
                                   "voucher if not. Nobody else gets anything.", compensation)},
            needs=("occupancy_pct", "loyalty_tier", "booked_direct", "room_ready", "prior_complaint")),
        Question(
            id="manager_welcome", type="noul", text="Should the duty manager greet this guest?",
            paraphrases=("Does this guest get a personal welcome from the manager?", "Manager greeting needed?",
                         "Should the duty manager meet this arrival?"),
            rules={"default": Rule("The duty manager greets platinum members and guests staying 7 or more nights.",
                                   lambda f: "yes" if f["loyalty_tier"] == "platinum" or f["nights"] >= 7 else "no"),
                   "nights_4": Rule("The duty manager greets platinum members and guests staying 4 or more nights.",
                                    lambda f: "yes" if f["loyalty_tier"] == "platinum" or f["nights"] >= 4 else
                                    "no")},
            needs=("loyalty_tier", "nights")),
    ],
)
