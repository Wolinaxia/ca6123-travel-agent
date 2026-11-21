"""Out-of-scope / unsafe topic blocking.

This is intentionally a *small* intent detector, not a general classifier. The
orchestrator handles harmless-but-out-of-scope travel questions; only requests
that ask for real-world harm or evasion are blocked here:

* making/acquiring weapons or explosives, or using them against people;
* evading, hiding things from, or getting around security/customs screening;
* illegal documents, smuggling, bribery, laundering;
* self-harm.

Asking **what the rules allow** is explicitly *not* blocked: "can I carry a
pocket knife in checked luggage?", "what are the customs rules for fruit?" and
"how do I declare cash at customs?" are ordinary travel questions. Matching is
case-insensitive and word-bounded, and each dangerous noun needs a verb that
shows harmful intent, so a news sentence such as "security line wait at JFK was
25 minutes" passes while "how do I hide a knife from security?" does not.

Getting a prohibited item (knife, blade, gun, explosive, lighter fluid, ...)
past screening covertly is an evasive intent: a covert transport verb with the
item and a checkpoint, or an item together with "undetected"/"without being
detected". Asking whether the item is allowed through carry-on or checked
baggage does not use those markers and therefore stays allowed.
"""

from __future__ import annotations

import re
from typing import Any

_WEAPON = (
    r"(?:bombs?|explosives?|firearms?|guns?|rifles?|pistols?|ammunition|"
    r"grenades?|silencers?|weapons?|machine\s*guns?|pipe\s*bombs?|nail\s*bombs?)"
)
_DOCUMENT = r"(?:passports?|visas?|documents?|ids?|licen[cs]es?|certificates?)"
_PROHIBITED_ITEM = (
    r"(?:knives|knife|blades?|box\s*cutters?|razors?|guns?|firearms?|pistols?|"
    r"rifles?|weapons?|explosives?|bombs?|ammunition|lighter\s*fluid|fireworks?|"
    r"flares?|tasers?|brass\s*knuckles?)"
)
_SCREENING = (
    r"(?:security|tsa|screening|screeners?|customs|checkpoints?|scanners?|"
    r"x[- ]?rays?|metal\s*detectors?|detection)"
)
# Verbs/markers that show the item is being moved covertly, as opposed to a
# question about the rules ("can I carry/pack/bring ...").
_COVERT_TRANSPORT = r"(?:sneak|slip|hide|conceal|get|smuggle)"
_DETECTION_EVASION = (
    r"(?:undetected|unnoticed|without\s+(?:being\s+)?(?:detected|noticed|caught)|"
    r"without\s+detection|avoid(?:ing)?\s+detection|not\s+(?:be\s+)?(?:detected|caught))"
)

_RULES: list[tuple[str, list[re.Pattern[str]], str]] = [
    (
        "weapons",
        [
            # Building/acquiring a weapon or explosive.
            re.compile(
                rf"\b(?:build|make|manufacture|assemble|create|produce|3d[- ]print|"
                rf"print|brew|construct)\b[^.\n]{{0,50}}\b{_WEAPON}\b"
            ),
            re.compile(
                rf"\b(?:buy|sell|acquire|obtain|purchase)\b[^.\n]{{0,30}}"
                rf"\b(?:illegal|untraceable|unregistered|ghost|homemade)\s+{_WEAPON}\b"
            ),
            # Using a weapon against people.
            re.compile(rf"\b(?:shoot|stab|kill|attack|hurt)\b[^.\n]{{0,30}}\b(?:someone|people|person|him|her|them)\b[^.\n]{{0,30}}\b{_WEAPON}\b"),
            re.compile(rf"\bpipe\s*bomb\b"),
            re.compile(r"\b3d[- ]printed (?:gun|firearm|weapon)\b"),
        ],
        "This request appears to involve weapons or explosives, which I can't help with.",
    ),
    (
        "security_evasion",
        [
            re.compile(
                r"\b(?:evade|bypass|get around|beat|dodge)\b[^.\n]{0,30}"
                r"\b(?:security|tsa|customs|screening|checkpoints?|scanners?|immigration)\b"
            ),
            re.compile(
                r"\b(?:hide|conceal|smuggle|sneak|slip)\b[^.\n]{0,40}"
                r"\b(?:from|past|through|onto|into)\b[^.\n]{0,30}"
                r"\b(?:security|tsa|customs|screening|checkpoints?|scanners?|plane|aircraft)\b"
            ),
            re.compile(rf"\bsneak (?:a |an |the )?{_WEAPON}\b"),
            # Covertly moving a prohibited item past screening.
            re.compile(
                rf"\b{_COVERT_TRANSPORT}\b[^.\n]{{0,60}}\b{_PROHIBITED_ITEM}\b"
                rf"[^.\n]{{0,40}}\b(?:past|through|around|onto|into|aboard|from)\b"
                rf"[^.\n]{{0,30}}\b{_SCREENING}\b"
            ),
            # "without being detected" on either side of the item.
            re.compile(rf"\b{_PROHIBITED_ITEM}\b[^.\n]{{0,60}}{_DETECTION_EVASION}\b"),
            re.compile(rf"\b{_DETECTION_EVASION}\b[^.\n]{{0,40}}\b{_PROHIBITED_ITEM}\b"),
            re.compile(r"\bthrough (?:airport )?security (?:with|undetected)\b"),
            re.compile(r"\bundeclared (?:cash|money|goods|items)\b"),
        ],
        "I can't help with evading security screening or customs.",
    ),
    (
        "illegal_activity",
        [
            re.compile(r"\b(?:smuggle|smuggling|smuggler)s?\b"),
            re.compile(rf"\b(?:fake|faked|forged|forge|counterfeit|counterfeited)\s+(?:a |an |the )?{_DOCUMENT}\b"),
            re.compile(r"\bbuy stolen\b"),
            re.compile(r"\b(?:launder|laundering)\s+(?:money|cash|funds)\b"),
            re.compile(r"\bbribe (?:a |an |the )?(?:official|customs|officer|guard)\b"),
        ],
        "I can't help with illegal activity such as fake documents or smuggling.",
    ),
    (
        "self_harm",
        [
            re.compile(r"\b(?:kill|hurt|harm) myself\b"),
            re.compile(r"\bsuicide\b"),
            re.compile(r"\bself[- ]harm\b"),
            re.compile(r"\bend (?:my|it) (?:own )?life\b"),
        ],
        "If you're thinking about harming yourself, please contact a local crisis line; I can't help with this.",
    ),
]


def check_topic(text: str) -> dict[str, Any]:
    """Return ``{"allowed", "category", "reason"}`` for a user message.

    ``allowed`` is False only for the four harmful-intent categories above; every
    other question (including out-of-scope but harmless ones, and "what is
    allowed" questions) is allowed here.
    """
    if not isinstance(text, str) or not text.strip():
        return {"allowed": True, "category": None, "reason": "allowed"}
    lowered = text.lower()
    for category, patterns, reason in _RULES:
        if any(pattern.search(lowered) for pattern in patterns):
            return {"allowed": False, "category": category, "reason": reason}
    return {"allowed": True, "category": None, "reason": "allowed"}
