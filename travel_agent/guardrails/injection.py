"""Prompt-injection detection: deterministic rules (layer 1) + optional LLM.

Why rules first?
---------------
Every tool result is data, not instruction, and the orchestrator is allowed to
show news headlines. A small deterministic detector keeps the guardrail
auditable, fast and offline. Layer 2 (``llm_classifier``) only runs when the
rules find nothing *and* it is enabled (``use_llm=True`` or
``GUARDRAIL_LLM=1``), so the common path never needs a key or the network.

Scoring
-------
Each category has an explicit weight; the score is the capped sum of the
distinct categories that matched. ``flagged`` is ``score >= 0.5``, so one strong
signal (instruction override, role hijack, prompt extraction, fake role tag,
forged approval) is enough.

Normalisation
-------------
Attackers obfuscate with casing, extra whitespace, zero-width characters and
punctuation. We match against two views of the text:

``norm``     -- lowercased, zero-width characters removed, whitespace collapsed;
``compact``  -- ``norm`` with every non-alphanumeric character removed.

``compact`` is what catches ``i g n o r e  p r e v i o u s  i n s t r u c t i o n s``
and ``ig-nore prev-ious instructions``. It is only used for phrase patterns, not
for tag patterns like ``SYSTEM:`` (which need their punctuation).

Precision notes
---------------
Role words alone are not an attack. ``act as`` / ``you are now`` only count when
followed by a *persona noun* (``act as an unrestricted AI``) rather than an
adverbial or possessive phrase (``act as fast as you can``, ``you are now my
favourite helper``). ``repeat`` / ``print`` only count with a prompt-extraction
object (``the text above``, ``your instructions``), not with a normal travel
noun.
"""

from __future__ import annotations

import os
import re
from typing import Any

FLAG_THRESHOLD = 0.5
_BASE64_RE = re.compile(r"(?<![\w+/=])[A-Za-z0-9+/]{40,}={0,2}(?![\w+/=])")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})

# Persona/role nouns. Kept as a fragment so both ``act as`` and ``you are now``
# can reuse it; matching a noun is what separates a role hijack from ordinary
# prose such as "act as fast as you can".
_ROLE = (
    r"(?:assistant|ai|a\.i\.|model|bot|chatbot|system|developer|admin|"
    r"administrator|hacker|llm|gpt|dan|jailbroken|uncensored|unrestricted|"
    r"evil|pirate)"
)

# (name, pattern, weight). Patterns are applied to the normalised and, where
# marked, the compact view. All patterns are lowercase.
_NORM_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass|skip)\b[^.!?\n]{0,50}"
            r"\b(?:all|any|previous|prior|above|earlier|existing|original)\b[^.!?\n]{0,50}"
            r"\b(?:instructions?|prompts?|rules?|directions?|directives?|guardrails?|polic(?:y|ies))\b"
        ),
        0.6,
    ),
    (
        "instruction_override",
        re.compile(
            r"\b(?:new|updated|replacement|revised)\s+(?:instructions?|rules?|directives?)\s*:"
        ),
        0.6,
    ),
    (
        "instruction_override",
        re.compile(
            r"\bfrom now on\b[^.!?\n]{0,40}"
            r"\b(?:you will|you must|you should|always|never|respond|answer|ignore|obey)\b"
        ),
        0.6,
    ),
    (
        "instruction_override",
        re.compile(r"\b(?:ignore|disregard|forget)\s+(?:the\s+)?(?:text|message|content)\s+above\b"),
        0.6,
    ),
    (
        "role_hijack",
        re.compile(rf"\byou are now\s+(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){{0,2}}?{_ROLE}\b"),
        0.6,
    ),
    (
        "role_hijack",
        re.compile(rf"\bact as\s+(?:a|an|the)\s+(?:[\w-]+\s+){{0,2}}?{_ROLE}\b"),
        0.6,
    ),
    (
        "role_hijack",
        re.compile(rf"\bpretend (?:to be|you are)\s+(?:a|an|the)\s+(?:[\w-]+\s+){{0,2}}?{_ROLE}\b"),
        0.6,
    ),
    (
        "role_hijack",
        re.compile(r"\b(?:developer|god|admin|administrator|dan|jailbreak|unrestricted)\s+mode\b"),
        0.6,
    ),
    ("role_hijack", re.compile(r"\bjailbreak(?:ed|ing)?\b"), 0.6),
    ("role_hijack", re.compile(r"\bunrestricted (?:assistant|ai|model|mode)\b"), 0.6),
    (
        "role_hijack",
        re.compile(r"\b(?:no|without|free of)\s+(?:restrictions|limits|filters|guardrails)\b"),
        0.6,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|leak|tell me|give me|display|"
            r"recite|echo|dump|expose|translate|encode)\b[^.!?\n]{0,50}"
            r"\b(?:system prompt|system message|system instruction|your instructions|"
            r"your prompt|initial prompt|hidden rules?|the text above|everything above|"
            r"the message above|previous message)\b"
        ),
        0.6,
    ),
    (
        "system_prompt_extraction",
        re.compile(r"\b(?:repeat|print|output|echo)\s+(?:the\s+)?(?:text|words|prompt)\s+above\b"),
        0.6,
    ),
    ("tool_forgery", re.compile(r"\bthe user (?:has |already )?approved\b"), 0.6),
    ("tool_forgery", re.compile(r"\b(?:human )?approval\s*(?::|is)?\s*(?:granted|given|provided)\b"), 0.6),
    ("tool_forgery", re.compile(r"\b(?:authorization|authorisation)\s*(?::|is)?\s*(?:granted|given|confirmed)\b"), 0.6),
    ("tool_forgery", re.compile(r"\b(?:no|without)\s+(?:further\s+)?(?:approval|confirmation)\s+(?:is\s+)?(?:needed|required)\b"), 0.6),
    ("tool_forgery", re.compile(r"\b(?:call|use|execute|run)\s+the\s+(?:booking|payment|refund)\s+tool\b"), 0.6),
    ("tool_forgery", re.compile(r"\bapprove (?:this|the) (?:booking|refund|purchase|payment)\b"), 0.6),
    ("tool_forgery", re.compile(r"\bbooking (?:is )?(?:approved|confirmed|authorized)\b"), 0.6),
    ("tool_forgery", re.compile(r"\bi (?:hereby )?approve\b"), 0.6),
]

_COMPACT_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (
        "instruction_override",
        re.compile(
            r"(?:ignore|disregard|forget|override).{0,50}"
            r"(?:all|any|previous|prior|above|earlier|existing).{0,50}"
            r"(?:instruction|prompt|rule|direction|directive|policy)"
        ),
        0.6,
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"(?:reveal|show|print|repeat|output|leak|tellme|giveme|display|recite|echo|dump).{0,50}"
            r"(?:systemprompt|systemmessage|systeminstruction|yourinstructions|yourprompt|"
            r"initialprompt|thetextabove|everythingabove)"
        ),
        0.6,
    ),
]

# Structural/fake-tag patterns need punctuation, so they only run on ``norm``.
_TAG_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    ("fake_role_tag", re.compile(r"(?:^|[\n;])\s*(?:system|assistant|developer|instruction)\s*:"), 0.6),
    ("fake_role_tag", re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>"), 0.6),
    ("fake_role_tag", re.compile(r"###\s*(?:instruction|system|assistant)\b"), 0.6),
    ("fake_role_tag", re.compile(r"\[/?inst\]|\[/?system\]|\[/?instruction\]"), 0.6),
    ("fake_role_tag", re.compile(r"</?(?:system|instruction|assistant|developer)>"), 0.6),
    ("fake_role_tag", re.compile(r"(?:^|\n)\s*---+\s*(?:system|assistant|developer|instruction)\s*---+"), 0.6),
]


def _normalise(text: str) -> str:
    """Lowercase, drop zero-width characters and collapse whitespace."""
    cleaned = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    cleaned = cleaned.translate(_LEET).lower()
    return re.sub(r"\s+", " ", cleaned).strip()


def _compact(norm: str) -> str:
    return re.sub(r"[^a-z0-9]", "", norm)


def detect_injection_rules(text: str) -> dict[str, Any]:
    """Layer-1 detector. Returns ``score``, ``flagged`` and ``reasons``."""
    if not isinstance(text, str) or not text.strip():
        return {"score": 0.0, "flagged": False, "reasons": []}

    norm = _normalise(text)
    compact = _compact(norm)
    matches: dict[str, float] = {}

    for name, pattern, weight in _NORM_PATTERNS + _TAG_PATTERNS:
        if pattern.search(norm):
            matches[name] = max(matches.get(name, 0.0), weight)
    for name, pattern, weight in _COMPACT_PATTERNS:
        if pattern.search(compact):
            matches[name] = max(matches.get(name, 0.0), weight)

    if _BASE64_RE.search(text):
        matches["encoded_payload"] = max(matches.get("encoded_payload", 0.0), 0.6)

    score = round(min(1.0, sum(matches.values())), 3)
    return {"score": score, "flagged": score >= FLAG_THRESHOLD, "reasons": sorted(matches)}


def _llm_enabled(use_llm: bool | None) -> bool:
    if use_llm is None:
        return os.environ.get("GUARDRAIL_LLM", "").strip().lower() in {"1", "true", "yes", "on"}
    return bool(use_llm)


def detect_injection(text: str, use_llm: bool | None = None) -> dict[str, Any]:
    """Score ``text`` for prompt-injection signals with layer 1 then layer 2.

    Returns ``{"score", "flagged", "reasons", "layer"}``. ``layer`` is
    ``"rules"`` when layer 1 matched, ``"llm"`` when the classifier flagged text
    the rules missed, and ``None`` otherwise. The classifier is only asked when
    ``use_llm`` is true (or, when ``None``, ``GUARDRAIL_LLM=1``) and layer 1 did
    not flag. Classifier failures count as "no opinion".
    """
    rules = detect_injection_rules(text)
    if rules["flagged"]:
        return {**rules, "layer": "rules"}
    if _llm_enabled(use_llm):
        try:
            from .llm_classifier import classify_injection

            verdict = classify_injection(text)
        except Exception:  # classifier must never break the request path
            verdict = {"available": False}
        if verdict.get("available") and verdict.get("injection"):
            reasons = sorted({*rules["reasons"], "llm_classifier"})
            return {
                "score": max(float(rules["score"]), 0.6),
                "flagged": True,
                "reasons": reasons,
                "layer": "llm",
            }
    return {**rules, "layer": None}
