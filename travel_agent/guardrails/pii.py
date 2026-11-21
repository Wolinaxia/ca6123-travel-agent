"""PII redaction.

Design notes
------------
The redactor is deliberately *pattern based* rather than a generic
"any long number" filter. Travel text is full of numbers that must survive:
flight numbers (``UA 857``), dates, times, prices and distances. So each
detector is anchored to the shape of a real identifier:

* emails       -- the only thing with an ``@``;
* payment cards -- 13-19 digits (optionally grouped) that also pass Luhn;
* Singapore NRIC/FIN -- one of ``S T F G M``, seven digits, one letter;
* passports    -- only when the word "passport" precedes the value, because a
  bare ``E12345678`` is indistinguishable from a booking reference;
* phones       -- explicit international ``+`` form, the US 3-3-4 form, or the
  Singapore 8-digit form (starting 6/8/9).

Every replacement is a typed placeholder so downstream prompts and traces never
carry the original value. The returned ``findings`` list is what the guardrail
trace records; it contains the original value on purpose (the trace is local and
git-ignored) so a human can audit what was removed.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Match

PLACEHOLDERS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "card": "[CARD]",
    "nric": "[NRIC]",
    "passport": "[PASSPORT]",
}

_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_NRIC_RE = re.compile(r"\b[STFGM]\d{7}[A-Z]\b")
_CARD_CANDIDATE_RE = re.compile(r"(?<![\w+])(?:\d[ -]?){12,18}\d(?![\w])")
_PASSPORT_RE = re.compile(
    r"(?i)(\bpassport(?:\s+(?:number|no\.?|num(?:ber)?))?\s*[:#-]?\s*)([A-Z0-9]{6,10})\b"
)
_INTL_PHONE_RE = re.compile(r"(?<![\w])\+\d{1,3}[\s.\-]?\d(?:[\s.\-]?\d){6,12}(?!\d)")
_US_PHONE_RE = re.compile(r"(?<!\d)(?:\(\d{3}\)\s*|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\d)")
_SG_PHONE_RE = re.compile(r"(?<!\d)[689]\d{7}(?!\d)")


def _luhn_valid(value: str) -> bool:
    """Return True when the digits of ``value`` satisfy the Luhn checksum."""
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _sub(
    text: str,
    pattern: re.Pattern[str],
    kind: str,
    findings: list[dict[str, Any]],
    validator: Callable[[str], bool] | None = None,
    value_group: int = 0,
) -> str:
    """Replace every match of ``pattern`` with the typed placeholder.

    When ``value_group`` is non-zero only that group is replaced (used by the
    passport rule, which must keep the word "passport" in the sentence).
    """
    placeholder = PLACEHOLDERS[kind]

    def _replace(match: Match[str]) -> str:
        whole = match.group(0)
        value = match.group(value_group)
        if validator is not None and not validator(value):
            return whole
        findings.append({"type": kind, "value": value, "placeholder": placeholder})
        if value_group == 0:
            return placeholder
        start, end = match.span(value_group)
        offset_start = start - match.start()
        offset_end = end - match.start()
        return whole[:offset_start] + placeholder + whole[offset_end:]

    return pattern.sub(_replace, text)


def redact_pii(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Replace personal identifiers in ``text`` with typed placeholders.

    Returns ``(redacted_text, findings)`` where each finding is
    ``{"type", "value", "placeholder"}``. The function is pure: the input string
    is never modified.

    Supported: emails, phones (international ``+``, US 3-3-4, Singapore 8-digit
    starting 6/8/9), Luhn-valid payment cards (13-19 digits, spaces/dashes
    allowed), Singapore NRIC/FIN, and passport numbers only when the word
    "passport" appears immediately before the value. Flight numbers, dates,
    times, airport codes, prices and distances are intentionally left alone.
    """
    if not isinstance(text, str) or not text:
        return text, []
    findings: list[dict[str, Any]] = []
    output = text

    output = _sub(output, _PASSPORT_RE, "passport", findings, value_group=2)
    output = _sub(output, _EMAIL_RE, "email", findings)
    output = _sub(output, _NRIC_RE, "nric", findings)
    output = _sub(output, _CARD_CANDIDATE_RE, "card", findings, validator=_luhn_valid)
    output = _sub(output, _INTL_PHONE_RE, "phone", findings)
    output = _sub(output, _US_PHONE_RE, "phone", findings)
    output = _sub(output, _SG_PHONE_RE, "phone", findings)
    return output, findings
