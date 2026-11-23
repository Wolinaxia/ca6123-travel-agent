"""Deterministic claim verification.

The Verifier answers one question: *does the evidence the agent cited actually
support what it said?* It deliberately uses **no LLM**. Ordered checks cover the
failure modes that matter for a travel agent:

1. the claim cites evidence at all;
2. every cited id was really returned by a tool (catches fabrication);
3. the tool call that returned the evidence did not fail;
4. at least one cited source is trusted and has not expired;
5. numeric claims match a number of the **same unit family** in trusted evidence;
6. airport codes mentioned in the claim appear in the cited trusted evidence;
7. claims of normal airport operations do not contradict cited FAA programs;
8. a risk judgement does not downplay the rule engine (cited findings *and* any
   summary for the same route, or for a route containing an airport the claim
   or its route-less citation names, in the store).

A future semantic judge belongs *after* these checks; see ``SEMANTIC_JUDGE_HOOK``
below. Checks themselves stay pure functions so they are trivial to test.

Numbers
-------
Claims and evidence are compared by unit family: ``minutes`` (``min``, ``h``,
``"1 h 23 min"`` -> 83, ``90-minute``, number words one-twenty plus tens),
``percent``, ``knots``, ``mph``, ``km`` and ``temperature``. Evidence numbers
come from ``observation`` and from ``data`` values whose key implies a unit
(``*_min``, ``*minutes*``, ``*_pct``, ``*rate*`` as a fraction -> percent,
``*_kt``/``gust*``, ``*_km``, ``*_mph``, ``temp*``). Ids, urls, timestamps,
``rule``, coordinates and ``source_status`` are never treated as support.

A claimed number with no same-family candidate in the trusted evidence is
``unsupported`` (the agent may be using an unknown source); a candidate exists
but none is within tolerance ``max(0.5, 5%)`` is ``contradicted``.

Places
------
Airport codes are matched on word boundaries. A three-letter token only counts
as an airport if it exists in the airport reference data and is not a common
abbreviation (USA, PDT, EST, ETA, ...). Lowercase tokens are accepted only when
they map to an airport and are not common English words; otherwise ignored.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import datetime, timezone
from typing import Any

from travel_agent.core.reference.airports import get as airport_by_iata

# Extension point for a future LLM judge. Deliberately not implemented here:
# deterministic checks decide support first.
SEMANTIC_JUDGE_HOOK = None

_CLAIM_RISK_RANK = {"low": 0, "moderate": 1, "high": 2}
# Claim "high" is deliberately equal to rule "high" *or* "severe": a claim may
# be at most as cautious as "high", so it cannot downplay a severe rule result.
_RULE_RISK_RANK = {"low": 0, "elevated": 1, "high": 2, "severe": 2}

_STOPLIST = {
    "USA", "PDT", "PST", "EDT", "EST", "CDT", "CST", "MDT", "MST", "ETA", "TSA",
    "SAT", "SUN", "HOT", "AND", "THE", "FOR", "FROM", "WITH", "YOU", "NOT", "ARE",
    "WAS", "HAS", "HAD", "CAN", "MAY", "NOW", "OUT", "TOP", "NEW", "OLD", "ALL",
    "ANY", "OUR", "OWN", "OFF", "ONTO", "PER", "VIA", "WAS", "WHO", "WHY", "HOW",
    "ITS", "IT'S", "DID", "GET", "GOT", "LET", "SET", "USE", "FAR", "LOW", "AIR",
}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD = r"(?:" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
_WORD_COMPOUND = rf"{_WORD}(?:[-\s]{_WORD})?"

# Ordered by priority: longer combined forms win over their pieces.
_NUMBER_SPECS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "combined_minutes",
        re.compile(
            rf"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\s*(?:and\s*)?"
            rf"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b",
            re.I,
        ),
        "combined",
    ),
    (
        "minutes",
        re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-\s]?\s*(?:minutes?|mins?|min)\b", re.I),
        "minutes",
    ),
    (
        "minutes_word",
        re.compile(rf"(?<![\w-])({_WORD_COMPOUND})\s*[-\s]?\s*(?:minutes?|mins?|min)\b", re.I),
        "minutes_word",
    ),
    (
        "hours",
        re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-\s]?\s*(?:hours?|hrs?|h)\b", re.I),
        "hours",
    ),
    (
        "hours_word",
        re.compile(rf"(?<![\w-])({_WORD_COMPOUND})\s*[-\s]?\s*(?:hours?|hrs?|h)\b", re.I),
        "hours_word",
    ),
    (
        "percent",
        re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|(?:per\s*cent|percent)\b)", re.I),
        "percent",
    ),
    (
        "knots",
        re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:kt|kts|knots?)\b", re.I),
        "knots",
    ),
    ("mph", re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*mph\b", re.I), "mph"),
    ("km", re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*km\b", re.I), "km"),
    (
        "temperature",
        re.compile(
            r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:°|degrees?\s*)?([CF])\b",
            re.I,
        ),
        "temperature",
    ),
    (
        "temperature_word",
        re.compile(
            rf"(?<![\w.])(-?\d+(?:\.\d+)?)\s*degrees?\s*(fahrenheit|celsius)",
            re.I,
        ),
        "temperature_word",
    ),
]

def _word_value(text: str) -> float:
    parts = re.split(r"[-\s]+", text.lower())
    return float(sum(_NUMBER_WORDS.get(part, 0) for part in parts))


def _extract_numbers(text: str) -> list[tuple[float, str]]:
    """Return ``(value, family)`` pairs, de-overlapping longer forms first."""
    if not isinstance(text, str) or not text:
        return []
    found: list[tuple[int, int, float, str]] = []
    for priority, (_name, pattern, kind) in enumerate(_NUMBER_SPECS):
        for match in pattern.finditer(text):
            start, end = match.span()
            if kind == "combined":
                value = float(match.group(1)) * 60 + float(match.group(2))
                family = "minutes"
            elif kind in {"minutes_word", "hours_word"}:
                value = _word_value(match.group(1))
                family = "minutes" if kind == "minutes_word" else "minutes"
                if kind == "hours_word":
                    value *= 60
            elif kind == "hours":
                value = float(match.group(1)) * 60
                family = "minutes"
            elif kind == "temperature":
                value = float(match.group(1))
                unit = match.group(2).upper()
                family = "temperature"
                if unit == "F":
                    value = (value - 32.0) * 5.0 / 9.0
            elif kind == "temperature_word":
                value = float(match.group(1))
                family = "temperature"
                if match.group(2).lower() == "fahrenheit":
                    value = (value - 32.0) * 5.0 / 9.0
            elif kind == "minutes":
                value = float(match.group(1))
                family = "minutes"
            else:
                value = float(match.group(1))
                family = kind
            found.append((start, end, value, family))

    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    accepted: list[tuple[int, int, float, str]] = []
    for candidate in found:
        start, end = candidate[0], candidate[1]
        if any(start < other[1] and end > other[0] for other in accepted):
            continue
        accepted.append(candidate)
    return [(value, family) for _start, _end, value, family in accepted]


def _as_dict(claim: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(claim):
        claim = dataclasses.asdict(claim)
    return dict(claim) if isinstance(claim, dict) else {}


def _normalise_store(store: Any) -> dict[str, Any]:
    if not isinstance(store, dict):
        return {}
    inner = store.get("evidence_store")
    return inner if isinstance(inner, dict) else store


def _evidence(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        nested = record.get("evidence")
        if isinstance(nested, dict):
            return nested
        if "id" in record:
            return record
    return {}


def _verdict(claim: dict[str, Any], status: str, reason: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"claim": claim, "status": status, "reason": reason, "checks": checks}


def _matches(left: float, right: float) -> bool:
    return abs(left - right) <= max(0.5, 0.05 * max(abs(left), abs(right)))


def _is_untrusted(evidence: dict[str, Any]) -> bool:
    return (evidence.get("data") or {}).get("untrusted") is True


def _verification_time() -> datetime:
    from travel_agent.tools.replay import demo_now

    return demo_now() or datetime.now(timezone.utc)


def _is_expired(evidence: dict[str, Any], as_of: datetime) -> bool:
    valid_to = evidence.get("valid_to")
    if valid_to is None:
        return False
    try:
        end = datetime.fromisoformat(str(valid_to).replace("Z", "+00:00"))
        return end.utcoffset() is None or end <= as_of
    except ValueError:
        return True


_NORMAL_AIRPORT = re.compile(
    r"\b(?P<traffic>arrivals?|departures?|flights?|operations?)\s+(?:are\s+)?(?:running|operating)\s+normally\b"
    r"|\bno active (?:faa )?programs?\b", re.I,
)


def _airport_status_conflict(text: str, evidence: list[dict[str, Any]]) -> bool:
    match = _NORMAL_AIRPORT.search(text)
    airports = set(_claim_places(text))
    if not match or len(airports) > 1:
        return False
    traffic = (match["traffic"] or "").lower()
    for item in evidence:
        subject = str(item.get("subject") or "")
        if len(subject) != 3 or airport_by_iata(subject) is None or (airports and subject not in airports):
            continue
        data = item.get("data") or {}
        program = data.get("program")
        if data.get("active") is False or data.get("affects_commercial") is False:
            continue
        if program in ("GDP", "ground_stop", "closure"):
            return True
        if program == "arrival_departure":
            kind = str(data.get("kind") or "").lower()
            if ((traffic.startswith("arrival") and kind.startswith("departure"))
                or (traffic.startswith("departure") and kind.startswith("arrival"))):
                continue
            return True
    return False


def _to_celsius(value: float, key: str) -> float:
    lowered = key.lower()
    if "f" in lowered.rsplit("temp", 1)[-1] or "fahrenheit" in lowered:
        return (value - 32.0) * 5.0 / 9.0
    return value


def _data_numbers(data: Any) -> list[tuple[float, str]]:
    """Numbers from ``data`` whose key implies a unit; everything else ignored."""
    numbers: list[tuple[float, str]] = []
    if not isinstance(data, dict):
        return numbers
    for key, value in data.items():
        lowered = str(key).lower()
        if lowered in {"lat", "lon", "latitude", "longitude", "rule", "source_status", "id", "url"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, dict):
            numbers.extend(_data_numbers(value))
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                numbers.extend(_data_numbers(item))
            continue

        scalar: float | None = None
        if isinstance(value, (int, float)):
            scalar = float(value)
        elif isinstance(value, str):
            try:
                scalar = float(value.strip())
            except ValueError:
                scalar = None

        parsed_from_string = (
            _extract_numbers(value) if scalar is None and isinstance(value, str) else []
        )
        if lowered.endswith("_min") or "minutes" in lowered or lowered.endswith("_mins"):
            family = "minutes"
        elif lowered.endswith("_pct") or "percent" in lowered:
            family = "percent"
        elif "_rate" in lowered or lowered.endswith("rate"):
            family = "percent"
            if scalar is not None and abs(scalar) <= 1.0:
                scalar *= 100.0
        elif lowered.endswith("_kt") or lowered.startswith("gust") or "knots" in lowered:
            family = "knots"
        elif lowered.endswith("_km"):
            family = "km"
        elif lowered.endswith("_mph"):
            family = "mph"
        elif lowered.startswith("temp") or "temperature" in lowered:
            family = "temperature"
            if scalar is not None:
                scalar = _to_celsius(scalar, lowered)
        else:
            family = ""

        if family and scalar is not None:
            numbers.append((scalar, family))
        elif family and parsed_from_string:
            numbers.extend(
                (value_, unit)
                for value_, unit in parsed_from_string
                if not family or unit == family
            )
    return numbers


def _evidence_numbers(evidence: dict[str, Any]) -> list[tuple[float, str]]:
    numbers = _extract_numbers(str(evidence.get("observation") or ""))
    numbers.extend(_data_numbers(evidence.get("data") or {}))
    return numbers


def _haystack(evidence: dict[str, Any]) -> str:
    import json

    return (
        f"{evidence.get('subject', '')} {evidence.get('observation', '')} "
        f"{json.dumps(evidence.get('data') or {}, default=str)}"
    ).upper()


def _claim_places(text: str) -> list[str]:
    places: list[str] = []
    for token in re.findall(r"\b[A-Za-z]{3}\b", text):
        upper = token.upper()
        if upper in _STOPLIST:
            continue
        # Only all-caps tokens count as airport codes. Lowercase matching was
        # dropped: ~66 ordinary English words (day, art, bed, row, pit, ...)
        # are also IATA codes, so correct claims got flagged.
        if token.isupper() and airport_by_iata(upper) is not None:
            places.append(upper)
    return list(dict.fromkeys(places))


def _route_tokens(evidence_ids: list[str]) -> set[str]:
    routes: set[str] = set()
    for evidence_id in evidence_ids:
        parts = str(evidence_id).split(":")
        if len(parts) > 1 and "-" in parts[1]:
            routes.add(parts[1].upper())
    return routes


def _evidence_routes(evidence_id: str, evidence: dict[str, Any]) -> set[str]:
    """Routes named by an evidence id or subject (``JFK-SFO``)."""
    routes = set(_route_tokens([evidence_id]))
    subject = str(evidence.get("subject") or "").upper()
    routes.update(re.findall(r"\b[A-Z]{3}(?:-[A-Z]{3})+\b", subject))
    return routes


def _route_contains_airport(route: str, airports: set[str]) -> bool:
    return bool(airports.intersection(route.upper().split("-")))


def _rule_levels(
    normalised: dict[str, Any],
    cited: list[dict[str, Any]],
    evidence_ids: list[str],
    claim_text: str = "",
) -> list[str]:
    """Levels from cited engine findings plus same-route or same-airport summaries.

    A citation that names a route (``engine:JFK-SFO:...`` or subject ``JFK-SFO``)
    lets us match summaries by route. When the claim cites only route-less
    evidence (for example an FAA item whose subject is ``SFO``), summaries for
    routes that contain an airport from the claim text or from a cited trusted
    subject (``JFK-SFO``, ``SFO-*``) are still compared, so a low-risk claim
    cannot hide a rule-engine warning.
    """
    levels: list[str] = []
    subjects = {str(item.get("subject") or "") for item in cited}
    subjects.discard("")
    routes = _route_tokens(evidence_ids)

    for item in cited:
        level = (item.get("data") or {}).get("level")
        if level in _RULE_RISK_RANK:
            levels.append(str(level))

    has_route_citation = False
    for evidence_id in evidence_ids:
        evidence = _evidence(normalised.get(evidence_id)) or {}
        source = str(evidence.get("source") or normalised[evidence_id].get("source") or "")
        if source.lower() == "engine" and _evidence_routes(evidence_id, evidence):
            has_route_citation = True
            break

    route_less_airports: set[str] = set()
    if not has_route_citation:
        route_less_airports.update(_claim_places(claim_text))
        for item in cited:
            if _is_untrusted(item):
                continue
            route_less_airports.update(_claim_places(str(item.get("subject") or "")))

    for evidence_id, record in normalised.items():
        evidence = _evidence(record)
        data = evidence.get("data") or {}
        level = data.get("level")
        if level not in _RULE_RISK_RANK:
            continue
        is_summary = str(evidence_id).endswith(":summary") or "legs" in data
        if not is_summary:
            continue
        same_subject = str(evidence.get("subject") or "") in subjects
        same_route = any(route in str(evidence_id).upper() for route in routes)
        same_airport = any(
            _route_contains_airport(route, route_less_airports)
            for route in _evidence_routes(evidence_id, evidence)
        )
        if same_subject or same_route or same_airport:
            levels.append(str(level))
    return levels


def verify_claim(claim: Any, store: dict[str, Any], *, as_of: datetime | None = None) -> dict[str, Any]:
    """Verify one claim against the evidence store.

    ``claim`` may be a ``Claim`` dataclass or a dict with ``text``,
    ``evidence_ids``, ``confidence`` and optional ``risk``. ``store`` is the
    mapping produced by ``verification.store.add_tool_result``.

    Returns ``{"claim", "status", "reason", "checks"}``. Ordered checks stop at
    the first failure, and every executed check is recorded in ``checks``.
    """
    claim_dict = _as_dict(claim)
    evidence_ids = [str(item) for item in (claim_dict.get("evidence_ids") or [])]
    text = str(claim_dict.get("text") or "")
    checks: list[dict[str, Any]] = []
    normalised = _normalise_store(store)
    as_of = as_of or _verification_time()

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    if not evidence_ids:
        record("citation_present", False, "no evidence ids")
        return _verdict(claim_dict, "unsupported", "no evidence cited", checks)
    record("citation_present", True, f"{len(evidence_ids)} evidence id(s)")

    missing = [item for item in evidence_ids if item not in normalised]
    if missing:
        record("citations_exist", False, f"unknown ids: {', '.join(missing)}")
        return _verdict(claim_dict, "unsupported", "cites evidence that no tool returned", checks)
    record("citations_exist", True, "all cited ids exist")

    failed = [
        item for item in evidence_ids if normalised[item].get("tool_ok") is False
    ]
    if failed:
        record("citations_trusted", False, f"failed tool ids: {', '.join(failed)}")
        return _verdict(
            claim_dict,
            "unsupported",
            "cites evidence from a failed tool call",
            checks,
        )
    record("citations_trusted", True, "all cited tool calls succeeded")

    cited = [_evidence(normalised[item]) for item in evidence_ids]
    cited = [item for item in cited if item]
    if cited and all(_is_untrusted(item) for item in cited):
        record("not_untrusted_only", False, "every cited item is untrusted news")
        return _verdict(claim_dict, "unsupported", "supported only by untrusted news text", checks)
    record("not_untrusted_only", True, "at least one trusted source")

    active_store = {key: value for key, value in normalised.items() if not _is_expired(_evidence(value), as_of)}
    active_ids = [key for key in evidence_ids if key in active_store]
    active_citations = [_evidence(active_store[key]) for key in active_ids]
    trusted = [item for item in active_citations if item and not _is_untrusted(item)]
    if not trusted:
        record("evidence_current", False, "no current trusted evidence")
        return _verdict(claim_dict, "unsupported", "no current trusted evidence cited", checks)

    evidence_numbers: list[tuple[float, str]] = []
    for item in trusted:
        evidence_numbers.extend(_evidence_numbers(item))
    for claimed, family in _extract_numbers(text):
        candidates = [value for value, unit in evidence_numbers if unit == family]
        if not candidates:
            record("numbers_match", False, f"no {family} value in cited evidence")
            return _verdict(
                claim_dict,
                "unsupported",
                f"states a {family} value but cited evidence has none",
                checks,
            )
        if not any(_matches(claimed, candidate) for candidate in candidates):
            record("numbers_match", False, f"{claimed:g} {family} not in cited evidence")
            return _verdict(
                claim_dict,
                "contradicted",
                f"states {claimed:g} {family} but cited evidence does not",
                checks,
            )
    record("numbers_match", True, "numeric claims match cited evidence")

    combined = " ".join(_haystack(item) for item in trusted)
    for code in _claim_places(text):
        if not re.search(rf"\b{re.escape(code)}\b", combined):
            record("places_match", False, f"{code} absent from cited evidence")
            return _verdict(
                claim_dict,
                "unsupported",
                f"mentions {code} but no cited evidence is about it",
                checks,
            )
    record("places_match", True, "airport codes match cited evidence")

    if _airport_status_conflict(text, trusted):
        record("airport_status", False, "cited evidence shows an active disruption")
        return _verdict(claim_dict, "contradicted", "cited evidence shows an active airport disruption", checks)

    risk = claim_dict.get("risk")
    levels = _rule_levels(active_store, active_citations, active_ids, text)
    if risk in _CLAIM_RISK_RANK and levels:
        claim_rank = _CLAIM_RISK_RANK[str(risk)]
        rule_rank = max(_RULE_RISK_RANK[str(level)] for level in levels)
        if claim_rank < rule_rank:
            record("risk_consistent", False, f"claim {risk} < rule {sorted(set(levels))}")
            return _verdict(claim_dict, "contradicted", "downplays rule-engine risk", checks)
        if claim_rank > rule_rank:
            record("risk_consistent", True, "more cautious than rule engine")
            return _verdict(claim_dict, "supported", "more cautious than rule engine", checks)
        record("risk_consistent", True, "matches rule-engine risk")
        return _verdict(claim_dict, "supported", "risk matches rule engine", checks)

    record("risk_consistent", True, "no risk comparison needed")
    return _verdict(claim_dict, "supported", "supported by cited evidence", checks)


def verify_claims(claims: list[Any], store: dict[str, Any], *, as_of: datetime | None = None) -> dict[str, Any]:
    """Verify a list of claims and summarise what a human must review.

    Returns ``{"verdicts", "needs_human", "summary", "counts"}``. ``needs_human``
    is true when any verdict is ``unsupported`` or ``contradicted``; ``summary``
    lists each problem for the HITL step. If claims cite ids while the evidence
    store is empty, the summary calls out the likely wiring mistake.
    """
    as_of = as_of or _verification_time()
    verdicts = [verify_claim(claim, store, as_of=as_of) for claim in (claims or [])]
    counts = {"supported": 0, "unsupported": 0, "contradicted": 0}
    problems: list[str] = []
    for index, verdict in enumerate(verdicts, start=1):
        status = str(verdict.get("status"))
        counts[status] = counts.get(status, 0) + 1
        if status != "supported":
            problems.append(f"claim {index}: {status} ({verdict.get('reason')})")

    needs_human = bool(problems)
    empty_store = not _normalise_store(store)
    cites_anything = any((_as_dict(claim).get("evidence_ids") or []) for claim in (claims or []))
    if not verdicts:
        summary = "no claims to verify"
    elif not problems:
        summary = f"all {len(verdicts)} claims supported"
    else:
        summary = f"{len(verdicts) - len(problems)}/{len(verdicts)} supported; " + "; ".join(problems)
    if empty_store and cites_anything:
        summary = (
            "evidence store is empty — is record_and_sanitize registered as "
            "after_tool_callback? " + summary
        )
    return {"verdicts": verdicts, "needs_human": needs_human, "summary": summary, "counts": counts}
