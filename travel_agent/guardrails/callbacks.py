"""ADK callbacks that wire the guardrails into an ``LlmAgent``.

ADK calls callbacks with **keyword** arguments:

* ``before_model_callback(callback_context=..., llm_request=...)``
* ``after_model_callback(callback_context=..., llm_response=...)``
* ``before_tool_callback(tool=..., args=..., tool_context=...)``
* ``after_tool_callback(tool=..., args=..., tool_context=..., tool_response=...)``

The parameter names below therefore match ADK exactly. Wire them like this::

    from travel_agent.guardrails import callbacks as guard
    from travel_agent.observability.trace import chain_callbacks, trace_callbacks

    trace = trace_callbacks(tracer)
    agent = LlmAgent(
        ...,
        before_model_callback=chain_callbacks(guard.input_guard, trace["before_model"]),
        after_model_callback=chain_callbacks(guard.output_guard, trace["after_model"]),
        before_tool_callback=chain_callbacks(guard.tool_gate, trace["before_tool"]),
        after_tool_callback=chain_callbacks(guard.record_and_sanitize, trace["after_tool"]),
    )

Failure policy
--------------
* ``input_guard`` / ``output_guard`` / ``tool_gate`` fail **closed**: an
  unexpected exception produces a refusal (or a blocked tool result) rather than
  letting unguarded input through. ``chain_callbacks`` honours the
  ``__fail_closed__`` marker on these functions.
* ``record_and_sanitize`` fails **closed for untrusted content** only: if the
  sanitizer raises, untrusted evidence items are dropped, trusted ones are
  kept, and the event is logged. A non-``ToolResult`` response is logged and
  passed through unchanged.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from travel_agent.observability.trace import get_active_tracer
from travel_agent.verification.store import add_tool_result

from .injection import detect_injection
from .pii import redact_pii
from .sanitize import sanitize_tool_result
from .topics import check_topic

_GENERIC_REFUSAL = "I can't help with that request."
_INJECTION_REFUSAL = (
    "I can't act on that message: it looks like it is trying to change how I work "
    "rather than asking about travel."
)
_APPROVAL_MESSAGE = (
    "I can't confirm or complete a booking yet: this action still needs human approval first."
)
_EARLIER_MARKER = "[removed: earlier message blocked by guardrail]"

# Deny-by-default side effects: a tool name with a gated **word** is gated even
# if nobody remembered to add it to ``GATED_TOOLS``. Names are split on ``_``,
# ``-`` and camelCase, so ``paypal_lookup`` (token "paypal") and
# ``display_hold_music`` ("hold" is a middle modifier) are not gated, while
# ``pay_fare``, ``hold_fare``, ``rebook_flight`` and ``mock_booking`` are.
_GATED_KEYWORDS = frozenset({"book", "reserve", "purchase", "pay", "ticket", "cancel", "hold"})
_GATED_NOUNS = {
    "booking": "book",
    "bookings": "book",
    "reservation": "reserve",
    "reservations": "reserve",
    "cancellation": "cancel",
    "cancellations": "cancel",
    "payment": "pay",
    "payments": "pay",
}
_NAME_TOKEN_RE = re.compile(r"[_\-]+|(?<=[a-z0-9])(?=[A-Z])")
BOOKING_ACTION_PATTERN = re.compile(r"book|reserve|purchase|pay|ticket", re.I)
GATED_TOOLS: frozenset[str] = frozenset()

_CLAIM_RE = re.compile(
    r"\b(?:booked|rebooked|reserved|confirmed|issued|purchased|ticketed)\b"
    r"|\bchanged your flight\b"
    r"|\byou(?:'re| are) all set\b",
    re.I,
)
# Noun-subject completion claims: "booking complete", "your reservation is
# confirmed", "the rebooking was successful". Negation is not part of this
# pattern, so "is not complete" / "has not been confirmed" do not match.
_NOUN_CLAIM_RE = re.compile(
    r"\b(?:booking|reservation|rebooking|ticket|purchase|change)\b"
    r"\s+(?:(?:is|was|has\s+been|had\s+been|been)\s+)?"
    r"(?:complete|completed|done|successful|finali[sz]ed|confirmed)\b",
    re.I,
)
# "Your booking reference is ABC123" / "confirmation number: 9XZ2PL" presents
# the booking as done. The captured token must look like a reference code, not
# an ordinary word such as "pending" or "cancelled".
_REFERENCE_CLAIM_RE = re.compile(
    r"\b(?:booking\s+(?:reference|confirmation)(?:\s+(?:number|code))?"
    r"|confirmation\s+(?:number|code))\b"
    r"\s*(?:is|:)\s*([A-Za-z0-9-]{4,})\b",
    re.I,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|haven'?t|hasn'?t|hadn'?t|isn'?t|aren'?t|wasn'?t|"
    r"weren'?t|didn'?t|doesn'?t|don'?t|cannot|can'?t|couldn'?t|won'?t|wouldn'?t)\b"
    r"(?:\s+\w+){0,2}\s*$",
    re.I,
)
_ADJECTIVE_BEFORE_RE = re.compile(
    r"\b(?:flights?|seats?|hotels?|rooms?|restaurants?|venues?|tables?|classes?|tickets?|"
    r"everything|it)\s+(?:are|is|were|was|get|got)\s+"
    r"(?:fully\s+|completely\s+|already\s+|mostly\s+)?$",
    re.I,
)
_ADJECTIVE_AFTER_RE = re.compile(r"^\s*(?:solid|up|out|full)\b", re.I)
_ADJECTIVE_NEAR_BEFORE_RE = re.compile(r"\b(?:fully|completely|sold out|booked up)\s*$", re.I)
_WEAK_CLAIM_VERBS = {"confirmed", "issued", "purchased", "reserved"}
_TRIP_CONTEXT_RE = re.compile(
    r"\b(?:your|flight|flights|booking|bookings|reservation|reservations|ticket|tickets|"
    r"seat|seats|itinerary|trip|travel|rebooking)\b",
    re.I,
)


def _refusal(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _state(ctx: Any) -> Any:
    return getattr(ctx, "state", None)


def _approved_actions(ctx: Any) -> list[dict[str, Any]]:
    state = _state(ctx)
    if state is None:
        return []
    try:
        approvals = state.get("approved_actions")
    except Exception:
        return []
    return approvals if isinstance(approvals, list) else []


def _completed_actions(ctx: Any) -> list[dict[str, Any]]:
    state = _state(ctx)
    if state is None:
        return []
    try:
        actions = state.get("completed_actions")
    except Exception:
        return []
    return actions if isinstance(actions, list) else []


def _log_event(ctx: Any, kind: str, decision: str, reasons: list[str]) -> None:
    """Append a guardrail event to state and, when configured, the tracer."""
    try:
        state = _state(ctx)
        if state is not None:
            events = state.get("guardrail_events")
            if not isinstance(events, list):
                events = []
                state["guardrail_events"] = events
            events.append({"kind": kind, "decision": decision, "reasons": list(reasons)})
    except Exception:
        pass
    try:
        tracer = get_active_tracer()
        if tracer is not None:
            tracer.log_guardrail(kind, decision, reasons)
    except Exception:
        pass


def _text_parts(content: Any) -> list[Any]:
    return [
        part
        for part in (getattr(content, "parts", None) or [])
        if isinstance(getattr(part, "text", None), str) and part.text
    ]


def _user_contents(llm_request: Any) -> list[Any]:
    return [
        content
        for content in (getattr(llm_request, "contents", None) or [])
        if getattr(content, "role", None) == "user"
    ]


def _scan_part(ctx: Any, text: str, *, use_llm: bool | None) -> dict[str, Any]:
    """Run PII redaction, injection and topic checks on one text part."""
    redacted, findings = redact_pii(text)
    if findings:
        _log_event(ctx, "pii", "redact", sorted({item["type"] for item in findings}))
    injection = detect_injection(redacted, use_llm=use_llm)
    topic = check_topic(redacted)
    return {"text": redacted, "injection": injection, "topic": topic}


def input_guard(callback_context: Any, llm_request: Any) -> Any:
    """Scan **every** user turn on **every** call; refuse only the latest attack.

    ADK rebuilds the request from session history for each model call, so
    redacting one request does not persist. We therefore walk all user contents:
    PII is redacted in place, an attack in the latest user turn returns a
    refusal, and an attack in an earlier turn has that part replaced with a
    marker so the conversation can continue. Text split across parts is also
    checked as a concatenation. Fails closed on any unexpected error.
    """
    try:
        contents = _user_contents(llm_request)
        if not contents:
            return None
        # ADK adds role="user" function-response contents after tool calls;
        # the latest *human* turn is the last user content that carries text.
        textual = [content for content in contents if _text_parts(content)]
        latest = textual[-1] if textual else contents[-1]

        for content in contents:
            is_latest = content is latest
            parts = _text_parts(content)
            if not parts:
                continue

            results = [
                _scan_part(
                    callback_context,
                    part.text,
                    use_llm=None if is_latest else False,
                )
                for part in parts
            ]
            for part, result in zip(parts, results):
                part.text = result["text"]

            composite = " ".join(result["text"] for result in results)
            composite_injection = detect_injection(composite, use_llm=None if is_latest else False)
            composite_topic = check_topic(composite)

            flagged_parts = [
                part
                for part, result in zip(parts, results)
                if result["injection"]["flagged"] or not result["topic"]["allowed"]
            ]

            if is_latest:
                if flagged_parts or composite_injection["flagged"] or not composite_topic["allowed"]:
                    reasons: list[str] = []
                    for result in results:
                        reasons.extend(result["injection"]["reasons"])
                    if not composite_topic["allowed"]:
                        reasons.append(str(composite_topic.get("category")))
                    _log_event(callback_context, "injection", "refuse", sorted(set(reasons)))
                    if composite_injection["flagged"] or any(
                        result["injection"]["flagged"] for result in results
                    ):
                        return _refusal(_INJECTION_REFUSAL)
                    return _refusal(str(composite_topic.get("reason") or _GENERIC_REFUSAL))
                continue

            # An earlier turn: neutralise the offending part(s) and continue.
            if flagged_parts:
                for part in flagged_parts:
                    part.text = _EARLIER_MARKER
                _log_event(
                    callback_context,
                    "injection",
                    "redact_earlier",
                    ["earlier_message_blocked"],
                )
            elif composite_injection["flagged"] or not composite_topic["allowed"]:
                for part in parts:
                    part.text = _EARLIER_MARKER
                _log_event(
                    callback_context,
                    "injection",
                    "redact_earlier",
                    ["earlier_message_blocked"],
                )
        return None
    except Exception:
        try:
            _log_event(callback_context, "input_guard", "refuse", ["internal_error"])
        except Exception:
            pass
        return _refusal(_GENERIC_REFUSAL)


def find_booking_claims(text: str) -> list[str]:
    """Return completion phrases that really claim a booking was completed.

    Negated forms ("not booked", "haven't booked", "not confirmed yet", "not
    complete", "has not been confirmed") and adjective uses ("flights are fully
    booked", "booked solid") are ignored. Noun-subject statements ("booking
    complete", "reservation is confirmed") and reference-number statements
    ("booking reference is ABC123") count as claims too.
    """
    claims: list[str] = []
    for match in _CLAIM_RE.finditer(text):
        start = match.start()
        before = text[max(0, start - 60) : start]
        after = text[match.end() : match.end() + 16]
        if _NEGATION_RE.search(before):
            continue
        verb = match.group(0).lower()
        context = text[max(0, start - 60) : match.end() + 30]
        if verb in _WEAK_CLAIM_VERBS and not _TRIP_CONTEXT_RE.search(context):
            continue
        if verb == "booked":
            # "flights are fully booked", "seats are booked solid" are
            # descriptions, not statements that a booking was completed.
            if _ADJECTIVE_NEAR_BEFORE_RE.search(before):
                continue
            if _ADJECTIVE_BEFORE_RE.search(before):
                continue
            if _ADJECTIVE_AFTER_RE.search(after):
                continue
        claims.append(match.group(0).lower())

    for match in _NOUN_CLAIM_RE.finditer(text):
        start = match.start()
        if _NEGATION_RE.search(text[max(0, start - 60) : start]):
            continue
        claims.append(match.group(0).lower())

    for match in _REFERENCE_CLAIM_RE.finditer(text):
        start = match.start()
        if _NEGATION_RE.search(text[max(0, start - 60) : start]):
            continue
        token = match.group(1)
        if not (any(char.isdigit() for char in token) or token.isupper()):
            continue
        claims.append(match.group(0).lower())
    return claims


def _redact_object(value: Any) -> tuple[Any, list[str]]:
    """Recursively redact PII in strings inside model function-call args."""
    if isinstance(value, str):
        redacted, findings = redact_pii(value)
        return redacted, [item["type"] for item in findings]
    if isinstance(value, dict):
        findings: list[str] = []
        out: dict[Any, Any] = {}
        for key, item in value.items():
            out[key], found = _redact_object(item)
            findings.extend(found)
        return out, findings
    if isinstance(value, (list, tuple)):
        findings = []
        out = []
        for item in value:
            redacted, found = _redact_object(item)
            out.append(redacted)
            findings.extend(found)
        return out, findings
    return value, []


def _has_completed_booking(ctx: Any) -> bool:
    for record in _completed_actions(ctx):
        if not isinstance(record, dict):
            continue
        if record.get("ok") is False:
            continue
        tool = str(record.get("tool") or "")
        if is_gated_tool(tool) and BOOKING_ACTION_PATTERN.search(tool):
            return True
    return False


def output_guard(callback_context: Any, llm_response: Any) -> Any:
    """Redact PII and block booking claims without a completed booking tool.

    A claim is allowed only when ``ctx.state["completed_actions"]`` contains a
    successful gated (booking-type) tool result, written by
    ``record_and_sanitize``. A non-empty ``approved_actions`` list is no longer
    enough: approval is permission to try, not proof the action happened. PII in
    ``function_call.args`` is redacted as well. Fails closed.
    """
    try:
        content = getattr(llm_response, "content", None)
        if content is None:
            return None
        parts = list(getattr(content, "parts", None) or [])
        if not parts:
            return None

        changed = False
        collected: list[str] = []
        for part in parts:
            text = part.text if isinstance(part.text, str) else ""
            if text:
                redacted, findings = redact_pii(text)
                if findings:
                    part.text = redacted
                    changed = True
                    _log_event(
                        callback_context,
                        "pii",
                        "redact",
                        sorted({item["type"] for item in findings}),
                    )
                collected.append(redacted)
            function_call = getattr(part, "function_call", None)
            if function_call is not None and getattr(function_call, "args", None):
                cleaned, findings = _redact_object(function_call.args)
                if findings:
                    function_call.args = cleaned
                    changed = True
                    _log_event(callback_context, "pii", "redact", sorted(set(findings)))

        claims = find_booking_claims(" ".join(collected))
        if claims and not _has_completed_booking(callback_context):
            text_parts = [part for part in parts if isinstance(part.text, str)]
            if text_parts:
                text_parts[0].text = _APPROVAL_MESSAGE
                for part in text_parts[1:]:
                    part.text = ""
            else:
                parts[0].text = _APPROVAL_MESSAGE
            _log_event(callback_context, "booking_claim", "block", sorted(set(claims)))
            changed = True

        return llm_response if changed else None
    except Exception:
        try:
            _log_event(callback_context, "output_guard", "refuse", ["internal_error"])
        except Exception:
            pass
        return _refusal(_GENERIC_REFUSAL)


def _name_tokens(name: str) -> list[str]:
    return [token.lower() for token in _NAME_TOKEN_RE.split(name) if token]


def _gated_keyword(token: str) -> str | None:
    """Return the gated keyword a name token represents, if any.

    Handles exact keywords, ``re``-prefixed verbs (``rebook`` -> ``book``),
    gerunds/past forms (``booking``, ``purchased``), and a few side-effect
    nouns (``reservation`` -> ``reserve``).
    """
    if token in _GATED_KEYWORDS:
        return token
    if token in _GATED_NOUNS:
        return _GATED_NOUNS[token]
    if token.startswith("re") and token[2:] in _GATED_KEYWORDS:
        return token[2:]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if stem in _GATED_KEYWORDS:
                return stem
            if stem + "e" in _GATED_KEYWORDS:
                return stem + "e"
    return None


def is_gated_tool(name: str) -> bool:
    """Deny-by-default gate: side-effect words or an explicit ``GATED_TOOLS`` entry.

    Matching is on whole name tokens, not raw substrings. The leading token is
    the action verb; a trailing token may name the action's object (``booking``,
    ``ticket``). Middle tokens are modifiers, so ``display_hold_music`` is not
    an action while ``hold_fare`` is.
    """
    if not name:
        return False
    if name in GATED_TOOLS:
        return True
    tokens = _name_tokens(name)
    if not tokens:
        return False
    if any(left == "change" and right == "flight" for left, right in zip(tokens, tokens[1:])):
        return True
    last = len(tokens) - 1
    for index, token in enumerate(tokens):
        if _gated_keyword(token) and (index == 0 or index == last):
            return True
    return False


def _take_approval(ctx: Any, name: str, args: dict[str, Any]) -> bool:
    """Consume (and remove) one single-use approval matching tool + args."""
    approvals = _approved_actions(ctx)
    for index, record in enumerate(approvals):
        if (
            isinstance(record, dict)
            and record.get("tool") == name
            and (record.get("args") or {}) == args
        ):
            remaining = approvals[:index] + approvals[index + 1 :]
            try:
                state = _state(ctx)
                if state is not None:
                    state["approved_actions"] = remaining
                else:
                    approvals.pop(index)
            except Exception:
                try:
                    approvals.pop(index)
                except Exception:
                    return False
            return True
    return False


def tool_gate(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """Block side-effecting tools unless a matching approval is recorded.

    Gated if the tool name has a gated word token (``book|reserve|purchase|
    pay|ticket|cancel|hold``, including ``re``-prefixed verbs such as
    ``rebook``), a ``change_flight`` action, or is listed in ``GATED_TOOLS``.
    An approval is a dict in ``ctx.state["approved_actions"]`` with the tool
    name in ``"tool"`` and the exact call arguments in ``"args"``. Matching
    arguments (not just the tool name) prevents an approval for one itinerary
    from authorising another, and each approval is **single-use**.
    """
    blocked = {
        "ok": False,
        "error": "blocked: human approval required",
        "evidence": [],
        "source": "guardrail",
    }
    try:
        name = str(getattr(tool, "name", "") or "")
        if not is_gated_tool(name):
            return None
        call_args = args if isinstance(args, dict) else {}
        if _take_approval(tool_context, name, call_args):
            _log_event(tool_context, "tool_gate", "allow", [name])
            return None
        _log_event(tool_context, "tool_gate", "block", [name])
        return blocked
    except Exception:
        try:
            _log_event(tool_context, "tool_gate", "block", ["internal_error"])
        except Exception:
            pass
        return blocked


def _is_untrusted(item: Any) -> bool:
    return isinstance(item, dict) and isinstance(item.get("data"), dict) and item["data"].get("untrusted") is True


def _drop_untrusted(tool_response: dict[str, Any]) -> dict[str, Any]:
    """Keep only trusted evidence when the sanitizer itself failed."""
    filtered = copy.deepcopy(tool_response)
    filtered["evidence"] = [
        item for item in (tool_response.get("evidence") or []) if not _is_untrusted(item)
    ]
    return filtered


def record_and_sanitize(
    tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any
) -> dict[str, Any] | None:
    """Sanitise a ``ToolResult`` (A4), store evidence (B1), record completions.

    Returns the sanitised dict only when sanitising changed something, so ADK
    keeps the original object on the common no-op path. If sanitising raises,
    untrusted evidence is dropped and trusted evidence is kept (fail closed for
    hostile text). A response that is not ``ToolResult``-shaped is logged and
    passed through.
    """
    try:
        if not isinstance(tool_response, dict) or not {
            "source",
            "ok",
            "evidence",
        }.issubset(tool_response.keys()):
            _log_event(
                tool_context,
                "record_and_sanitize",
                "unrecognised",
                ["unrecognised tool response shape"],
            )
            return None

        try:
            sanitized, events = sanitize_tool_result(tool_response)
        except Exception:
            sanitized = _drop_untrusted(tool_response)
            events = [
                {
                    "evidence_id": None,
                    "kind": "sanitize_failed",
                    "detail": ["dropped_untrusted_evidence"],
                }
            ]

        state = _state(tool_context)
        if state is not None:
            add_tool_result(state, sanitized)
            name = str(getattr(tool, "name", "") or "")
            if is_gated_tool(name) and bool(sanitized.get("ok", False)):
                completed = state.get("completed_actions")
                if not isinstance(completed, list):
                    completed = []
                    state["completed_actions"] = completed
                completed.append({"tool": name, "args": args if isinstance(args, dict) else {}, "ok": True})

        for event in events:
            _log_event(
                tool_context,
                event.get("kind", "sanitize"),
                "sanitized",
                list(event.get("detail") or []),
            )
        return sanitized if sanitized != tool_response else None
    except Exception:
        try:
            _log_event(tool_context, "record_and_sanitize", "error", ["internal_error"])
        except Exception:
            pass
        return None


def _iter_callbacks(callback: Any) -> Iterable[Any]:
    if callback is None:
        return
    if isinstance(callback, (list, tuple)):
        for entry in callback:
            yield from _iter_callbacks(entry)
        return
    chained = getattr(callback, "__chain_callbacks__", None)
    if chained:
        for entry in chained:
            yield from _iter_callbacks(entry)
        return
    yield callback


def _callback_name(callback: Any) -> str:
    return str(getattr(callback, "__guardrail_name__", None) or getattr(callback, "__name__", ""))


_REQUIRED = (
    ("before_model_callback", "input_guard"),
    ("after_model_callback", "output_guard"),
    ("before_tool_callback", "tool_gate"),
    ("after_tool_callback", "record_and_sanitize"),
)


def assert_guarded(agent: Any) -> None:
    """Raise with a clear message if any guardrail callback is missing.

    Works with a single callable, a list of callables, and
    ``chain_callbacks(...)`` compositions. Teammates should call this in their
    agent tests so a refactor cannot silently drop a guardrail.
    """
    missing: list[str] = []
    for attr, required in _REQUIRED:
        names = {_callback_name(entry) for entry in _iter_callbacks(getattr(agent, attr, None))}
        if required not in names:
            missing.append(f"{attr} (needs {required})")
    if missing:
        raise AssertionError(
            "agent is not fully guarded: missing " + ", ".join(missing) + "; call assert_guarded()"
        )


# Marks used by ``chain_callbacks`` to fail closed when a guard raises.
input_guard.__guardrail_name__ = "input_guard"
input_guard.__fail_closed__ = True
output_guard.__guardrail_name__ = "output_guard"
output_guard.__fail_closed__ = True
tool_gate.__guardrail_name__ = "tool_gate"
tool_gate.__fail_closed__ = True
record_and_sanitize.__guardrail_name__ = "record_and_sanitize"
