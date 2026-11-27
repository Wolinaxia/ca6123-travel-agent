"""Human-in-the-loop approval for a proposed (mock) booking (M4).

Flow::

    result   = parse_agent_result(agent_text)            # Alternatives agent output
    proposal = build_proposal(state, result)             # runs the Verifier checks
    print(render_card_text(proposal))                    # human reads the card
    approve(state, proposal, acknowledge_warnings=...)   # or reject / request_change / explain
    execute_approved(state, proposal)                    # gated mock booking

Safety rules enforced here (the Responsible AI story):

* **Verifier first.** The card shows a verdict next to every claim, read from
  ``state["last_verification"]`` (the key the Verifier writes), never from
  model prose. If anything is unsupported/contradicted, or the option is not
  "comfortable", approving requires an explicit ``acknowledge_warnings=True``.
* **Exact, single-use approval.** ``approve`` writes one entry to
  ``state["approved_actions"]`` with the exact booking args. M1's
  ``tool_gate`` consumes it, so one approval = one booking of that option.
* **Approvals expire** (``APPROVAL_TTL_MIN``). An old "yes" does not stay valid.
* **Stale evidence blocks booking** (``MAX_EVIDENCE_AGE_MIN``). If the data
  behind the option is older than that, the human must re-check first.
* **Code books, not the model.** ``execute_approved`` calls the mock booking
  tool through the same ``tool_gate`` + ``record_and_sanitize`` callbacks the
  agents use; the LLM never holds the booking tool.
* **Learning needs consent.** ``reject`` feeds the reason to the preference
  memory, which may *suggest* a preference; it is stored only if confirmed.
* Every decision is appended to ``state["hitl_log"]`` and the trace log.
"""

from __future__ import annotations

import html
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from google.adk.tools import FunctionTool

from travel_agent.guardrails import callbacks as guard
from travel_agent.memory.text_safety import looks_like_instruction
from travel_agent.memory import cases as case_library
from travel_agent.memory.preferences import PreferenceStore
from travel_agent.observability.trace import get_active_tracer
from travel_agent.tools import replay
from travel_agent.tools.booking import book_trip_option
from travel_agent.verification.checks import verify_claims

APPROVAL_TTL_MIN = 15
MAX_EVIDENCE_AGE_MIN = 30
BOOKING_TOOL = "book_trip_option"
VERIFICATION_DISABLED = "verification disabled"
_BOOKING_FUNCTION_TOOL = FunctionTool(book_trip_option)

_STATUS_MARK = {"supported": "OK", "unsupported": "CHECK", "contradicted": "WRONG"}
_VERDICT_TEXT = {
    "comfortable": "Comfortable margin",
    "tight": "Tight margin",
    "at_risk": "At risk (FAA ground stop or closure)",
    "likely_miss": "Likely to miss the deadline",
}


class ProposalError(ValueError):
    """The agent result cannot be turned into a proposal (escalate to human)."""


# --------------------------------------------------------------------------
# Extra M4 claim checks (run after M1's Verifier, never instead of it)
# --------------------------------------------------------------------------

_BOOKED_RE = re.compile(
    r"\b(?:booked|reserved|ticketed|purchased|confirmation(?:\s+(?:number|code))?|you(?:'re| are) all set)\b",
    re.IGNORECASE,
)
# Minimum honest risk for an active FAA program at a cited airport.
_FAA_PROGRAM_FLOOR = {"GDP": "moderate", "arrival_departure": "moderate", "ground_stop": "high", "closure": "high"}
_RISK_RANK = {"low": 0, "moderate": 1, "high": 2}


def _m4_checks(claims: list[dict[str, Any]], verification: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
    """Two dangers M1's generic Verifier does not cover for this flow.

    1. A claim that something was **booked** is only true if a booking tool
       result is cited. Before approval no booking exists, so such a claim is
       false reassurance.
    2. A claim citing an active FAA delay program must not call the risk "low".
    """
    verdicts = [dict(item) for item in verification["verdicts"]]
    for index, (claim, verdict) in enumerate(zip(claims, verdicts)):
        if verdict.get("status") != "supported":
            continue
        text = str(claim.get("text") or "")
        ids = [str(item) for item in claim.get("evidence_ids") or []]
        if _BOOKED_RE.search(text) and not any(item.startswith("booking:") and item in store for item in ids):
            verdicts[index] = {**verdict, "status": "unsupported",
                               "reason": "says the trip is booked, but no booking exists; only an approved booking can confirm that"}
            continue
        risk = str(claim.get("risk") or "")
        # Only for claims about the FAA program itself. When a plan option is
        # cited, M1's Verifier already compares the risk with that option's
        # level (and "OAK avoids the SFO delay" may honestly be low risk).
        if risk in _RISK_RANK and not any(item.startswith("plan:") for item in ids):
            floors = []
            clear_airport_cited = False   # e.g. faa_nas:OAK:status = "no program at OAK"
            for evidence_id in ids:
                evidence = ((store.get(evidence_id) or {}).get("evidence") or {})
                if evidence.get("source") != "faa_nas":
                    continue
                floor = _FAA_PROGRAM_FLOOR.get(str((evidence.get("data") or {}).get("program")))
                if floor:
                    floors.append(floor)
                else:
                    clear_airport_cited = True
            # A comparison ("OAK has no program and avoids the SFO delay") may
            # honestly be low risk; only claims ABOUT the delayed airport get the floor.
            if floors and not clear_airport_cited and _RISK_RANK[risk] < max(_RISK_RANK[item] for item in floors):
                verdicts[index] = {**verdict, "status": "contradicted",
                                   "reason": f"calls the risk '{risk}' while citing an active FAA delay program"}
    problems = [f"claim {i}: {v['status']} ({v.get('reason')})" for i, v in enumerate(verdicts, start=1) if v.get("status") != "supported"]
    counts = {"supported": 0, "unsupported": 0, "contradicted": 0}
    for item in verdicts:
        counts[str(item.get("status"))] = counts.get(str(item.get("status")), 0) + 1
    if not verdicts:
        summary = "no claims to verify"
    elif not problems:
        summary = f"all {len(verdicts)} claims supported"
    else:
        summary = f"{len(verdicts) - len(problems)}/{len(verdicts)} supported; " + "; ".join(problems)
    return {**verification, "verdicts": verdicts, "needs_human": bool(problems) or verification.get("needs_human", False),
            "summary": summary, "counts": counts}


def _clock(now: datetime | None = None) -> datetime:
    return now or replay.demo_now() or datetime.now(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _log(state: Any, event: str, **fields: Any) -> None:
    entry = {"event": event, "at": _clock().isoformat(), **fields}
    try:
        log = state.get("hitl_log")
        if not isinstance(log, list):
            log = []
        log.append(entry)
        state["hitl_log"] = log
    except Exception:
        pass
    try:
        tracer = get_active_tracer()
        if tracer is not None:
            tracer.log("hitl", **{"decision": event, **{k: v for k, v in fields.items() if isinstance(v, (str, int, float, bool))}})
    except Exception:
        pass


def _store(state: Any) -> dict[str, Any]:
    try:
        value = state.get("evidence_store")
    except Exception:
        value = None
    return value if isinstance(value, dict) else {}


def _evidence(state: Any, evidence_id: str | None) -> dict[str, Any] | None:
    if not evidence_id:
        return None
    record = _store(state).get(evidence_id)
    return (record or {}).get("evidence") if record else None


@dataclass
class Proposal:
    """Everything the human needs to decide, in one object."""

    proposal_id: str
    option_id: str
    option: dict[str, Any]
    observation: str
    plan_b: dict[str, Any] | None
    contingency: dict[str, Any]
    claims: list[dict[str, Any]]
    verdicts: list[dict[str, Any]]
    verification_summary: str
    warnings: list[str]
    needs_human: bool
    summary: str
    lessons_used: list[str]
    preferences_used: list[str]
    data_gaps: list[str]
    evidence_fetched_at: str | None
    created_at: str
    booking_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Building the proposal
# --------------------------------------------------------------------------

def build_proposal(
    state: Any,
    result: dict[str, Any],
    now: datetime | None = None,
    *,
    verify: bool = True,
) -> Proposal:
    """Turn the Alternatives agent's JSON into a checked proposal.

    Runs the same deterministic checks as the Verifier agent's tool and writes
    them to ``state["last_verification"]`` so the card and the audit log use
    the authoritative verdicts. ``verify=False`` (the ablation's OFF arm) skips
    both checks, writes no ``last_verification`` and leaves ``verdicts`` empty.
    """
    if not isinstance(result, dict):
        raise ProposalError("the agent's answer is not a JSON object; escalate to a human")
    option_id = result.get("recommended_option_id")
    if not isinstance(option_id, str) or not option_id.startswith("plan:"):
        raise ProposalError(f"recommended option {option_id!r} is not a plan id; escalate to a human")
    option_evidence = _evidence(state, option_id)
    if not option_evidence:
        raise ProposalError(
            f"recommended option {option_id!r} was not returned by any tool in this session; escalate to a human"
        )
    option = dict(option_evidence.get("data") or {})

    raw_claims = result.get("claims")
    malformed = not isinstance(raw_claims, list)
    claims = []
    for item in raw_claims if isinstance(raw_claims, list) else []:
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
            claims.append({**item, "evidence_ids": [str(x) for x in item.get("evidence_ids") or [] if isinstance(x, str)]})
        else:
            malformed = True
    if verify:
        verification = _m4_checks(claims, verify_claims(claims, _store(state)), _store(state))
        state["last_verification"] = verification
    else:
        verification = {"verdicts": [], "needs_human": False, "summary": VERIFICATION_DISABLED,
                        "counts": {"supported": 0, "unsupported": 0, "contradicted": 0}}

    contingency_evidence = _evidence(state, "plan:contingency") or {}
    contingency = dict(contingency_evidence.get("data") or {})
    plan_b_id = result.get("plan_b_option_id") or contingency.get("plan_b")
    plan_b_evidence = _evidence(state, plan_b_id)
    plan_b = {"id": plan_b_id, "observation": plan_b_evidence.get("observation"), **(plan_b_evidence.get("data") or {})} if plan_b_evidence else None
    gaps = list(((_evidence(state, "plan:data_gaps") or {}).get("data") or {}).get("gaps") or [])

    warnings: list[str] = []
    for index, verdict in enumerate(verification["verdicts"], start=1):
        if verdict.get("status") != "supported":
            warnings.append(f"Claim {index} is {verdict.get('status')}: {verdict.get('reason')}")
    if option.get("verdict") != "comfortable":
        warnings.append(f"The recommended option is '{option.get('verdict')}', not comfortable.")
    if gaps:
        warnings.append("Some data sources were unavailable: " + "; ".join(gaps))
    if result.get("parse_error"):
        warnings.append("The agent's answer could not be read as JSON.")
    if malformed:
        warnings.append("Some of the agent's claims were malformed and were left out; check the option yourself.")

    fetched: list[datetime] = []
    for evidence_id in [option_id, *(option.get("based_on") or [])]:
        item = _evidence(state, evidence_id)
        moment = _parse_time((item or {}).get("fetched_at"))
        if moment:
            fetched.append(moment)
    oldest = min(fetched).isoformat() if fetched else None

    proposal = Proposal(
        proposal_id=uuid.uuid4().hex[:8],
        option_id=str(option_id),
        option=option,
        observation=str(option_evidence.get("observation") or ""),
        plan_b=plan_b,
        contingency=contingency,
        claims=claims,
        verdicts=verification["verdicts"],
        verification_summary=verification["summary"],
        warnings=warnings,
        needs_human=bool(warnings or result.get("needs_human") or verification["needs_human"]),
        summary=str(result.get("summary") or ""),
        lessons_used=list(result.get("lessons_used") or []),
        preferences_used=list(result.get("preferences_used") or []),
        data_gaps=gaps,
        evidence_fetched_at=oldest,
        created_at=_clock(now).isoformat(),
        booking_args={"option_id": str(option_id)},
    )
    _log(state, "proposal_built", proposal_id=proposal.proposal_id, option_id=proposal.option_id,
         needs_human=proposal.needs_human, verification=verification["summary"])
    return proposal


# --------------------------------------------------------------------------
# Showing it to the human
# --------------------------------------------------------------------------

def render_card_text(proposal: Proposal) -> str:
    """Plain-text approval card (works in any terminal or notebook)."""
    option = proposal.option
    lines = [
        "=" * 64,
        "PROPOSED ACTION (needs your approval)  -  MOCK booking, no real ticket",
        "=" * 64,
        f"Option : {proposal.observation}",
        f"Verdict: {_VERDICT_TEXT.get(option.get('verdict'), option.get('verdict'))}",
        f"Arrive : {option.get('earliest_arrival_local')}  (deadline {option.get('deadline_local')})",
        f"Leave by: {option.get('latest_departure_local') or 'no departure keeps your buffer'}",
    ]
    if proposal.plan_b:
        lines.append(f"Plan B : {proposal.plan_b.get('observation')}")
    if proposal.contingency.get("switch_trigger"):
        decide = "now" if proposal.contingency.get("decide_now") else proposal.contingency.get("decide_by_local")
        lines.append(f"Switch if: {proposal.contingency['switch_trigger']}" + (f"  (decide by {decide})" if decide else ""))
    if proposal.summary:
        if proposal.verification_summary == VERIFICATION_DISABLED:
            lines += ["", "Agent summary (not fact-checked): " + proposal.summary]
        else:
            lines += ["", "Agent summary (not fact-checked, read the checked claims below): " + proposal.summary]
    lines += ["", f"Fact check ({proposal.verification_summary}):"]
    if proposal.verdicts:
        for claim, verdict in zip(proposal.claims, proposal.verdicts):
            mark = _STATUS_MARK.get(verdict.get("status"), "?")
            lines.append(f"  [{mark:5}] {claim.get('text')}")
    elif proposal.verification_summary == VERIFICATION_DISABLED:
        lines.append("  Claims were not verified.")
    if proposal.warnings:
        lines += ["", "WARNINGS - read before approving:"] + [f"  ! {item}" for item in proposal.warnings]
    lines += ["", "Choices: approve | reject (say why) | change (ask for something else) | why?", "=" * 64]
    return "\n".join(lines)


def render_card_html(proposal: Proposal) -> str:
    """HTML approval card for Jupyter / Colab (``IPython.display.HTML``)."""
    esc = html.escape
    option = proposal.option
    colour = {"comfortable": "#1b7f3b", "tight": "#b26a00", "at_risk": "#b3261e", "likely_miss": "#b3261e"}.get(option.get("verdict"), "#555")
    rows = []
    for claim, verdict in zip(proposal.claims, proposal.verdicts):
        status = verdict.get("status")
        icon = {"supported": "&#9989;", "unsupported": "&#9888;&#65039;", "contradicted": "&#10060;"}.get(status, "?")
        reason = "" if status == "supported" else f"<br><small style='color:#b3261e'>{esc(str(verdict.get('reason')))}</small>"
        rows.append(f"<tr><td style='vertical-align:top;padding:2px 6px'>{icon}</td><td style='padding:2px 6px'>{esc(str(claim.get('text')))}{reason}</td></tr>")
    if not proposal.verdicts and proposal.verification_summary == VERIFICATION_DISABLED:
        rows.append("<tr><td colspan='2' style='padding:2px 6px'>Claims were not verified.</td></tr>")
    warn = ""
    if proposal.warnings:
        items = "".join(f"<li>{esc(item)}</li>" for item in proposal.warnings)
        warn = f"<div style='background:#fff4e5;border-left:4px solid #b26a00;padding:6px 10px;margin-top:8px'><b>Read before approving</b><ul style='margin:4px 0'>{items}</ul></div>"
    plan_b = ""
    if proposal.plan_b:
        decide = "now" if proposal.contingency.get("decide_now") else proposal.contingency.get("decide_by_local")
        plan_b = (
            f"<p style='margin:6px 0'><b>Plan B:</b> {esc(str(proposal.plan_b.get('observation')))}<br>"
            f"<b>Switch if:</b> {esc(str(proposal.contingency.get('switch_trigger')))}"
            + (f" &middot; <b>decide by</b> {esc(str(decide))}" if decide else "")
            + "</p>"
        )
    return f"""
<div style="font-family:system-ui,sans-serif;border:1px solid #ccc;border-radius:10px;padding:14px;max-width:760px;background:#fff;color:#111">
  <div style="font-size:12px;color:#666">PROPOSED ACTION &middot; needs your approval &middot; MOCK booking (no real ticket)</div>
  <div style="font-size:18px;font-weight:600;margin:4px 0">{esc(proposal.observation.split(':', 1)[0])}</div>
  <span style="background:{colour};color:#fff;border-radius:12px;padding:2px 10px;font-size:13px">{esc(_VERDICT_TEXT.get(option.get('verdict'), str(option.get('verdict'))))}</span>
  <p style="margin:8px 0">Arrive <b>{esc(str(option.get('earliest_arrival_local')))}</b> for a deadline of <b>{esc(str(option.get('deadline_local')))}</b>.
     Leave by <b>{esc(str(option.get('latest_departure_local') or 'n/a'))}</b>.</p>
  {plan_b}
  <p style="margin:6px 0;color:#333"><small style="color:#666">Agent summary (not fact-checked):</small> <i>{esc(proposal.summary)}</i></p>
  <div style="font-weight:600;margin-top:8px">Fact check: {esc(proposal.verification_summary)}</div>
  <table style="border-collapse:collapse;font-size:14px">{''.join(rows)}</table>
  {warn}
</div>"""


def explain(state: Any, proposal: Proposal) -> str:
    """Answer "why this option?" from evidence only (no extra model call)."""
    option = proposal.option
    lines = [f"Why {proposal.option_id}?"]
    lines.append(
        f"- Door to door about {option.get('door_to_door_min')} min, plus {option.get('expected_delay_min')} min delay "
        f"already announced by the FAA, which leaves {option.get('slack_min')} min before your deadline."
    )
    lines.append(f"- It can absorb up to {option.get('delay_tolerance_min')} min of total delay before you would miss the deadline.")
    lines.append(f"- Ranked #{option.get('rank')} of all options the planner compared.")
    for evidence_id in option.get("based_on") or []:
        item = _evidence(state, evidence_id)
        if item:
            lines.append(f"- Based on {evidence_id}: {item.get('observation')}")
    if option.get("notes"):
        lines.append("- Your preferences: " + "; ".join(option["notes"]))
    for evidence_id in proposal.preferences_used:
        item = _evidence(state, evidence_id)
        if item:
            lines.append(f"- Preference used: {item.get('observation')}")
    for evidence_id in proposal.lessons_used:
        item = _evidence(state, evidence_id)
        if item:
            lesson = (item.get("data") or {}).get("lesson")
            lines.append(f"- Lesson from a past case ({evidence_id}, guidance only): {lesson}")
    lines.append(f"- Assumption: {option.get('assumption')}")
    if proposal.data_gaps:
        lines.append("- Missing data: " + "; ".join(proposal.data_gaps))
    _log(state, "explain_requested", proposal_id=proposal.proposal_id)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Human decisions
# --------------------------------------------------------------------------

def approve(
    state: Any,
    proposal: Proposal,
    *,
    acknowledge_warnings: bool = False,
    approver: str = "human",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record a single-use, expiring approval for exactly this booking."""
    if proposal.option.get("verdict") == "likely_miss":
        _log(state, "approve_refused", proposal_id=proposal.proposal_id, reason="likely_miss")
        return {"ok": False, "error": "This option is expected to miss the deadline; it cannot be approved."}
    if proposal.needs_human and not acknowledge_warnings:
        _log(state, "approve_refused", proposal_id=proposal.proposal_id, reason="warnings_not_acknowledged")
        return {"ok": False, "error": "This proposal has warnings. Read them and approve with acknowledge_warnings=True."}
    approved_at = _clock(now)
    record = {
        "tool": BOOKING_TOOL,
        "args": dict(proposal.booking_args),
        "proposal_id": proposal.proposal_id,
        "approver": approver,
        "approved_at": approved_at.isoformat(),
        "expires_at": (approved_at + timedelta(minutes=APPROVAL_TTL_MIN)).isoformat(),
        "acknowledged_warnings": list(proposal.warnings) if acknowledge_warnings else [],
    }
    approvals = state.get("approved_actions")
    approvals = list(approvals) if isinstance(approvals, list) else []
    approvals = [item for item in approvals if item.get("proposal_id") != proposal.proposal_id]
    approvals.append(record)
    state["approved_actions"] = approvals
    _log(state, "approved", proposal_id=proposal.proposal_id, option_id=proposal.option_id,
         acknowledged_warnings=bool(acknowledge_warnings and proposal.warnings))
    return {"ok": True, "approval": record}


def _drop_approval(state: Any, proposal_id: str) -> None:
    approvals = state.get("approved_actions")
    if isinstance(approvals, list):
        state["approved_actions"] = [item for item in approvals if item.get("proposal_id") != proposal_id]


def reject(state: Any, proposal: Proposal, reason: str, prefs: PreferenceStore | None = None) -> dict[str, Any]:
    """Record a rejection and let memory *suggest* (not save) a preference."""
    _drop_approval(state, proposal.proposal_id)
    suggestions = []
    if prefs is not None and reason:
        suggestions = prefs.observe_rejection(reason, option=proposal.option)
    _log(state, "rejected", proposal_id=proposal.proposal_id, option_id=proposal.option_id, suggestions=len(suggestions))
    return {"ok": True, "suggestions": suggestions}


def request_change(state: Any, proposal: Proposal, request: str, prefs: PreferenceStore | None = None) -> dict[str, Any]:
    """Human wants something different: build the message to send to the agent.

    The request text is forwarded to the model, so it goes through the same
    injection filter as any user turn. A flagged request is not forwarded.
    """
    rejected = reject(state, proposal, request, prefs)
    if looks_like_instruction(request or ""):
        _log(state, "change_refused", proposal_id=proposal.proposal_id, reason="injection_filter")
        return {"ok": False, "message_for_agent": None, "suggestions": [],
                "error": "That request looks like an instruction to the AI rather than a travel change. "
                         "Please describe what you want instead (e.g. 'avoid Oakland', 'take the train')."}
    message = (
        f"The traveller did not accept {proposal.option_id}. Their request: \"{request.strip()[:300]}\". "
        "Re-plan with plan_trip_options if needed and answer in the same JSON format."
    )
    _log(state, "change_requested", proposal_id=proposal.proposal_id)
    return {"ok": True, "message_for_agent": message, "suggestions": rejected["suggestions"]}


def execute_approved(
    state: Any,
    proposal: Proposal,
    *,
    now: datetime | None = None,
    save_case: bool = True,
) -> dict[str, Any]:
    """Run the MOCK booking if, and only if, a valid approval exists.

    Checks, in order: approval present -> not expired -> evidence still fresh
    -> M1's ``tool_gate`` (consumes the approval) -> booking tool ->
    ``record_and_sanitize`` (writes ``completed_actions``).
    """
    clock = _clock(now)
    approvals = state.get("approved_actions")
    approvals = approvals if isinstance(approvals, list) else []
    approval = next(
        (item for item in approvals if item.get("proposal_id") == proposal.proposal_id and item.get("args") == proposal.booking_args),
        None,
    )
    if approval is None:
        _log(state, "booking_blocked", proposal_id=proposal.proposal_id, reason="no_approval")
        return {"ok": False, "blocked": True, "error": "No approval recorded for this proposal."}

    fetched = _parse_time(proposal.evidence_fetched_at)
    if fetched is not None and clock - fetched > timedelta(minutes=MAX_EVIDENCE_AGE_MIN):
        _drop_approval(state, proposal.proposal_id)
        _log(state, "booking_blocked", proposal_id=proposal.proposal_id, reason="stale_evidence")
        age = int((clock - fetched).total_seconds() // 60)
        return {
            "ok": False,
            "blocked": True,
            "needs_recheck": True,
            "error": f"The data behind this option is {age} min old (limit {MAX_EVIDENCE_AGE_MIN}). Conditions may have changed: re-check before booking.",
        }

    expires = _parse_time(approval.get("expires_at"))
    if expires is not None and clock > expires:
        _drop_approval(state, proposal.proposal_id)
        _log(state, "booking_blocked", proposal_id=proposal.proposal_id, reason="approval_expired")
        return {"ok": False, "blocked": True, "error": f"Approval expired after {APPROVAL_TTL_MIN} min. Please review and approve again."}

    ctx = SimpleNamespace(state=state)
    args = dict(proposal.booking_args)
    blocked = guard.tool_gate(_BOOKING_FUNCTION_TOOL, args, ctx)
    if blocked is not None:
        _log(state, "booking_blocked", proposal_id=proposal.proposal_id, reason="tool_gate")
        return {"ok": False, "blocked": True, "error": blocked.get("error")}

    result = book_trip_option(**args, tool_context=ctx)
    guard.record_and_sanitize(_BOOKING_FUNCTION_TOOL, args, ctx, result)
    if not result.get("ok"):
        _log(state, "booking_failed", proposal_id=proposal.proposal_id, error=str(result.get("error")))
        return {"ok": False, "blocked": False, "error": result.get("error"), "result": result}

    booking = result["evidence"][0]
    _log(state, "booked_mock", proposal_id=proposal.proposal_id, confirmation=booking["data"]["confirmation"])
    learned = None
    if save_case:
        learned = _learn_case(proposal)
    return {"ok": True, "booking": booking, "learned_case": learned}


def _learn_case(proposal: Proposal) -> dict[str, Any] | None:
    """Save the approved decision as a new case for future retrieval.

    Only tool-generated text is stored (the planner's observation and switch
    trigger), never the model's free-text summary: a hallucinated sentence
    must not become a "past case" that later guides the agent.
    """
    option = proposal.option
    airports = [code for code in str(option.get("route") or "").split("-") if len(code) == 3]
    trigger = proposal.contingency.get("switch_trigger")
    try:
        saved = case_library.record_case(
            title=f"Approved {option.get('mode')} {option.get('route')} for a deadline",
            situation=proposal.observation + (f" Switch trigger: {trigger}." if trigger else ""),
            decision=f"Chose {proposal.option_id} ({option.get('verdict')}, {option.get('slack_min')} min spare)"
            + (f"; Plan B {proposal.plan_b.get('id')}" if proposal.plan_b else ""),
            outcome="Approved by the traveller; final outcome not recorded yet.",
            lesson="; ".join(option.get("notes") or []) or "Traveller accepted the planner's top option.",
            airports=airports,
            modes=[str(option.get("mode") or "")],
            disruption="learned_decision",
        )
        return saved.get("case") if saved.get("ok") else None
    except Exception:
        return None
