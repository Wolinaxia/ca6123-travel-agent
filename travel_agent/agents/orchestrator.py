"""Root orchestrator agent (Reason stage, M2).

One user turn runs through these steps::

    screen ─► plan ─► ┌ flight_airport ┐ ─► alternatives? ─► verify ─► answer
                      ├ weather        ┤    (conditional)
                      └ rule engine    ┘

* **screen**: M1's deterministic guardrails run *before* the planner's Gemini
  call, so PII never reaches the model and injection/harmful requests are
  refused without spending tokens.
* **plan**: ``planner.plan_request`` classifies the intent, grounds slots and
  builds the sub-task graph (see planner.py). ``out_of_scope`` and missing slots
  end the turn here with a polite answer or a clarifying question.
* **flight_airport ‖ weather**: run concurrently (ADK ``ParallelAgent``). They
  are independent, and running them in parallel roughly halves wall time.
* **rule engine**: at the same time, the deterministic risk engine
  (``tools.live.assess_trip_risk``) scores the whole trip in a worker thread.
  The specialists report *observations* (M3 sets ``risk=null`` on factual
  claims), so the trip-level risk comes from tested rules, not from an LLM
  label. Its summary also gives the Verifier something to compare risk claims
  against.
* **alternatives**: runs after both, because it needs their findings. For
  ``assess_trip`` it only runs when the rule engine or some claim says
  moderate/high risk, which saves a model call on calm days; for
  ``plan_trip``/``rebook`` it always runs.
* **verify**: every claim is checked against the evidence the tools actually
  returned plus the rule engine's evidence (M1's ``verify_claims``). It is called directly rather than through
  an LLM Verifier agent: re-typing the claims through a model costs tokens and
  can itself introduce copy errors. ``verifier_enabled=False`` turns this step
  off for M1's ablation study.
* **answer**: assembled in code from *supported* claims only. Held-back claims
  and the reason are shown to the human.
* **approval card** (``plan_trip``/``rebook``): M4's ``hitl.build_proposal``
  turns the recommended option into a card with a verdict per claim. Nothing
  is booked in this turn: the session keeps ``pending_proposal`` and the
  human's next reply ("approve", "reject: too early", "why?", "take the train
  instead") is applied by ``hitl.approve``/``execute_approved``/``reject``/
  ``explain``/``request_change`` instead of planning a new trip. Only code
  holds the (mock) booking tool, behind M1's ``tool_gate``.

Why a custom ``BaseAgent`` instead of one big ``SequentialAgent``: the route
depends on the intent and on the specialists' risk levels, which is control
flow that is clearer (and testable) in Python than in prompts.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import warnings
from typing import Any, AsyncGenerator, Callable, Optional

from google.adk.agents import BaseAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from travel_agent import hitl
from travel_agent.agents import handoff
from travel_agent.agents import planner as planner_mod
from travel_agent.agents.flight_airport import build_flight_airport_agent
from travel_agent.agents.specialists import (
    DEFAULT_MODEL,
    SPECIALIST_NAMES,
    parse_agent_result,
)
from travel_agent.agents.weather import build_weather_agent
from travel_agent.guardrails.injection import detect_injection
from travel_agent.guardrails.pii import redact_pii
from travel_agent.guardrails.topics import check_topic
from travel_agent.guardrails.sanitize import sanitize_tool_result
from travel_agent.memory.preferences import DEFAULT_USER, PreferenceStore
from travel_agent.observability.trace import Tracer, get_active_tracer, set_active_tracer
from travel_agent.tools import live
from travel_agent.verification.checks import verify_claims
from travel_agent.verification.store import add_tool_result

RISK_ORDER = {"low": 0, "moderate": 1, "high": 2}
# The rule engine has four levels; the answer uses the claim scale (three).
ENGINE_TO_RISK = {"low": "low", "elevated": "moderate", "high": "high", "severe": "high"}
RULE_ENGINE = "rule_engine"
# Intents whose answer ends in a proposal the human approves or rejects.
PROPOSAL_INTENTS = ("plan_trip", "rebook")
HITL_KEYS = ("evidence_store", "approved_actions", "completed_actions", "hitl_log")
SECTION_TITLES = {"flight_airport": "Flight & airport", "weather": "Weather", "alternatives": "Options",
                  "rule_engine": "Rule engine"}
SLOT_QUESTIONS = {
    "origin": "Where are you leaving from?",
    "destination": "Where are you going?",
    "departure_date": "Which day do you travel?",
    "arrive_by": "What date and time do you need to arrive?",
}
OUT_OF_SCOPE_TEXT = (
    "I can only help with the disruption risk of U.S. domestic trips (flights, weather, "
    "airport delays and alternatives). Try: \"How risky is JFK to SFO tomorrow evening?\""
)
INJECTION_REFUSAL = (
    "I can't act on that message: it looks like it is trying to change how I work "
    "rather than asking about travel."
)


# --------------------------------------------------------------------------
# Step helpers (pure functions, unit-tested without ADK)
# --------------------------------------------------------------------------


def screen_request(text: str) -> dict[str, Any]:
    """Run the input guardrails. Returns ``{"text", "refusal", "reasons"}``."""
    clean, findings = redact_pii(text)
    reasons = [f"pii:{item.get('kind', 'redacted')}" for item in findings]
    topic = check_topic(clean)
    if not topic.get("allowed", True):
        return {"text": clean, "refusal": str(topic.get("reason") or OUT_OF_SCOPE_TEXT),
                "reasons": reasons + [f"topic:{topic.get('category')}"]}
    injection = detect_injection(clean)
    if injection.get("flagged"):
        return {"text": clean, "refusal": INJECTION_REFUSAL,
                "reasons": reasons + list(injection.get("reasons") or [])}
    return {"text": clean, "refusal": None, "reasons": reasons}


def _max_risk(claims: list[dict[str, Any]]) -> Optional[str]:
    risks = [c["risk"] for c in claims if c.get("risk") in RISK_ORDER]
    return max(risks, key=RISK_ORDER.__getitem__) if risks else None


def _higher(*risks: Optional[str]) -> Optional[str]:
    known = [r for r in risks if r in RISK_ORDER]
    return max(known, key=RISK_ORDER.__getitem__) if known else None


def should_run_alternatives(
    plan: list[dict[str, Any]],
    signal_claims: list[dict[str, Any]],
    rule_risk: Optional[str] = None,
) -> bool:
    """Apply the plan's condition on the alternatives step.

    ``rule_risk`` is the rule engine's level on the claim scale. It matters
    because the specialists usually leave ``risk`` null on factual claims, so a
    real ground delay program would otherwise never trigger alternatives.
    """
    task = next((t for t in plan if t.get("agent") == "alternatives"), None)
    if task is None:
        return False
    if not task.get("condition"):
        return True
    risk = _higher(_max_risk(signal_claims), rule_risk) or "low"
    return RISK_ORDER[risk] >= RISK_ORDER["moderate"]


def assess_with_rules(goal: dict[str, Any], engine: Callable[..., dict]) -> dict[str, Any]:
    """Score the trip with the deterministic rule engine.

    Returns ``{"risk", "engine_level", "confidence", "action", "evidence_id",
    "result"}`` on success or ``{"error", "result"}`` on failure, where
    ``result`` is the sanitised ``ToolResult``. Never raises: a broken engine
    only means the answer falls back to the specialists' claim risks.
    """
    try:
        raw = engine(goal["origin_airport"], goal["destination_airport"],
                     goal["departure_date"], goal.get("departure_window") or "whole_day")
        # Engine evidence can carry operator text (TomTom/MTA): same sanitiser as tool callbacks.
        result, _events = sanitize_tool_result(raw)
    except Exception as exc:
        return {"error": f"rule engine failed ({type(exc).__name__})", "result": None}
    if not result.get("ok"):
        return {"error": f"rule engine: {result.get('error') or 'no result'}", "result": result}
    summary = next((e for e in result.get("evidence") or []
                    if str(e.get("id", "")).endswith(":summary")), None)
    level = ((summary or {}).get("data") or {}).get("level")
    if level not in ENGINE_TO_RISK:
        return {"error": "rule engine returned no overall level", "result": result}
    data = summary["data"]
    return {"risk": ENGINE_TO_RISK[level], "engine_level": level, "confidence": data.get("confidence"),
            "action": data.get("action"), "evidence_id": summary["id"], "result": result}


def clarify_text(goal: dict[str, Any]) -> str:
    questions = [SLOT_QUESTIONS.get(slot, f"Please tell me the {slot}.") for slot in goal.get("missing", [])]
    notes = goal.get("notes") or []
    prefix = ("I could not use part of your request: " + "; ".join(notes) + ".\n") if notes else ""
    return prefix + "To check this trip I need a bit more detail:\n" + "\n".join(f"- {q}" for q in questions)


def _trip_line(goal: dict[str, Any]) -> str:
    origin = goal.get("origin_airport") or goal.get("origin")
    destination = goal.get("destination_airport") or goal.get("destination")
    line = f"{origin} → {destination}, {goal.get('departure_date')}"
    if goal.get("departure_window") not in (None, "whole_day"):
        line += f" ({goal['departure_window']})"
    if goal.get("arrive_by"):
        line += f", arrive by {goal['arrive_by']}"
    if goal.get("flight_number"):
        line += f", booked flight {goal['flight_number']}"
    return line


def assemble_answer(
    intent: str,
    goal: dict[str, Any],
    results: dict[str, dict[str, Any]],
    verification: Optional[dict[str, Any]],
    rules: Optional[dict[str, Any]] = None,
    proposal: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the final answer from the specialists' results.

    With a verification: only ``supported`` claims are shown; the others are
    listed under "held back" with the Verifier's reason. Without one (ablation
    "Verifier OFF"): every claim is shown as-is.

    ``rules`` is the output of :func:`assess_with_rules`. The overall risk is
    the higher of the rule engine's level and the shown claims' risks, so a
    specialist can raise the level but never talk it down.

    ``proposal`` is a ``hitl.Proposal`` dict; its approval card is appended and
    the answer is marked ``awaiting_approval``.
    """
    claims = [claim for name in SPECIALIST_NAMES for claim in results.get(name, {}).get("claims", [])]
    shown: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    if verification is None:
        shown = list(claims)
    else:
        for claim, verdict in zip(claims, verification.get("verdicts", [])):
            if verdict.get("status") == "supported":
                shown.append(claim)
            else:
                held.append({**claim, "status": verdict.get("status"), "reason": verdict.get("reason")})

    errors = [r["error"] for r in results.values() if r.get("error")]
    signal_claims = [c for c in shown if c.get("agent") in ("flight_airport", "weather")]
    rules = rules or {}
    overall = _higher(_max_risk(signal_claims), rules.get("risk")) or "unknown"
    needs_human = bool(held or errors or any(r.get("needs_human") for r in results.values())
                       or (verification or {}).get("needs_human"))

    risk_line = f"Overall risk: {overall}"
    if rules.get("risk"):
        risk_line += (f" (rule engine: {rules['engine_level']}, confidence {rules.get('confidence')}; "
                      f"evidence: {rules['evidence_id']})")
    lines = [f"Trip: {_trip_line(goal)}", risk_line]
    for name in SPECIALIST_NAMES:
        section = [c for c in shown if c.get("agent") == name]
        if section:
            lines.append(f"\n{SECTION_TITLES[name]}:")
            for claim in section:
                risk = f" [{claim['risk']} risk]" if claim.get("risk") else ""
                lines.append(f"- {claim['text']}{risk} (evidence: {', '.join(claim['evidence_ids']) or 'none'})")
    if not shown:
        lines.append("\nNo finding could be confirmed from the data sources right now.")
    unavailable = {name: r.get("unavailable") for name, r in results.items() if r.get("unavailable")}
    if rules.get("error"):
        unavailable[RULE_ENGINE] = [rules["error"]]
    if unavailable:
        # Specialists list failed tools *and* coverage gaps (e.g. no forecast
        # for part of the trip window), so each entry is shown in full.
        lines.append("\nCoverage gaps (not checked, or no data for the trip window):")
        lines.extend(f"- {SECTION_TITLES.get(name, name)}: {gap}"
                     for name, gaps in unavailable.items() for gap in gaps)
    if held or errors:
        lines.append("\nNeeds your check (not confirmed by the evidence):")
        lines.extend(f"- {c['text']} — {c['status']}: {c['reason']}" for c in held)
        lines.extend(f"- {error}" for error in errors)
    if proposal:
        lines.append("\n" + hitl.render_card_text(hitl.Proposal(**proposal)))
    elif intent == "rebook":
        lines.append("\nConfirm a specific replacement with your airline before making a change.")
    return {
        "text": "\n".join(lines),
        "overall_risk": overall,
        "rule_risk": rules.get("engine_level"),
        "shown_claims": shown,
        "held_back": held,
        "unavailable": unavailable,
        "needs_human": needs_human,
        # Only a verified M4 proposal can be approved; claims alone never are.
        "awaiting_approval": bool(proposal),
        "proposal": proposal,
        "verifier_enabled": verification is not None,
    }


# --------------------------------------------------------------------------
# Human in the loop (M4's hitl package, driven by the orchestrator)
# --------------------------------------------------------------------------


def _summarise(verdicts: list[dict[str, Any]], needs_human: bool = False) -> dict[str, Any]:
    counts = {"supported": 0, "unsupported": 0, "contradicted": 0}
    problems = []
    for index, verdict in enumerate(verdicts, start=1):
        status = str(verdict.get("status"))
        counts[status] = counts.get(status, 0) + 1
        if status != "supported":
            problems.append(f"claim {index}: {status} ({verdict.get('reason')})")
    if not verdicts:
        summary = "no claims to verify"
    elif not problems:
        summary = f"all {len(verdicts)} claims supported"
    else:
        summary = f"{len(verdicts) - len(problems)}/{len(verdicts)} supported; " + "; ".join(problems)
    return {"verdicts": verdicts, "counts": counts, "summary": summary,
            "needs_human": bool(problems) or needs_human}


def combine_verifications(*parts: dict[str, Any]) -> dict[str, Any]:
    """Concatenate ``verify_claims`` results, in claim order."""
    verdicts = [verdict for part in parts for verdict in part.get("verdicts") or []]
    return _summarise(verdicts, any(part.get("needs_human") for part in parts))


def verify_results(results: dict[str, dict[str, Any]], store: dict[str, Any],
                   rules_result: Optional[dict[str, Any]]) -> dict[str, Any]:
    """flight_airport + weather claims vs store + rule-engine evidence; alternatives claims vs store only."""
    with_rules = {"evidence_store": dict(store)}
    if rules_result:
        add_tool_result(with_rules, rules_result)
    signal_claims = [c for name in ("flight_airport", "weather")
                     for c in results.get(name, {}).get("claims", [])]
    alternative_claims = results.get("alternatives", {}).get("claims", [])
    return combine_verifications(verify_claims(signal_claims, with_rules["evidence_store"]),
                                 verify_claims(alternative_claims, store))


def merge_verdicts(verification: dict[str, Any], tail: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace the alternatives verdicts (the last claims) with M4's stricter ones.

    ``hitl.build_proposal`` re-runs M1's Verifier on the alternatives claims
    and adds two checks (no "booked" claim without a booking id, no low risk
    on an active FAA program). The claims of the other specialists keep their
    verdicts.
    """
    verdicts = list(verification.get("verdicts") or [])
    if tail and len(verdicts) >= len(tail):
        verdicts[-len(tail):] = tail
    return {**verification, **_summarise(verdicts, bool(verification.get("needs_human")))}


def build_hitl_proposal(
    state: Any,
    results: dict[str, dict[str, Any]],
    store: dict[str, Any],
    verification: Optional[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], dict[str, Any]]:
    """Build M4's approval card for the recommended option.

    Returns ``(proposal dict or None, verification, state delta)``. Works on a
    copy of the state so ``build_proposal`` cannot overwrite the orchestrator's
    full ``last_verification`` with an alternatives-only one.
    """
    alternatives = results["alternatives"]
    fields = alternatives.get("proposal") or {}
    if not fields.get("recommended_option_id"):
        return None, verification, {}
    m4_result = {**fields, "needs_human": alternatives.get("needs_human", False),
                 "claims": [{k: v for k, v in claim.items() if k != "agent"} for claim in alternatives["claims"]]}
    view = {"evidence_store": store, "hitl_log": list(state.get("hitl_log") or [])}
    try:
        proposal = hitl.build_proposal(view, m4_result, verify=verification is not None)
    except hitl.ProposalError as exc:
        alternatives["error"] = f"no approval card: {exc}"
        return None, verification, {"hitl_log": view["hitl_log"]}
    if verification is not None:
        verification = merge_verdicts(verification, proposal.verdicts)
    return proposal.to_dict(), verification, {"hitl_log": view["hitl_log"]}


def apply_decision(state: Any, decision: dict[str, Any], pending: dict[str, Any]) -> dict[str, Any]:
    """Run one human decision on the pending proposal.

    Returns ``{"ok", "text", "answer", "delta", "change_request"}``; ``delta``
    is the state update (approvals, booking evidence, log, pending proposal).
    """
    view = {key: copy.deepcopy(state.get(key)) for key in HITL_KEYS if state.get(key) is not None}
    view.setdefault("evidence_store", {})
    proposal = hitl.Proposal(**pending)
    prefs = PreferenceStore(str(state.get("user_id") or DEFAULT_USER))
    kind = decision["kind"]
    keep: Optional[dict[str, Any]] = pending
    change: Optional[str] = None
    answer: dict[str, Any] = {"needs_human": True, "awaiting_approval": True, "proposal": pending}
    ok = True

    if kind == "explain":
        text = hitl.explain(view, proposal)
    elif kind == "approve":
        approved = hitl.approve(view, proposal, acknowledge_warnings=decision.get("acknowledge", False))
        if not approved["ok"]:
            ok = False
            if proposal.option.get("verdict") == "likely_miss":
                text = approved["error"]
            else:
                text = ("This proposal has warnings:\n" + "\n".join(f"- {w}" for w in proposal.warnings)
                        + '\nReply "approve anyway" if you accept them, or "reject" with a reason.')
        else:
            done = hitl.execute_approved(view, proposal)
            ok = bool(done.get("ok"))
            if ok:
                booking = done["booking"]
                text = (f"MOCK booking made for {proposal.option_id} (reference {booking['data']['confirmation']}, "
                        "no real ticket was bought). Plan B stays available if the switch trigger happens.")
                keep, answer = None, {"needs_human": False, "awaiting_approval": False, "booking": booking}
            else:
                text = str(done.get("error") or "The booking was blocked.")
                if done.get("needs_recheck"):
                    keep = None
                    text += " Send your trip request again to re-check the options."
    elif kind == "reject":
        rejected = hitl.reject(view, proposal, decision.get("reason") or "", prefs)
        text = "OK, nothing was booked."
        questions = [s["question"] for s in rejected.get("suggestions") or [] if s.get("question")]
        if questions:
            text += "\n" + "\n".join(questions)
        keep, answer = None, {"needs_human": False, "awaiting_approval": False,
                              "suggestions": rejected.get("suggestions") or []}
    else:  # change
        changed = hitl.request_change(view, proposal, decision.get("request") or "", prefs)
        keep = None
        if changed["ok"]:
            text, change = "Looking for another option.", changed["message_for_agent"]
        else:
            ok, text = False, str(changed.get("error"))
            answer = {"needs_human": False, "awaiting_approval": False}
    delta = {**{key: view[key] for key in HITL_KEYS if key in view}, "pending_proposal": keep}
    return {"ok": ok, "text": text, "answer": answer, "delta": delta, "change_request": change}


# --------------------------------------------------------------------------
# The ADK agent
# --------------------------------------------------------------------------


def _internal_event(event: Event) -> Event:
    """Keep ADK tool traffic and state updates, and withhold specialist prose."""
    if not event.content or not any(part.text for part in event.content.parts or []):
        return event
    parts = [part for part in event.content.parts if not part.text]
    content = event.content.model_copy(update={"parts": parts}) if parts else None
    return event.model_copy(update={"content": content})


class TravelOrchestrator(BaseAgent):
    """Root agent. Sub-agents: ``signals`` (parallel flight+weather), ``alternatives``."""

    verifier_enabled: bool = True
    # ``planner(text) -> plan_request dict``; injectable so tests need no Gemini.
    planner: Any = None
    # ``risk_engine(origin, destination, date, window) -> ToolResult dict``.
    # ``None`` turns the rule engine off (offline unit tests).
    risk_engine: Any = None

    def _sub(self, name: str) -> BaseAgent:
        return next(agent for agent in self.sub_agents if agent.name == name)

    def _event(self, ctx: InvocationContext, text: Optional[str] = None, **state: Any) -> Event:
        content = types.Content(role="model", parts=[types.Part(text=text)]) if text else None
        return Event(author=self.name, invocation_id=ctx.invocation_id, branch=ctx.branch,
                     content=content, actions=EventActions(state_delta=state))

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        tracer = get_active_tracer()
        text = " ".join(p.text for p in (ctx.user_content.parts or []) if getattr(p, "text", None)) \
            if ctx.user_content else ""

        # 1. Screen. Refusals end the turn before any model call.
        screened = screen_request(text)
        if tracer and screened["reasons"]:
            tracer.log_guardrail("orchestrator_input", "refuse" if screened["refusal"] else "redact", screened["reasons"])
        if screened["refusal"]:
            answer = {"text": screened["refusal"], "blocked": True, "needs_human": False}
            yield self._event(ctx, screened["refusal"], intent="blocked", final_answer=answer,
                              pending_clarification=None, last_verification=None)
            return

        # 1b. A reply to the approval card (approve / reject / why / change).
        pending_proposal = ctx.session.state.get("pending_proposal")
        decision = handoff.parse_decision(screened["text"]) if pending_proposal else None
        if decision:
            async for event in self._decide(ctx, decision, pending_proposal):
                yield event
            return

        # 2. Plan (blocking Gemini call -> worker thread so the event loop stays free).
        planner: Callable[[str], dict] = self.planner or planner_mod.plan_request
        pending = ctx.session.state.get("pending_clarification")
        planned = await asyncio.to_thread(planner_mod.resume_clarification, screened["text"], pending)
        if planned is None:
            planned = await asyncio.to_thread(planner, screened["text"])
        goal, plan = planned["trip_goal"], planned["plan"]
        if tracer:
            tracer.log("plan", intent=planned["intent"], classified_by=goal.get("classified_by"),
                       agents=[t["agent"] for t in plan], missing=goal.get("missing"))
        # A new request replaces any open proposal (its approval was never given).
        yield self._event(ctx, intent=planned["intent"], trip_goal=goal, plan=plan,
                          pending_clarification=planned.get("pending_clarification"),
                          last_verification=None, specialist_results={}, evidence_store={},
                          rule_assessment=None, pending_proposal=None, change_request=None,
                          user_id=ctx.session.state.get("user_id") or ctx.session.user_id,
                          **{key: None for key in (*handoff.RESULT_KEYS,
                                                   *(f"result_{name}" for name in SPECIALIST_NAMES))})

        if planned["intent"] == "out_of_scope" or planned["needs_clarification"]:
            reply = OUT_OF_SCOPE_TEXT if planned["intent"] == "out_of_scope" else clarify_text(goal)
            yield self._event(ctx, reply, final_answer={"text": reply, "needs_human": False})
            return

        # 3. Flight + weather in parallel (each writes state["result_<name>"]),
        #    with the rule engine running in a worker thread at the same time.
        rules_task = (asyncio.ensure_future(asyncio.to_thread(assess_with_rules, goal, self.risk_engine))
                      if self.risk_engine else None)
        try:
            async for event in self._sub("signals").run_async(ctx):
                yield _internal_event(event)
        finally:
            rules = await rules_task if rules_task else {}
        if tracer and rules:
            tracer.log("rule_engine", level=rules.get("engine_level"), error=rules.get("error"))
        # The summary is shown to the alternatives step; the full result goes to the Verifier.
        rule_view = {k: v for k, v in rules.items() if k != "result"}
        yield self._event(ctx, rule_assessment=rule_view or None)
        state = ctx.session.state
        results = {name: parse_agent_result(state.get(f"result_{name}"), name) for name in ("flight_airport", "weather")}

        # 4.-6. Alternatives (if the plan says so), verify, answer.
        signal_claims = results["flight_airport"]["claims"] + results["weather"]["claims"]
        run_alternatives = should_run_alternatives(plan, signal_claims, rules.get("risk"))
        async for event in self._finish(ctx, planned["intent"], goal, results, rules, run_alternatives):
            yield event

    async def _run_alternatives(self, ctx: InvocationContext, results: dict[str, Any]) -> AsyncGenerator[Event, None]:
        async for event in self._sub("alternatives").run_async(ctx):
            yield _internal_event(event)
        raw = handoff.raw_alternatives(ctx.session.state)
        results["alternatives"] = {**parse_agent_result(raw, "alternatives"),
                                   "proposal": handoff.proposal_fields(raw)}

    async def _finish(self, ctx: InvocationContext, intent: str, goal: dict[str, Any],
                      results: dict[str, Any], rules: dict[str, Any],
                      run_alternatives: bool) -> AsyncGenerator[Event, None]:
        """Steps 4-6: alternatives, verification, approval card, answer."""
        tracer = get_active_tracer()
        if run_alternatives:
            async for event in self._run_alternatives(ctx, results):
                yield event

        # 5. Verify against the evidence the tools really returned.
        #    The rule engine scores the *planned* route, so its evidence joins
        #    the check of the flight/weather claims only. Alternatives claims
        #    are about other routes (EWR-OAK...) and are checked against their
        #    own planner level; the trip summary would wrongly flag "OAK avoids
        #    the SFO delay, low risk" as downplaying the JFK-SFO risk.
        store = dict(ctx.session.state.get("evidence_store") or {})
        verification = None
        if self.verifier_enabled:
            verification = verify_results(results, store, rules.get("result"))

        # 5b. Approval card (M4 HITL): only for intents that end in a booking.
        proposal, extra = None, {}
        if intent in PROPOSAL_INTENTS and "alternatives" in results:
            proposal, verification, extra = build_hitl_proposal(ctx.session.state, results, store, verification)
        if tracer and verification is not None:
            tracer.log_verdicts(verification["counts"], verification["needs_human"])

        # 6. Answer.
        rule_view = {k: v for k, v in rules.items() if k != "result"}
        answer = assemble_answer(intent, goal, results, verification, rule_view, proposal)
        if tracer:
            tracer.log("answer", overall_risk=answer["overall_risk"], shown=len(answer["shown_claims"]),
                       held_back=len(answer["held_back"]), needs_human=answer["needs_human"],
                       awaiting_approval=answer["awaiting_approval"])
        yield self._event(ctx, answer["text"], specialist_results=results, last_verification=verification,
                          final_answer=answer, pending_proposal=proposal, change_request=None, **extra)

    async def _decide(self, ctx: InvocationContext, decision: dict[str, Any],
                      pending: dict[str, Any]) -> AsyncGenerator[Event, None]:
        """Apply the human's decision to the pending proposal (M4 HITL)."""
        state = ctx.session.state
        outcome = apply_decision(state, decision, pending)
        if tracer := get_active_tracer():
            tracer.log("decision", kind=decision["kind"], ok=outcome["ok"])
        if outcome.get("change_request") is None:
            yield self._event(ctx, outcome["text"], final_answer={"text": outcome["text"], **outcome["answer"]},
                              **outcome["delta"])
            return

        # "Take the train instead": re-run alternatives with the request, on
        # the same trip and the same flight/weather findings.
        # (delta clears pending_proposal: the old card was rejected by request_change)
        yield self._event(ctx, change_request=outcome["change_request"],
                          **{key: None for key in handoff.RESULT_KEYS}, **outcome["delta"])
        goal = state.get("trip_goal") or {}
        previous = state.get("specialist_results") or {}
        results = {name: previous.get(name) or {"agent": name, "claims": []} for name in ("flight_airport", "weather")}
        rules = (await asyncio.to_thread(assess_with_rules, goal, self.risk_engine)) if self.risk_engine else {}
        async for event in self._finish(ctx, state.get("intent") or "plan_trip", goal, results, rules, True):
            yield event


def build_orchestrator(
    model: Any = DEFAULT_MODEL,
    specialists: Optional[dict[str, BaseAgent]] = None,
    verifier_enabled: bool = True,
    planner: Optional[Callable[[str], dict]] = None,
    risk_engine: Optional[Callable[..., dict]] = live.assess_trip_risk,
) -> TravelOrchestrator:
    """Wire the root agent with the default specialists and any caller overrides.

    ``risk_engine=None`` skips the deterministic rule engine (offline tests).
    """
    overrides = specialists or {}
    flight = overrides.get("flight_airport") or build_flight_airport_agent(model)
    weather = overrides.get("weather") or build_weather_agent(model)
    alternatives = overrides.get("alternatives") or handoff.build_m4_alternatives(model)
    with warnings.catch_warnings():
        # ParallelAgent is marked deprecated in ADK 2.x in favour of Workflow,
        # but it is what the course uses and still fully supported.
        warnings.simplefilter("ignore", DeprecationWarning)
        signals = ParallelAgent(name="signals", sub_agents=[flight, weather])
    return TravelOrchestrator(
        name="orchestrator",
        description="Routes U.S. domestic trip-disruption requests to specialists.",
        sub_agents=[signals, alternatives],
        verifier_enabled=verifier_enabled,
        planner=planner,
        risk_engine=risk_engine,
    )


# --------------------------------------------------------------------------
# One-call entry point (notebook, evaluation, M1's ablation)
# --------------------------------------------------------------------------

RESULT_KEYS = ("intent", "trip_goal", "plan", "specialist_results", "last_verification",
               "final_answer", "evidence_store", "rule_assessment", "pending_proposal", "hitl_log",
               "completed_actions", "guardrail_events", "trace_run_id")


class Conversation:
    """Several turns in one ADK session, e.g. a request and then "approve".

    ``scenario`` replays a recorded demo scenario (frozen clock) during each
    turn. One trace file per conversation. ``send`` works in scripts and in
    Jupyter/Colab.
    """

    def __init__(
        self,
        *,
        scenario: Optional[str] = None,
        verifier_enabled: bool = True,
        model: Any = DEFAULT_MODEL,
        specialists: Optional[dict[str, BaseAgent]] = None,
        planner: Optional[Callable[[str], dict]] = None,
        risk_engine: Optional[Callable[..., dict]] = live.assess_trip_risk,
        log_dir: str = "logs",
        user_id: str = "user",
    ) -> None:
        from google.adk.runners import InMemoryRunner

        self.scenario = scenario
        self.user_id = user_id
        self.tracer = Tracer(log_dir=log_dir)
        root = build_orchestrator(model, specialists, verifier_enabled, planner, risk_engine)
        self.runner = InMemoryRunner(agent=root, app_name="travel_agent")
        self.session_id: Optional[str] = None

    async def send_async(self, text: str) -> dict[str, Any]:
        """Run one user turn; return the session state keys in ``RESULT_KEYS``."""
        from travel_agent.tools import replay

        if self.scenario:
            replay.use_demo_scenario(self.scenario)
        previous = get_active_tracer()
        set_active_tracer(self.tracer)
        service = self.runner.session_service
        try:
            if self.session_id is None:
                session = await service.create_session(app_name="travel_agent", user_id=self.user_id,
                                                       state={"trace_run_id": self.tracer.run_id})
                self.session_id = session.id
            message = types.Content(role="user", parts=[types.Part(text=text)])
            async for _event in self.runner.run_async(user_id=self.user_id, session_id=self.session_id,
                                                      new_message=message):
                pass
            final = await service.get_session(app_name="travel_agent", user_id=self.user_id,
                                              session_id=self.session_id)
            return {key: final.state.get(key) for key in RESULT_KEYS}
        finally:
            set_active_tracer(previous)
            if self.scenario:
                replay.use_demo_scenario(None)

    def send(self, text: str) -> dict[str, Any]:
        """Synchronous :meth:`send_async`."""
        return _run_sync(lambda: self.send_async(text))


async def run_pipeline_async(
    request: str,
    *,
    scenario: Optional[str] = None,
    verifier_enabled: bool = True,
    model: Any = DEFAULT_MODEL,
    specialists: Optional[dict[str, BaseAgent]] = None,
    planner: Optional[Callable[[str], dict]] = None,
    risk_engine: Optional[Callable[..., dict]] = live.assess_trip_risk,
    log_dir: str = "logs",
) -> dict[str, Any]:
    """Run one request end to end and return the final state (see RESULT_KEYS).

    ``scenario`` replays a recorded demo scenario (deterministic, frozen clock);
    ``None`` uses live data. Each call writes a trace to ``logs/<run_id>.jsonl``.
    For an approval after the answer, use :class:`Conversation` instead.
    """
    chat = Conversation(scenario=scenario, verifier_enabled=verifier_enabled, model=model,
                        specialists=specialists, planner=planner, risk_engine=risk_engine, log_dir=log_dir)
    return await chat.send_async(request)


def _run_sync(make: Callable[[], Any]) -> Any:
    """Run a coroutine from sync code, also inside Jupyter's running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["result"] = asyncio.run(make())
        except BaseException as exc:  # re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def run_pipeline(request: str, **kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper around :func:`run_pipeline_async`.

    Works in scripts and in Jupyter/Colab (where an event loop is already
    running) by using a helper thread in the latter case.
    """
    return _run_sync(lambda: run_pipeline_async(request, **kwargs))
