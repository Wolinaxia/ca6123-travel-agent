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
  and the reason are shown to the human. A rebooking is only ever *proposed*.

Why a custom ``BaseAgent`` instead of one big ``SequentialAgent``: the route
depends on the intent and on the specialists' risk levels, which is control
flow that is clearer (and testable) in Python than in prompts.
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from typing import Any, AsyncGenerator, Callable, Optional

from google.adk.agents import BaseAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from travel_agent.agents import planner as planner_mod
from travel_agent.agents.flight_airport import build_flight_airport_agent
from travel_agent.agents.specialists import (
    DEFAULT_MODEL,
    SPECIALIST_NAMES,
    build_placeholder_specialist,
    parse_agent_result,
)
from travel_agent.agents.weather import build_weather_agent
from travel_agent.guardrails.injection import detect_injection
from travel_agent.guardrails.pii import redact_pii
from travel_agent.guardrails.topics import check_topic
from travel_agent.guardrails.sanitize import sanitize_tool_result
from travel_agent.observability.trace import Tracer, get_active_tracer, set_active_tracer
from travel_agent.tools import live
from travel_agent.verification.checks import verify_claims
from travel_agent.verification.store import add_tool_result

RISK_ORDER = {"low": 0, "moderate": 1, "high": 2}
# The rule engine has four levels; the answer uses the claim scale (three).
ENGINE_TO_RISK = {"low": "low", "elevated": "moderate", "high": "high", "severe": "high"}
RULE_ENGINE = "rule_engine"
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
) -> dict[str, Any]:
    """Build the final answer from the specialists' results.

    With a verification: only ``supported`` claims are shown; the others are
    listed under "held back" with the Verifier's reason. Without one (ablation
    "Verifier OFF"): every claim is shown as-is.

    ``rules`` is the output of :func:`assess_with_rules`. The overall risk is
    the higher of the rule engine's level and the shown claims' risks, so a
    specialist can raise the level but never talk it down.
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
    if intent == "rebook":
        lines.append("\nConfirm a specific replacement with your airline before making a change.")
    return {
        "text": "\n".join(lines),
        "overall_risk": overall,
        "rule_risk": rules.get("engine_level"),
        "shown_claims": shown,
        "held_back": held,
        "unavailable": unavailable,
        "needs_human": needs_human,
        # Claims describe evidence. A concrete, validated proposal must come from the booking workflow.
        "awaiting_approval": False,
        "verifier_enabled": verification is not None,
    }


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
        yield self._event(ctx, intent=planned["intent"], trip_goal=goal, plan=plan,
                          pending_clarification=planned.get("pending_clarification"),
                          last_verification=None, specialist_results={}, evidence_store={},
                          rule_assessment=None,
                          **{f"result_{name}": None for name in SPECIALIST_NAMES})

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

        # 4. Alternatives, if the plan says so.
        signal_claims = results["flight_airport"]["claims"] + results["weather"]["claims"]
        if should_run_alternatives(plan, signal_claims, rules.get("risk")):
            async for event in self._sub("alternatives").run_async(ctx):
                yield _internal_event(event)
            results["alternatives"] = parse_agent_result(ctx.session.state.get("result_alternatives"), "alternatives")

        # 5. Verify against the evidence the tools really returned.
        verification = None
        if self.verifier_enabled:
            claims = [c for name in SPECIALIST_NAMES for c in results.get(name, {}).get("claims", [])]
            # Copy: the engine's evidence informs the Verifier's risk check but
            # stays out of the specialists' own evidence_store.
            view = {"evidence_store": dict(ctx.session.state.get("evidence_store") or {})}
            if rules.get("result"):
                add_tool_result(view, rules["result"])
            verification = verify_claims(claims, view["evidence_store"])
            if tracer:
                tracer.log_verdicts(verification["counts"], verification["needs_human"])

        # 6. Answer.
        answer = assemble_answer(planned["intent"], goal, results, verification, rule_view)
        if tracer:
            tracer.log("answer", overall_risk=answer["overall_risk"], shown=len(answer["shown_claims"]),
                       held_back=len(answer["held_back"]), needs_human=answer["needs_human"])
        yield self._event(ctx, answer["text"], specialist_results=results,
                          last_verification=verification, final_answer=answer)


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
    alternatives = overrides.get("alternatives") or build_placeholder_specialist("alternatives", model)
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
               "final_answer", "evidence_store", "rule_assessment", "guardrail_events", "trace_run_id")


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
    """
    from google.adk.runners import InMemoryRunner

    from travel_agent.tools import replay

    if scenario:
        replay.use_demo_scenario(scenario)
    tracer = Tracer(log_dir=log_dir)
    previous = get_active_tracer()
    set_active_tracer(tracer)
    try:
        root = build_orchestrator(model, specialists, verifier_enabled, planner, risk_engine)
        runner = InMemoryRunner(agent=root, app_name="travel_agent")
        session = await runner.session_service.create_session(
            app_name="travel_agent", user_id="user", state={"trace_run_id": tracer.run_id})
        message = types.Content(role="user", parts=[types.Part(text=request)])
        async for _event in runner.run_async(user_id="user", session_id=session.id, new_message=message):
            pass
        final = await runner.session_service.get_session(app_name="travel_agent", user_id="user", session_id=session.id)
        return {key: final.state.get(key) for key in RESULT_KEYS}
    finally:
        set_active_tracer(previous)
        if scenario:
            replay.use_demo_scenario(None)


def run_pipeline(request: str, **kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper around :func:`run_pipeline_async`.

    Works in scripts and in Jupyter/Colab (where an event loop is already
    running) by using a helper thread in the latter case.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_pipeline_async(request, **kwargs))
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["result"] = asyncio.run(run_pipeline_async(request, **kwargs))
        except BaseException as exc:  # re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]
