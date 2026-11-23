"""Root orchestrator agent (Reason stage, M2).

One user turn runs through these steps::

    screen ─► plan ─► ┌ flight_airport ┐ ─► alternatives? ─► verify ─► answer
                      └ weather        ┘    (conditional)

* **screen**: M1's deterministic guardrails run *before* the planner's Gemini
  call, so PII never reaches the model and injection/harmful requests are
  refused without spending tokens.
* **plan**: ``planner.plan_request`` classifies the intent, grounds slots and
  builds the sub-task graph (see planner.py). ``out_of_scope`` and missing slots
  end the turn here with a polite answer or a clarifying question.
* **flight_airport ‖ weather**: run concurrently (ADK ``ParallelAgent``). They
  are independent, and running them in parallel roughly halves wall time.
* **alternatives**: runs after both, because it needs their findings. For
  ``assess_trip`` it only runs when some claim is moderate/high risk, which
  saves a model call on calm days; for ``plan_trip``/``rebook`` it always runs.
* **verify**: every claim is checked against the evidence the tools actually
  returned (M1's ``verify_claims``). It is called directly rather than through
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
from travel_agent.agents.specialists import (
    DEFAULT_MODEL,
    SPECIALIST_NAMES,
    build_placeholder_specialist,
    parse_agent_result,
)
from travel_agent.guardrails.injection import detect_injection
from travel_agent.guardrails.pii import redact_pii
from travel_agent.guardrails.topics import check_topic
from travel_agent.observability.trace import Tracer, get_active_tracer, set_active_tracer
from travel_agent.verification.checks import verify_claims

RISK_ORDER = {"low": 0, "moderate": 1, "high": 2}
SECTION_TITLES = {"flight_airport": "Flight & airport", "weather": "Weather", "alternatives": "Options"}
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


def should_run_alternatives(plan: list[dict[str, Any]], signal_claims: list[dict[str, Any]]) -> bool:
    """Apply the plan's condition on the alternatives step."""
    task = next((t for t in plan if t.get("agent") == "alternatives"), None)
    if task is None:
        return False
    if not task.get("condition"):
        return True
    return RISK_ORDER.get(_max_risk(signal_claims) or "low", 0) >= RISK_ORDER["moderate"]


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
) -> dict[str, Any]:
    """Build the final answer from the specialists' results.

    With a verification: only ``supported`` claims are shown; the others are
    listed under "held back" with the Verifier's reason. Without one (ablation
    "Verifier OFF"): every claim is shown as-is.
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
    overall = _max_risk(signal_claims) or "unknown"
    needs_human = bool(held or errors or any(r.get("needs_human") for r in results.values())
                       or (verification or {}).get("needs_human"))

    lines = [f"Trip: {_trip_line(goal)}", f"Overall risk: {overall}"]
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
    if unavailable:
        lines.append("\nNot checked (data source unavailable): " + "; ".join(
            f"{SECTION_TITLES.get(name, name)}: {', '.join(tools)}" for name, tools in unavailable.items()))
    if held or errors:
        lines.append("\nNeeds your check (not confirmed by the evidence):")
        lines.extend(f"- {c['text']} — {c['status']}: {c['reason']}" for c in held)
        lines.extend(f"- {error}" for error in errors)
    if intent == "rebook":
        lines.append("\nConfirm a specific replacement with your airline before making a change.")
    return {
        "text": "\n".join(lines),
        "overall_risk": overall,
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
                          **{f"result_{name}": None for name in SPECIALIST_NAMES})

        if planned["intent"] == "out_of_scope" or planned["needs_clarification"]:
            reply = OUT_OF_SCOPE_TEXT if planned["intent"] == "out_of_scope" else clarify_text(goal)
            yield self._event(ctx, reply, final_answer={"text": reply, "needs_human": False})
            return

        # 3. Flight + weather in parallel; each writes state["result_<name>"].
        async for event in self._sub("signals").run_async(ctx):
            yield _internal_event(event)
        state = ctx.session.state
        results = {name: parse_agent_result(state.get(f"result_{name}"), name) for name in ("flight_airport", "weather")}

        # 4. Alternatives, if the plan says so.
        signal_claims = results["flight_airport"]["claims"] + results["weather"]["claims"]
        if should_run_alternatives(plan, signal_claims):
            async for event in self._sub("alternatives").run_async(ctx):
                yield _internal_event(event)
            results["alternatives"] = parse_agent_result(ctx.session.state.get("result_alternatives"), "alternatives")

        # 5. Verify against the evidence the tools really returned.
        verification = None
        if self.verifier_enabled:
            claims = [c for name in SPECIALIST_NAMES for c in results.get(name, {}).get("claims", [])]
            verification = verify_claims(claims, ctx.session.state.get("evidence_store") or {})
            if tracer:
                tracer.log_verdicts(verification["counts"], verification["needs_human"])

        # 6. Answer.
        answer = assemble_answer(planned["intent"], goal, results, verification)
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
) -> TravelOrchestrator:
    """Wire the root agent. ``specialists`` overrides any of the placeholders."""
    agents = {name: build_placeholder_specialist(name, model) for name in SPECIALIST_NAMES}
    agents.update(specialists or {})
    with warnings.catch_warnings():
        # ParallelAgent is marked deprecated in ADK 2.x in favour of Workflow,
        # but it is what the course uses and still fully supported.
        warnings.simplefilter("ignore", DeprecationWarning)
        signals = ParallelAgent(name="signals", sub_agents=[agents["flight_airport"], agents["weather"]])
    return TravelOrchestrator(
        name="orchestrator",
        description="Routes U.S. domestic trip-disruption requests to specialists.",
        sub_agents=[signals, agents["alternatives"]],
        verifier_enabled=verifier_enabled,
        planner=planner,
    )


# --------------------------------------------------------------------------
# One-call entry point (notebook, evaluation, M1's ablation)
# --------------------------------------------------------------------------

RESULT_KEYS = ("intent", "trip_goal", "plan", "specialist_results", "last_verification",
               "final_answer", "evidence_store", "guardrail_events", "trace_run_id")


async def run_pipeline_async(
    request: str,
    *,
    scenario: Optional[str] = None,
    verifier_enabled: bool = True,
    model: Any = DEFAULT_MODEL,
    specialists: Optional[dict[str, BaseAgent]] = None,
    planner: Optional[Callable[[str], dict]] = None,
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
        root = build_orchestrator(model, specialists, verifier_enabled, planner)
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
