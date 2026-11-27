"""Glue between the orchestrator (M2) and M4's Alternatives agent + HITL.

Three jobs, all deterministic:

* **Context in.** M4's prompt asks the model to call
  ``plan_trip_options(origin, destination, deadline, earliest_departure)``.
  The planner has already resolved those values, so they are appended to the
  prompt instead of letting the model re-read (or invent) them. A trip with no
  stated deadline gets an explicit *assumed* deadline: the latest arrival of a
  flight inside the requested departure window.
* **Result out.** The fields the approval card needs (recommended option,
  Plan B, switch trigger, ...) are kept next to the parsed claims.
* **Decisions.** A short reply to a pending proposal ("approve", "reject: too
  early", "why?", "take the train instead") is classified with simple rules;
  anything else is treated as a new request.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional

from google.adk.agents import LlmAgent

from travel_agent.agents.alternatives import (
    OUTPUT_KEY as M4_OUTPUT_KEY,
    build_alternatives_agent,
)
from travel_agent.agents.alternatives import parse_agent_result as parse_m4_result
from travel_agent.agents.perceive import airport_windows
from travel_agent.tools import replay

# Fields of M4's answer that ``hitl.build_proposal`` reads, beyond AgentResult.
PROPOSAL_FIELDS = ("recommended_option_id", "plan_b_option_id", "switch_trigger", "decide_by",
                   "lessons_used", "preferences_used", "summary")
RESULT_KEYS = ("result_alternatives", M4_OUTPUT_KEY)  # M2 contract first, then M4's own key


def raw_alternatives(state: Any) -> Any:
    """The alternatives agent's final message, whichever output key it used."""
    for key in RESULT_KEYS:
        value = state.get(key)
        if value:
            return value
    return None


def _local(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def deadline_inputs(goal: dict[str, Any], now: Optional[datetime] = None) -> dict[str, Any]:
    """``deadline`` (destination local) and ``earliest_departure`` (origin local).

    ``assumed`` is True when the user gave no arrival deadline.
    """
    windows = airport_windows(goal)
    origin, destination = goal.get("origin_airport"), goal.get("destination_airport")
    if not windows or origin not in windows or destination not in windows:
        return {}
    departure_start = windows[origin][0]
    now = now or replay.demo_now() or datetime.now(departure_start.tzinfo)
    earliest = max(departure_start, now.astimezone(departure_start.tzinfo))
    if goal.get("arrive_by"):
        deadline = datetime.fromisoformat(goal["arrive_by"])
        assumed = False
    else:
        deadline = windows[destination][1]  # last flight in the window, landed
        assumed = True
    return {"deadline": _local(deadline), "earliest_departure": _local(earliest), "assumed": assumed}


def alternatives_context(state: Any) -> str:
    """Trip facts appended to M4's prompt (all values come from code, not from the model)."""
    goal = state.get("trip_goal") or {}
    task = next((t.get("goal") for t in state.get("plan") or [] if t.get("agent") == "alternatives"), "")
    times = deadline_inputs(goal)
    lines = ["", "Trip resolved by the planner (use these values exactly, do not re-interpret the request):",
             f"- Your sub-task: {task}",
             f"- origin: {goal.get('origin_airport')}  destination: {goal.get('destination_airport')}"]
    if times:
        lines.append(f"- Call plan_trip_options with deadline=\"{times['deadline']}\" and "
                     f"earliest_departure=\"{times['earliest_departure']}\".")
        if times["assumed"]:
            lines.append("- The traveller gave no arrival deadline. This deadline is ASSUMED (the latest arrival "
                         "of a flight in their departure window); say so in the summary.")
    if goal.get("flight_number"):
        lines.append(f"- Booked flight: {goal['flight_number']}")
    if goal.get("constraints"):
        lines.append(f"- Traveller constraints: {', '.join(goal['constraints'])}")
    lines += [f"- Findings so far (flight_airport): {state.get('result_flight_airport') or ''}",
              f"- Findings so far (weather): {state.get('result_weather') or ''}",
              f"- Rule-engine trip assessment: {json.dumps(state.get('rule_assessment'))}"]
    if state.get("change_request"):
        lines.append(f"- {state['change_request']}")
    return "\n".join(lines) + "\n"


def build_m4_alternatives(model: Any) -> LlmAgent:
    """M4's agent with the planner's trip context appended to its own prompt."""
    agent = build_alternatives_agent(model)
    base = agent.instruction

    def instruction(ctx: Any) -> str:
        return base(ctx) + alternatives_context(ctx.state)

    agent.instruction = instruction
    return agent


def proposal_fields(raw: Any) -> dict[str, Any]:
    """The approval-card fields of M4's answer (empty when it has none)."""
    if not raw:
        return {}
    try:
        payload = parse_m4_result(raw)
    except Exception:
        return {}
    return {key: payload[key] for key in PROPOSAL_FIELDS if payload.get(key) is not None}


# --------------------------------------------------------------------------
# Replies to a pending proposal
# --------------------------------------------------------------------------

_APPROVE = re.compile(r"^(?:i\s+)?(?:approve|approved|yes|yep|ok(?:ay)?|confirm|go ahead|book it|do it)\b", re.I)
_ACKNOWLEDGE = re.compile(r"\b(?:anyway|despite|accept(?:ing)? the warnings?|acknowledge[ds]?|i understand)\b", re.I)
_REJECT = re.compile(r"^(?:reject|rejected|no|nope|don'?t|do not|cancel)\b[\s:,.-]*", re.I)
_EXPLAIN = re.compile(r"^(?:why|explain|how come)\b", re.I)
_CHANGE = re.compile(r"^(?:change|instead|what about|how about|can you|could you|i'?d rather|i prefer|prefer|avoid|"
                     r"take|try|show me)\b", re.I)


def parse_decision(text: str) -> Optional[dict[str, Any]]:
    """Classify a reply to the approval card; ``None`` means a new request."""
    reply = (text or "").strip()
    if not reply or len(reply) > 300:
        return None
    if _APPROVE.match(reply):
        return {"kind": "approve", "acknowledge": bool(_ACKNOWLEDGE.search(reply))}
    if _EXPLAIN.match(reply):
        return {"kind": "explain"}
    reject = _REJECT.match(reply)
    if reject:
        return {"kind": "reject", "reason": reply[reject.end():].strip()}
    if _CHANGE.match(reply):
        return {"kind": "change", "request": reply}
    return None
