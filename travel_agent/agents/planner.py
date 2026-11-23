"""Task breakdown for goal-driven requests (Reason stage, M2).

Pipeline for one user turn::

    user text ──► classify_intent (1 Gemini call)  ──► IntentResult (raw phrases)
              ──► ground()   (deterministic)         ──► TripGoal (airports, dates)
              ──► build_plan() (deterministic)       ──► list[SubTask]

Why the breakdown is code and not a second LLM call:

* **Grounding reduces hallucination.** Places and dates are resolved with the
  same tested resolver the rule engine uses (``core.engine.parse``), so the plan
  can only mention airports that exist and dates that the calendar agrees with.
  If something cannot be resolved, it is listed in ``missing`` and the plan
  becomes a single *clarify* step instead of a guess.
* **Plans are auditable.** Each intent maps to a fixed sub-task graph whose
  shape is unit-tested (e.g. the Verifier always runs before the answer is
  assembled; a rebooking always ends with human approval).
* **Cheap.** No extra tokens per turn.

The sub-task graph also encodes the routing decision the orchestrator follows:
``flight_airport`` and ``weather`` share a ``parallel_group`` because neither
needs the other's output; ``alternatives`` depends on both because whether and
how hard to look for alternatives depends on the risk they found; the Verifier
depends on every specialist because it must see every claim.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from travel_agent.agents.intent import CLOCK_PHRASE, DATE_PHRASE, Intent, IntentResult, classify_intent

# Slots each intent needs before specialists can do useful work.
REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    "assess_trip": ("origin", "destination", "departure_date"),
    "plan_trip": ("origin", "destination", "departure_date"),
    "rebook": ("origin", "destination", "departure_date"),
    "out_of_scope": (),
}

# Specialist names; they must match the ``AgentResult.agent`` values of M3/M4.
FLIGHT_AIRPORT = "flight_airport"
WEATHER = "weather"
ALTERNATIVES = "alternatives"
VERIFIER = "verifier"
SYNTHESIZER = "synthesizer"
HUMAN_APPROVAL = "human_approval"
CLARIFY = "clarify"


@dataclass
class TripGoal:
    """The user's goal with every slot resolved (or listed in ``missing``)."""

    intent: Intent
    request: str
    origin: Optional[str] = None
    destination: Optional[str] = None
    origin_airport: Optional[str] = None
    destination_airport: Optional[str] = None
    departure_date: Optional[str] = None  # ISO date, origin local time
    departure_window: str = "whole_day"
    arrive_by: Optional[str] = None  # ISO datetime, destination local time
    days_ahead: Optional[int] = None  # lets specialists pick a forecast source
    flight_number: Optional[str] = None
    constraints: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    classified_by: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubTask:
    """One step of the plan, addressed to one agent."""

    id: str
    agent: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    parallel_group: Optional[str] = None
    condition: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Grounding: raw phrases -> airports and dates
# --------------------------------------------------------------------------

_DEADLINE_PARTS = re.compile(
    rf"(?:(?P<day>{DATE_PHRASE})(?:\s+(?:at\s+)?(?P<clock>{CLOCK_PHRASE}))?|"
    rf"(?P<clock_only>{CLOCK_PHRASE}))", re.I,
)
_DEPARTURE_REPLY = re.compile(rf"(?P<day>{DATE_PHRASE})(?:\s+(?P<window>morning|afternoon|evening))?", re.I)
_PLACE_REPLY = re.compile(r"[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*(?:,\s*[A-Z]{2})?")


def _resolve_day(phrase: Optional[str], timezone: str, now: datetime) -> tuple[Optional[date], Optional[str]]:
    """Resolve a day phrase with the rule engine's resolver; returns (date, error)."""
    from travel_agent.core.engine.parse import _date_from_text  # tested in tests/core

    if not phrase:
        return None, None
    # "tonight" is common in user text but not in the shared resolver.
    phrase = re.sub(r"\btonight\b", "today", phrase, flags=re.I)
    try:
        day, _, error = _date_from_text(phrase, timezone, now)
    except OverflowError:
        return None, "date out of range"
    return day, error


def _resolve_clock(phrase: Optional[str]) -> time:
    if phrase is None:
        return time(23, 59)
    lowered = phrase.lower()
    if lowered == "noon":
        return time(12, 0)
    if lowered == "midnight":
        return time(0, 0)
    suffix = lowered[-2:]
    clock = lowered[:-2].strip() if suffix in {"am", "pm"} else lowered
    hours, _, minutes = clock.partition(":")
    hour = int(hours)
    if suffix in {"am", "pm"}:
        if not 1 <= hour <= 12:
            raise ValueError("12-hour clock hour out of range")
        hour = hour % 12 + (12 if suffix == "pm" else 0)
    return time(hour, int(minutes or 0))


def _resolve_deadline(phrase: str, departure: date | None, zone: ZoneInfo, now: datetime) -> datetime | None:
    from travel_agent.core.engine.parse import WEEKDAYS

    parts = _DEADLINE_PARTS.fullmatch(phrase.strip().rstrip(".?!"))
    if parts is None:
        return None
    day_phrase = parts["day"]
    day = departure
    if day_phrase:
        words = day_phrase.lower().split()
        weekday = WEEKDAYS.get(words[-1])
        if weekday is None:
            day = _resolve_day(day_phrase, zone.key, now)[0]
        else:
            today = now.astimezone(zone).date()
            delta = (weekday - today.weekday()) % 7
            day = today + timedelta(days=7 if words[0] == "next" and delta == 0 else delta)
    if day is None:
        return None
    try:
        clock = _resolve_clock(parts["clock"] or parts["clock_only"])
    except ValueError:
        return None
    return datetime.combine(day, clock, zone)


def _resolve_place(phrase: Optional[str]) -> tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Return (city, airport, timezone, error) using the shared place resolver."""
    from travel_agent.core.engine.parse import _resolve_place as resolve

    if not phrase:
        return None, None, "America/New_York", None
    try:
        city, airport, _latlon, timezone, error = resolve(phrase)
    except Exception as exc:  # geocoder outage must not crash routing
        return None, None, "America/New_York", f"could not resolve place: {type(exc).__name__}"
    return city, airport, timezone, error


def _own_airport(phrase: Optional[str], chosen: Optional[str]) -> Optional[str]:
    """Correct the metro pick when the user named a city with its own airport.

    The rule engine maps a city to the busiest airport pair of its *metro*
    area. That is right for "Chicago" (ORD vs MDW) but wrong for "Newark", which
    sits inside the New York metro and would become JFK. So: if the chosen
    airport is not in the named city, and the named city has exactly one
    airport of its own, use that one. ("Dallas" keeps DFW, whose municipality
    is "Dallas-Fort Worth".)
    """
    from travel_agent.core.reference import airports

    if not phrase or not chosen or airports.get(phrase.upper()):
        return None  # explicit IATA codes are already honoured by build_trip
    name = phrase.strip().lower()
    chosen_city = str((airports.get(chosen) or {}).get("municipality") or "").lower()
    if name in chosen_city:
        return None
    own = [a for a in airports._load()  # offline reference table, read-only
           if a.get("municipality", "").lower() == name and a.get("iata")]
    return str(own[0]["iata"]) if len(own) == 1 else None


def ground(result: IntentResult, request: str, now: datetime) -> TripGoal:
    """Turn the classifier's raw phrases into a checked ``TripGoal``."""
    from travel_agent.core.engine.parse import build_trip

    goal = TripGoal(
        intent=result.intent,
        request=request,
        departure_window=result.window,
        flight_number=result.flight_number,
        constraints=list(result.constraints),
        classified_by=result.source,
    )
    if result.intent == "out_of_scope":
        return goal

    origin_city, _, origin_tz, origin_error = _resolve_place(result.origin)
    dest_city, _, dest_tz, dest_error = _resolve_place(result.destination)
    for slot, phrase, error in (("origin", result.origin, origin_error), ("destination", result.destination, dest_error)):
        if not phrase:
            goal.missing.append(slot)
        elif error:
            goal.missing.append(slot)
            goal.notes.append(f"{slot} '{phrase}' is not a known U.S. place")

    departure, date_error = _resolve_day(result.date_text, origin_tz, now)
    # Deadline: day phrase resolved in the destination time zone, because
    # "be in SF by 6pm" means 6pm in San Francisco.
    # (skipped when the destination is unknown: its time zone would be a guess)
    arrive_by = None
    if result.deadline_text and dest_city and not dest_error:
        arrive_by = _resolve_deadline(result.deadline_text, departure, ZoneInfo(dest_tz), now)
        if arrive_by is None:
            goal.notes.append("the arrival deadline needs a valid date and time")
            goal.missing.append("arrive_by")
        elif arrive_by < now:
            goal.notes.append("the arrival deadline is already in the past")
            goal.missing.append("arrive_by")
        else:
            goal.arrive_by = arrive_by.isoformat(timespec="minutes")

    # Departure day: explicit phrase first, otherwise the deadline day.
    if departure is None and not result.date_text and goal.arrive_by and arrive_by is not None:
        departure = arrive_by.astimezone(ZoneInfo(origin_tz)).date()
    if departure is None:
        goal.missing.append("departure_date")
        if date_error and result.date_text:
            goal.notes.append(f"could not understand the date '{result.date_text}'")

    goal.origin, goal.destination = origin_city, dest_city
    if departure is not None:
        goal.departure_date = departure.isoformat()
        goal.days_ahead = (departure - now.astimezone(ZoneInfo(origin_tz)).date()).days

    # Pick the metro airports with the rule engine's logic (busiest BTS pair).
    if not goal.missing:
        trip = build_trip(result.origin, result.destination, departure, goal.departure_window, now)
        if isinstance(trip, dict):
            goal.notes.append(trip.get("error", "could not build the trip"))
            if "past" in trip.get("error", ""):
                goal.missing.append("departure_date")
            else:
                goal.missing.append("origin" if not origin_city else "destination")
        else:
            goal.origin_airport = _own_airport(result.origin, trip.origin_airport) or trip.origin_airport
            goal.destination_airport = (_own_airport(result.destination, trip.destination_airport)
                                        or trip.destination_airport)
            if arrive_by is not None:
                earliest = datetime.combine(departure, time(trip.window_hours[0]), ZoneInfo(origin_tz))
                if earliest.astimezone(timezone.utc) >= arrive_by.astimezone(timezone.utc):
                    goal.notes.append("the departure window starts at or after the arrival deadline")
                    goal.missing.append("departure_date")

    # Keep only slots this intent actually needs, deduplicated, stable order.
    needed = set(REQUIRED_SLOTS[goal.intent]) | {"arrive_by"}
    goal.missing = [slot for slot in dict.fromkeys(goal.missing) if slot in needed]
    return goal


# --------------------------------------------------------------------------
# Plan: TripGoal -> sub-task graph
# --------------------------------------------------------------------------


def _trip_label(goal: TripGoal) -> str:
    origin = goal.origin_airport or goal.origin
    destination = goal.destination_airport or goal.destination
    when = goal.departure_date + ("" if goal.departure_window == "whole_day" else f" {goal.departure_window}")
    return f"{origin}→{destination} on {when}"


def build_plan(goal: TripGoal) -> list[SubTask]:
    """Return the ordered sub-tasks for ``goal`` (see module docstring)."""
    if goal.intent == "out_of_scope":
        return [SubTask("t1", SYNTHESIZER, "Explain that the assistant only assesses U.S. domestic trip "
                        "disruption risk and give an example request.")]
    if goal.missing:
        return [SubTask("t1", CLARIFY, "Ask the user for: " + ", ".join(goal.missing)
                        + (". " + "; ".join(goal.notes) if goal.notes else "."))]

    trip = _trip_label(goal)
    flight_goal = f"Check FAA delay programs, airport weather (METAR/TAF) and on-time history for {trip}."
    if goal.intent == "rebook" and goal.flight_number:
        flight_goal = f"Assess the booked flight {goal.flight_number} ({trip}): FAA programs, airport weather, on-time history."
    plan = [
        SubTask("t1", FLIGHT_AIRPORT, flight_goal, parallel_group="signals"),
        SubTask("t2", WEATHER, f"Check NWS alerts and the forecast at both ends of {trip}.", parallel_group="signals"),
    ]

    if goal.intent == "assess_trip":
        plan.append(SubTask("t3", ALTERNATIVES, f"Suggest other airports, rail or driving for {trip}.",
                            depends_on=["t1", "t2"], condition="only if any claim has risk moderate or high"))
    else:
        deadline = f" arriving before {goal.arrive_by}" if goal.arrive_by else ""
        prefs = f" Respect: {', '.join(goal.constraints)}." if goal.constraints else ""
        verb = "Propose replacement options" if goal.intent == "rebook" else "Compare flying, rail and driving options"
        plan.append(SubTask("t3", ALTERNATIVES, f"{verb} for {trip}{deadline}; rank by risk.{prefs}",
                            depends_on=["t1", "t2"]))

    plan.append(SubTask("t4", VERIFIER, "Check every specialist claim against the cited evidence.",
                        depends_on=["t1", "t2", "t3"]))
    plan.append(SubTask("t5", SYNTHESIZER, "Write the answer from supported claims only; flag anything a human must check.",
                        depends_on=["t4"]))
    if goal.intent == "rebook":
        # The proposal is shown first; only the human can trigger the mock booking.
        plan.append(SubTask("t6", HUMAN_APPROVAL, "Human approves or rejects the proposed rebooking. "
                            "Nothing is booked without approval.", depends_on=["t5"]))
    return plan


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _now() -> datetime:
    """Use the replay scenario's frozen clock in demos, the real clock otherwise."""
    from travel_agent.tools import replay

    return replay.demo_now() or datetime.now().astimezone()


def plan_request(text: str, now: Optional[datetime] = None, *, client: Any = None) -> dict[str, Any]:
    """Classify, ground and plan one user request.

    Returns ``{"intent", "trip_goal", "plan", "needs_clarification"}`` with plain
    dicts, ready to store in ADK state or print in the notebook.
    """
    now = now or _now()
    return _plan_result(classify_intent(text, now, client=client), text, now)


def _plan_result(result: IntentResult, text: str, now: datetime) -> dict[str, Any]:
    goal = ground(result, text, now)
    plan = build_plan(goal)
    return {
        "intent": goal.intent,
        "trip_goal": goal.to_dict(),
        "plan": [task.to_dict() for task in plan],
        "needs_clarification": bool(goal.missing) and goal.intent != "out_of_scope",
        "pending_clarification": {"intent": result.to_dict(), "request": text, "missing": goal.missing}
        if goal.missing else None,
    }


def resume_clarification(text: str, pending: Any) -> dict[str, Any] | None:
    """Apply a short answer to a missing slot. A full request is planned afresh."""
    if not isinstance(pending, dict) or not pending.get("missing"):
        return None
    missing = pending["missing"]
    result = IntentResult(**pending["intent"])
    reply = text.strip().rstrip(".?!")
    departure = _DEPARTURE_REPLY.fullmatch(reply)
    if "departure_date" in missing and departure:
        window = departure["window"] or ("evening" if reply.lower() == "tonight" else result.window)
        result = replace(result, date_text=departure["day"], window=window.lower())
    elif "arrive_by" in missing and _DEADLINE_PARTS.fullmatch(reply):
        result = replace(result, deadline_text=reply)
    elif missing[0] in {"origin", "destination"} and _PLACE_REPLY.fullmatch(reply):
        result = replace(result, **{missing[0]: reply})
    else:
        return None
    return _plan_result(result, pending["request"], _now())


def plan_request_tool(request: str, tool_context: Any) -> dict[str, Any]:
    """Understand the user's travel request and break it into sub-tasks.

    Call this first, once per user request, with the user's message verbatim.
    Returns the intent (assess_trip, plan_trip, rebook or out_of_scope), the
    resolved trip goal (airports, dates, deadline, constraints, missing slots)
    and the ordered sub-task plan. If ``needs_clarification`` is true, ask the
    user for the missing details instead of calling specialists.
    """
    result = plan_request(request)
    state = getattr(tool_context, "state", None)
    if state is not None:
        # State keys documented in docs/INTERFACES.md ("Orchestrator keys").
        state["intent"] = result["intent"]
        state["trip_goal"] = result["trip_goal"]
        state["plan"] = result["plan"]
    return result
