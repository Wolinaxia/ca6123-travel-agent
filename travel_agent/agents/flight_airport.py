"""Flight & Airport specialist for observed status, forecasts and route history."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import BaseLlm
from google.adk.tools.tool_context import ToolContext

from travel_agent.agents.perceive import filter_tool_result
from travel_agent.agents.specialists import guarded_callbacks, parse_agent_result
from travel_agent.schemas import ToolResult
from travel_agent.tools import live


_NYC_DEPARTURES = {"JFK", "EWR", "LGA"}
_GAP_PREFIX = "temp:flight_airport_gap:"


def _departure_goal(goal: dict, airport: str) -> dict | None:
    origin = goal["origin_airport"]
    if airport == origin:
        return goal
    if origin in _NYC_DEPARTURES and airport in _NYC_DEPARTURES and airport != goal["destination_airport"]:
        return {**goal, "origin_airport": airport}
    return None


def _airport_result(provider: Callable[[str], dict], source: str, airport: str, ctx: ToolContext) -> dict:
    goal = ctx.state["trip_goal"]
    code = airport.strip().upper()
    candidate = goal if code == goal["destination_airport"] else _departure_goal(goal, code)
    if candidate is None:
        return ToolResult(source, ok=False, error=f"{code} is outside the planned route and departure options").to_dict()
    result = filter_tool_result(provider(code), candidate)
    if result["ok"]:
        evidence = result["evidence"]
        if source == "faa_nas" and not any(
            item["subject"] == code and item["data"].get("affects_commercial") is not False
            for item in evidence
        ):
            result["coverage_gap"] = f"{code} FAA: no status evidence for trip window"
        if source == "aviation_wx" and not any(item["id"].endswith(":metar") for item in evidence):
            result["coverage_gap"] = f"{code} METAR: no observation for trip window"
    # Separate keys survive parallel calls. A successful retry replaces its own gap.
    if result["ok"] or result.get("coverage_gap"):
        ctx.state[f"{_GAP_PREFIX}{code}:{source}"] = result.get("coverage_gap")
    return result


def _include_coverage_gaps(callback_context: CallbackContext) -> None:
    """Merge tool-detected gaps into the guarded output before downstream agents read it."""
    state = callback_context.state
    gaps = [state[key] for key in state if key.startswith(_GAP_PREFIX) and state[key]]
    if not gaps:
        return
    result = parse_agent_result(state.get("result_flight_airport"), "flight_airport")
    for claim in result["claims"]:
        claim.pop("agent")  # M2 adds this grouping field when it reads the Claim contract.
    gaps.extend(result.get("unavailable", []))
    if "error" in result:
        gaps.append(result["error"])  # Keep the parser diagnostic visible with the tool gaps.
    result["unavailable"] = list(dict.fromkeys(gaps))
    state["result_flight_airport"] = json.dumps(result)


def _instruction(ctx: ReadonlyContext) -> str:
    trip = ctx.state["trip_goal"]
    task = next(task for task in ctx.state["plan"] if task["agent"] == "flight_airport")
    return (
        f"You are the Flight & Airport specialist for {trip['origin_airport']} to "
        f"{trip['destination_airport']} on {trip['departure_date']} ({trip.get('departure_window', 'whole_day')}). "
        f"Your assigned task is: {task['goal']}\n"
        "Use get_airport_delays and get_aviation_weather at both endpoints and "
        "get_route_ontime_history for the departure airport. Quote snapshot status as of "
        "as_of_local. FAA averages describe the program at that time; its continued operation "
        "at departure or arrival is unknown without a validity interval. For METAR use the "
        "station observation time. Use TAF only for its stated forecast period. "
        "Report any tool coverage_gap in unavailable. An empty FAA result means the "
        "status for the trip window is unknown. Only report no active FAA program when "
        "an applicable FAA record explicitly says so. "
        "For BTS report the directional route, actual basis (year-month or annual average), "
        "sample size and observed rates as historical context. Do not predict a flight's outcome. "
        "For a requested NYC departure comparison, check JFK, EWR and LGA separately to "
        "the same destination and for the same month. Cite each airport's own FAA or aviation "
        "record and each directional route's own BTS record. Distinguish annual from monthly "
        "samples. Report any missing BTS route and month explicitly under unavailable; "
        "do not assign it a zero rate, assume no flights or rank it using another route's data. "
        "Return AgentResult JSON with agent=flight_airport, claims, and boolean needs_human. "
        "Each Claim has text, exact evidence_ids, confidence (low, medium or high), and "
        "risk (null for factual observations). Label simulated results as mock. If a source "
        "is unavailable, name the tool, route or airport and reason in the optional unavailable "
        "list. Preserve supported claims and set needs_human if the decision lacks essential evidence."
    )


def build_flight_airport_agent(
    model: str | BaseLlm = "gemini-3.5-flash-lite",
    *,
    faa_tool: Callable[[str], dict] = live.get_faa_airport_status,
    aviation_tool: Callable[[str], dict] = live.get_aviation_weather,
    history_tool: Callable[[str, str, int], dict] = live.get_route_ontime_history,
) -> LlmAgent:
    """Inject data providers below the same Perceive and M1 callbacks used in production."""
    def get_airport_delays(airport: str, tool_context: ToolContext) -> dict:
        """Check FAA status at an itinerary airport."""
        return _airport_result(faa_tool, "faa_nas", airport, tool_context)

    def get_aviation_weather(airport: str, tool_context: ToolContext) -> dict:
        """Check observed METAR and applicable TAF periods at an itinerary airport."""
        return _airport_result(aviation_tool, "aviation_wx", airport, tool_context)

    def get_route_ontime_history(origin_airport: str, tool_context: ToolContext) -> dict:
        """Look up BTS history to the trip destination for the planned departure month."""
        origin = origin_airport.strip().upper()
        goal = _departure_goal(tool_context.state["trip_goal"], origin)
        if goal is None:
            return ToolResult("bts", ok=False, error=f"{origin} is outside the planned departure options").to_dict()
        month = date.fromisoformat(goal["departure_date"]).month
        result = history_tool(origin, goal["destination_airport"], month)
        return filter_tool_result(result, goal)

    return LlmAgent(
        name="flight_airport",
        model=model,
        instruction=_instruction,
        tools=[get_airport_delays, get_aviation_weather, get_route_ontime_history],
        output_key="result_flight_airport",
        after_agent_callback=_include_coverage_gaps,
        **guarded_callbacks(),
    )
