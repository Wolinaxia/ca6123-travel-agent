"""Weather Agent acceptance tests with the recorded JFK/SFO weather and a scripted model."""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone

import pytest
from google import genai
from google.adk.runners import InMemoryRunner

from tests.test_adk_wiring import ScriptedModel, _call_response, _run, _text_response
from travel_agent.agents.specialists import parse_agent_result
from travel_agent.agents.weather import _instruction as weather_instruction
from travel_agent.agents.weather import _missing_forecast_intervals, build_weather_agent
from travel_agent.core.models import NwsAlert, SourceResult
from travel_agent.guardrails.callbacks import assert_guarded
from travel_agent.schemas import AgentResult, Claim, ToolResult
from travel_agent.tools import live, replay
from travel_agent.verification.checks import verify_claims

ALERT = "get_weather_alerts"
FORECAST = "get_weather_forecast"
DAILY = "get_extended_forecast"
JFK_CLEAR = "nws:JFK:no_alerts"
SFO_PREFIX = "nws:SFO:https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.7c3009fb0ba23631e2ec9388cd109962b0cd83b6.001."
SFO_WARNING = SFO_PREFIX + "1:alert:0"
SFO_EXPIRED = SFO_PREFIX + "2:alert:1"
JFK_PERIOD = "nws_forecast:JFK:period:6"
SFO_PERIOD = "nws_forecast:SFO:period:6"
NEAR = [(ALERT, "JFK"), (FORECAST, "JFK"), (ALERT, "SFO"), (FORECAST, "SFO")]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def unexpected_client(*args, **kwargs):
        pytest.fail("A Weather test attempted to create a Gemini client")

    monkeypatch.setenv("GUARDRAIL_LLM", "0")
    monkeypatch.setattr(genai.Client, "__init__", unexpected_client)
    replay.use_demo_scenario("calm_baseline_20261121")
    try:
        yield
    finally:
        replay.use_demo_scenario(None)


def run_weather(calls, claims, *, day="2026-11-23", days_ahead=2, **providers):
    goal = {"origin_airport": "JFK", "destination_airport": "SFO", "departure_window": "evening",
            "departure_date": day, "days_ahead": days_ahead}
    task = "Check both airport windows and disclose coverage gaps."
    # Leave all gap reporting to the implementation, including essential escalation.
    reply = AgentResult(agent="weather", claims=claims)
    model = ScriptedModel(script=[
        *[_call_response(name, {"airport": airport}) for name, airport in calls],
        _text_response(json.dumps(asdict(reply))),
    ])
    agent = build_weather_agent(model, **providers)
    assert_guarded(agent)
    runner = InMemoryRunner(agent=agent, app_name="test")

    async def scenario():
        session = await runner.session_service.create_session(
            app_name="test", user_id="u",
            state={"trip_goal": goal, "plan": [{"agent": "flight_airport", "goal": "Inspect FAA."},
                                               {"agent": "weather", "goal": task}]},
        )
        await _run(runner, session, "Assess this trip.")
        fresh = await runner.session_service.get_session(app_name="test", user_id="u", session_id=session.id)
        return fresh.state

    state = asyncio.run(scenario())
    assert len(model.seen) == len(calls) + 1
    instruction = model.seen[0].model_dump_json()
    assert task in instruction and day in instruction and "Inspect FAA." not in instruction
    responses = {}
    for (name, airport), request in zip(calls, model.seen[1:]):
        parts = [p.function_response for c in request.contents for p in c.parts if p.function_response]
        assert parts[-1].name == name
        responses[name, airport] = parts[-1].response
    visible = {e["id"]: e for response in responses.values() for e in response["evidence"]}
    store = state.get("evidence_store", {})
    assert {key: record["evidence"] for key, record in store.items() if record["tool_ok"]} == visible
    output = json.loads(state["result_weather"])
    assert output["agent"] == "weather"
    assert output["claims"] == [asdict(claim) for claim in claims]
    verdicts = verify_claims(output["claims"], store)["verdicts"]
    assert [v["status"] for v in verdicts] == ["supported"] * len(claims), verdicts
    return output, visible, responses, model.seen[-1].model_dump_json()


def test_nws_both_endpoints_and_zero_alert_snapshot():
    claims = [
        Claim("At 02:18 EST on November 21, JFK had no active weather alerts.", [JFK_CLEAR], "high"),
        Claim("JFK is forecast to be mostly clear during departure.", [JFK_PERIOD], "high"),
        Claim("A coastal flood warning overlaps the SFO arrival window.", [SFO_WARNING], "high"),
        Claim("The SFO forecast includes a chance of light rain during arrival.", [SFO_PERIOD], "high"),
    ]
    output, visible, _, request = run_weather(NEAR, claims)
    assert set(visible) == {JFK_CLEAR, JFK_PERIOD, SFO_WARNING, SFO_PERIOD}
    assert SFO_EXPIRED not in request and "Coastal Flood Advisory" not in request
    assert visible[JFK_CLEAR]["data"]["as_of_local"] == "2026-11-21T02:18:58.336764-05:00"
    assert visible[JFK_CLEAR]["data"]["temporal_scope"] == "snapshot"
    assert not output.get("unavailable") and not output["needs_human"]


def test_daily_forecasts_follow_each_local_date():
    jfk = "open_meteo:JFK:day:2026-11-29"
    sfo_day = "open_meteo:SFO:day:2026-11-29"
    sfo_next = "open_meteo:SFO:day:2026-11-30"
    claims = [
        Claim("The JFK daily forecast for November 29 is overcast.", [jfk], "high"),
        Claim("The SFO daily forecasts indicate rain on November 29 and overcast skies "
              "on November 30.", [sfo_day, sfo_next], "high"),
    ]
    output, visible, _, request = run_weather(
        [(DAILY, "JFK"), (DAILY, "SFO")], claims, day="2026-11-29", days_ahead=8,
    )
    assert set(visible) == {jfk, sfo_day, sfo_next}
    assert all(e["data"]["temporal_scope"] == "daily" for e in visible.values())
    assert visible[sfo_next]["data"]["date"] == "2026-11-30"
    assert "open_meteo:JFK:day:2026-11-30" not in request   # JFK window ends exactly at midnight
    assert "open_meteo:SFO:day:2026-11-28" not in request
    assert not output.get("unavailable") and not output["needs_human"]


def test_failure_and_empty_window_preserve_other_claims():
    def forecasts(airport):
        if airport == "SFO":
            return ToolResult("nws_forecast", ok=False, error="forecast fixture offline").to_dict()
        return live.get_weather_forecast(airport)

    def alerts(airport):
        result = live.get_weather_alerts(airport)
        if airport == "SFO":
            # Include a valid record for the other itinerary endpoint and an expired SFO alert.
            wrong = dict(live.get_weather_alerts("JFK")["evidence"][0], id="nws:wrong-endpoint",
                         observation="Wrong endpoint fixture")
            result["evidence"] = [wrong, result["evidence"][1]]
        return result

    output, visible, responses, request = run_weather(
        NEAR, [Claim("JFK is forecast to be mostly clear during departure.", [JFK_PERIOD], "high")],
        forecast_tool=forecasts, alerts_tool=alerts,
    )
    assert set(visible) == {JFK_PERIOD, JFK_CLEAR}
    assert all(text not in request for text in ("nws:wrong-endpoint", "Wrong endpoint fixture", SFO_EXPIRED))
    empty, failed = responses[ALERT, "SFO"], responses[FORECAST, "SFO"]
    assert empty["ok"] and not empty["evidence"] and "no applicable evidence" in empty["coverage_gap"]
    assert not failed["ok"] and "forecast fixture offline" in failed["coverage_gap"]
    assert set(output["unavailable"]) == {empty["coverage_gap"], failed["coverage_gap"]}
    assert output["needs_human"]


def test_optional_alert_gap_preserves_forecast_decision():
    def alerts(airport):
        if airport == "JFK":
            return ToolResult("nws", ok=False, error="alert feed unavailable").to_dict()
        return live.get_weather_alerts(airport)

    output, _, responses, _ = run_weather(
        NEAR, [Claim("The SFO forecast includes a chance of light rain.", [SFO_PERIOD], "high")],
        alerts_tool=alerts,
    )
    assert output["unavailable"] == [responses[ALERT, "JFK"]["coverage_gap"]]
    assert not output["needs_human"]


def test_beyond_horizon_does_not_call_a_provider():
    def unexpected(airport):
        pytest.fail("A weather source was queried beyond its forecast horizon")

    output, visible, responses, _ = run_weather(
        [(DAILY, "SFO")], [], day="2026-12-10", days_ahead=19, extended_tool=unexpected,
    )
    assert not visible and not responses[DAILY, "SFO"]["ok"]
    assert output["needs_human"] and len(output["unavailable"]) == 2


def test_horizon_boundary_uses_airport_local_query_date():
    # The frozen clock reads Nov 21 in New York and Nov 20 in San Francisco.
    # For a Nov 28 departure, JFK is day 7 while SFO is already day 8.
    calls = [(ALERT, "JFK"), (FORECAST, "JFK"), (DAILY, "SFO")]
    output, visible, responses, _ = run_weather(calls, [], day="2026-11-28", days_ahead=7)
    assert responses[FORECAST, "JFK"]["ok"] and responses[DAILY, "SFO"]["ok"]
    assert "open_meteo:SFO:day:2026-11-28" in visible
    assert "open_meteo:SFO:day:2026-11-29" in visible
    # The NWS feed ends before the requested JFK departure. That gap remains
    # explicit even though the horizon qualified it for a lookup.
    assert responses[FORECAST, "JFK"]["coverage_gap"] in output["unavailable"]
    assert output["needs_human"]


def test_missing_sfo_daily_forecast_after_midnight():
    def daily(airport):
        result = live.get_extended_forecast(airport)
        if airport == "SFO":
            result["evidence"] = [e for e in result["evidence"] if e["data"]["date"] != "2026-11-30"]
        return result

    jfk = "open_meteo:JFK:day:2026-11-29"
    sfo = "open_meteo:SFO:day:2026-11-29"
    claims = [Claim("The JFK daily forecast for November 29 is overcast.", [jfk], "high"),
              Claim("The SFO daily forecast for November 29 indicates rain.", [sfo], "high")]
    output, visible, _, request = run_weather(
        [(DAILY, "JFK"), (DAILY, "SFO")], claims, day="2026-11-29", days_ahead=8, extended_tool=daily,
    )
    assert set(visible) == {jfk, sfo}
    assert "open_meteo:SFO:day:2026-11-30" not in request
    assert output["needs_human"]
    gap, = output["unavailable"]
    assert "SFO" in gap and "2026-11-30" in gap and "JFK" not in gap


def test_same_nws_alert_at_both_airports(monkeypatch):
    alert = NwsAlert(id="shared-wind", event="Wind Advisory", severity="Moderate",
                     onset=datetime(2026, 11, 23, tzinfo=timezone.utc),
                     ends=datetime(2026, 11, 25, tzinfo=timezone.utc))
    feed = SourceResult(source="nws", ok=True, data=[alert], fetched_at=replay.demo_now(), url="fixture://alerts")
    monkeypatch.setattr(live.nws, "alerts", lambda lat, lon: feed)
    ids = {code: f"nws:{code}:shared-wind:alert:0" for code in ("JFK", "SFO")}
    claims = [Claim(f"A wind advisory overlaps the {code} window.", [eid], "high") for code, eid in ids.items()]
    output, visible, responses, _ = run_weather(NEAR, claims)
    assert not output["needs_human"]
    for code, eid in ids.items():
        record, = responses[ALERT, code]["evidence"]
        assert record["id"] == eid and visible[eid]["subject"] == code
        assert record["data"]["nws_alert_id"] == alert.id


def test_numeric_confidence_from_the_model_keeps_weather_claims():
    goal = {"origin_airport": "JFK", "destination_airport": "SFO", "departure_window": "evening",
            "departure_date": "2026-11-23", "days_ahead": 2}
    task = "Check both airport windows and disclose coverage gaps."
    reply = {"agent": "weather", "claims": [
        {"text": "At 02:18 EST on November 21, JFK had no active weather alerts.",
         "evidence_ids": [JFK_CLEAR], "confidence": 1.0},
    ]}
    model = ScriptedModel(script=[
        *[_call_response(name, {"airport": airport}) for name, airport in NEAR],
        _text_response(json.dumps(reply)),
    ])
    agent = build_weather_agent(model)
    assert_guarded(agent)
    runner = InMemoryRunner(agent=agent, app_name="test")

    async def scenario():
        session = await runner.session_service.create_session(
            app_name="test", user_id="u",
            state={"trip_goal": goal, "plan": [{"agent": "flight_airport", "goal": "Inspect FAA."},
                                               {"agent": "weather", "goal": task}]},
        )
        await _run(runner, session, "Assess this trip.")
        fresh = await runner.session_service.get_session(app_name="test", user_id="u", session_id=session.id)
        return fresh.state

    state = asyncio.run(scenario())
    output = parse_agent_result(state["result_weather"], "weather")
    assert "error" not in output
    assert output["claims"] == [{"text": reply["claims"][0]["text"], "evidence_ids": [JFK_CLEAR],
                                 "confidence": "high", "risk": None, "agent": "weather"}]
    instruction = model.seen[0].model_dump_json()
    assert all(word in instruction for word in ("low", "medium", "high"))


@pytest.mark.parametrize("second_start, expected_gap", [
    ("22:00", []),   # Overlap.
    ("22:30", []),   # Touching endpoints.
    ("23:00", [("22:30", "23:00")]),
])
def test_nws_periods_and_daily_forecast_cover_a_window_together(second_start, expected_gap):
    def local(day, clock):
        return datetime.fromisoformat(f"2026-11-{day}T{clock}:00-08:00")

    window = (local(29, "21:30"), local(30, "03:30"))
    # Reverse chronological order. Only the daily record covers the part after midnight.
    forecasts = [
        {"source": "open_meteo", "data": {"date": "2026-11-30"}},
        {"source": "nws_forecast", "valid_from": local(29, second_start).isoformat(),
         "valid_to": local(30, "00:00").isoformat()},
        {"source": "nws_forecast", "valid_from": local(29, "18:00").isoformat(),
         "valid_to": local(29, "22:30").isoformat()},
    ]
    missing = _missing_forecast_intervals(window, forecasts)
    assert missing == [(local(29, start), local(29, end)) for start, end in expected_gap]


class _PromptContext:
    def __init__(self, state):
        self.state = state


def test_prompt_names_the_evidence_key_literally():
    instruction = weather_instruction(_PromptContext({
        "trip_goal": {"origin_airport": "JFK", "destination_airport": "SFO",
                      "departure_date": "2026-11-23", "days_ahead": 2},
        "plan": [{"agent": "weather", "goal": "Check both airport windows."}],
    }))
    assert '"evidence_ids"' in instruction
    assert "exact evidence_ids" not in instruction
