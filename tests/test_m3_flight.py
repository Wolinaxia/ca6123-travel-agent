"""Exercise Flight specialist tool responses through the ADK runner and M1 Verifier."""

import asyncio
import json
from dataclasses import asdict

import pytest
from google import genai
from google.adk.runners import InMemoryRunner

from tests.test_adk_wiring import ScriptedModel, _call_response, _run, _text_response
from travel_agent.agents.flight_airport import _instruction as flight_instruction
from travel_agent.agents.flight_airport import build_flight_airport_agent
from travel_agent.agents.orchestrator import run_pipeline
from travel_agent.agents.perceive import filter_tool_result
from travel_agent.agents.specialists import build_placeholder_specialist, parse_agent_result
from travel_agent.core.models import FaaClosure
from travel_agent.guardrails.callbacks import assert_guarded
from travel_agent.schemas import AgentResult, Claim
from travel_agent.tools import live, replay
from travel_agent.verification.checks import verify_claims

GOAL = {
    "origin_airport": "JFK", "destination_airport": "SFO",
    "departure_date": "2026-09-23", "departure_window": "evening",
}


@pytest.fixture(autouse=True)
def offline_scenario(monkeypatch):
    def unexpected_client(*args, **kwargs):
        pytest.fail("Flight tests must never create a Gemini client")

    monkeypatch.setenv("GUARDRAIL_LLM", "0")
    # A key in .env would make the planner build a Gemini client in run_pipeline tests.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.setattr(genai.Client, "__init__", unexpected_client)
    replay.use_demo_scenario("sfo_gdp_20260923")
    try:
        yield
    finally:
        replay.use_demo_scenario(None)


def run_case(calls, claims, *, task="Assess the JFK to SFO flight.", unavailable=()):
    reply = asdict(AgentResult(agent="flight_airport", claims=claims))
    if unavailable:
        reply["unavailable"] = list(unavailable)
    script = [_call_response(name, args) for name, args in calls]
    model = ScriptedModel(script=[*script, _text_response(json.dumps(reply))])
    agent = build_flight_airport_agent(model)
    assert_guarded(agent)
    runner = InMemoryRunner(agent=agent, app_name="test")

    async def scenario():
        state = {"trip_goal": GOAL, "plan": [{"agent": "flight_airport", "goal": task}]}
        session = await runner.session_service.create_session(app_name="test", user_id="u", state=state)
        await _run(runner, session, task)
        fresh = await runner.session_service.get_session(app_name="test", user_id="u", session_id=session.id)
        return fresh.state

    state = asyncio.run(scenario())
    assert len(model.seen) == len(calls) + 1
    # Inspect the tool response in the very next model request after each call.
    responses = []
    for (name, _), request in zip(calls, model.seen[1:]):
        parts = [p.function_response for c in request.contents for p in c.parts if p.function_response]
        assert parts[-1].name == name
        responses.append(parts[-1].response)
    visible = {e["id"]: e for response in responses for e in response["evidence"]}
    store = state["evidence_store"]
    assert {k: r["evidence"] for k, r in store.items() if r["tool_ok"]} == visible
    output = json.loads(state["result_flight_airport"])
    assert output == reply
    verdicts = verify_claims(output["claims"], store)["verdicts"]
    assert [v["status"] for v in verdicts] == ["supported"] * len(claims), verdicts
    return visible, responses


def test_sfo_same_day_faa_and_september_bts_reach_model():
    faa = "faa_nas:SFO:GDP:0"
    bts = "bts:JFK-SFO:month:09"
    visible, responses = run_case(
        [("get_airport_delays", {"airport": "SFO"}),
         ("get_route_ontime_history", {"origin_airport": "JFK"})],
        [Claim("At 09:10 PDT on September 23, the SFO ground delay program averaged 34 minutes.",
               [faa], "high"),
         Claim("Of 579 historical JFK-SFO flights in September 2025, 11.9% departed at least "
               "15 minutes late and 14.2% arrived at least 15 minutes late.", [bts], "high")],
    )
    assert set(visible) == {faa, bts}
    assert "coverage_gap" not in responses[0]
    record = visible[faa]
    assert record["valid_to"] is None
    assert record["data"]["as_of_local"] == "2026-09-23T09:10:00-07:00"
    assert record["data"]["temporal_scope"] == "snapshot"
    assert record["data"]["avg_delay_min"] == 34
    history = visible[bts]["data"]
    assert (history["origin"], history["destination"], history["basis"], history["flights"]) == (
        "JFK", "SFO", "2025-09", 579,
    )
    assert round(100 * history["dep_del15"], 1) == 11.9
    assert round(100 * history["arr_del15"], 1) == 14.2


@pytest.mark.parametrize("route, month", [(("SFO", "JFK"), 9), (("JFK", "SFO"), 8)])
def test_bts_requires_direction_and_month(route, month):
    raw = live.get_route_ontime_history(*route, month)
    assert raw["ok"]
    assert filter_tool_result(raw, GOAL)["evidence"] == []


def test_station_times_and_taf_windows_reach_model(monkeypatch):
    # This scenario recorded one batch URL. Use its real decoded weather for both
    # single-station calls while still exercising the production ToolResult adapter.
    batch = live.aviation_wx.fetch(["KJFK", "KSFO"])
    assert batch.ok
    monkeypatch.setattr(live.aviation_wx, "fetch", lambda ids: batch)
    visible, _ = run_case(
        [("get_aviation_weather", {"airport": a}) for a in ("JFK", "SFO")],
        [Claim("At 08:56 PDT the SFO station observed wind of 5 kt.", ["aviation_wx:SFO:metar"], "high"),
         Claim("SFO forecast wind of 10 kt in the period from 23:00 to 02:00 local time.",
               ["aviation_wx:SFO:taf:4"], "high")],
    )
    assert set(visible) == {
        "aviation_wx:JFK:metar", "aviation_wx:JFK:taf:0", "aviation_wx:JFK:taf:1",
        "aviation_wx:SFO:metar", "aviation_wx:SFO:taf:3", "aviation_wx:SFO:taf:4", "aviation_wx:SFO:taf:5",
    }
    metar = visible["aviation_wx:SFO:metar"]
    assert metar["data"]["as_of_local"] == "2026-09-23T08:56:00-07:00"
    assert metar["data"]["observed_at"] == "2026-09-23T15:56:00+00:00"
    assert metar["fetched_at"] == "2026-09-23T16:10:00+00:00"
    assert visible["aviation_wx:JFK:metar"]["data"]["as_of_local"] == "2026-09-23T11:51:00-04:00"
    taf = visible["aviation_wx:SFO:taf:4"]
    assert (taf["valid_from"], taf["valid_to"]) == ("2026-09-24T06:00:00+00:00", "2026-09-24T09:00:00+00:00")


@pytest.mark.parametrize(("commercial_flags", "expect_faa_gap"), [
    pytest.param([], True, id="expired-snapshot"),
    pytest.param([False], True, id="general-aviation-only"),
    pytest.param([False, True], False, id="commercial-and-general-aviation"),
])
def test_future_trip_gaps_reach_final_answer(monkeypatch, tmp_path, commercial_flags, expect_faa_gap):
    batch = live.aviation_wx.fetch(["KJFK", "KSFO"])
    assert batch.ok
    monkeypatch.setattr(live.aviation_wx, "fetch", lambda ids: batch)

    feed = live.faa_nas.fetch(now=replay.demo_now())
    assert feed.ok
    feed.data.closures = [
        FaaClosure(airport="SFO", reason="Scheduled restriction", affects_commercial=flag,
                   start="2026-09-23T12:00:00Z", reopen="2026-09-25T12:00:00Z")
        for flag in commercial_flags
    ]
    monkeypatch.setattr(live.faa_nas, "fetch", lambda now: feed)

    taf_id = "aviation_wx:SFO:taf:5"
    claim = Claim("A TAF forecast period is available for SFO.", [taf_id], "high")
    # The scripted reply cites a valid forecast and omits both tool gaps.
    reply = json.dumps(asdict(AgentResult("flight_airport", [claim])))
    flight = ScriptedModel(script=[
        _call_response("get_airport_delays", {"airport": "SFO"}),
        _call_response("get_aviation_weather", {"airport": "SFO"}),
        _text_response(f"```json\n{reply}\n```"),
    ])
    weather = ScriptedModel(script=[_text_response(json.dumps(asdict(AgentResult("weather"))))])
    # The rule engine rates this GDP scenario elevated, so the alternatives step
    # runs too; it needs its own model so `flight.seen` stays the flight agent's.
    alternatives = ScriptedModel(script=[_text_response(json.dumps(asdict(AgentResult("alternatives"))))])
    out = run_pipeline(
        "How bad are delays from JFK to SFO on 2026-09-24 morning?",
        model=flight, specialists={"weather": build_placeholder_specialist("weather", weather, tools=[]),
                                   "alternatives": build_placeholder_specialist("alternatives", alternatives,
                                                                                tools=[])},
        scenario="sfo_gdp_20260923", log_dir=str(tmp_path),
    )
    responses = {p.function_response.name: p.function_response.response
                 for content in flight.seen[-1].contents for p in content.parts if p.function_response}
    faa, aviation = responses["get_airport_delays"], responses["get_aviation_weather"]
    assert faa["ok"] and aviation["ok"]
    assert [e["data"]["affects_commercial"] for e in faa["evidence"]] == commercial_flags
    assert [e["id"] for e in aviation["evidence"]] == [taf_id]
    assert all(out["evidence_store"][e["id"]]["tool_ok"] for e in faa["evidence"] + aviation["evidence"])
    faa_gap = "SFO FAA: no status evidence for trip window"
    metar_gap = "SFO METAR: no observation for trip window"
    assert faa.get("coverage_gap") == (faa_gap if expect_faa_gap else None)
    assert aviation["coverage_gap"] == metar_gap

    expected = {metar_gap, faa_gap} if expect_faa_gap else {metar_gap}
    answer = out["final_answer"]
    assert set(out["specialist_results"]["flight_airport"]["unavailable"]) == expected
    assert set(answer["unavailable"]["flight_airport"]) == expected
    assert metar_gap in answer["text"] and (faa_gap in answer["text"]) == expect_faa_gap
    assert answer["shown_claims"] == [{**asdict(claim), "agent": "flight_airport"}]


def test_monthly_and_annual_history_remain_distinct():
    raw = live.get_route_ontime_history("JFK", "SFO", 9)
    record, = raw["evidence"]
    for basis, keep in (("2025-08", False), ("annual average", True)):
        sample = {**record, "data": {**record["data"], "basis": basis}}
        selected = filter_tool_result({**raw, "evidence": [sample]}, GOAL)
        assert selected["evidence"] == ([sample] if keep else [])


@pytest.mark.parametrize("observed_at", [None, "bad", "2026-09-22T16:00:00Z", "2026-09-23T17:00:00Z"])
def test_metar_fetch_time_cannot_replace_station_time(observed_at):
    record = {
        "source": "aviation_wx", "subject": "SFO", "id": "aviation_wx:SFO:metar",
        "fetched_at": "2026-09-23T16:10:00Z", "data": {"observed_at": observed_at},
    }
    assert filter_tool_result({"ok": True, "evidence": [record]}, GOAL)["evidence"] == []


def test_compare_nyc_departures_uses_each_route_and_discloses_missing_history():
    airports = ("JFK", "EWR", "LGA")
    calls = [("get_airport_delays", {"airport": a}) for a in airports]
    calls.extend(("get_route_ontime_history", {"origin_airport": a}) for a in airports)
    status_ids = [f"faa_nas:{a}:status" for a in airports]
    history_ids = ["bts:JFK-SFO:month:09", "bts:EWR-SFO:month:09"]
    claims = [
        Claim("At 12:10 EDT, JFK, EWR and LGA each reported no active FAA program.", status_ids, "high"),
        Claim("In September 2025, historical departure delays of at least 15 minutes affected "
              "11.9% of 579 JFK-SFO flights and 24.9% of 357 EWR-SFO flights.", history_ids, "high"),
    ]
    gap = "no historical BTS data for LGA-SFO in month 9"
    visible, responses = run_case(
        calls, claims, task="Compare JFK, EWR and LGA departures to SFO.", unavailable=[gap],
    )
    assert set(visible) == set(status_ids + history_ids)
    for airport, evidence_id in zip(airports, status_ids):
        item = visible[evidence_id]
        assert item["subject"] == airport
        assert item["data"]["active"] is False
        assert item["data"]["as_of_local"] == "2026-09-23T12:10:00-04:00"
    for evidence_id, origin, flights, percent in zip(history_ids, airports, (579, 357), (11.9, 24.9)):
        data = visible[evidence_id]["data"]
        assert (data["origin"], data["destination"], data["basis"], data["flights"]) == (origin, "SFO", "2025-09", flights)
        assert round(100 * data["dep_del15"], 1) == percent
    assert responses[-1] == {"source": "bts", "ok": False, "evidence": [], "error": gap}


class _PromptContext:
    def __init__(self, state):
        self.state = state


def test_prompt_names_the_evidence_key_literally():
    instruction = flight_instruction(_PromptContext({
        "trip_goal": GOAL, "plan": [{"agent": "flight_airport", "goal": "Assess the flight."}]}))
    assert '"evidence_ids"' in instruction
    assert "exact evidence_ids" not in instruction


def test_exact_evidence_ids_from_the_model_carries_the_ids():
    faa = "faa_nas:SFO:GDP:0"
    reply = {"agent": "flight_airport", "needs_human": False, "claims": [
        {"text": "At 09:10 PDT on September 23, the SFO ground delay program averaged 34 minutes.",
         "exact_evidence_ids": [faa], "confidence": "high", "risk": "moderate"},
    ]}
    model = ScriptedModel(script=[
        _call_response("get_airport_delays", {"airport": "SFO"}),
        _text_response(json.dumps(reply)),
    ])
    agent = build_flight_airport_agent(model)
    assert_guarded(agent)
    runner = InMemoryRunner(agent=agent, app_name="test")

    async def scenario():
        state = {"trip_goal": GOAL, "plan": [{"agent": "flight_airport", "goal": "Assess the flight."}]}
        session = await runner.session_service.create_session(app_name="test", user_id="u", state=state)
        await _run(runner, session, "Assess the flight.")
        return await runner.session_service.get_session(app_name="test", user_id="u", session_id=session.id)

    state = asyncio.run(scenario())
    parsed = parse_agent_result(state.state["result_flight_airport"], "flight_airport")
    claim, = parsed["claims"]
    assert claim["evidence_ids"] == [faa]
    verdicts = verify_claims(parsed["claims"], state.state["evidence_store"])["verdicts"]
    assert [verdict["status"] for verdict in verdicts] == ["supported"]
