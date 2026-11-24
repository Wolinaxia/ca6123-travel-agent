"""Offline Perceive tests for airport-local windows and the real ADK tool boundary."""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime

import pytest
from google import genai
from google.adk.runners import InMemoryRunner

from tests.test_adk_wiring import ScriptedModel, _call_response, _run, _text_response
from travel_agent.agents import perceive
from travel_agent.agents.flight_airport import build_flight_airport_agent
from travel_agent.schemas import AgentResult, Claim, Evidence, ToolResult
from travel_agent.tools import mock
from travel_agent.verification.checks import verify_claims

TRIP = {
    "origin_airport": "JFK",
    "destination_airport": "SFO",
    "departure_date": "2026-09-23",
    "departure_window": "evening",
}


@pytest.fixture
def six_hour_flight(monkeypatch):
    # A fixed duration makes the exact boundaries independent of BTS data updates.
    monkeypatch.setattr(perceive.bts, "block_minutes", lambda origin, destination: 360)


def test_airport_windows_cross_midnight(six_hour_flight):
    windows = perceive.airport_windows(TRIP)
    assert [t.isoformat() for t in windows["JFK"]] == [
        "2026-09-23T18:00:00-04:00", "2026-09-24T00:00:00-04:00",
    ]
    assert [t.isoformat() for t in windows["SFO"]] == [
        "2026-09-23T21:00:00-07:00", "2026-09-24T03:00:00-07:00",
    ]


def test_arrival_uses_elapsed_time_across_dst(six_hour_flight):
    windows = perceive.airport_windows({**TRIP, "departure_date": "2026-10-31"})
    assert windows["SFO"][1].isoformat() == "2026-11-01T02:00:00-08:00"


def _record(**fields):
    return {
        "source": "mock_faa", "id": "faa:test", "subject": "SFO",
        "fetched_at": "2026-09-24T04:00:00Z", **fields,
    }


@pytest.mark.parametrize(("fields", "keep"), [
    ({"valid_from": "2026-09-24T03:00:00Z", "valid_to": "2026-09-24T04:00:00Z"}, False),
    ({"valid_from": "2026-09-24T10:00:00Z", "valid_to": "2026-09-24T11:00:00Z"}, False),
    ({"valid_from": "2026-09-24T04:00:00Z", "valid_to": "2026-09-24T04:01:00Z"}, True),
    ({"valid_to": "2026-09-24T06:00:00Z"}, True),
    ({"fetched_at": "2026-09-24T10:00:00Z"}, False),
    ({"fetched_at": None}, False),
    ({"fetched_at": "invalid"}, False),
    ({"valid_from": "bad", "valid_to": "2026-09-24T08:00:00Z"}, False),
    ({"valid_from": "2026-09-24T06:00:00+00:00", "valid_to": None}, False),
    ({"valid_from": "2026-09-24T06:00:00", "valid_to": "2026-09-24T08:00:00Z"}, False),
    ({"valid_from": "2026-09-24T08:00:00Z", "valid_to": "2026-09-24T07:00:00Z"}, False),
    ({"subject": "JFK", "fetched_at": "2026-09-23T23:00:00Z"}, True),
    ({"subject": "SFO", "fetched_at": "2026-09-23T23:00:00Z"}, True),
    ({"subject": "SFO", "fetched_at": "2026-09-22T23:00:00Z"}, False),
    ({"subject": "LGA"}, False),
])
def test_filter_boundaries_and_missing_times(six_hour_flight, fields, keep):
    record = _record(**fields)
    result = {"source": "mock_faa", "ok": True, "evidence": [record]}
    selected = perceive.filter_tool_result(result, TRIP)["evidence"]
    assert [item["id"] for item in selected] == ([record["id"]] if keep else [])
    if keep and not fields.get("valid_to"):
        assert selected[0]["data"]["temporal_scope"] == "snapshot"
    assert "data" not in record
    assert result["evidence"] == [record]


def test_missing_departure_date_excludes_evidence():
    result = {"ok": True, "source": "mock_faa", "evidence": [_record()]}
    assert perceive.filter_tool_result(result, {**TRIP, "departure_date": None})["evidence"] == []


def test_only_applicable_sfo_record_reaches_model(monkeypatch):
    def unexpected_client(*args, **kwargs):
        pytest.fail("The offline scenario attempted to create a Gemini client")

    monkeypatch.setenv("GUARDRAIL_LLM", "0")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.setattr(genai.Client, "__init__", unexpected_client)

    def mock_programs(airport):
        assert airport == "SFO"
        return ToolResult(source="mock_faa", ok=True, evidence=[applicable, expired]).to_dict()

    applicable = Evidence(
        id="mock_faa:SFO:overnight", source="mock_faa", subject="SFO",
        observation="SFO overnight ground delay program averages 34 minutes.",
        data={"avg_delay_min": 34}, fetched_at="2026-09-23T23:00:00Z",
        valid_from="2026-09-24T00:00:00-07:00", valid_to="2026-09-24T02:00:00-07:00",
    )
    expired = Evidence(
        id="mock_faa:SFO:earlier", source="mock_faa", subject="SFO",
        observation="The earlier SFO ground stop was lifted.",
        fetched_at="2026-09-23T23:00:00Z",
        valid_from="2026-09-23T16:00:00-07:00", valid_to="2026-09-23T20:00:00-07:00",
    )
    monkeypatch.setattr(mock, "get_airport_delays", mock_programs)
    claim = Claim(text=applicable.observation, evidence_ids=[applicable.id], confidence="high")
    reply = AgentResult(agent="flight_airport", claims=[claim])
    model = ScriptedModel(script=[
        _call_response("get_airport_delays", {"airport": "SFO"}),
        _text_response(json.dumps(asdict(reply))),
    ])
    agent = build_flight_airport_agent(model=model, faa_tool=mock.get_airport_delays)
    runner = InMemoryRunner(agent=agent, app_name="test")

    async def run_trip():
        session = await runner.session_service.create_session(
            app_name="test", user_id="u",
            state={"trip_goal": TRIP, "plan": [{"agent": "flight_airport", "goal": "Check FAA delays."}]},
        )
        await _run(runner, session, "Assess my evening trip from JFK to SFO.")
        return await runner.session_service.get_session(app_name="test", user_id="u", session_id=session.id)

    state = asyncio.run(run_trip()).state
    assert len(model.seen) == 2
    follow_up = model.seen[1].model_dump_json()
    assert applicable.id in follow_up and applicable.observation in follow_up
    assert expired.id not in follow_up and expired.observation not in follow_up

    store = state["evidence_store"]
    assert set(store) == {applicable.id}
    assert store[applicable.id] == {"evidence": asdict(applicable), "tool_ok": True, "source": "mock_faa"}
    output = json.loads(state["result_flight_airport"])
    assert output["claims"] == [asdict(claim)]
    # Verify at the recorded overnight program time, independent of the host clock.
    as_of = datetime.fromisoformat("2026-09-24T00:30:00-07:00")
    verdict, = verify_claims(output["claims"], store, as_of=as_of)["verdicts"]
    assert verdict["status"] == "supported"
