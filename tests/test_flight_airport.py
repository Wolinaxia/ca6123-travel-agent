"""Exercise the offline JFK-to-SFO FAA claim through ADK and the M1 Verifier."""

import asyncio
import json
from dataclasses import asdict, replace

from google import genai
from google.adk.runners import InMemoryRunner

from tests.test_adk_wiring import ScriptedModel, _call_response, _run, _text_response
from travel_agent.agents.flight_airport import build_flight_airport_agent
from travel_agent.guardrails.callbacks import assert_guarded
from travel_agent.schemas import AgentResult, Claim
from travel_agent.tools import mock
from travel_agent.verification.checks import verify_claims


def test_mock_sfo_ground_delay_claim(monkeypatch):
    def unexpected_client(*args, **kwargs):
        raise AssertionError("The offline test attempted to create a Gemini client")

    monkeypatch.setenv("GUARDRAIL_LLM", "0")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.setattr(genai.Client, "__init__", unexpected_client)
    monkeypatch.setattr(mock, "_now", lambda: "2026-09-23T16:10:00+00:00")

    scripted_claim = Claim(
        text="The mock SFO ground delay program has an average delay of 34 minutes.",
        evidence_ids=["mock_faa:SFO:GDP"],
        confidence="high",
    )
    reply = AgentResult(agent="flight_airport", claims=[scripted_claim])
    model = ScriptedModel(
        script=[
            _call_response("get_airport_delays", {"airport": "SFO"}),
            _text_response(json.dumps(asdict(reply))),
        ]
    )
    agent = build_flight_airport_agent(model=model, faa_tool=mock.get_airport_delays)
    assert_guarded(agent)
    runner = InMemoryRunner(agent=agent, app_name="test")

    async def run_trip():
        session = await runner.session_service.create_session(
            app_name="test",
            user_id="u",
            state={
                "trip_goal": {
                    "origin_airport": "JFK", "destination_airport": "SFO",
                    "departure_date": "2026-09-23", "departure_window": "whole_day",
                },
                "plan": [{"agent": "flight_airport", "goal": "Check destination FAA delays."}],
            },
        )
        await _run(runner, session, "Check airport delays for my trip from JFK to SFO.")
        return await runner.session_service.get_session(app_name="test", user_id="u", session_id=session.id)

    session = asyncio.run(run_trip())
    assert len(model.seen) == 2
    responses = [
        part.function_response
        for content in model.seen[1].contents
        for part in content.parts
        if part.function_response
    ]
    response, = responses
    assert response.name == "get_airport_delays"
    evidence, = response.response["evidence"]
    assert evidence["id"] == "mock_faa:SFO:GDP"
    assert evidence["subject"] == "SFO"
    assert evidence["data"]["program"] == "GDP"
    assert evidence["data"]["avg_delay_min"] == 34

    store = session.state["evidence_store"]
    assert store[evidence["id"]] == {"evidence": evidence, "tool_ok": True, "source": "mock_faa"}
    result = json.loads(session.state["result_flight_airport"])
    assert result["agent"] == "flight_airport"
    assert result["needs_human"] is False
    raw_claim, = result["claims"]
    claim = Claim(**raw_claim)
    assert claim.text == scripted_claim.text
    assert claim.evidence_ids == [evidence["id"]]

    missing_citation = replace(claim, evidence_ids=["mock_faa:SFO:missing"])
    verdicts = verify_claims([claim, missing_citation], store)["verdicts"]
    assert [verdict["status"] for verdict in verdicts] == ["supported", "unsupported"]
