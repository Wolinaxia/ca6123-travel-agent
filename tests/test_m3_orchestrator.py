"""Run both M3 specialists through the team's actual parallel orchestrator."""

import json
from dataclasses import asdict

import pytest
from google import genai

from tests.test_adk_wiring import ScriptedModel, _call_response, _text_response
from travel_agent.agents.flight_airport import build_flight_airport_agent
from travel_agent.agents.orchestrator import build_orchestrator, run_pipeline
from travel_agent.agents.specialists import build_placeholder_specialist
from travel_agent.agents.weather import build_weather_agent
from travel_agent.schemas import AgentResult, Claim
from travel_agent.tools import live

BTS = "bts:JFK-SFO:month:11"
JFK = "open_meteo:JFK:day:2026-11-29"
SFO = "open_meteo:SFO:day:2026-11-29"
GAP = "SFO forecast gap 2026-11-30 00:00:00/03:37:09 PST"


def test_default_orchestrator_uses_m3_specialists():
    model = ScriptedModel()
    runner = build_orchestrator(model=model)
    flight, weather = runner.sub_agents[0].sub_agents
    assert flight.name == "flight_airport"
    assert flight.instruction is build_flight_airport_agent(model).instruction
    assert weather.name == "weather"
    assert weather.instruction is build_weather_agent(model).instruction
    assert flight.model is weather.model is model


@pytest.mark.parametrize(("fenced", "include_bad_claim"), [
    pytest.param(False, False, id="plain"),
    pytest.param(True, False, id="fenced"),
    pytest.param(True, True, id="verifier-holds-back-wrong-rate"),
])
def test_real_specialists_and_overnight_gap(tmp_path, monkeypatch, fenced, include_bad_claim):
    def unexpected_client(*args, **kwargs):
        pytest.fail("Offline integration attempted to construct a Gemini client")

    monkeypatch.setattr(genai.Client, "__init__", unexpected_client)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.setenv("GUARDRAIL_LLM", "0")

    def daily(airport):
        result = live.get_extended_forecast(airport)
        if airport == "SFO":
            result["evidence"] = [e for e in result["evidence"] if e["data"]["date"] != "2026-11-30"]
        return result

    flight_claim = Claim("In November 2025, 15.6% of 546 JFK-SFO flights departed at least 15 minutes late.",
                         [BTS], "high")
    flight_claims = [flight_claim]
    bad = Claim("95% of JFK-SFO flights were cancelled.", [BTS], "high")
    if include_bad_claim:
        flight_claims.append(bad)
    flight_reply = AgentResult("flight_airport", flight_claims)
    flight = ScriptedModel(script=[
        _call_response("get_route_ontime_history", {"origin_airport": "JFK"}),
        _text_response(json.dumps(asdict(flight_reply))),
    ])
    weather_claims = [Claim("The JFK daily forecast for November 29 is overcast.", [JFK], "high"),
                      Claim("The SFO daily forecast for November 29 indicates rain.", [SFO], "high")]
    # No model-supplied gap or escalation: both must come from Weather's callback.
    weather_reply = json.dumps(asdict(AgentResult("weather", weather_claims)))
    if fenced:
        weather_reply = f"```json\n{weather_reply}\n```"
    weather = ScriptedModel(script=[
        _call_response("get_extended_forecast", {"airport": "JFK"}),
        _call_response("get_extended_forecast", {"airport": "SFO"}),
        _text_response(weather_reply),
    ])
    # M4 is a placeholder on this main. Use its actual instruction provider so
    # the captured request proves what the downstream specialist receives.
    alternatives = ScriptedModel(script=[_text_response(json.dumps(asdict(AgentResult("alternatives"))))])
    out = run_pipeline(
        "Find the least risky way from JFK to SFO on 2026-11-29 evening.",
        scenario="calm_baseline_20261121", log_dir=str(tmp_path),
        specialists={
            "flight_airport": build_flight_airport_agent(flight),
            "weather": build_weather_agent(weather, extended_tool=daily),
            "alternatives": build_placeholder_specialist("alternatives", alternatives, tools=[]),
        },
    )

    assert out["intent"] == "plan_trip"
    assert any(task["agent"] == "alternatives" for task in out["plan"])
    results = out["specialist_results"]
    for name, expected in (("flight_airport", flight_claims), ("weather", weather_claims)):
        assert results[name]["claims"] == [{**asdict(c), "agent": name} for c in expected]
    assert results["weather"]["needs_human"] is True
    assert results["weather"]["unavailable"] == [GAP]
    store = out["evidence_store"]
    assert {key for key, record in store.items() if record["tool_ok"]} == {BTS, JFK, SFO}

    answer = out["final_answer"]
    supported = {flight_claim.text, *(claim.text for claim in weather_claims)}
    verdicts = {v["claim"]["text"]: v["status"] for v in out["last_verification"]["verdicts"]}
    assert {text for text, status in verdicts.items() if status == "supported"} == supported
    assert {c["text"] for c in answer["shown_claims"]} == supported
    if include_bad_claim:
        assert verdicts[bad.text] == "contradicted"
        held, = answer["held_back"]
        assert (held["text"], held["status"]) == (bad.text, "contradicted")
    else:
        assert answer["held_back"] == []
        assert out["last_verification"]["needs_human"] is False
    assert answer["needs_human"] is True
    assert answer["unavailable"]["weather"] == [GAP]
    assert GAP in answer["text"]

    request, = alternatives.seen
    instruction = request.config.system_instruction
    for finding in (flight_claim.text, weather_claims[0].text, weather_claims[1].text, BTS, JFK, SFO, GAP):
        assert finding in instruction
