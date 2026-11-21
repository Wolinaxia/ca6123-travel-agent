"""Tests for the evidence store, claim checks and Verifier agent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from travel_agent.agents.verifier import build_verifier_agent, verify_claims_tool
from travel_agent.schemas import Claim
from travel_agent.tools import live, replay
from travel_agent.verification.checks import verify_claim, verify_claims
from travel_agent.verification.store import add_tool_result, all_ids, get

FAA_SFO = "faa_nas:SFO:GDP:0"
FAA_JFK = "faa_nas:JFK:status"
SUMMARY = "assess:JFK-SFO:2026-09-23:evening:summary"
ENGINE = "engine:JFK-SFO:2026-09-23:evening:FAA_GDP_AVG_15:1"


@pytest.fixture()
def state() -> dict:
    """Evidence store built from the recorded replay scenario (offline)."""
    state: dict = {}
    try:
        replay.use_demo_scenario("sfo_gdp_20260923")
        add_tool_result(state, live.assess_trip_risk("New York", "San Francisco", "2026-09-23", "evening"))
        add_tool_result(state, live.get_faa_airport_status("SFO"))
        add_tool_result(state, live.get_faa_airport_status("JFK"))
    finally:
        replay.use_demo_scenario(None)
    return state


def _claim(text: str, ids: list[str], risk: str | None = None) -> dict:
    claim = {"text": text, "evidence_ids": ids, "confidence": "high"}
    if risk is not None:
        claim["risk"] = risk
    return claim


def test_store_add_get_and_ids(state):
    ids = all_ids(state)
    assert FAA_SFO in ids and SUMMARY in ids
    record = get(state, FAA_SFO)
    assert record["source"] == "faa_nas"
    assert record["tool_ok"] is True
    assert get(state, "missing") is None


def test_citation_present_check(state):
    verdict = verify_claim(_claim("SFO is delayed.", []), state["evidence_store"])
    assert verdict["status"] == "unsupported"
    assert verdict["reason"] == "no evidence cited"
    assert verdict["checks"][0]["check"] == "citation_present"


def test_citations_exist_check(state):
    verdict = verify_claim(_claim("SFO is delayed 34 min.", ["faa_nas:SFO:GDP:999"]), state["evidence_store"])
    assert verdict["status"] == "unsupported"
    assert verdict["reason"] == "cites evidence that no tool returned"


def test_untrusted_only_check(state):
    news = {
        "source": "news",
        "ok": True,
        "evidence": [
            {
                "id": "news:abc",
                "source": "news",
                "subject": "news",
                "observation": "News headline (untrusted text)",
                "fetched_at": "now",
                "data": {"text": "SFO is delayed", "untrusted": True},
            }
        ],
    }
    add_tool_result(state, news)
    verdict = verify_claim(_claim("SFO is delayed.", ["news:abc"]), state["evidence_store"])
    assert verdict["status"] == "unsupported"
    assert verdict["reason"] == "supported only by untrusted news text"


def test_numbers_match_supports(state):
    verdict = verify_claim(_claim("SFO average delay is 34 min.", [FAA_SFO]), state["evidence_store"])
    assert verdict["status"] == "supported"


def test_numbers_mismatch_contradicts(state):
    verdict = verify_claim(_claim("SFO average delay is 90 min.", [FAA_SFO]), state["evidence_store"])
    assert verdict["status"] == "contradicted"
    assert "90" in verdict["reason"]


def test_places_match_supports(state):
    verdict = verify_claim(_claim("SFO has a ground delay program.", [FAA_SFO]), state["evidence_store"])
    assert verdict["status"] == "supported"


def test_places_mismatch_is_unsupported(state):
    verdict = verify_claim(_claim("OAK is delayed 34 min.", [FAA_SFO]), state["evidence_store"])
    assert verdict["status"] == "unsupported"
    assert "OAK" in verdict["reason"]


def test_risk_downplayed_is_contradicted(state):
    verdict = verify_claim(_claim("Overall trip risk is low.", [SUMMARY], risk="low"), state["evidence_store"])
    assert verdict["status"] == "contradicted"
    assert "downplays" in verdict["reason"]


def test_risk_equal_is_supported(state):
    verdict = verify_claim(_claim("Overall trip risk is elevated.", [SUMMARY], risk="moderate"), state["evidence_store"])
    assert verdict["status"] == "supported"
    assert any("matches rule-engine" in check["detail"] for check in verdict["checks"])


def test_risk_more_cautious_is_supported_with_note(state):
    verdict = verify_claim(_claim("Treat this as high risk.", [SUMMARY], risk="high"), state["evidence_store"])
    assert verdict["status"] == "supported"
    assert verdict["reason"] == "more cautious than rule engine"


def test_route_less_citation_uses_cited_airport_to_find_summary(state):
    verdict = verify_claim(
        _claim("Conditions at SFO are low risk.", [FAA_SFO], risk="low"),
        state["evidence_store"],
    )
    assert verdict["status"] == "contradicted"
    assert "downplays" in verdict["reason"]


def test_route_less_citation_uses_claim_place_to_find_summary(state):
    _store(
        state,
        "regional:bay-area",
        {
            "source": "nws",
            "subject": "Bay Area",
            "observation": "SFO conditions are normal",
            "data": {},
        },
    )
    verdict = verify_claim(
        _claim("SFO conditions are low risk.", ["regional:bay-area"], risk="low"),
        state["evidence_store"],
    )
    assert verdict["status"] == "contradicted"
    assert "downplays" in verdict["reason"]


def test_engine_delay_claim_supported(state):
    verdict = verify_claim(_claim("The rule engine reports 34 min at SFO.", [ENGINE]), state["evidence_store"])
    assert verdict["status"] == "supported"


def test_claim_high_matches_severe_rule(state):
    state["evidence_store"]["assess:synthetic:summary"] = {
        "evidence": {
            "id": "assess:synthetic:summary",
            "source": "engine",
            "subject": "JFK-SFO",
            "observation": "Overall trip risk severe",
            "fetched_at": "now",
            "data": {"level": "severe", "legs": {"flight": "severe"}},
        },
        "tool_ok": True,
        "source": "engine",
    }
    verdict = verify_claim(
        _claim("The trip risk is severe.", ["assess:synthetic:summary"], risk="high"),
        state["evidence_store"],
    )
    assert verdict["status"] == "supported"


def test_verify_claims_needs_human_and_summary(state):
    result = verify_claims(
        [
            _claim("SFO average delay is 34 min.", [FAA_SFO]),
            _claim("SFO is delayed.", []),
            _claim("OAK is delayed 34 min.", [FAA_SFO]),
        ],
        state["evidence_store"],
    )
    assert result["needs_human"] is True
    assert result["counts"] == {"supported": 1, "unsupported": 2, "contradicted": 0}
    assert "claim 2" in result["summary"] and "claim 3" in result["summary"]


def test_verify_claims_all_supported_has_no_human(state):
    result = verify_claims([_claim("SFO average delay is 34 min.", [FAA_SFO])], state["evidence_store"])
    assert result["needs_human"] is False
    assert result["summary"] == "all 1 claims supported"


def test_verify_claim_accepts_dataclass(state):
    claim = Claim(text="SFO average delay is 34 min.", evidence_ids=[FAA_SFO], confidence="high")
    verdict = verify_claim(claim, state["evidence_store"])
    assert verdict["status"] == "supported"
    assert verdict["claim"]["text"] == claim.text


def test_verify_claims_tool_reads_state(state):
    context = SimpleNamespace(state=state)
    result = verify_claims_tool([_claim("SFO average delay is 34 min.", [FAA_SFO])], context)
    assert result["verdicts"][0]["status"] == "supported"


def test_build_verifier_agent_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    agent = build_verifier_agent()
    assert agent.name == "verifier"
    assert agent.model == "gemini-2.5-flash-lite"
    assert agent.tools


def test_mixed_trusted_and_untrusted_is_allowed(state):
    news = {
        "source": "news",
        "ok": True,
        "evidence": [
            {
                "id": "news:trusted-plus",
                "source": "news",
                "subject": "news",
                "observation": "News headline (untrusted text)",
                "fetched_at": "now",
                "data": {"text": "SFO is delayed", "untrusted": True},
            }
        ],
    }
    add_tool_result(state, news)
    verdict = verify_claim(_claim("SFO average delay is 34 min.", [FAA_SFO, "news:trusted-plus"]), state["evidence_store"])
    assert verdict["status"] == "supported"


# --- G6: trusted, relevant, unit-aware evidence --------------------------


def _store(state, evidence_id: str, evidence: dict, *, tool_ok: bool = True) -> None:
    state["evidence_store"][evidence_id] = {
        "evidence": {"id": evidence_id, "fetched_at": "now", **evidence},
        "tool_ok": tool_ok,
        "source": evidence.get("source", "synthetic"),
    }


def test_failed_tool_evidence_cannot_support(state):
    _store(
        state,
        "failed:sfo",
        {
            "source": "faa_nas",
            "subject": "SFO",
            "observation": "SFO average delay 34 min",
            "data": {"avg_delay_min": 34},
        },
        tool_ok=False,
    )
    verdict = verify_claim(_claim("SFO average delay is 34 min.", ["failed:sfo"]), state["evidence_store"])
    assert verdict["status"] == "unsupported"
    assert "failed tool" in verdict["reason"]


def test_numbers_only_from_trusted_evidence(state):
    news = {
        "source": "news",
        "ok": True,
        "evidence": [
            {
                "id": "news:numeric",
                "source": "news",
                "subject": "news",
                "observation": "News headline (untrusted text)",
                "fetched_at": "now",
                "data": {"text": "SFO is delayed 90 min", "untrusted": True},
            }
        ],
    }
    add_tool_result(state, news)
    verdict = verify_claim(_claim("SFO is delayed 90 min.", ["news:numeric"]), state["evidence_store"])
    assert verdict["status"] == "unsupported"


def test_unit_aware_hours_and_hyphenated_minutes(state):
    assert (
        verify_claim(_claim("The maximum delay is 1 h 23 min.", [FAA_SFO]), state["evidence_store"])["status"]
        == "supported"
    )
    assert (
        verify_claim(_claim("The maximum delay is 83-minute.", [FAA_SFO]), state["evidence_store"])["status"]
        == "supported"
    )
    assert (
        verify_claim(_claim("The maximum delay is 90-minute.", [FAA_SFO]), state["evidence_store"])["status"]
        == "contradicted"
    )


def test_unit_family_mismatch_is_unsupported(state):
    _store(
        state,
        "weather:wind",
        {
            "source": "aviation_wx",
            "subject": "SFO",
            "observation": "SFO wind 30 kt",
            "data": {"gust_kt": 30},
        },
    )
    assert (
        verify_claim(_claim("SFO gusts reach 30 kt.", ["weather:wind"]), state["evidence_store"])["status"]
        == "supported"
    )
    assert (
        verify_claim(_claim("SFO gusts reach 30 mph.", ["weather:wind"]), state["evidence_store"])["status"]
        == "unsupported"
    )


def test_temperature_is_unit_aware(state):
    _store(
        state,
        "weather:temp",
        {
            "source": "nws",
            "subject": "SFO",
            "observation": "SFO temperature 20 C",
            "data": {"temp_c": 20},
        },
    )
    assert (
        verify_claim(_claim("It is 68 °F at SFO.", ["weather:temp"]), state["evidence_store"])["status"]
        == "supported"
    )


def test_numbers_in_coordinates_and_ids_are_ignored(state):
    _store(
        state,
        "place:sfo",
        {
            "source": "airports",
            "subject": "SFO",
            "observation": "SFO is at 37.6188, -122.3750",
            "data": {"lat": 37.6188, "lon": -122.375},
        },
    )
    verdict = verify_claim(_claim("SFO is 37.62 km away.", ["place:sfo"]), state["evidence_store"])
    assert verdict["status"] == "unsupported"


def test_common_word_codes_are_not_airports(state):
    verdict = verify_claim(
        _claim("This USA trip carries elevated risk.", [SUMMARY]),
        state["evidence_store"],
    )
    assert verdict["status"] == "supported"


def test_airport_codes_match_on_word_boundaries(state):
    _store(
        state,
        "synthetic:sfoo",
        {
            "source": "faa_nas",
            "subject": "SFOO",
            "observation": "SFOO has a ground delay program averaging 34 min",
            "data": {"avg_delay_min": 34},
        },
    )
    verdict = verify_claim(
        _claim("SFO has a ground delay program averaging 34 min.", ["synthetic:sfoo"]),
        state["evidence_store"],
    )
    assert verdict["status"] == "unsupported"


def test_non_summary_finding_downplay_is_contradicted(state):
    verdict = verify_claim(
        _claim("The FAA program is low risk.", [ENGINE], risk="low"),
        state["evidence_store"],
    )
    assert verdict["status"] == "contradicted"
    assert "downplays" in verdict["reason"]


def test_uncited_summary_for_same_route_blocks_downplay(state):
    _store(
        state,
        "engine:JFK-SFO:custom:finding",
        {
            "source": "engine",
            "subject": "JFK-SFO",
            "observation": "A finding without a level",
            "data": {},
        },
    )
    verdict = verify_claim(
        _claim("This route is low risk.", ["engine:JFK-SFO:custom:finding"], risk="low"),
        state["evidence_store"],
    )
    assert verdict["status"] == "contradicted"


def test_empty_store_summary_names_the_wiring_mistake():
    result = verify_claims([_claim("SFO is delayed 34 min.", [FAA_SFO])], {})
    assert "evidence store is empty" in result["summary"]
    assert "record_and_sanitize" in result["summary"]


def test_verify_claims_tool_persists_last_verification(state):
    from travel_agent.agents.verifier import verify_claims_tool as tool

    context = SimpleNamespace(state=state)
    tool([_claim("SFO average delay is 34 min.", [FAA_SFO])], context)
    assert context.state["last_verification"]["counts"]["supported"] == 1


def test_build_verifier_agent_registers_input_and_output_guards(monkeypatch):
    from travel_agent.guardrails import callbacks as guard

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    agent = build_verifier_agent()
    assert agent.before_model_callback is guard.input_guard
    assert agent.after_model_callback is guard.output_guard


def test_percent_rates_are_unit_aware(state):
    _store(
        state,
        "bts:rate",
        {
            "source": "bts",
            "subject": "JFK-SFO",
            "observation": "JFK-SFO history: 15.0% departures delayed",
            "data": {"dep_del15_pct": 15.0, "cancel_rate": 0.05},
        },
    )
    assert (
        verify_claim(_claim("About 15% of departures are late.", ["bts:rate"]), state["evidence_store"])["status"]
        == "supported"
    )
    assert (
        verify_claim(_claim("About 5% of flights are cancelled.", ["bts:rate"]), state["evidence_store"])["status"]
        == "supported"
    )
    assert (
        verify_claim(_claim("About 40% of departures are late.", ["bts:rate"]), state["evidence_store"])["status"]
        == "contradicted"
    )
