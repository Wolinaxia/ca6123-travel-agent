"""Tests for the evidence store, claim checks and Verifier agent."""

from __future__ import annotations

from datetime import datetime, timezone
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
def state(monkeypatch) -> dict:
    """Evidence store built from the recorded replay scenario (offline)."""
    state: dict = {}
    try:
        replay.use_demo_scenario("sfo_gdp_20260923")
        as_of = replay.demo_now()
        add_tool_result(state, live.assess_trip_risk("New York", "San Francisco", "2026-09-23", "evening"))
        add_tool_result(state, live.get_faa_airport_status("SFO"))
        add_tool_result(state, live.get_faa_airport_status("JFK"))
    finally:
        replay.use_demo_scenario(None)
    monkeypatch.setattr("travel_agent.verification.checks._verification_time", lambda: as_of)
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


def test_normal_arrivals_contradict_active_ground_delay(state):
    verdict = verify_claim(_claim("SFO arrivals are running normally.", [FAA_SFO]), state["evidence_store"])
    assert verdict["status"] == "contradicted"
    assert "active airport disruption" in verdict["reason"]


def test_expired_airport_evidence_cannot_support_a_current_claim(state):
    store = state["evidence_store"]
    store[FAA_SFO]["evidence"]["valid_to"] = "2026-09-23T15:00:00Z"
    claim = _claim("The SFO ground delay program averages 34 min this evening.", [FAA_SFO])
    before = datetime(2026, 9, 23, 14, tzinfo=timezone.utc)
    after = datetime(2026, 9, 23, 16, tzinfo=timezone.utc)
    assert verify_claim(claim, store, as_of=before)["status"] == "supported"
    assert verify_claim(claim, store, as_of=after)["status"] == "unsupported"


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
    assert agent.model == "gemini-3.5-flash-lite"
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


@pytest.mark.parametrize(
    "text",
    [
        "Rail is a reliable same-day option any day this week.",
        "Book a bed near the station and sit in row 5; the art museum is a short walk.",
    ],
)
def test_lowercase_words_that_are_iata_codes_are_not_airports(state, text):
    # "day" is DAY (Dayton), "bed" BED, "row" ROW, "art" ART, "sit" SIT.
    verdict = verify_claim(_claim(text, [SUMMARY]), state["evidence_store"])
    assert verdict["status"] == "supported", verdict


def test_uppercase_code_without_evidence_is_still_flagged(state):
    verdict = verify_claim(
        _claim("DAY has a ground stop tonight.", [SUMMARY]), state["evidence_store"]
    )
    assert verdict["status"] == "unsupported"


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


# --- Verifier false holds fix (docs/VERIFIER_FALSE_HOLDS_FIX.md, _FIX_1.md, _FIX_2.md) ---
#
# After the Verifier ablation: an observation about one airport is no longer
# compared with the trip-level summary, provided it is provably that (one
# airport code, citations only about it, no trip or judgement wording, and
# every active FAA program at that airport cited). It then gets a floor from
# its own evidence. Everything else keeps the original trip-level comparison.


def _put(store: dict, evidence_id: str, source: str, subject: str, observation: str, **data) -> None:
    _store(store, evidence_id, {"source": source, "subject": subject, "observation": observation, "data": data})


def _trip_store(level: str = "high", lga_program: str | None = "GDP") -> dict:
    """LGA-BOS trip rated ``level``. BOS is calm; LGA has one FAA program (or none)."""
    store: dict = {"evidence_store": {}}
    _put(store, "assess:LGA-BOS:2026-11-22:evening:summary", "engine", "LGA-BOS", f"Overall trip risk {level}",
         level=level, legs={"LGA": level, "BOS": "low"})
    if lga_program == "GDP":
        _put(store, "faa_nas:LGA:GDP:0", "faa_nas", "LGA", "Ground Delay Program at LGA: avg 148 min (low visibility)",
             program="GDP", avg_delay_min=148, max_delay_min=362)
    elif lga_program:
        _put(store, "faa_nas:LGA:GS:0", "faa_nas", "LGA", "Ground stop at LGA", program=lga_program)
    _put(store, "nws:LGA:no_alerts", "nws", "LGA", "No active NWS alerts near LGA", active_alerts=0)
    _put(store, "nws:LGA:alert:0", "nws", "LGA", "Severe Thunderstorm Warning (severity Severe)",
         event="Severe Thunderstorm Warning", severity="Severe")
    _put(store, "aviation_wx:LGA:metar", "aviation_wx", "LGA", "LGA METAR: LIFR, wind 8 kt, gusts 38 kt",
         flight_category="LIFR", wind_speed_kt=8, gust_kt=38)
    _put(store, "nws_forecast:LGA:period:1", "nws_forecast", "LGA", "Light Rain (wind N 14 mph)",
         short_forecast="Light Rain", wind="N 14 mph", precipitation_probability=80,
         detailed_forecast=("Rain. Cloudy, with a low around 57. North wind around 14 mph, with gusts as high as "
                            "26 mph. Chance of precipitation is 80%."))
    _put(store, "faa_nas:BOS:status", "faa_nas", "BOS", "No active FAA program at BOS", active=False, airport="BOS")
    _put(store, "nws:BOS:no_alerts", "nws", "BOS", "No active NWS alerts near BOS", active_alerts=0)
    _put(store, "aviation_wx:BOS:metar", "aviation_wx", "BOS", "BOS METAR: VFR, wind 0 kt",
         flight_category="VFR", wind_speed_kt=0)
    return store


def _status(store: dict, text: str, ids: list[str], risk: str | None = "low") -> dict:
    return verify_claim(_claim(text, ids, risk=risk), store["evidence_store"])


@pytest.mark.parametrize("text, ids, risk", [
    ("As of 12:08 local time, there are no active NWS alerts near BOS.", ["nws:BOS:no_alerts"], "low"),
    ("BOS currently has no active FAA delay programs.", ["faa_nas:BOS:status"], "low"),
    ("BOS weather is currently VFR with calm winds.", ["aviation_wx:BOS:metar"], "low"),
    # The airport's only FAA program is cited, and the risk is not below what it shows.
    ("LGA has a Ground Delay Program averaging 148 minutes.", ["faa_nas:LGA:GDP:0"], "moderate"),
    ("LGA has a Ground Delay Program averaging 148 minutes and is reporting LIFR conditions.",
     ["faa_nas:LGA:GDP:0", "aviation_wx:LGA:metar"], "moderate"),
    # The GDP is not cited, but moderate is not below what it requires.
    ("LGA is reporting LIFR conditions.", ["aviation_wx:LGA:metar"], "moderate"),
    ("The LGA forecast shows light rain tonight.", ["nws_forecast:LGA:period:1"], "moderate"),
])
def test_single_airport_observation_is_not_compared_with_the_trip_summary(text, ids, risk):
    verdict = _status(_trip_store("high"), text, ids, risk)
    assert verdict["status"] == "supported", verdict["reason"]


@pytest.mark.parametrize("text", [
    "As of 12:08 local time, there are no active NWS alerts near LGA.",
    "LGA is operating normally.",
    "LGA flights are on schedule.",
    "LGA has zero delays.",
    "LGA is clear for departures tonight.",
])
def test_observation_cannot_be_low_risk_while_the_airport_has_an_uncited_program(text):
    # LGA has an active GDP in the store; the claim cites only the benign no-alerts record.
    verdict = _status(_trip_store("elevated"), text, ["nws:LGA:no_alerts"], "low")
    assert verdict["status"] == "contradicted", text
    assert verdict["reason"] == "downplays rule-engine risk"
    # A ground stop needs high, so moderate is still too low on a trip rated high.
    verdict = _status(_trip_store("high", lga_program="ground_stop"), text, ["nws:LGA:no_alerts"], "moderate")
    assert verdict["status"] == "contradicted", text
    # The fix only removes holds: without a trip summary this was never held, and still is not.
    store = _trip_store("elevated")
    del store["evidence_store"]["assess:LGA-BOS:2026-11-22:evening:summary"]
    assert _status(store, text, ["nws:LGA:no_alerts"], "low")["status"] == "supported"
    # With no program at LGA the same factual sentence is a plain observation.
    if text.startswith("As of"):
        assert _status(_trip_store("elevated", lga_program=None), text, ["nws:LGA:no_alerts"], "low")["status"] == "supported"


@pytest.mark.parametrize("text, ids", [
    # Trip-level reassurance, with or without two upper-case codes.
    ("LGA to BOS looks fine tonight.", ["faa_nas:BOS:status", "nws:LGA:no_alerts"]),
    ("Your flight tonight should be fine.", ["faa_nas:BOS:status"]),
    ("The trip looks low risk.", ["nws:BOS:no_alerts"]),
    ("New York to Boston looks fine tonight.", ["faa_nas:BOS:status"]),
    ("lga to bos looks fine tonight.", ["faa_nas:BOS:status"]),
    ("No disruption is expected for this journey.", ["faa_nas:BOS:status", "nws:BOS:no_alerts"]),
    # One code, only that airport cited, no program there: the wording makes it a trip-level judgement.
    ("Everything at BOS is calm so the trip is safe.", ["faa_nas:BOS:status", "nws:BOS:no_alerts"]),
    ("With BOS clear, your trip tonight is low risk.", ["faa_nas:BOS:status"]),
    ("Flying into BOS tonight is low risk.", ["nws:BOS:no_alerts"]),
    ("BOS is calm, so tonight will be easy.", ["nws:BOS:no_alerts"]),
    ("The flight into BOS will be on time.", ["faa_nas:BOS:status"]),
    ("Travel through BOS tonight carries little risk.", ["nws:BOS:no_alerts"]),
    ("BOS is clear, so the whole route is fine.", ["faa_nas:BOS:status"]),
    ("Nothing at BOS threatens tonight's flight.", ["nws:BOS:no_alerts"]),
    ("Go ahead as planned; BOS is clear.", ["faa_nas:BOS:status"]),
    ("Travelers through BOS face minimal risk tonight.", ["nws:BOS:no_alerts"]),
    ("Departing LGA tonight should be risk-free.", ["nws:LGA:no_alerts"]),
    # One airport named, but another airport's evidence cited.
    ("BOS looks calm for arrivals.", ["faa_nas:BOS:status", "nws:LGA:no_alerts"]),
    # Citing the engine summary keeps the original path.
    ("Overall risk for this trip is low.", ["assess:LGA-BOS:2026-11-22:evening:summary"]),
])
def test_trip_level_reassurance_is_still_held(text, ids):
    verdict = _status(_trip_store("elevated", lga_program=None), text, ids, "low")
    assert verdict["status"] != "supported", text


@pytest.mark.parametrize("program, ids, risk, expected", [
    ("GDP", ["faa_nas:LGA:GDP:0"], "low", "contradicted"),
    ("GDP", ["faa_nas:LGA:GDP:0"], "moderate", "supported"),
    ("ground_stop", ["faa_nas:LGA:GS:0"], "moderate", "contradicted"),
    ("ground_stop", ["faa_nas:LGA:GS:0"], "high", "supported"),
    ("Ground Stop", ["faa_nas:LGA:GS:0"], "moderate", "contradicted"),
    (None, ["nws:LGA:alert:0"], "low", "contradicted"),
    (None, ["nws:LGA:alert:0"], "moderate", "supported"),
    (None, ["aviation_wx:LGA:metar"], "low", "contradicted"),
    (None, ["aviation_wx:LGA:metar"], "moderate", "supported"),
])
def test_floor_from_the_cited_evidence(program, ids, risk, expected):
    verdict = _status(_trip_store("high", lga_program=program), "Conditions at LGA are as reported.", ids, risk)
    assert verdict["status"] == expected, verdict["reason"]
    if expected == "contradicted":
        assert verdict["reason"] == "downplays cited evidence"


def test_ga_only_closure_and_calm_observations_have_no_floor():
    store = _trip_store("high", lga_program=None)
    _put(store, "faa_nas:LGA:closure:0", "faa_nas", "LGA", "Runway closed to general aviation",
         program="closure", affects_commercial=False)
    assert _status(store, "LGA has a general aviation restriction.", ["faa_nas:LGA:closure:0"], "low")["status"] == "supported"
    assert _status(store, "BOS is VFR.", ["aviation_wx:BOS:metar"], "low")["status"] == "supported"


@pytest.mark.parametrize("text, ids", [
    # Alternatives that avoid the delayed airport: the low risk describes the alternative.
    ("Flying into BOS avoids the LGA delay.", ["faa_nas:BOS:status", "faa_nas:LGA:GDP:0"]),
    ("The EWR to PVD option avoids the 148 minute delay at LGA.", ["plan:flight:EWR-PVD", "faa_nas:LGA:GDP:0"]),
    ("The Amtrak option avoids the 148 minute delay at LGA and is on time.", ["plan:rail:NYP-BOS", "faa_nas:LGA:GDP:0"]),
    ("The rail alternative is unaffected by the delay program at LGA.", ["plan:rail:NYP-BOS", "faa_nas:LGA:GDP:0"]),
    ("Taking the train avoids LGA, so there are no delays from the delay program.", ["plan:rail:NYP-BOS", "faa_nas:LGA:GDP:0"]),
])
def test_comparison_claims_are_not_held(text, ids):
    store = _trip_store("high")
    del store["evidence_store"]["assess:LGA-BOS:2026-11-22:evening:summary"]  # alternatives are verified without it
    _put(store, "plan:flight:EWR-PVD", "deadline_planner", "EWR-PVD", "Fly EWR to PVD: comfortable", verdict="comfortable")
    _put(store, "plan:rail:NYP-BOS", "deadline_planner", "NYP-BOS", "Amtrak NYP to BOS: comfortable", verdict="comfortable")
    verdict = _status(store, text, ids, "low")
    assert verdict["status"] == "supported", verdict["reason"]


@pytest.mark.parametrize("text, ids, risk", [
    # True sentences with reassuring words must not be held because a program exists (unchanged from before the fix).
    ("EWR arrivals are delayed 15-30 min but there are no delays for departures.", ["faa_nas:EWR:ARRDEP:0"], "moderate"),
    ("There are no delays for EWR departures.", ["faa_nas:EWR:ARRDEP:0"], None),
    ("Security wait at LGA is normal.", ["nws:LGA:no_alerts"], None),
    ("LGA weather is fine: no active NWS alerts.", ["nws:LGA:no_alerts"], None),
])
def test_true_reassuring_sentences_are_not_newly_held(text, ids, risk):
    store = _trip_store("high")
    del store["evidence_store"]["assess:LGA-BOS:2026-11-22:evening:summary"]
    _put(store, "faa_nas:EWR:ARRDEP:0", "faa_nas", "EWR", "Arrival delays at EWR 15-30 min",
         program="arrival_departure", kind="Arrival", min_delay_min=15, max_delay_min=30)
    verdict = _status(store, text, ids, risk)
    assert verdict["status"] == "supported", verdict["reason"]


def test_engine_citation_does_not_inherit_another_routes_summary():
    store: dict = {"evidence_store": {}}
    _put(store, "assess:JFK-SFO:2026-09-23:evening:summary", "engine", "JFK-SFO", "Overall trip risk low", level="low", legs={})
    _put(store, "assess:JFK-OAK:2026-09-23:evening:summary", "engine", "JFK-OAK", "Overall trip risk high", level="high", legs={})
    _put(store, "engine:JFK-SFO:2026-09-23:evening:NWS_NONE:1", "engine", "JFK-SFO", "No alerts on JFK-SFO or near OAK", level="low")
    verdict = _status(store, "JFK to SFO is low risk, better than OAK.",
                      ["engine:JFK-SFO:2026-09-23:evening:NWS_NONE:1"], "low")
    assert verdict["status"] == "supported", verdict["reason"]


def test_forecast_text_and_probability_numbers_are_read():
    store = _trip_store("high")
    forecast = ["nws_forecast:LGA:period:1"]
    assert _status(store, "LGA forecast: gusts up to 26 mph.", forecast, None)["status"] == "supported"
    assert _status(store, "LGA forecast: 80% chance of precipitation.", forecast, None)["status"] == "supported"
    assert _status(store, "LGA forecast: gusts up to 46 mph.", forecast, None)["status"] != "supported"
    assert _status(store, "LGA forecast: 30% chance of precipitation.", forecast, None)["status"] != "supported"


def test_numbers_in_other_free_text_do_not_support_a_claim():
    store = _trip_store("high")
    _put(store, "faa_nas:EWR:GDP:0", "faa_nas", "EWR", "Ground Delay Program at EWR",
         program="GDP", avg_delay_min=115, reason="volume; expect 15 minutes taxi", raw="RWY 22 CLSD 40 minutes")
    _put(store, "metro:pop", "census", "EWR", "Metro area", metro_pop=8000000)
    assert _status(store, "EWR ground delay program averages 15 minutes.", ["faa_nas:EWR:GDP:0"], None)["status"] != "supported"
    assert _status(store, "EWR ground delay program averages 40 minutes.", ["faa_nas:EWR:GDP:0"], None)["status"] != "supported"
    assert _status(store, "EWR ground delay program averages 115 minutes.", ["faa_nas:EWR:GDP:0"], None)["status"] == "supported"
    assert _status(store, "About 8000000% of EWR flights are late.", ["metro:pop"], None)["status"] != "supported"


@pytest.mark.parametrize("text", [
    "LGA operations are running normally.",
    "LGA has no active FAA programs.",
])
def test_program_level_acceptance_does_not_cover_a_claim_that_the_airport_is_normal(text):
    # LGA has a GDP (uncited). Moderate is not below the GDP, but the sentence denies it.
    verdict = _status(_trip_store("high"), text, ["nws:LGA:no_alerts"], "moderate")
    assert verdict["status"] == "contradicted", verdict["reason"]


def test_flight_category_wording_is_still_an_observation():
    verdict = _status(_trip_store("high"), "BOS METAR shows VFR flight category.", ["aviation_wx:BOS:metar"], "low")
    assert verdict["status"] == "supported", verdict["reason"]


def test_rate_keys_keep_their_rescale_and_probability_keys_are_percent():
    store = _trip_store("high")
    _put(store, "x:LGA:rates", "bts", "LGA", "rates", probability_rate=0.5, precipitation_probability=80)
    assert _status(store, "50% at LGA.", ["x:LGA:rates"], None)["status"] == "supported"
    assert _status(store, "80% at LGA.", ["x:LGA:rates"], None)["status"] == "supported"
