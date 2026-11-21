from datetime import date, datetime, timezone, timedelta
import json
from pathlib import Path
import pytest

from travel_agent.core.engine.evaluate import evaluate
from travel_agent.core.engine.models import Policy, SourceInputs, TripRequest
from travel_agent.core.engine.weather_rules import open_meteo_level
from travel_agent.core.engine.parse import parse
from travel_agent.core.models import SourceResult, FaaGroundDelay, FaaNasData, TrafficIncident, NewsItem, TrainStatus, TransitAlert
from travel_agent.core.sources.faa_nas import parse as parse_faa
from travel_agent.core.sources.aviation_wx import parse as parse_wx


ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 23, 16, 10, tzinfo=timezone.utc)
FAA = (ROOT / "travel_agent/snapshots/raw_20260923_ny_sf/faa_nas.xml").read_text()
TAF = (ROOT / "travel_agent/snapshots/raw_20260923_ny_sf/taf.json").read_text()
METAR = (ROOT / "travel_agent/snapshots/raw_20260923_ny_sf/metar.json").read_text()


def _result(source, data, url="snapshot://fixture"):
    return SourceResult(source=source, ok=True, fetched_at=NOW, url=url, data=data)


def _trip(days=1, window="whole_day"):
    return TripRequest(origin_text="New York", destination_text="San Francisco", origin_city="New York", destination_city="San Francisco", departure_date=date(2026, 9, 23) + timedelta(days=days), departure_window=window, origin_airport="JFK", destination_airport="SFO", origin_latlon=(40.7128, -74.006), destination_latlon=(37.7749, -122.4194), origin_timezone="America/New_York", destination_timezone="America/Los_Angeles")


def _prior_trip(day=date(2026, 11, 21)):
    return TripRequest(
        origin_text="New York",
        destination_text="Boston",
        origin_city="New York",
        destination_city="Boston",
        departure_date=day,
        departure_window="evening",
        origin_airport="JFK",
        destination_airport="BOS",
        origin_latlon=(40.6413, -73.7781),
        destination_latlon=(42.3656, -71.0096),
        origin_timezone="America/New_York",
        destination_timezone="America/New_York",
    )


def test_bts_prior_is_info_and_does_not_raise_level():
    assessment = evaluate(_prior_trip(), SourceInputs(), datetime(2026, 11, 21, 12, tzinfo=timezone.utc))
    bts = [item for item in assessment.evidence if item.rule_id == "BTS_BASE_CAP"]
    assert bts
    assert bts[0].level == "info"
    assert bts[0].category == "delay"
    assert "Departure delay" in bts[0].detail
    assert assessment.level == "low"


def test_holiday_peak_prior_is_info_and_does_not_raise_level():
    assessment = evaluate(_prior_trip(date(2026, 11, 25)), SourceInputs(), datetime(2026, 11, 25, 12, tzinfo=timezone.utc))
    peak = [item for item in assessment.evidence if item.rule_id == "HOLIDAY_PEAK"]
    assert peak
    assert peak[0].level == "info"
    assert peak[0].category == "delay"
    assert assessment.level == "low"


def test_prior_amplifier_note_does_not_raise_live_level():
    faa = FaaNasData(ground_delays=[FaaGroundDelay(airport="BOS", reason="capacity", avg_minutes=20, max_minutes=40)])
    assessment = evaluate(_prior_trip(), SourceInputs(faa=_result("faa", faa)), datetime(2026, 11, 21, 12, tzinfo=timezone.utc))
    amplifier = [item for item in assessment.evidence if "make recovery slower" in item.detail]
    assert len(amplifier) == 1
    assert amplifier[0].level == "info"
    assert amplifier[0].leg == "flight"
    assert assessment.level == "elevated"


def test_scenario_a_destination_gdp_drives_flight():
    weather = parse_wx(TAF, METAR)
    ok_alerts = _result("alerts", [])
    assessment = evaluate(_trip(0), SourceInputs(faa=_result("faa", parse_faa(FAA, NOW)), taf_origin=_result("wx", weather), taf_destination=_result("wx", weather), nws_origin_alerts=ok_alerts, nws_destination_alerts=ok_alerts), NOW)
    assert assessment.level == "elevated"
    assert assessment.driving_leg == "flight"
    assert any(item.rule_id.startswith("FAA_GDP") and "SFO" in item.detail for item in assessment.evidence)
    assert assessment.confidence == "high"
    assert all("LAX" not in item.detail and "SAN" not in item.detail for item in assessment.evidence)


def test_scenario_d_failed_faa_is_explicit():
    failed = SourceResult(source="faa", ok=False, fetched_at=NOW, url="snapshot://faa", error="offline")
    assessment = evaluate(_trip(), SourceInputs(faa=failed), NOW)
    assert any(item.rule_id == "SOURCE_UNAVAILABLE" for item in assessment.evidence)
    assert assessment.confidence != "high"


def test_scenario_c_far_horizon_recheck():
    assessment = evaluate(_trip(30), SourceInputs(), NOW)
    assert assessment.confidence == "low"
    assert assessment.recheck_on == date(2026, 10, 16)


def test_parser_examples(monkeypatch):
    class Result:
        ok = True
        error = None
        data = {"latitude": 40.7, "longitude": -74.0, "timezone": "America/New_York"}
    monkeypatch.setattr("travel_agent.core.engine.parse.geocode", lambda name: Result())
    result = parse("NYC to SF tomorrow afternoon", NOW)
    assert result.departure_date == date(2026, 9, 24)
    assert result.departure_window == "afternoon"
    assert parse("nonsense", NOW)["error"]


def test_parser_ops_timezone_does_not_shift_origin_date(monkeypatch):
    class Result:
        ok = True
        error = None
        data = {"latitude": 40.7, "longitude": -74.0, "timezone": "America/New_York"}
    monkeypatch.setattr("travel_agent.core.engine.parse.geocode", lambda name: Result())
    result = parse("New York to San Francisco tomorrow", datetime(2026, 9, 23, 20, tzinfo=timezone.utc))
    assert result.departure_date == date(2026, 9, 24)


def test_source_failure_never_becomes_confident_low():
    failed = SourceResult(source="taf", ok=False, fetched_at=NOW, url="x", error="timeout")
    assessment = evaluate(_trip(), SourceInputs(faa=_result("faa", parse_faa(FAA, NOW)), taf_origin=failed, taf_destination=failed), NOW)
    assert assessment.confidence != "high"


def test_bus_alternative_is_explicitly_unavailable():
    assessment = evaluate(_trip(), SourceInputs(), NOW)
    bus = next(item for item in assessment.alternatives if item.mode == "bus")
    assert bus.not_practical
    assert "no open national data" in bus.reason


def test_horizon_excludes_current_faa_gdp():
    assessment = evaluate(_trip(10), SourceInputs(faa=_result("faa", parse_faa(FAA, NOW))), NOW)
    assert not any(item.rule_id.startswith("FAA_GDP") for item in assessment.evidence)
    assert any("outside horizon" in reason.get("reason", "") for reason in assessment.filtered if isinstance(reason, dict))


def test_taf_periods_overlap_tomorrow_window():
    weather = parse_wx(TAF, METAR)
    assessment = evaluate(_trip(), SourceInputs(taf_origin=_result("wx", weather), taf_destination=_result("wx", weather)), NOW)
    assert "TAF/METAR outside horizon" not in assessment.filtered


def test_high_flight_with_feasible_rail_changes_plan():
    faa = FaaNasData(ground_delays=[FaaGroundDelay(airport="SFO", reason="capacity", avg_minutes=60, max_minutes=90)])
    ok = _result("core", [])
    inputs = SourceInputs(faa=_result("faa", faa), taf_origin=ok, taf_destination=ok, nws_origin_alerts=ok, nws_destination_alerts=ok)
    trip = _trip(0).model_copy(update={"destination_text": "Washington DC", "destination_city": "Washington DC", "destination_airport": "DCA", "destination_latlon": (38.8521, -77.0377), "destination_timezone": "America/New_York"})
    faa.ground_delays[0].airport = "DCA"
    assessment = evaluate(trip, inputs, NOW)
    assert assessment.level == "high"
    assert assessment.action == "change_plan"
    assert assessment.action_detail


def test_duty_of_care_is_separate():
    assessment = evaluate(_trip(), SourceInputs(), NOW)
    assert assessment.duty_of_care is False
    assert "+duty_of_care" not in assessment.action


def test_low_assessment_has_no_driving_leg():
    assessment = evaluate(_trip(30), SourceInputs(), NOW)
    assert assessment.level == "low"
    assert assessment.driving_leg is None


def test_news_is_separate_and_requires_place_and_disruption():
    items = [NewsItem(title="Airport delay in New York", url="u", provider="gdelt", published_at=NOW), NewsItem(title="Airport disruption in Iran", url="u2", provider="gdelt", published_at=NOW)]
    assessment = evaluate(_trip(), SourceInputs(news=_result("news", items)), NOW)
    assert [item.title for item in assessment.news] == ["Airport delay in New York"]
    assert all(item.leg != "alternative" or item.rule_id != "NEWS_DISPLAY" for item in assessment.evidence)


def test_amtraker_nationwide_noise_not_trip_evidence():
    train = TrainStatus(number="1", route="Unrelated route", state="Active", delay_minutes=90)
    assessment = evaluate(_trip(), SourceInputs(amtraker=_result("amtraker", [train])), NOW)
    assert not any(item.rule_id == "AMTRAKER_DELAY" for item in assessment.evidence)


def test_tomtom_requires_numbered_road_and_deduplicates():
    incidents = [
        TrafficIncident(description="Closed ramp", road_numbers=["I-95"], magnitude_of_delay=1, distance_km=1),
        TrafficIncident(description="Closed ramp", road_numbers=["I-95"], magnitude_of_delay=1, distance_km=1),
        TrafficIncident(description="Street delay", road_numbers=[], magnitude_of_delay=5, distance_km=1),
    ]
    assessment = evaluate(_trip(0), SourceInputs(tomtom_origin=_result("tomtom", incidents)), NOW)
    traffic = [item for item in assessment.evidence if item.rule_id == "TOMTOM_MAJOR"]
    assert len(traffic) == 1
    assert traffic[0].level == "high"


def test_policy_rule_table_is_renderable():
    policy = Policy()
    assessment = evaluate(_trip(), SourceInputs(faa=_result("faa", parse_faa(FAA, NOW))), NOW, policy)
    assert {item.rule_id for item in assessment.evidence} <= set(policy.rules)
    assert all(description for description in policy.rules.values())


@pytest.mark.parametrize("average,expected", [(14, "low"), (15, "elevated"), (59, "elevated"), (60, "high")])
def test_faa_gdp_boundaries(average, expected):
    faa = FaaNasData(ground_delays=[FaaGroundDelay(airport="SFO", reason="x", avg_minutes=average, max_minutes=average)])
    assessment = evaluate(_trip(0), SourceInputs(faa=_result("faa", faa)), NOW)
    assert assessment.legs[1].level == expected


@pytest.mark.parametrize("code,snowfall,expected", [(0, 0, "low"), (95, 0, "elevated"), (0, 5, "high")])
def test_open_meteo_boundaries(code, snowfall, expected):
    assert open_meteo_level({"weather_code": [code], "snowfall_sum": [snowfall]}) == expected


def _mta_trip(window="whole_day"):
    return TripRequest(
        origin_text="New York",
        destination_text="Boston",
        origin_city="New York",
        destination_city="Boston",
        departure_date=date(2026, 9, 23),
        departure_window=window,
        origin_airport="JFK",
        destination_airport="BOS",
        origin_latlon=(40.6413, -73.7781),
        destination_latlon=(42.3656, -71.0096),
        origin_timezone="America/New_York",
        destination_timezone="America/New_York",
    )


def _mta_alert(header, start=None, end=None, routes=("A",)):
    periods = [(start, end)] if start is not None or end is not None else []
    return TransitAlert(header_text=header, affected_routes=list(routes), active_periods=periods)


def test_mta_future_planned_work_outside_window_is_filtered():
    alert = _mta_alert(
        "No [A] between 168 St and 207 St",
        datetime(2026, 9, 25, 12, tzinfo=timezone.utc),
        datetime(2026, 9, 25, 18, tzinfo=timezone.utc),
    )
    assessment = evaluate(_mta_trip("evening"), SourceInputs(mta=_result("mta", [alert])), NOW)
    assert not [item for item in assessment.evidence if item.source == "mta"]
    assert any(reason.get("reason") == "MTA alert outside trip window" for reason in assessment.filtered if isinstance(reason, dict))
    assert assessment.legs[0].level == "low"


def test_mta_duplicate_headers_are_deduped():
    alerts = [_mta_alert("No [A] trains to Howard Beach-JFK Airport") for _ in range(10)]
    assessment = evaluate(_mta_trip(), SourceInputs(mta=_result("mta", alerts)), NOW)
    elevated = [item for item in assessment.evidence if item.source == "mta" and item.level == "elevated"]
    assert len(elevated) == 1


def test_mta_non_airport_work_is_info_and_does_not_raise_leg():
    alert = _mta_alert("No [A] between 168 St and 207 St")
    assessment = evaluate(_mta_trip(), SourceInputs(mta=_result("mta", [alert])), NOW)
    items = [item for item in assessment.evidence if item.source == "mta"]
    assert len(items) == 1
    assert items[0].level == "info"
    assert "none on the airport branch" in items[0].detail
    assert assessment.legs[0].level == "low"
    assert any(reason.get("reason") == "MTA alert not on airport branch" for reason in assessment.filtered if isinstance(reason, dict))


def test_mta_airport_branch_work_is_elevated():
    alerts = [
        _mta_alert("No [A] trains to Howard Beach-JFK Airport"),
        _mta_alert("No [A] between 168 St and 207 St"),
    ]
    assessment = evaluate(_mta_trip(), SourceInputs(mta=_result("mta", alerts)), NOW)
    elevated = [item for item in assessment.evidence if item.source == "mta" and item.level == "elevated"]
    assert len(elevated) == 1
    assert "Howard Beach" in elevated[0].detail
    info = [item for item in assessment.evidence if item.source == "mta" and item.level == "info"]
    assert len(info) == 1
    assert "1 on the airport branch" in info[0].detail
    assert assessment.legs[0].level == "elevated"


def test_mta_active_period_without_bounds_counts_as_active():
    alerts = [_mta_alert("No [A] trains to Howard Beach-JFK Airport", None, None)] * 2
    alerts.append(_mta_alert("No [A] between 168 St and 207 St", None, None))
    assert all(alert.active_periods == [] for alert in alerts)
    assessment = evaluate(_mta_trip(), SourceInputs(mta=_result("mta", alerts)), NOW)
    elevated = [item for item in assessment.evidence if item.source == "mta" and item.level == "elevated"]
    info = [item for item in assessment.evidence if item.source == "mta" and item.level == "info"]
    assert len(elevated) == 1
    assert len(info) == 1
    assert info[0].detail.startswith("2 active service changes")
    assert assessment.legs[0].level == "elevated"
