from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from travel_agent.core.reference import amtrak, airports, bts, holidays
from travel_agent.core.sources.faa_nas import parse
import json
import statistics


ROOT = Path(__file__).resolve().parents[2]


def test_ground_stop_program_end_time():
    xml = "<AIRPORT_STATUS_INFORMATION><Delay_type><Name>Ground Stops</Name><Ground_Stop_List><Program><ARPT>JFK</ARPT><Reason>weather</Reason><End_Time>Sep 23 at 18:00 UTC.</End_Time></Program></Ground_Stop_List></Delay_type></AIRPORT_STATUS_INFORMATION>"
    item = parse(xml, datetime(2026, 9, 23, 16, tzinfo=timezone.utc)).ground_stops[0]
    assert item.airport == "JFK"
    assert item.end.hour == 18


def test_ground_stop_ground_stop_end():
    xml = "<AIRPORT_STATUS_INFORMATION><Delay_type><Name>Ground Stops</Name><Ground_Stop_List><Ground_Stop><ARPT>SFO</ARPT><Reason>fog</Reason><End>Sep 23 at 17:00 UTC.</End></Ground_Stop></Ground_Stop_List></Delay_type></AIRPORT_STATUS_INFORMATION>"
    assert parse(xml).ground_stops[0].airport == "SFO"


def test_unknown_faa_delay_type_is_preserved():
    xml = "<AIRPORT_STATUS_INFORMATION><Delay_type><Name>Airspace Flow Programs</Name><Thing>value</Thing></Delay_type></AIRPORT_STATUS_INFORMATION>"
    item = parse(xml).unparsed[0]
    assert item["name"] == "Airspace Flow Programs"
    assert "Thing" in item["raw_xml_snippet"]


def test_december_january_rollover():
    xml = "<AIRPORT_STATUS_INFORMATION><Delay_type><Name>Airport Closures</Name><Airport_Closure_List><Airport><ARPT>ABC</ARPT><Reason>maintenance</Reason><Start>Dec 28 at 10:00 UTC.</Start><Reopen>Jan 02 at 10:00 UTC.</Reopen></Airport></Airport_Closure_List></Delay_type></AIRPORT_STATUS_INFORMATION>"
    item = parse(xml, datetime(2026, 12, 29, tzinfo=timezone.utc)).closures[0]
    assert item.reopen.year == 2027
    assert item.active is True


def test_airports_near_nyc():
    result = airports.airports_near(40.7128, -74.006, 60)
    codes = {item["iata"] for item in result}
    assert {"JFK", "LGA", "EWR"} <= codes


def test_airports_near_san_francisco():
    codes = {item["iata"] for item in airports.airports_near(37.7749, -122.4194, 70)}
    assert {"SFO", "OAK", "SJC"} <= codes


def test_holiday_dates_and_peak():
    assert date(2026, 11, 26) in holidays.holidays(2026)
    assert date(2026, 7, 3) in holidays.holidays(2026)
    assert holidays.is_peak_travel(date(2026, 11, 25))


def test_bts_loader_missing_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(bts, "ROOT", tmp_path)
    bts._load.cache_clear()
    try:
        assert bts.route_base_rate("JFK", "SFO", 9) is None
    finally:
        bts._load.cache_clear()


def test_amtrak_empty_without_processed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(amtrak, "DATA", tmp_path / "missing.json")
    assert amtrak.direct_services((40.7, -74), (38.9, -77), 40) == []


def test_bts_august_truth_rates():
    path = ROOT / "travel_agent/data/processed/bts_airports.json"
    if not path.exists():
        pytest.skip("processed BTS data is not available")
    records = json.loads(path.read_text())
    august = [value for key, value in records.items() if key.endswith("-2025-08")]
    total = sum(item["flights"] for item in august)
    dep_rate = sum(item["dep_del15"] * item["flights"] for item in august) / total
    cancel_rate = sum(item["cancel_rate"] * item["flights"] for item in august) / total
    assert 0.12 < dep_rate < 0.40
    assert 0.002 < cancel_rate < 0.05
    assert any(item["cancel_rate"] > 0 for item in august)


def test_bts_winter_storm_daily_truth():
    path = ROOT / "travel_agent/data/processed/bts_daily.json"
    if not path.exists():
        pytest.skip("processed BTS daily data is not available")
    records = json.loads(path.read_text())
    for airport in ("DEN", "MDW"):
        values = [value["cancel_rate"] for key, value in records.items() if key.endswith(f"-{airport}") and key.startswith("2022-12-")]
        storm = [value["cancel_rate"] for key, value in records.items() if key.endswith(f"-{airport}") and any(key.startswith(f"2022-12-{day:02d}") for day in range(22, 27))]
        assert storm
        assert max(storm) > 3 * statistics.median(values)


def test_amtrak_truth_services():
    nyc = (40.7128, -74.006)
    dc = (38.9072, -77.0369)
    boston = (42.3601, -71.0589)
    dc_services = amtrak.direct_services(nyc, dc)
    assert dc_services[0]["from_station"]["id"] == "NYP"
    assert dc_services[0]["to_station"]["id"] == "WAS"
    assert dc_services[0]["trains_per_day"] >= 20
    assert 150 <= dc_services[0]["min_duration_min"] <= 240
    bos_services = amtrak.direct_services(nyc, boston)
    assert any(item["to_station"]["id"] in {"BOS", "BBY"} and item["trains_per_day"] >= 10 for item in bos_services)
    pairs = [(item["from_station"]["id"], item["to_station"]["id"]) for item in bos_services]
    assert len(pairs) == len(set(pairs))
