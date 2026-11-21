import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from travel_agent.core.http import redact_url, request, source_result
from travel_agent.core.models import SourceResult
from travel_agent.core.sources import amtraker, aviation_wx, faa_nas, mta, news, nws, open_meteo, osrm, tomtom


ROOT = Path(__file__).resolve().parents[2]
FAA_XML = (ROOT / "travel_agent/snapshots/raw_20260923_ny_sf/faa_nas.xml").read_text()
TAF_JSON = (ROOT / "travel_agent/snapshots/raw_20260923_ny_sf/taf.json").read_text()
METAR_JSON = (ROOT / "travel_agent/snapshots/raw_20260923_ny_sf/metar.json").read_text()
NOW = datetime(2026, 9, 23, 16, tzinfo=timezone.utc)


def test_duration_minutes_variants():
    assert faa_nas.parse_duration_minutes("45 minutes") == 45
    assert faa_nas.parse_duration_minutes("2 hours") == 120
    assert faa_nas.parse_duration_minutes("1 hour and 23 minutes") == 83


def test_faa_sfo_ground_delay():
    result = faa_nas.parse(FAA_XML, NOW)
    sfo = next(item for item in result.ground_delays if item.airport == "SFO")
    assert sfo.avg_minutes == 34
    assert sfo.max_minutes == 83
    assert sfo.reason == "low ceilings"


def test_faa_bos_ground_delay():
    result = faa_nas.parse(FAA_XML, NOW)
    bos = next(item for item in result.ground_delays if item.airport == "BOS")
    assert bos.avg_minutes == 52
    assert bos.max_minutes == 162


def test_faa_arrival_departure_delay():
    item = faa_nas.parse(FAA_XML, NOW).arrival_departure_delays[0]
    assert (item.airport, item.kind, item.min_minutes, item.max_minutes) == ("TEB", "Departure", 15, 29)
    assert item.trend == "Decreasing"


def test_faa_lax_ga_only_stale():
    item = next(item for item in faa_nas.parse(FAA_XML, NOW).closures if item.airport == "LAX")
    assert item.affects_commercial is False
    assert item.active is True
    assert item.reopen.year == 2027


def test_faa_san_ga_only():
    item = next(item for item in faa_nas.parse(FAA_XML, NOW).closures if item.airport == "SAN")
    assert item.affects_commercial is False
    assert item.active is True


def test_faa_alo_commercial_closure():
    item = next(item for item in faa_nas.parse(FAA_XML, NOW).closures if item.airport == "ALO")
    assert item.affects_commercial is True
    assert item.start.year == 2026


@pytest.mark.parametrize(
    ("ceiling", "visibility", "expected"),
    [(5000, 10, "VFR"), (2500, 10, "MVFR"), (3000, 4, "MVFR"), (900, 10, "IFR"), (1000, 2, "IFR"), (400, 10, "LIFR"), (1000, 0.5, "LIFR")],
)
def test_flight_category_boundaries(ceiling, visibility, expected):
    assert aviation_wx.flight_category(ceiling, visibility) == expected


def test_aviation_snapshot_parse():
    result = aviation_wx.parse(TAF_JSON, METAR_JSON)
    assert result
    assert all(item.taf for item in result.values())


def test_taf_gust_parsing():
    payload = [{"icaoId": "KAAA", "fcsts": [{"timeFrom": 1, "timeTo": 2, "wdir": 180, "wspd": 12, "wgst": 25, "visib": 6, "clouds": [{"cover": "BKN", "base": 900}], "wxString": "TS SN"}]}]
    result = aviation_wx.parse(payload, [{"icaoId": "KAAA", "visib": "10+", "fltCat": "VFR"}])
    period = result["KAAA"].taf[0]
    assert period.gust_kt == 25
    assert period.ceiling_ft == 900
    assert period.weather_codes == ["TS", "SN"]


def test_nws_points_parser():
    result = nws.parse_points({"properties": {"forecast": "https://api.weather.gov/gridpoints/X/1,2/forecast", "timeZone": "America/New_York"}})
    assert result.timezone == "America/New_York"


def test_nws_forecast_parser():
    payload = {"properties": {"periods": [{"startTime": "2026-09-23T12:00:00Z", "endTime": "2026-09-23T18:00:00Z", "shortForecast": "Sunny", "detailedForecast": "Clear", "windDirection": "NW", "windSpeed": "10 mph", "probabilityOfPrecipitation": {"value": 20}}]}}
    item = nws.parse_forecast(payload)[0]
    assert item.short_forecast == "Sunny"
    assert item.precipitation_probability == 20


def test_nws_alert_fields():
    payload = {"features": [{"id": "alert-1", "properties": {"event": "Flood Warning", "severity": "Severe", "urgency": "Immediate", "certainty": "Observed", "onset": "2026-09-23T12:00:00Z", "ends": "2026-09-24T12:00:00Z", "expires": "2026-09-24T13:00:00Z", "headline": "Take action", "areaDesc": "New York"}}]}
    alert = nws.parse_alerts(payload)[0]
    assert alert.event == "Flood Warning"
    assert alert.id == "alert-1"
    assert alert.area_desc == "New York"


def test_open_meteo_daily_shape(monkeypatch):
    payload = {"daily": {"time": ["2026-09-23"], "weather_code": [1], "precipitation_sum": [0], "snowfall_sum": [0], "wind_gusts_10m_max": [20]}}
    monkeypatch.setattr(open_meteo, "request", lambda *args, **kwargs: (200, json.dumps(payload), "url"))
    result = open_meteo.forecast(40, -74)
    assert result.ok
    assert result.data["weather_code"] == [1]


def test_tomtom_distance_and_fields():
    payload = {"incidents": [{"properties": {"iconCategory": 6, "magnitudeOfDelay": 4, "description": "Crash", "from": "A", "to": "B", "roadNumbers": ["I-95"], "delay": 120}, "geometry": {"coordinates": [-74.0, 40.7]}}]}
    item = tomtom.parse(payload, 40.7, -74.0)[0]
    assert item.icon_category == 6
    assert item.distance_km == 0
    assert item.delay_seconds == 120


def test_osrm_parser():
    result = osrm.parse({"routes": [{"duration": 123.4, "distance": 5000, "geometry": {"coordinates": [[-74, 40], [-73, 41]]}}]})
    assert result.duration_s == 123.4
    assert result.coordinates[0] == (-74, 40)


def test_amtraker_parser():
    result = amtraker.parse({"1": {"trainNum": 1, "routeName": "Northeast Regional", "trainState": "Enroute", "delayMinutes": 12, "stations": [{"code": "NYP", "scheduled": "2026-09-23T12:00:00Z"}]}})
    assert result[0].number == "1"
    assert result[0].delay_minutes == 12
    assert result[0].stations[0].station_code == "NYP"


def test_gdelt_parser():
    result = news.parse_gdelt({"articles": [{"title": "Delay", "url": "https://example.com/a", "domain": "example.com", "seendate": "2026-09-23T12:00:00Z"}]})
    assert result[0].provider == "gdelt"
    assert result[0].source == "example.com"


def test_gdelt_compact_seendate_is_utc():
    result = news.parse_gdelt({"articles": [{"title": "Delay", "url": "https://example.com", "seendate": "20260923120000"}]})
    assert result[0].published_at.isoformat() == "2026-09-23T12:00:00+00:00"


def test_google_rss_parser():
    xml = "<rss><channel><item><title>News</title><link>https://example.com</link><source>Example</source><pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate></item></channel></rss>"
    result = news.parse_google_rss(xml)
    assert result[0].provider == "google_news"
    assert result[0].published_at.year == 2026


def test_google_rss_recency_fixture_has_fresh_old_and_undated():
    xml = """<rss><channel>
      <item><title>Fresh delay</title><link>https://example.com/fresh</link><pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate></item>
      <item><title>Old delay</title><link>https://example.com/old</link><pubDate>Thu, 23 Apr 2026 12:00:00 GMT</pubDate></item>
      <item><title>Undated delay</title><link>https://example.com/none</link></item>
    </channel></rss>"""
    parsed = news.parse_google_rss(xml)
    assert parsed[0].published_at is not None
    assert parsed[1].published_at is not None
    assert parsed[2].published_at is None


def test_gdelt_plain_text_falls_back(monkeypatch):
    calls = []
    def fake_request(method, url, **kwargs):
        calls.append(url)
        if "gdelt" in url:
            return 200, "rate limit", url
        return 200, "<rss><channel><item><title>Fallback</title><link>https://example.com</link></item></channel></rss>", url
    monkeypatch.setattr(news, "request", fake_request)
    monkeypatch.setattr(news.time, "sleep", lambda seconds: None)
    result = news.search("airport")
    assert result.ok
    assert result.data[0].provider == "google_news"
    assert len(calls) == 1
    assert "news.google.com" in calls[0]


def test_mta_parser():
    payload = {"entity": [{"alert": {"header_text": {"translation": [{"text": "Delays"}]}, "informed_entity": [{"route_id": "A", "stop_id": "123"}], "active_period": [{"start": 1, "end": 2}]}}]}
    result = mta.parse(payload)[0]
    assert result.header_text == "Delays"
    assert result.affected_routes == ["A"]


def test_key_redaction():
    assert "secret" not in redact_url("https://example.test/path?key=secret&x=1")
    assert "key=REDACTED" in redact_url("https://example.test/path?key=secret&x=1")


def test_replay_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("TDA_HTTP_MODE", "replay")
    monkeypatch.setenv("TDA_SNAPSHOT_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="replay snapshot miss"):
        request("GET", "https://example.test?key=secret")


def test_source_failure_is_result(monkeypatch):
    monkeypatch.setattr(nws, "request", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    result = nws.alerts(1, 2)
    assert isinstance(result, SourceResult)
    assert not result.ok
    assert result.error == "boom"


def test_http_500_is_result(monkeypatch):
    monkeypatch.setattr(open_meteo, "request", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("HTTP 500")))
    result = open_meteo.forecast(1, 2)
    assert not result.ok
    assert "HTTP 500" in result.error


def test_holidays_not_part_of_step_1a():
    from travel_agent.core.reference.holidays import holidays
    assert date(2026, 11, 26) in holidays(2026)
