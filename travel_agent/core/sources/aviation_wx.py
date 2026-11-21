"""AviationWeather.gov TAF and METAR adapter."""

from datetime import datetime, timezone
import json
import re
from typing import Any

from ..http import request, source_result
from ..models import AirportWeather, SourceResult, TafPeriod


def _parse_visibility(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("+", "")
    try:
        return float(text)
    except ValueError:
        return None


def flight_category(ceiling_ft: int | None, visibility_sm: float | None) -> str:
    ceiling = ceiling_ft if ceiling_ft is not None else 99999
    visibility = visibility_sm if visibility_sm is not None else 999
    if ceiling < 500 or visibility < 1:
        return "LIFR"
    if ceiling < 1000 or visibility < 3:
        return "IFR"
    if ceiling < 3000 or visibility < 5:
        return "MVFR"
    return "VFR"


def _weather_codes(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [code for code in ("TS", "SN", "FZRA", "RA", "BR", "FG", "-SHRA") if code in raw]


def _period(raw: dict[str, Any]) -> TafPeriod:
    wind = str(raw.get("wdir", ""))
    speed = str(raw.get("wspd", ""))
    gust = str(raw.get("wgst", ""))
    clouds = [cloud.get("base") for cloud in raw.get("clouds", []) if cloud.get("cover") in {"BKN", "OVC", "OVX"} and cloud.get("base") is not None]
    ceiling = min(clouds) if clouds else None
    return TafPeriod(
        start=datetime.fromtimestamp(raw["timeFrom"], timezone.utc) if raw.get("timeFrom") else None,
        end=datetime.fromtimestamp(raw["timeTo"], timezone.utc) if raw.get("timeTo") else None,
        wind_direction=int(wind) if wind.isdigit() else None,
        wind_speed_kt=int(speed) if speed.isdigit() else None,
        gust_kt=int(gust) if gust.isdigit() else None,
        visibility_sm=_parse_visibility(raw.get("visib")),
        ceiling_ft=int(raw["ceil"]) if str(raw.get("ceil", "")).isdigit() else ceiling,
        weather_codes=_weather_codes(raw.get("wxString") or raw.get("rawTaf")),
    )


def parse(taf_payload: str | list[dict[str, Any]], metar_payload: str | list[dict[str, Any]]) -> dict[str, AirportWeather]:
    taf = json.loads(taf_payload) if isinstance(taf_payload, str) else taf_payload
    metar = json.loads(metar_payload) if isinstance(metar_payload, str) else metar_payload
    metar_by_icao = {item.get("icaoId"): item for item in metar}
    output: dict[str, AirportWeather] = {}
    for item in taf:
        icao = item.get("icaoId", "")
        observation = metar_by_icao.get(icao, {})
        ceiling = observation.get("ceiling") or observation.get("ceil")
        if ceiling is None:
            cloud_bases = [cloud.get("base") for cloud in observation.get("clouds", []) if cloud.get("cover") in {"BKN", "OVC", "OVX"} and cloud.get("base") is not None]
            ceiling = min(cloud_bases) if cloud_bases else None
        visibility = _parse_visibility(observation.get("visib"))
        output[icao] = AirportWeather(
            icao=icao,
            flight_category=observation.get("flightCategory") or observation.get("fltCat") or flight_category(ceiling, visibility),
            metar=observation,
            taf=[_period(period) for period in item.get("fcsts", [])],
        )
    return output


def fetch(icaos: list[str]) -> SourceResult[dict[str, AirportWeather]]:
    query = ",".join(icaos)
    taf_url = f"https://aviationweather.gov/api/data/taf?ids={query}&format=json"
    metar_url = f"https://aviationweather.gov/api/data/metar?ids={query}&format=json"

    def load() -> dict[str, AirportWeather]:
        taf = request("GET", taf_url)[1]
        metar = request("GET", metar_url)[1]
        return parse(taf, metar)

    return source_result("aviation_wx", taf_url, load)
