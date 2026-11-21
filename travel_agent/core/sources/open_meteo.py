"""Open-Meteo geocoding, forecast, and archive adapter."""

import json
from typing import Any
from urllib.parse import quote

from ..http import request, source_result
from ..models import SourceResult


def geocode(name: str) -> SourceResult[dict[str, Any] | None]:
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote(name)}&count=1&language=en&format=json&countryCode=US"

    def load() -> dict[str, Any] | None:
        result = json.loads(request("GET", url)[1]).get("results", [])
        if not result:
            return None
        item = result[0]
        return {key: item.get(key) for key in ("latitude", "longitude", "timezone", "population", "admin1")}

    return source_result("open_meteo_geocode", url, load)


def _weather_url(lat: float, lon: float, archive: bool, start: str | None, end: str | None) -> str:
    host = "archive-api.open-meteo.com/v1/archive" if archive else "api.open-meteo.com/v1/forecast"
    url = f"https://{host}?latitude={lat}&longitude={lon}&daily=weather_code,precipitation_sum,snowfall_sum,wind_gusts_10m_max&timezone=auto"
    if archive:
        url += f"&start_date={start}&end_date={end}"
    else:
        url += "&forecast_days=16"
    return url


def _weather(lat: float, lon: float, archive: bool, start: str | None, end: str | None) -> SourceResult[dict[str, Any]]:
    url = _weather_url(lat, lon, archive, start, end)
    return source_result("open_meteo", url, lambda: json.loads(request("GET", url)[1]).get("daily", {}))


def forecast(lat: float, lon: float) -> SourceResult[dict[str, Any]]:
    return _weather(lat, lon, False, None, None)


def archive(lat: float, lon: float, start: str, end: str) -> SourceResult[dict[str, Any]]:
    return _weather(lat, lon, True, start, end)
