"""National Weather Service adapter."""

from datetime import datetime
import json
from typing import Any

from ..http import request, source_result
from ..models import NwsAlert, NwsForecastPeriod, NwsPoint, SourceResult


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def parse_points(payload: dict[str, Any]) -> NwsPoint:
    if "properties" not in payload:
        raise ValueError("NWS points response missing properties")
    properties = payload["properties"]
    return NwsPoint(forecast_url=properties["forecast"], timezone=properties["timeZone"])


def parse_forecast(payload: dict[str, Any]) -> list[NwsForecastPeriod]:
    if "properties" not in payload:
        raise ValueError("NWS forecast response missing properties")
    periods = []
    for item in payload.get("properties", {}).get("periods", []):
        probability = item.get("probabilityOfPrecipitation", {}).get("value")
        periods.append(
            NwsForecastPeriod(
                start=_dt(item["startTime"]),
                end=_dt(item["endTime"]),
                short_forecast=item.get("shortForecast", ""),
                detailed_forecast=item.get("detailedForecast", ""),
                wind=f"{item.get('windDirection', '')} {item.get('windSpeed', '')}".strip(),
                precipitation_probability=probability,
            )
        )
    return periods


def parse_alerts(payload: dict[str, Any]) -> list[NwsAlert]:
    alerts = []
    for feature in payload.get("features", []):
        item = feature.get("properties", {})
        alerts.append(
            NwsAlert(
                event=item.get("event", ""),
                severity=item.get("severity"),
                urgency=item.get("urgency"),
                certainty=item.get("certainty"),
                onset=_dt(item.get("onset")),
                ends=_dt(item.get("ends")),
                expires=_dt(item.get("expires")),
                headline=item.get("headline"),
                area_desc=item.get("areaDesc"),
                id=feature.get("id") or item.get("id"),
            )
        )
    return alerts


def points(lat: float, lon: float) -> SourceResult[NwsPoint]:
    url = f"https://api.weather.gov/points/{lat},{lon}"
    result = source_result("nws_points", url, lambda: parse_points(json.loads(request("GET", url)[1])))
    if not result.ok:
        rounded_url = f"https://api.weather.gov/points/{lat:.3f},{lon:.3f}"
        return source_result("nws_points", rounded_url, lambda: parse_points(json.loads(request("GET", rounded_url)[1])))
    return result


def forecast(lat: float, lon: float) -> SourceResult[list[NwsForecastPeriod]]:
    point = points(lat, lon)
    if not point.ok:
        point = points(lat, lon)
    if not point.ok or point.data is None:
        return point
    if isinstance(point.data, dict):
        url = point.data.get("properties", {}).get("forecast", "")
        if not url:
            raise ValueError("NWS points response missing forecast URL")
    else:
        url = point.data.forecast_url
    return source_result("nws_forecast", url, lambda: parse_forecast(json.loads(request("GET", url)[1])))


def alerts(lat: float, lon: float) -> SourceResult[list[NwsAlert]]:
    url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
    return source_result("nws_alerts", url, lambda: parse_alerts(json.loads(request("GET", url)[1])))
