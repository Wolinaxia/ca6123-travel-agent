"""Amtraker live train adapter."""

from datetime import datetime
import json
from typing import Any

from ..http import request, source_result
from ..models import SourceResult, TrainStationTime, TrainStatus


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse(payload: dict[str, Any]) -> list[TrainStatus]:
    trains = []
    values = payload.values() if isinstance(payload, dict) else payload
    flattened = []
    for value in values:
        flattened.extend(value if isinstance(value, list) else [value])
    for item in flattened:
        if not isinstance(item, dict):
            continue
        stations = []
        for stop in item.get("stations", item.get("route", [])):
            if not isinstance(stop, dict):
                continue
            stations.append(
                TrainStationTime(
                    station_code=stop.get("code", stop.get("station", "")),
                    scheduled=_time(stop.get("scheduled", stop.get("schArr"))),
                    actual=_time(stop.get("actual", stop.get("arr", stop.get("actArr")))),
                    estimated=_time(stop.get("estimated", stop.get("estArr"))),
                )
            )
        delay = item.get("delayMinutes", item.get("delay"))
        if delay is None:
            delays = []
            for station in stations:
                if station.actual and station.scheduled:
                    delays.append((station.actual - station.scheduled).total_seconds() / 60)
            delay = round(max(delays)) if delays else None
        trains.append(
            TrainStatus(
                number=str(item.get("trainNum", item.get("number", ""))),
                route=item.get("routeName", item.get("route")) if isinstance(item.get("route"), str) else item.get("routeName"),
                state=item.get("trainState", item.get("state")),
                stations=stations,
                delay_minutes=int(delay) if isinstance(delay, (int, float)) else None,
            )
        )
    return trains


def trains() -> SourceResult[list[TrainStatus]]:
    url = "https://api-v3.amtraker.com/v3/trains"
    return source_result("amtraker", url, lambda: parse(json.loads(request("GET", url)[1])))
