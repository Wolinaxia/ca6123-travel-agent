"""MTA GTFS-realtime JSON alert adapter."""

from datetime import datetime, timezone
import json
from typing import Any

from ..http import request, source_result
from ..models import SourceResult, TransitAlert


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromtimestamp(int(value), tz=timezone.utc)


def parse(payload: dict[str, Any]) -> list[TransitAlert]:
    alerts = []
    for entity in payload.get("entity", []):
        item = entity.get("alert", {})
        header = item.get("header_text", {}).get("translation", [{}])[0].get("text", "")
        routes = []
        stops = []
        for informed in item.get("informed_entity", []):
            if informed.get("route_id"):
                routes.append(informed["route_id"])
            if informed.get("stop_id"):
                stops.append(informed["stop_id"])
        periods = [(_time(period.get("start")), _time(period.get("end"))) for period in item.get("active_period", [])]
        alerts.append(TransitAlert(header_text=header, affected_routes=routes, affected_stops=stops, active_periods=periods))
    return alerts


def alerts() -> SourceResult[list[TransitAlert]]:
    url = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/camsys%2Fsubway-alerts.json"
    return source_result("mta", url, lambda: parse(json.loads(request("GET", url)[1])))
