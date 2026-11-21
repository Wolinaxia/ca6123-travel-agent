"""TomTom traffic incident adapter."""

import json
import math
import os
from typing import Any

from ..http import request, source_result
from ..models import SourceResult, TrafficIncident


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radians = math.pi / 180
    x = (lon2 - lon1) * radians * math.cos((lat1 + lat2) * radians / 2)
    y = (lat2 - lat1) * radians
    return 6371 * math.sqrt(x * x + y * y)


def parse(payload: dict[str, Any], center_lat: float, center_lon: float) -> list[TrafficIncident]:
    output = []
    for item in payload.get("incidents", []):
        properties = item.get("properties", item)
        geometry = item.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        if coordinates and isinstance(coordinates[0], list):
            first = coordinates[0]
            last = coordinates[-1]
            coordinates = [(first[0] + last[0]) / 2, (first[1] + last[1]) / 2]
        midpoint = (float(coordinates[1]), float(coordinates[0])) if len(coordinates) >= 2 else None
        distance = _distance_km(center_lat, center_lon, *midpoint) if midpoint else None
        output.append(
            TrafficIncident(
                icon_category=properties.get("iconCategory"),
                magnitude_of_delay=properties.get("magnitudeOfDelay"),
                description=properties.get("events", [{}])[0].get("description") if properties.get("events") else properties.get("description"),
                from_location=properties.get("from"),
                to_location=properties.get("to"),
                road_numbers=properties.get("roadNumbers", []),
                delay_seconds=properties.get("delay", properties.get("delaySeconds")),
                midpoint=midpoint,
                distance_km=distance,
            )
        )
    return output


def incidents(lat: float, lon: float, radius_km: float = 15) -> SourceResult[list[TrafficIncident]]:
    delta = radius_km / 111
    bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
    key = os.getenv("TOMTOM_API_KEY", "")
    if not key:
        from datetime import datetime, timezone
        return SourceResult(source="tomtom", ok=False, fetched_at=datetime.now(timezone.utc), url="", error="TOMTOM_API_KEY not configured (optional)")
    url = f"https://api.tomtom.com/traffic/services/5/incidentDetails?bbox={bbox}&key={key}"
    return source_result("tomtom", url, lambda: parse(json.loads(request("GET", url)[1]), lat, lon))
