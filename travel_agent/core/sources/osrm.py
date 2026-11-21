"""OSRM driving route adapter."""

import json

from ..http import request, source_result
from ..models import DriveRoute, SourceResult


def parse(payload: dict) -> DriveRoute:
    route = payload["routes"][0]
    coordinates = [tuple(point) for point in route.get("geometry", {}).get("coordinates", [])]
    return DriveRoute(duration_s=route["duration"], distance_m=route["distance"], coordinates=coordinates)


def route(origin: tuple[float, float], destination: tuple[float, float]) -> SourceResult[DriveRoute]:
    url = f"https://router.project-osrm.org/route/v1/driving/{origin[1]},{origin[0]};{destination[1]},{destination[0]}?overview=simplified&geometries=geojson"
    return source_result("osrm", url, lambda: parse(json.loads(request("GET", url)[1])))
