"""Offline Amtrak station and direct-service lookups."""

import json
import math
from pathlib import Path
import statistics
from typing import Any


DATA = Path(__file__).parents[2] / "data/processed/amtrak.json"


def _load() -> dict[str, Any]:
    return json.loads(DATA.read_text()) if DATA.exists() else {"stations": [], "trips": []}


def _distance(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    lat1, lon1 = origin
    lat2, lon2 = destination
    radians = math.pi / 180
    x = (lon2 - lon1) * radians * math.cos((lat1 + lat2) * radians / 2)
    y = (lat2 - lat1) * radians
    return 6371 * math.sqrt(x * x + y * y)


def stations_near(lat: float, lon: float, radius_km: float) -> list[dict[str, Any]]:
    stations = []
    for station in _load()["stations"]:
        distance = _distance((lat, lon), (station["lat"], station["lon"]))
        if distance <= radius_km:
            stations.append({**station, "distance_km": distance})
    return sorted(stations, key=lambda item: item["distance_km"])


def _minutes(value: str) -> int:
    parts = value.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def direct_services(origin_latlon: tuple[float, float], dest_latlon: tuple[float, float], radius_km: float = 40) -> list[dict[str, Any]]:
    origins = stations_near(*origin_latlon, radius_km)
    destinations = stations_near(*dest_latlon, radius_km)
    origin_ids = {station["id"]: station for station in origins}
    destination_ids = {station["id"]: station for station in destinations}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for trip in _load()["trips"]:
        stops = trip.get("stops", [])
        for start_index, start in enumerate(stops):
            if start[0] not in origin_ids:
                continue
            for end in stops[start_index + 1 :]:
                if end[0] not in destination_ids:
                    continue
                duration = _minutes(end[1]) - _minutes(start[2])
                if duration <= 0:
                    duration += 24 * 60
                key = (start[0], end[0])
                service = grouped.setdefault(
                    key,
                    {
                        "from_station": origin_ids[start[0]],
                        "to_station": destination_ids[end[0]],
                        "durations": [],
                        "trip_ids": set(),
                        "route_names": set(),
                    },
                )
                service["durations"].append(duration)
                service["trip_ids"].add(trip.get("id"))
                service["route_names"].add(trip.get("route", ""))
    services = []
    for service in grouped.values():
        durations = service.pop("durations")
        trip_ids = service.pop("trip_ids")
        route_names = service.pop("route_names")
        services.append(
            {
                **service,
                "min_duration_min": min(durations),
                "median_duration_min": statistics.median(durations),
                "trains_per_day": len(trip_ids),
                "route_names": sorted(name for name in route_names if name),
            }
        )
    return sorted(
        services,
        key=lambda item: (
            -item["trains_per_day"],
            item["from_station"]["distance_km"] + item["to_station"]["distance_km"],
        ),
    )
