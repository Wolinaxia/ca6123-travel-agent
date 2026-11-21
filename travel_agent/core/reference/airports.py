"""Lookup helpers for the processed OurAirports US airport list."""

import csv
import math
from pathlib import Path


DATA_FILE = Path(__file__).parents[2] / "data/processed/airports.csv"


def _load() -> list[dict[str, str]]:
    if not DATA_FILE.exists():
        return []
    with DATA_FILE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radians = math.pi / 180
    x = (lon2 - lon1) * radians * math.cos((lat1 + lat2) * radians / 2)
    y = (lat2 - lat1) * radians
    return 6371 * math.sqrt(x * x + y * y)


def airports_near(lat: float, lon: float, radius_km: float) -> list[dict[str, str | float]]:
    airports = []
    for airport in _load():
        distance = _distance_km(lat, lon, float(airport["lat"]), float(airport["lon"]))
        if distance <= radius_km:
            airports.append({**airport, "distance_km": distance})
    return sorted(airports, key=lambda item: float(item["distance_km"]))


def get(iata: str) -> dict[str, str] | None:
    code = iata.upper()
    return next((airport for airport in _load() if airport.get("iata", "").upper() == code), None)


def by_municipality(name: str) -> dict[str, str] | None:
    target = name.strip().lower()
    matches = [airport for airport in _load() if airport.get("municipality", "").lower() == target]
    return next((airport for airport in matches if airport.get("scheduled_service", "").lower() == "yes"), None) or (matches[0] if matches else None)
