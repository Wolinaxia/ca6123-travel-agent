"""Load compact BTS reference statistics."""

import json
from pathlib import Path
from typing import Any
from functools import lru_cache


ROOT = Path(__file__).parents[2] / "data/processed"


@lru_cache(maxsize=4)
def _load(name: str) -> dict[str, Any]:
    path = ROOT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def route_base_rate(origin: str, dest: str, month: int | str) -> dict[str, Any] | None:
    records = _load("bts_routes.json")
    prefix = f"{origin}-{dest}-"
    matches = [(key, value) for key, value in records.items() if key.startswith(prefix) and key.endswith(f"-{int(month):02d}") and "-2022-12" not in key]
    if matches:
        key, value = max(matches, key=lambda item: item[0])
        return {**value, "basis": key.rsplit("-", 2)[-2] + "-" + key.rsplit("-", 1)[-1]}
    annual = [value for key, value in records.items() if key.startswith(prefix) and "-2022-12" not in key]
    if not annual:
        return None
    total = sum(item.get("flights", 0) for item in annual)
    if not total:
        return None
    weighted = lambda field: sum(item.get(field, 0) * item.get("flights", 0) for item in annual) / total
    return {"flights": total, "dep_del15": weighted("dep_del15"), "arr_del15": weighted("arr_del15"), "cancel_rate": weighted("cancel_rate"), "basis": "annual average"}


def airport_base_rate(iata: str, month: int | str) -> dict[str, Any] | None:
    suffix = f"-{int(month):02d}"
    return next((value for key, value in _load("bts_airports.json").items() if key.startswith(f"{iata}-") and key.endswith(suffix)), None)


def airport_flights(iata: str) -> int:
    """Total scheduled flights for an airport across every recorded month.

    Summing (rather than taking a single month) keeps the ranking stable when an
    airport is missing the newest month in the reference file.
    """
    code = (iata or "").upper()
    records = _load("bts_airports.json")
    return sum(int(value.get("flights", 0)) for key, value in records.items() if key.startswith(f"{code}-"))



def block_minutes(origin: str, dest: str) -> float | None:
    records = _load("bts_routes.json")
    blocks = [value["median_block_min"] for key, value in records.items() if key.startswith(f"{origin}-{dest}-") and value.get("median_block_min") is not None]
    return sum(blocks) / len(blocks) if blocks else None
