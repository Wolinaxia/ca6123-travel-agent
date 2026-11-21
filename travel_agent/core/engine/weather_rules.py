"""Shared weather-to-level mappings used by evaluation and backtests."""

from typing import Any


def open_meteo_level(daily: dict[str, list[Any]]) -> str:
    """Map one Open-Meteo daily payload to the highest weather level."""
    snowfall = max((float(value or 0) for value in daily.get("snowfall_sum", [])), default=0)
    codes = {int(value) for value in daily.get("weather_code", []) if value is not None}
    if snowfall >= 5 or codes.intersection({66, 67, 71, 73, 75, 77, 85, 86}):
        return "high"
    if codes.intersection({95, 96, 99}):
        return "elevated"
    return "low"
