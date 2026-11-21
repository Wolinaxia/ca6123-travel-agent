"""Scenario metadata and replay fixture helpers."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import SourceResult
from .sources.aviation_wx import parse as parse_weather
from .sources.faa_nas import parse as parse_faa


ROOT = Path(__file__).parents[1] / "snapshots/scenarios"


def list_scenarios() -> list[dict]:
    output = []
    for path in sorted(ROOT.glob("*/scenario.json")):
        output.append(json.loads(path.read_text(encoding="utf-8")))
    return output


def replay_inputs(name: str, now: datetime):
    return {}
