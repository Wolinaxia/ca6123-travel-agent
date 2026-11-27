"""Collect the REAL flights that landed during real FAA ground delay programs (run once).

    python -m eval.collect_bts_gdp_flights            # months 2025-01 .. 2026-08

Joins two public US government datasets:

* the FAA ground delay program advisories saved by ``eval/collect_faa_advisories.py``
  (which airport, from when, what average delay the FAA announced);
* the BTS "Reporting Carrier On-Time Performance" per-flight files
  (https://transtats.bts.gov/PREZIP/), which say how late every US airline flight
  actually arrived, or whether it was cancelled or diverted.

For every program it keeps the flights scheduled to ARRIVE at the program airport,
or at another airport of the same metro area (the planner's alternatives), during
the program window, that had not yet departed when the program was announced, i.e.
flights a traveller could still have chosen at that moment.

Output (small, git-ignored raw folder): ``travel_agent/data/raw/bts_gdp_flights.json``.
``eval/backtest_gdp_history.py`` uses it; the planner never reads it.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parents[1]
ADVISORIES = ROOT / "travel_agent" / "data" / "processed" / "faa_gdp_advisories.json"
RAW = ROOT / "travel_agent" / "data" / "raw"
OUT = RAW / "bts_gdp_flights.json"
PREZIP = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
METRO_KM = 60


def _airport_table() -> dict[str, dict]:
    table = {}
    with open(ROOT / "travel_agent" / "data" / "processed" / "airports.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["iata"]:
                table[row["iata"].upper()] = row
    return table


def _period(text: str | None, adv_date: str) -> tuple[datetime, datetime] | None:
    """'23/2100Z - 24/0359Z' with adv_date 'MMDDYYYY' -> UTC datetimes."""
    match = re.match(r"(\d{2})/(\d{4})Z\s*-\s*(\d{2})/(\d{4})Z", (text or "").strip())
    if not match:
        return None
    sent = datetime(int(adv_date[4:]), int(adv_date[:2]), int(adv_date[2:4]), tzinfo=timezone.utc)

    def at(day: str, hhmm: str) -> datetime:
        # The period gives only day-of-month: pick the month (previous, same,
        # next) that puts it closest to the day the advisory was sent.
        options = []
        for shift in (-1, 0, 1):
            first = (sent.replace(day=1) + timedelta(days=32 * shift)).replace(day=1) if shift >= 0 else \
                (sent.replace(day=1) - timedelta(days=1)).replace(day=1)
            try:
                options.append(first.replace(day=int(day), hour=int(hhmm[:2]) % 24, minute=int(hhmm[2:])))
            except ValueError:
                continue
        return min(options, key=lambda value: abs(value - sent))

    start, end = at(match.group(1), match.group(2)), at(match.group(3), match.group(4))
    return (start, end) if end > start else None


def episodes() -> list[dict]:
    """First 'issued' GDP advisory of each program (US airports), with its window."""
    data = json.loads(ADVISORIES.read_text(encoding="utf-8"))
    table = _airport_table()
    groups = defaultdict(list)
    for item in data["advisories"]:
        if item["airport"] in table and item["kind"] in {"issued", "cancelled"}:
            groups[(item["adv_date"], item["airport"])].append(item)
    out = []
    for (adv_date, airport), items in groups.items():
        items.sort(key=lambda item: datetime.strptime(item["sent"], "%m/%d/%y %H:%M"))
        current = None
        for item in items:
            sent = datetime.strptime(item["sent"], "%m/%d/%y %H:%M").replace(tzinfo=timezone.utc)
            if item["kind"] == "issued" and item["avg_delay"] is not None:
                window = _period(item["period"], adv_date)
                if window is None:
                    continue
                if current is None:
                    current = {"airport": airport, "adv_date": adv_date, "sent": sent.isoformat(), "avg_delay": item["avg_delay"],
                               "max_delay": item["max_delay"], "condition": item["condition"], "start": window[0].isoformat(),
                               "end": window[1].isoformat(), "revisions": [], "cancelled_at": None, "url": item["url"]}
                    out.append(current)
                else:
                    current["revisions"].append({"sent": sent.isoformat(), "avg_delay": item["avg_delay"], "end": window[1].isoformat()})
            elif item["kind"] == "cancelled" and current is not None:
                current["cancelled_at"] = sent.isoformat()
                current = None
    return out


def metro_alternates(code: str, table: dict[str, dict]) -> list[str]:
    from travel_agent.core.reference import bts
    from travel_agent.core.reference.airports import airports_near

    home = table[code]
    near = airports_near(float(home["lat"]), float(home["lon"]), METRO_KM)
    return sorted({str(item["iata"]).upper() for item in near
                   if item.get("iata") and str(item["iata"]).upper() != code and bts.airport_flights(str(item["iata"])) > 0})


def _month_rows(year: int, month: int, wanted: dict, table: dict) -> list[list]:
    """Stream one BTS month and keep flights inside a wanted (dest, window)."""
    request = urllib.request.Request(PREZIP.format(y=year, m=month), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        blob = response.read()
    rows = []
    tz_cache: dict[str, ZoneInfo] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(name) as handle:
            for row in csv.DictReader(io.TextIOWrapper(handle, encoding="latin-1")):
                dest = row["Dest"]
                if dest not in wanted:
                    continue
                origin = row["Origin"]
                if origin not in table:
                    continue
                tz = tz_cache.setdefault(origin, ZoneInfo(table[origin]["timezone"]))
                hhmm = row["CRSDepTime"].zfill(4)
                day = datetime.strptime(row["FlightDate"], "%Y-%m-%d")
                dep_utc = day.replace(hour=int(hhmm[:2]) % 24, minute=int(hhmm[2:]), tzinfo=tz).astimezone(timezone.utc)
                arr_utc = dep_utc + timedelta(minutes=float(row["CRSElapsedTime"] or 0))
                for index, (start, end, sent) in wanted[dest]:
                    if start <= arr_utc <= end and dep_utc >= sent:
                        delay = row["ArrDelay"]
                        rows.append([index, dest, origin, dep_utc.isoformat()[:16], arr_utc.isoformat()[:16],
                                     None if delay == "" else float(delay), int(float(row["Cancelled"] or 0)),
                                     int(float(row["Diverted"] or 0))])
    return rows


def main() -> None:
    first = sys.argv[1] if len(sys.argv) > 1 else "2025-01"
    last = sys.argv[2] if len(sys.argv) > 2 else "2026-08"
    table = _airport_table()
    eps = episodes()
    months = []
    y, m = map(int, first.split("-"))
    while f"{y}-{m:02d}" <= last:
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    month_keys = {f"{y}-{m:02d}" for y, m in months}
    eps = [ep for ep in eps if ep["start"][:7] in month_keys]
    alt_cache: dict[str, list[str]] = {}
    for index, ep in enumerate(eps):
        ep["index"] = index
        ep["alternates"] = alt_cache.setdefault(ep["airport"], metro_alternates(ep["airport"], table))
    RAW.mkdir(parents=True, exist_ok=True)
    flights: list[list] = []

    def run(month: tuple[int, int]) -> list[list]:
        y, m = month
        wanted: dict[str, list] = defaultdict(list)
        for ep in eps:
            start = datetime.fromisoformat(ep["start"])
            if (start.year, start.month) != (y, m):
                continue
            window = (start, datetime.fromisoformat(ep["end"]), datetime.fromisoformat(ep["sent"]))
            for code in [ep["airport"], *ep["alternates"]]:
                wanted[code].append((ep["index"], window))
        for attempt in range(3):
            try:
                rows = _month_rows(y, m, wanted, table)
                print(f"  {y}-{m:02d}: {len(rows)} flights", flush=True)
                return rows
            except Exception as exc:
                print(f"  {y}-{m:02d}: retry ({str(exc)[:80]})", flush=True)
        return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        for rows in pool.map(run, months):
            flights.extend(rows)
    OUT.write_text(json.dumps({
        "_about": "BTS on-time flights arriving during real FAA GDP windows (and at same-metro alternates). "
                  "Row: [episode index, dest, origin, sched dep UTC, sched arr UTC, arrival delay min, cancelled, diverted].",
        "months": sorted(month_keys), "episodes": eps, "flights": flights}), encoding="utf-8")
    print(f"wrote {len(eps)} programs and {len(flights)} flights to {OUT}")


if __name__ == "__main__":
    main()
