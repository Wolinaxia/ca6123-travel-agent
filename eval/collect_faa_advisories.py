"""Collect real FAA Ground Delay Program (GDP) advisories (run once; output is saved).

    python -m eval.collect_faa_advisories            # 2024-01-01 .. 2026-10-31, every 3rd day

Source: the FAA ATCSCC Advisories Database (https://www.fly.faa.gov/adv/advADB),
the official public archive of every ground delay program, revision and
cancellation the FAA Command Center sent. No API key, no personal data.

For each sampled day the script reads the list of GDP advisories and then each
advisory's text, keeping only the fields M4 needs:
airport, send time, kind (issued / proposed / cancelled), AVERAGE DELAY,
MAXIMUM DELAY, program period and the impacting condition.

Output: ``travel_agent/data/processed/faa_gdp_advisories.json``. The planner never
calls the FAA site at run time; ``eval/build_gdp_history.py`` turns this file into
the small statistics table the planner uses.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

BASE = "https://www.fly.faa.gov/adv"
OUT = Path(__file__).parents[1] / "travel_agent" / "data" / "processed" / "faa_gdp_advisories.json"
HEADERS = {"User-Agent": "ca6123-course-project-research/1.0"}


def _get(url: str, tries: int = 3) -> str:
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=40) as response:
                return response.read().decode("latin-1")
        except Exception:
            time.sleep(2 + 3 * attempt)
    return ""


def _clean(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def list_gdp_advisories(day: date) -> list[dict]:
    """GDP advisories (number, airport, title, send time) sent on ``day``."""
    url = (f"{BASE}/adv_list?whichAdvisories=ATCSCC&advisoryCategory=NotAll&date={day.isoformat()}"
           "&gDelay=true&_gDelay=on&_gStop=on&_airflow=on&_ctop=on&_route=on&_other=on")
    page = _get(url)
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        link = re.search(r"adv_otherdis\?adv_date=(\d+)&advn=(\d+)", row)
        cells = [_clean(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if not link or len(cells) < 5:
            continue
        title = cells[3].upper()
        if "GROUND DELAY" not in title and "GDP" not in title:
            continue
        rows.append({"adv_date": link.group(1), "advn": int(link.group(2)), "element": cells[1],
                     "title": title, "sent": cells[4]})
    return rows


def _field(text: str, name: str) -> str | None:
    match = re.search(name + r":\s*([^\n]+)", text)
    return match.group(1).strip() if match else None


def read_advisory(row: dict) -> dict:
    url = f"{BASE}/adv_otherdis?adv_date={row['adv_date']}&advn={row['advn']}"
    page = _get(url)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", page)).replace("\xa0", " ")
    def number(name: str) -> int | None:
        value = _field(text, name)
        match = re.match(r"(\d+)", value or "")
        return int(match.group(1)) if match else None
    title = row["title"]
    kind = "cancelled" if "CNX" in title else "proposed" if "PROPOSED" in title else "issued"
    return {
        **row,
        "url": url,
        "kind": kind,
        "airport": (_field(text, "CTL ELEMENT") or row["element"].split("/")[0]).split()[0],
        "adl_time": _field(text, "ADL TIME"),
        "period": _field(text, "CUMULATIVE PROGRAM PERIOD") or _field(text, "ARRIVALS ESTIMATED FOR") or _field(text, "GDP CNX PERIOD"),
        "avg_delay": number("AVERAGE DELAY"),
        "max_delay": number("MAXIMUM DELAY"),
        "condition": _field(text, "IMPACTING CONDITION"),
    }


def main() -> None:
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2024, 1, 1)
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 10, 31)
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    days = []
    day = start
    while day <= end:
        days.append(day)
        day += timedelta(days=step)
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"advisories": []}
    done_days = set(existing.get("days") or [])
    advisories = existing["advisories"]
    todo = [d for d in days if d.isoformat() not in done_days]
    print(f"{len(todo)} days to fetch ({len(done_days)} already saved)")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, day in enumerate(todo, start=1):
            rows = list_gdp_advisories(day)
            advisories.extend(pool.map(read_advisory, rows))
            done_days.add(day.isoformat())
            if index % 10 == 0 or index == len(todo):
                OUT.write_text(json.dumps({
                    "_about": "Real FAA ATCSCC ground delay program advisories (issued, proposed, cancelled), "
                              "sampled days; collected by eval/collect_faa_advisories.py from "
                              "https://www.fly.faa.gov/adv/advADB. Fields parsed from each advisory's text.",
                    "days": sorted(done_days), "advisories": advisories}, indent=0), encoding="utf-8")
                print(f"  {index}/{len(todo)} days, {len(advisories)} advisories", flush=True)


if __name__ == "__main__":
    main()
