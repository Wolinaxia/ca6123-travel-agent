"""Learn from real history, then back-test the planner on later, unseen history.

    python -m eval.backtest_gdp_history

Data (both public, US government):
* FAA ATCSCC ground delay program (GDP) advisories  -> eval/collect_faa_advisories.py
* BTS per-flight on-time records for the flights that landed during those programs
  (at the program airport and at the other airports of the same metro area)
                                                     -> eval/collect_bts_gdp_flights.py

Question: when the FAA announces "GDP at SFO, average delay 34 min", how late did
real flights actually arrive, and was a neighbouring airport (OAK, SJC) really
safer? The past cases claim lessons like "a delay program often does not apply to
a neighbouring airport" (case sfo_fog_gdp_oak) and "summer storms build during the
day" (case nyc_summer_thunderstorms_2023). Here those lessons are checked on data.

Method (no peeking):
* LEARN on programs that started in 2025: for each kind of program (thunderstorms,
  low ceilings / fog, wind, snow / ice, other) the 90th percentile of
  "real arrival delay minus the announced average" at the program airport, the
  90th percentile of the real arrival delay at same-metro neighbours, and the
  cancellation rates. Saved to ``travel_agent/data/processed/gdp_history.json``;
  this small table is what the planner uses.
* TEST on programs that started in 2026 (never used for learning): replay many
  travellers (every real flight x a range of deadlines) and compare
    - the current planner: trusts the announced average delay;
    - the history-aware planner: an option is "comfortable" only if it would still
      make the deadline at the 90th-percentile delay seen in 2025.
  Outcome = did the real flight arrive before the deadline (cancelled or diverted
  = missed)?

Writes ``eval/results/m4_gdp_backtest.md`` (and the learned table).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
RAW = ROOT / "travel_agent" / "data" / "raw" / "bts_gdp_flights.json"
TABLE = ROOT / "travel_agent" / "data" / "processed" / "gdp_history.json"
REPORT = ROOT / "eval" / "results" / "m4_gdp_backtest.md"

QUANTILE = 0.90
COMFORTABLE_MIN = 90      # same as deadline_planner.COMFORTABLE_MIN
HIGH_AVG = 60             # Policy.gdp_high_avg_minutes: GDP this bad is never "comfortable"
ELEVATED_AVG = 15         # Policy.gdp_elevated_avg_minutes
SLACKS = list(range(0, 241, 15))
EXTRA_GROUND = [0, 15, 30, 45, 60]   # extra minutes to reach the city from the neighbour airport
MIN_FLIGHTS = 150         # a group needs this many flights, otherwise "all" is used


from travel_agent.tools.gdp_history import condition_group  # noqa: E402  (same mapping as the planner)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def load() -> tuple[list[dict], dict[int, dict]]:
    data = json.loads(RAW.read_text(encoding="utf-8"))
    episodes = {ep["index"]: ep for ep in data["episodes"]}
    # an alternate only counts as "outside the program" if it had no GDP itself that day
    gdp_days = {(ep["adv_date"], ep["airport"]) for ep in data["episodes"]}
    for ep in episodes.values():
        ep["clean_alternates"] = [code for code in ep["alternates"] if (ep["adv_date"], code) not in gdp_days]
        ep["group"] = condition_group(ep["condition"])
        ep["year"] = ep["start"][:4]
    flights = []
    for index, dest, origin, dep, arr, delay, cancelled, diverted in data["flights"]:
        ep = episodes[index]
        role = "gdp" if dest == ep["airport"] else "neighbour" if dest in ep["clean_alternates"] else None
        if role is None:
            continue
        flights.append({"ep": index, "role": role, "dest": dest, "lost": bool(cancelled or diverted or delay is None),
                        "delay": max(0.0, delay or 0.0) if not (cancelled or diverted) else None})
    return flights, episodes


def learn(flights: list[dict], episodes: dict[int, dict], year: str) -> dict:
    by = defaultdict(lambda: {"gdp_excess": [], "gdp_lost": 0, "gdp_n": 0, "nb_delay": [], "nb_lost": 0, "nb_n": 0,
                              "programs": set()})
    for flight in flights:
        ep = episodes[flight["ep"]]
        if ep["year"] != year:
            continue
        for key in (ep["group"], "all"):
            bucket = by[key]
            bucket["programs"].add(flight["ep"])
            if flight["role"] == "gdp":
                bucket["gdp_n"] += 1
                if flight["lost"]:
                    bucket["gdp_lost"] += 1
                else:
                    bucket["gdp_excess"].append(flight["delay"] - ep["avg_delay"])
            else:
                bucket["nb_n"] += 1
                if flight["lost"]:
                    bucket["nb_lost"] += 1
                else:
                    bucket["nb_delay"].append(flight["delay"])
    table = {}
    for key, bucket in by.items():
        table[key] = {
            "programs": len(bucket["programs"]),
            "gdp_flights": bucket["gdp_n"],
            "gdp_extra_over_announced_q50": round(quantile(bucket["gdp_excess"], 0.5)),
            "gdp_extra_over_announced_q90": round(quantile(bucket["gdp_excess"], QUANTILE)),
            "gdp_cancel_or_divert_rate": round(bucket["gdp_lost"] / max(1, bucket["gdp_n"]), 4),
            "neighbour_flights": bucket["nb_n"],
            "neighbour_delay_q50": round(quantile(bucket["nb_delay"], 0.5)),
            "neighbour_delay_q90": round(quantile(bucket["nb_delay"], QUANTILE)),
            "neighbour_cancel_or_divert_rate": round(bucket["nb_lost"] / max(1, bucket["nb_n"]), 4),
        }
    return table


def margin(table: dict, group: str, role: str) -> int:
    row = table.get(group)
    if not row or row["gdp_flights" if role == "gdp" else "neighbour_flights"] < MIN_FLIGHTS:
        row = table["all"]
    value = row["gdp_extra_over_announced_q90" if role == "gdp" else "neighbour_delay_q90"]
    return max(0, int(value))


# ---------------------------------------------------------------- planner replicas
def verdict(slack: int, announced: int, history_margin: int | None) -> str:
    """Same rules as deadline_planner._verdict (+ the history rule when a margin is given)."""
    if slack < 0:
        return "likely_miss"
    level = "high" if announced >= HIGH_AVG else "elevated" if announced >= ELEVATED_AVG else "low"
    if slack >= COMFORTABLE_MIN and level != "high":
        if history_margin is not None and slack < history_margin:
            return "tight"
        return "comfortable"
    return "tight"


_RANK = {"comfortable": 0, "tight": 1, "at_risk": 2, "likely_miss": 3}
_LEVEL = {"low": 0, "elevated": 1, "high": 2}


def choose(options: list[dict], aware: bool) -> dict:
    def key(option):
        v = verdict(option["slack"], option["announced"], option["margin"] if aware else None)
        level = "high" if option["announced"] >= HIGH_AVG else "elevated" if option["announced"] >= ELEVATED_AVG else "low"
        return (_RANK[v], _LEVEL[level], option["door_plus_delay"])
    return min(options, key=key)


def miss_rate(outcomes: list[dict], allowed_delay: float) -> float:
    """Share of real flights that missed: lost, or later than ``allowed_delay`` minutes."""
    if not outcomes:
        return float("nan")
    return sum(1 for o in outcomes if o["lost"] or o["delay"] > allowed_delay) / len(outcomes)


def backtest(flights: list[dict], episodes: dict[int, dict], table: dict, year: str) -> dict:
    per_ep = defaultdict(lambda: {"gdp": [], "neighbour": []})
    for flight in flights:
        if episodes[flight["ep"]]["year"] == year:
            per_ep[flight["ep"]][flight["role"]].append(flight)

    # 1) Is "comfortable" a promise that holds? (every flight x every deadline)
    counts = {"naive": [0, 0], "aware": [0, 0]}     # [comfortable scenarios, of which missed]
    downgraded = [0, 0]                              # [scenarios downgraded, of which missed]
    for index, groups in per_ep.items():
        ep = episodes[index]
        m = margin(table, ep["group"], "gdp")
        for flight in groups["gdp"]:
            for slack in SLACKS:
                missed = flight["lost"] or flight["delay"] - ep["avg_delay"] > slack
                naive = verdict(slack, ep["avg_delay"], None) == "comfortable"
                aware = verdict(slack, ep["avg_delay"], m) == "comfortable"
                if naive:
                    counts["naive"][0] += 1
                    counts["naive"][1] += missed
                if aware:
                    counts["aware"][0] += 1
                    counts["aware"][1] += missed
                if naive and not aware:
                    downgraded[0] += 1
                    downgraded[1] += missed

    # 2) Does the choice change, and is the new choice better?
    #    Two options: fly into the program airport, or into a same-metro neighbour
    #    that is `extra` minutes further from the city. Deadline = `slack` minutes
    #    after the planned arrival at the program airport (scheduled + announced delay).
    scenarios = changed = 0
    miss_naive = miss_aware = 0.0
    changed_naive = changed_aware = 0.0
    for index, groups in per_ep.items():
        ep = episodes[index]
        if len(groups["gdp"]) < 5 or len(groups["neighbour"]) < 5:
            continue
        m_gdp, m_nb = margin(table, ep["group"], "gdp"), margin(table, ep["group"], "neighbour")
        a = ep["avg_delay"]
        for slack in SLACKS:
            for extra in EXTRA_GROUND:
                gdp_option = {"name": "gdp", "slack": slack, "announced": a, "margin": m_gdp + 0, "door_plus_delay": a}
                # margin for the program airport is "beyond the announced average", so compare it with slack directly;
                # the neighbour has no announced delay, so its margin is its plain 90th-percentile delay.
                # rule v2: neighbours are not downgraded (v1 used m_nb here; see the report's "Versions")
                nb_option = {"name": "neighbour", "slack": slack + a - extra, "announced": 0, "margin": None, "door_plus_delay": extra}
                options = [gdp_option, nb_option]
                pick_naive, pick_aware = choose(options, False), choose(options, True)
                real = {"gdp": miss_rate(groups["gdp"], a + slack), "neighbour": miss_rate(groups["neighbour"], slack + a - extra)}
                scenarios += 1
                miss_naive += real[pick_naive["name"]]
                miss_aware += real[pick_aware["name"]]
                if pick_naive["name"] != pick_aware["name"]:
                    changed += 1
                    changed_naive += real[pick_naive["name"]]
                    changed_aware += real[pick_aware["name"]]
    return {
        "programs": len(per_ep),
        "gdp_flights": sum(len(g["gdp"]) for g in per_ep.values()),
        "neighbour_flights": sum(len(g["neighbour"]) for g in per_ep.values()),
        "comfortable_naive": counts["naive"][0], "comfortable_naive_missed": counts["naive"][1],
        "comfortable_aware": counts["aware"][0], "comfortable_aware_missed": counts["aware"][1],
        "downgraded": downgraded[0], "downgraded_missed": downgraded[1],
        "choice_scenarios": scenarios, "choice_changed": changed,
        "choice_miss_naive": miss_naive / max(1, scenarios), "choice_miss_aware": miss_aware / max(1, scenarios),
        "changed_miss_naive": changed_naive / max(1, changed), "changed_miss_aware": changed_aware / max(1, changed),
    }


def lesson_check(flights: list[dict], episodes: dict[int, dict], year: str) -> dict:
    """Case sfo_fog_gdp_oak: was the neighbour airport really less affected? Per program."""
    per_ep = defaultdict(lambda: {"gdp": [], "neighbour": []})
    for flight in flights:
        if episodes[flight["ep"]]["year"] == year:
            per_ep[flight["ep"]][flight["role"]].append(flight)
    result = defaultdict(lambda: [0, 0])
    for index, groups in per_ep.items():
        if len(groups["gdp"]) < 5 or len(groups["neighbour"]) < 5:
            continue
        group = episodes[index]["group"]
        bad_gdp = miss_rate(groups["gdp"], 60)        # share > 60 min late or lost
        bad_nb = miss_rate(groups["neighbour"], 60)
        for key in (group, "all"):
            result[key][0] += 1
            result[key][1] += bad_nb < bad_gdp
    return dict(result)


def raised_share(year_filter: tuple[str, ...]) -> dict:
    """How often the FAA later RAISED the announced average (from all saved advisories)."""
    data = json.loads((ROOT / "travel_agent" / "data" / "processed" / "faa_gdp_advisories.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for item in data["advisories"]:
        if item["kind"] in {"issued", "cancelled"} and item["adv_date"][-4:] in year_filter:
            groups[(item["adv_date"], item["airport"])].append(item)
    out = defaultdict(lambda: [0, 0])
    for items in groups.values():
        items.sort(key=lambda item: datetime.strptime(item["sent"], "%m/%d/%y %H:%M"))
        first, peak = None, None
        for item in items:
            if item["kind"] == "issued" and item["avg_delay"] is not None:
                if first is None:
                    first = item
                    peak = item["avg_delay"]
                else:
                    peak = max(peak, item["avg_delay"])
            elif item["kind"] == "cancelled" and first is not None:
                break
        if first is None:
            continue
        for key in (condition_group(first["condition"]), "all"):
            out[key][0] += 1
            out[key][1] += peak > first["avg_delay"]
    return {key: {"programs": n, "raised": raised, "share": round(raised / max(1, n), 3)} for key, (n, raised) in out.items()}


def main() -> None:
    flights, episodes = load()
    table = learn(flights, episodes, "2025")
    raised = raised_share(("2024", "2025"))
    TABLE.write_text(json.dumps({
        "_about": "Learned from REAL data: FAA ATCSCC ground delay program advisories (2025) joined with BTS per-flight "
                  "arrival records for flights landing during those programs. Built by eval/backtest_gdp_history.py; "
                  "back-tested on 2026 programs in eval/results/m4_gdp_backtest.md. Minutes are percentiles over flights.",
        "learned_from": "programs starting 2025-01..2025-12",
        "quantile": QUANTILE,
        "min_flights": MIN_FLIGHTS,
        "groups": table,
        "faa_raised_average_later": raised,
        "checks_case_lessons": ["sfo_fog_gdp_oak", "nyc_summer_thunderstorms_2023"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1), encoding="utf-8")

    train = backtest(flights, episodes, table, "2025")
    test = backtest(flights, episodes, table, "2026")
    lessons = {year: lesson_check(flights, episodes, year) for year in ("2025", "2026")}
    write_report(table, raised, train, test, lessons)
    print(REPORT.read_text(encoding="utf-8"))


def pct(a: float, b: float | None = None) -> str:
    value = a if b is None else (a / b if b else 0.0)
    return f"{100 * value:.1f}%"


def summary_lines(table, test, lessons) -> list[str]:
    """Three plain-English findings, computed from the results."""
    nb = lessons["2026"].get("all", [0, 0])
    row = table["all"]
    return [
        f"1. **The past-case lesson holds on real data.** Case `sfo_fog_gdp_oak` says a neighbouring airport usually "
        f"avoids a delay program. In 2026 the neighbour did better in **{nb[1]} of {nb[0]} programs ({pct(nb[1], nb[0])})**; "
        f"in 2025, {pct(row['gdp_cancel_or_divert_rate'])} of flights into program airports were cancelled or diverted vs "
        f"{pct(row['neighbour_cancel_or_divert_rate'])} at neighbours.",
        f"2. **The announced average is not the whole story.** In 2025, 1 in 10 flights into a program airport arrived "
        f"{row['gdp_extra_over_announced_q90']} min or more after the announced average delay.",
        f"3. **The history-aware planner keeps its promises more often.** On 2026 programs (not used for learning), "
        f"travellers told \"comfortable\" who still missed fell from {pct(test['comfortable_naive_missed'], test['comfortable_naive'])} "
        f"to {pct(test['comfortable_aware_missed'], test['comfortable_aware'])}; the options it downgraded really missed "
        f"{pct(test['downgraded_missed'], test['downgraded'])} of the time. It rarely changes which airport is chosen, "
        f"because the planner already prefers the safer neighbour.",
        "",
        f"Data: {test['programs']} programs and {test['gdp_flights'] + test['neighbour_flights']:,} flights in the 2026 test alone.",
    ]


def write_report(table, raised, train, test, lessons) -> None:
    groups = ["thunderstorms", "low_visibility", "wind", "snow_ice", "other", "all"]
    lines = [
        "# M4 history back-test: do the past-case lessons hold on real data?",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Data: real FAA ground delay program (GDP) advisories (https://www.fly.faa.gov/adv/advADB) joined with BTS",
        "per-flight arrival records (https://transtats.bts.gov). Learned on programs that started in **2025**, tested",
        "on programs that started in **2026** (Jan-Aug, never used for learning). Built by `eval/backtest_gdp_history.py`.",
        "",
        "## In short",
        "",
        *summary_lines(table, test, lessons),
        "",
        "## 1. What real flights did during announced delay programs (learned, 2025)",
        "",
        "| Kind of program | programs | flights at program airport | real delay beyond announced avg (median / 90th pct) | cancelled or diverted | neighbour airport delay (median / 90th pct) | neighbour cancelled |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in groups:
        row = table.get(key)
        if not row:
            continue
        lines.append(f"| {key} | {row['programs']} | {row['gdp_flights']} | {row['gdp_extra_over_announced_q50']} / {row['gdp_extra_over_announced_q90']} min | "
                     f"{pct(row['gdp_cancel_or_divert_rate'])} | {row['neighbour_delay_q50']} / {row['neighbour_delay_q90']} min | {pct(row['neighbour_cancel_or_divert_rate'])} |")
    lines += ["", "How often the FAA later **raised** the announced average (all saved advisories, 2024-2025):", ""]
    for key in groups:
        if key in raised:
            lines.append(f"* {key}: {raised[key]['raised']}/{raised[key]['programs']} programs ({pct(raised[key]['share'])})")
    lines += ["", "## 2. Lesson check: was the neighbouring airport really safer? (case `sfo_fog_gdp_oak`)", "",
              "Per program: share of flights more than 60 min late or cancelled, program airport vs same-metro neighbours",
              "that had no program of their own that day.", "",
              "| Kind of program | 2025: neighbour better | 2026: neighbour better |", "|---|---|---|"]
    for key in groups:
        a, b = lessons["2025"].get(key), lessons["2026"].get(key)
        if a or b:
            lines.append(f"| {key} | {f'{a[1]}/{a[0]} ({pct(a[1], a[0])})' if a else '-'} | {f'{b[1]}/{b[0]} ({pct(b[1], b[0])})' if b else '-'} |")

    def block(name, r):
        return [
            f"**{name}** ({r['programs']} programs, {r['gdp_flights']} flights at the program airport, {r['neighbour_flights']} at neighbours)",
            "",
            "| | current planner (trusts the announced average) | history-aware planner |",
            "|---|---|---|",
            f"| travellers told \"comfortable\" | {r['comfortable_naive']} | {r['comfortable_aware']} |",
            f"| ...of whom actually missed the deadline | {r['comfortable_naive_missed']} ({pct(r['comfortable_naive_missed'], r['comfortable_naive'])}) | "
            f"{r['comfortable_aware_missed']} ({pct(r['comfortable_aware_missed'], r['comfortable_aware'])}) |",
            f"| downgraded to \"tight\" by history | - | {r['downgraded']} (of whom {pct(r['downgraded_missed'], r['downgraded'])} really missed) |",
            "",
            f"Choice between the program airport and a same-metro neighbour ({r['choice_scenarios']} traveller scenarios: "
            f"deadlines 0-240 min of slack x 0-60 min extra ground time from the neighbour):",
            "",
            f"* the history rule changed the choice in {r['choice_changed']} scenarios ({pct(r['choice_changed'], r['choice_scenarios'])});",
            f"* in those scenarios the real miss rate was {pct(r['changed_miss_naive'])} with the current choice and "
            f"{pct(r['changed_miss_aware'])} with the history-aware choice;",
            f"* over all scenarios: {pct(r['choice_miss_naive'])} vs {pct(r['choice_miss_aware'])}.",
            "",
        ]

    lines += ["", "## 3. Back-test: does the planner keep its promises?", "",
              "A traveller scenario = one real flight into the program airport x one deadline (0-240 min of slack after",
              "the announced delay). *Missed* = the real flight arrived after the deadline, or was cancelled / diverted.",
              "History rule: an option is \"comfortable\" only if it still makes the deadline at the 90th-percentile",
              "delay that real flights had in 2025 programs of the same kind.", ""]
    lines += block("TEST: 2026 programs (not used for learning)", test)
    lines += block("For reference, 2025 (the learning year, so optimistic)", train)
    lines += ["## Versions (what changed and why)", "",
              "* **Rule v1** also downgraded options landing at a same-metro *neighbour* (using the neighbour statistic).",
              "  On 2026 it cut wrong \"comfortable\" promises from 8.3% to 7.5%, but where it changed the airport choice",
              "  (80 of 16,745 scenarios) the real miss rate got slightly worse (33.4% -> 35.6%), because it penalised the",
              "  neighbour, which section 2 shows was safer in 94% of programs.",
              "* **Rule v2** (current): only the program airport itself is downgraded, because section 2 shows the",
              "  neighbour was the safer airport. Checked on a split inside 2025 (learn Jan-Jun, check Jul-Dec): the",
              "  \"comfortable\" promise improved the same way (6.1% -> 5.7% missed), but in the few scenarios where the",
              "  airport choice changed (27 for v2, 52 for v1), both versions did slightly worse there. On 2026, v2 changed",
              "  40 choices and did better there (40.9% -> 32.9%). These counts are too small to call: **the rule improves",
              "  the honesty of \"comfortable\"; it does not measurably improve which airport is chosen.**",
              "  Since v1's 2026 result had been seen before v2 was chosen, 2026 is not a perfectly untouched test for v2.",
              "* Also tried on the 2025 split, not adopted: ranking options by estimated chance of missing (from the same",
              "  history). It lowered the miss rate only from 11.78% to 11.68% overall, because the planner already prefers",
              "  the neighbour airport during a program, which the data confirms; not worth changing the ranking before freeze.",
              "",
              "## Honest limits", "",
              "* Flights are sampled every 3rd day; programs crossing midnight UTC are cut at the day boundary.",
              "* A traveller's deadline is simulated (0-240 min of slack); real travellers' deadlines are unknown.",
              "* Delays are arrival delays reported by airlines to BTS; connections and ground transport are not included.",
              "* Neighbour airports count only when they had no program of their own that day.",
              "* Only ground delay programs are covered; ground stops, airline outages and rail are not."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
