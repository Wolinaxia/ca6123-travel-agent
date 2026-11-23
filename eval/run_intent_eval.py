"""Score intent routing and slot grounding (M2, Reason stage).

Every case in ``intent_cases.jsonl`` is run through the same two steps the
orchestrator uses, ``classify_intent`` then ``planner.ground``, with a fixed
clock (Sunday 2026-11-22 12:00 New York) so relative dates are reproducible.

Two classifiers are compared on the same cases:

* **rules**: the keyword fallback (offline, no tokens): the baseline;
* **llm**: Gemini few-shot classifier (one call per case, needs GOOGLE_API_KEY).

None of the cases repeats a few-shot example from the prompt (checked by
``tests/test_intent_eval.py``), so the LLM is not graded on data it was shown.

    python -m eval.run_intent_eval           # rules + llm (llm skipped without a key)
    python -m eval.run_intent_eval --rules   # rules only, offline

Writes ``eval/results/intent_latest.md`` (tables + every failure) and
``intent_latest.json``. If any LLM call fell back to rules, the LLM column is
labelled NOT VALID rather than silently mixing the two.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import travel_agent.core  # noqa: F401  (loads .env)
from travel_agent.agents.intent import INTENTS, IntentResult, classify_intent, classify_with_rules
from travel_agent.agents.planner import ground

CASES = Path(__file__).with_name("intent_cases.jsonl")
RESULTS = Path(__file__).with_name("results")
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=ZoneInfo("America/New_York"))

# Fields a case may specify, and how to compare them with the grounded goal.
SLOT_CHECKS: dict[str, Callable[[Any, dict[str, Any]], bool]] = {
    "origin_airport": lambda expected, goal: goal.get("origin_airport") in expected,
    "destination_airport": lambda expected, goal: goal.get("destination_airport") in expected,
    "departure_date": lambda expected, goal: goal.get("departure_date") == expected,
    "departure_window": lambda expected, goal: goal.get("departure_window") == expected,
    "arrive_by": lambda expected, goal: goal.get("arrive_by") == expected,
    "flight_number": lambda expected, goal: (goal.get("flight_number") or "").replace(" ", "").upper()
    == expected.replace(" ", "").upper(),
    "has_constraints": lambda expected, goal: bool(goal.get("constraints")) == expected,
    "missing": lambda expected, goal: set(expected) <= set(goal.get("missing") or []),
}


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_case(case: dict[str, Any], classify: Callable[[str], IntentResult]) -> dict[str, Any]:
    result = classify(case["text"])
    goal = ground(result, case["text"], NOW).to_dict()
    slot_results = {
        field: check(case[field], goal)
        for field, check in SLOT_CHECKS.items()
        if field in case and result.intent == case["intent"]  # slots only graded when routed right
    }
    return {
        "id": case["id"],
        "text": case["text"],
        "expected": case["intent"],
        "predicted": result.intent,
        "source": result.source,
        "intent_ok": result.intent == case["intent"],
        "slots": slot_results,
        "all_ok": result.intent == case["intent"] and all(slot_results.values()),
        "goal": {k: goal.get(k) for k in ("origin_airport", "destination_airport", "departure_date",
                                           "departure_window", "arrive_by", "flight_number", "missing")},
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {e: {p: 0 for p in INTENTS} for e in INTENTS}
    for row in rows:
        confusion[row["expected"]][row["predicted"]] += 1
    per_intent = {}
    for intent in INTENTS:
        tp = confusion[intent][intent]
        actual = sum(confusion[intent].values())
        predicted = sum(confusion[e][intent] for e in INTENTS)
        per_intent[intent] = {"precision": tp / predicted if predicted else None,
                              "recall": tp / actual if actual else None, "support": actual}
    slot_totals: dict[str, list[int]] = {}
    for row in rows:
        for field, ok in row["slots"].items():
            slot_totals.setdefault(field, [0, 0])
            slot_totals[field][0] += int(ok)
            slot_totals[field][1] += 1
    return {
        "cases": len(rows),
        "intent_correct": sum(r["intent_ok"] for r in rows),
        "all_correct": sum(r["all_ok"] for r in rows),
        "fallbacks": sum(r["source"] == "rules" for r in rows),
        "per_intent": per_intent,
        "confusion": confusion,
        "slots": slot_totals,
    }


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(runs: dict[str, dict[str, Any]], model: str) -> str:
    names = list(runs)
    lines = ["# Intent routing evaluation (M2)", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
             f"Fixed clock: {NOW.isoformat()} · cases: `eval/intent_cases.jsonl` · LLM: `{model}`", ""]
    for name in names:
        if name == "llm" and runs[name]["summary"]["fallbacks"]:
            lines.append(f"**LLM column NOT VALID:** {runs[name]['summary']['fallbacks']} call(s) fell back to rules.\n")

    lines += ["## Summary", "", "| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    measures = [
        ("Intent accuracy", lambda s: _pct(s["intent_correct"], s["cases"])),
        ("Intent + all slots correct", lambda s: _pct(s["all_correct"], s["cases"])),
    ]
    fields = sorted({f for run in runs.values() for f in run["summary"]["slots"]}, key=list(SLOT_CHECKS).index)
    measures += [(f"Slot `{f}`", lambda s, f=f: _pct(*s["slots"].get(f, [0, 0]))) for f in fields]
    for label, fn in measures:
        lines.append(f"| {label} | " + " | ".join(fn(runs[n]["summary"]) for n in names) + " |")

    for name in names:
        summary = runs[name]["summary"]
        lines += ["", f"## Per intent ({name})", "", "| Intent | Precision | Recall | Cases |", "|---|---|---|---|"]
        for intent, m in summary["per_intent"].items():
            lines.append(f"| {intent} | {_f(m['precision'])} | {_f(m['recall'])} | {m['support']} |")
        lines += ["", f"Confusion matrix ({name}; rows = expected, columns = predicted):", "",
                  "| | " + " | ".join(INTENTS) + " |", "|---|" + "---|" * len(INTENTS)]
        for expected, row in summary["confusion"].items():
            lines.append(f"| **{expected}** | " + " | ".join(str(row[p]) for p in INTENTS) + " |")

    lines += ["", "## Failures", ""]
    any_failure = False
    for name in names:
        for row in runs[name]["rows"]:
            if row["all_ok"]:
                continue
            any_failure = True
            wrong = [f for f, ok in row["slots"].items() if not ok]
            what = (f"intent {row['predicted']} (expected {row['expected']})" if not row["intent_ok"]
                    else "slots " + ", ".join(wrong))
            lines.append(f"- **{name}** `{row['id']}`: {what}. \"{row['text']}\" → `{json.dumps(row['goal'])}`")
    if not any_failure:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main() -> int:
    cases = load_cases()
    runs: dict[str, dict[str, Any]] = {}
    rules_rows = [score_case(c, classify_with_rules) for c in cases]
    runs["rules"] = {"rows": rules_rows, "summary": summarise(rules_rows)}

    model = os.environ.get("ORCHESTRATOR_MODEL") or "gemini-3.5-flash-lite"
    if "--rules" not in sys.argv:
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY"):
            llm_rows = [score_case(c, lambda text: classify_intent(text, NOW)) for c in cases]
            runs["llm"] = {"rows": llm_rows, "summary": summarise(llm_rows)}
        else:
            print("no GOOGLE_API_KEY: LLM column skipped (rules only)")

    RESULTS.mkdir(exist_ok=True)
    report = render_markdown(runs, model)
    (RESULTS / "intent_latest.md").write_text(report, encoding="utf-8")
    (RESULTS / "intent_latest.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print(report)
    invalid = "llm" in runs and runs["llm"]["summary"]["fallbacks"] > 0
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
