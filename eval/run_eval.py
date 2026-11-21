"""Offline evaluation of the guardrails and the Verifier.

Run with ``.venv/bin/python -m eval.run_eval``. No network and no API key: the
Verifier cases are evaluated against the recorded ``sfo_gdp_20260923`` replay
scenario, exactly the evidence the demo produces.

Outputs
-------
* ``eval/results/latest.json`` -- machine-readable numbers and every failure;
* ``eval/results/latest.md``   -- the same numbers as a small table.

The runner reports honestly: a case that fails is listed, never hidden. The
detectors are heuristic and general; nothing here keys off a case id or exact
text.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import argparse
import os

from travel_agent.guardrails.injection import detect_injection, detect_injection_rules
from travel_agent.guardrails.pii import redact_pii
from travel_agent.guardrails.sanitize import REMOVED_MARKER, sanitize_tool_result
from travel_agent.guardrails.topics import check_topic
from travel_agent.tools import live, replay
from travel_agent.verification.checks import verify_claim
from travel_agent.verification.store import add_tool_result

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SCENARIO = "sfo_gdp_20260923"
UNTRUSTED_FIXTURE_ID = "news:0badc0de"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _guardrail_outcome(case: dict[str, Any], use_llm: bool = False) -> tuple[bool, str]:
    """Return ``(blocked, detail)`` for one guardrail case.

    ``use_llm=False`` measures layer 1 only; ``use_llm=True`` measures
    layer 1 + layer 2 (the classifier is only asked when the rules miss).
    """
    text = str(case.get("text") or "")
    if case.get("channel") == "tool":
        result = {
            "source": "news",
            "ok": True,
            "evidence": [
                {
                    "id": "news:eval",
                    "source": "news",
                    "subject": "news",
                    "observation": "News headline (untrusted text)",
                    "fetched_at": "eval",
                    "data": {"text": text, "untrusted": True},
                }
            ],
        }
        sanitized, _ = sanitize_tool_result(result, use_llm=use_llm)
        data = sanitized["evidence"][0]["data"]
        blocked = data.get("injection_removed") is True and data.get("text") == REMOVED_MARKER
        return blocked, "tool sanitizer" if blocked else "headline kept"

    injection = detect_injection(text, use_llm=use_llm) if use_llm else detect_injection_rules(text)
    topic = check_topic(text)
    reasons = list(injection["reasons"])
    if not topic["allowed"]:
        reasons.append(str(topic.get("category")))
    return injection["flagged"] or not topic["allowed"], ", ".join(reasons)


def evaluate_guardrails(cases: list[dict[str, Any]], use_llm: bool = False) -> dict[str, Any]:
    """Score block/allow/redact cases for one detector configuration."""
    failures: list[dict[str, Any]] = []
    categories: dict[str, dict[str, int]] = {}

    block_total = block_passed = 0
    benign_total = benign_blocked = 0
    pii_total = pii_passed = 0
    over_redaction = 0

    for case in cases:
        category = str(case.get("category") or "unknown")
        entry = categories.setdefault(category, {"total": 0, "passed": 0})
        entry["total"] += 1
        expect = case.get("expect")
        text = str(case.get("text") or "")
        blocked, detail = _guardrail_outcome(case, use_llm=use_llm)
        passed = True

        if expect == "block":
            block_total += 1
            passed = blocked
            if passed:
                block_passed += 1
            else:
                failures.append({"id": case.get("id"), "category": category, "expected": "block", "detail": detail or "not blocked"})

        elif expect == "redact":
            pii_total += 1
            redacted, findings = redact_pii(text)
            missing = [value for value in case.get("must_redact", []) if value in redacted]
            kept = [value for value in case.get("must_keep", []) if value not in redacted]
            passed = blocked is False and not missing and not kept
            if passed:
                pii_passed += 1
            else:
                failures.append(
                    {
                        "id": case.get("id"),
                        "category": category,
                        "expected": "redact",
                        "detail": f"left={missing} lost={kept} blocked={blocked} findings={[f['type'] for f in findings]}",
                    }
                )

        else:  # allow
            benign_total += 1
            redacted, _ = redact_pii(text)
            lost = [value for value in case.get("must_keep", []) if value not in redacted]
            over_redaction += len(lost)
            passed = not blocked and not lost
            if blocked:
                benign_blocked += 1
            if not passed:
                failures.append(
                    {
                        "id": case.get("id"),
                        "category": category,
                        "expected": "allow",
                        "detail": f"blocked={blocked} over_redacted={lost} reasons={detail}",
                    }
                )

        if passed:
            entry["passed"] += 1

    per_category = {
        name: {
            "total": values["total"],
            "passed": values["passed"],
            "accuracy": round(values["passed"] / values["total"], 3) if values["total"] else 0.0,
        }
        for name, values in sorted(categories.items())
    }

    return {
        "total_cases": len(cases),
        "block_cases": block_total,
        "blocked_recall": round(block_passed / block_total, 3) if block_total else 0.0,
        "benign_cases": benign_total,
        "false_positive_rate": round(benign_blocked / benign_total, 3) if benign_total else 0.0,
        "pii_cases": pii_total,
        "pii_redaction_recall": round(pii_passed / pii_total, 3) if pii_total else 0.0,
        "over_redaction_count": over_redaction,
        "per_category": per_category,
        "failures": failures,
    }


def _build_verifier_store() -> dict[str, Any]:
    """Run the replay tools and return ``state["evidence_store"]``.

    The scenario has no recorded news tool, so one clearly-labelled synthetic
    untrusted headline is added to exercise the "untrusted-only" check.
    """
    state: dict[str, Any] = {}
    try:
        replay.use_demo_scenario(SCENARIO)
        add_tool_result(state, live.assess_trip_risk("New York", "San Francisco", "2026-09-23", "evening"))
        add_tool_result(state, live.get_faa_airport_status("SFO"))
        add_tool_result(state, live.get_faa_airport_status("JFK"))
    finally:
        replay.use_demo_scenario(None)

    store = state.setdefault("evidence_store", {})
    store[UNTRUSTED_FIXTURE_ID] = {
        "evidence": {
            "id": UNTRUSTED_FIXTURE_ID,
            "source": "news",
            "subject": "news",
            "observation": "News headline (untrusted text)",
            "fetched_at": "2026-09-23T16:10:00+00:00",
            "data": {"text": "SFO is delayed 34 min", "untrusted": True},
        },
        "tool_ok": True,
        "source": "news",
    }
    # Synthetic records for the G6 edge cases. They are labelled trusted/untrusted
    # and failed/succeeded exactly like real store entries.
    store["failed:tool:0"] = {
        "evidence": {
            "id": "failed:tool:0",
            "source": "faa_nas",
            "subject": "SFO",
            "observation": "SFO average delay 34 min",
            "fetched_at": "2026-09-23T16:10:00+00:00",
            "data": {"avg_delay_min": 34},
        },
        "tool_ok": False,
        "source": "faa_nas",
    }
    store["coords:place"] = {
        "evidence": {
            "id": "coords:place",
            "source": "airports",
            "subject": "SFO",
            "observation": "SFO is at 37.6188, -122.3750",
            "fetched_at": "2026-09-23T16:10:00+00:00",
            "data": {"lat": 37.6188, "lon": -122.375},
        },
        "tool_ok": True,
        "source": "airports",
    }
    store["engine:JFK-SFO:custom:finding"] = {
        "evidence": {
            "id": "engine:JFK-SFO:custom:finding",
            "source": "engine",
            "subject": "JFK-SFO",
            "observation": "A finding without a risk level",
            "fetched_at": "2026-09-23T16:10:00+00:00",
            "data": {},
        },
        "tool_ok": True,
        "source": "engine",
    }
    store["synthetic:sfoo"] = {
        "evidence": {
            "id": "synthetic:sfoo",
            "source": "faa_nas",
            "subject": "SFOO",
            "observation": "SFOO has a ground delay program averaging 34 min",
            "fetched_at": "2026-09-23T16:10:00+00:00",
            "data": {"avg_delay_min": 34},
        },
        "tool_ok": True,
        "source": "faa_nas",
    }
    return store


def evaluate_verifier(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score labelled claims; return accuracy, confusion matrix and failures."""
    store = _build_verifier_store()
    confusion: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    passed = 0

    for case in cases:
        expected = str(case.get("expect"))
        verdict = verify_claim(case.get("claim") or {}, store)
        actual = str(verdict.get("status"))
        row = confusion.setdefault(expected, {})
        row[actual] = row.get(actual, 0) + 1
        if actual == expected:
            passed += 1
        else:
            failures.append(
                {
                    "id": case.get("id"),
                    "category": case.get("category"),
                    "expected": expected,
                    "actual": actual,
                    "reason": verdict.get("reason"),
                }
            )

    return {
        "total_cases": len(cases),
        "accuracy": round(passed / len(cases), 3) if cases else 0.0,
        "confusion": confusion,
        "failures": failures,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    guard = payload["guardrails"]
    verifier = payload["verifier"]
    lines = [
        "# Evaluation results",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Guardrails",
        "",
        f"- Cases: {guard['total_cases']}",
        f"- Layer 1 blocked recall: {guard['layer1']['blocked_recall']:.1%} ({guard['layer1']['block_cases']} attack cases)",
        f"- Layer 1 benign false-positive rate: {guard['layer1']['false_positive_rate']:.1%} ({guard['layer1']['benign_cases']} allow cases)",
        f"- Layer 1 PII redaction recall: {guard['layer1']['pii_redaction_recall']:.1%} ({guard['layer1']['pii_cases']} PII cases)",
        f"- Layer 1 over-redaction count: {guard['layer1']['over_redaction_count']}",
        f"- Layer 1+2 blocked recall: {guard['layer1_plus_2']['blocked_recall']:.1%}",
        f"- Layer 1+2 benign false-positive rate: {guard['layer1_plus_2']['false_positive_rate']:.1%}",
        f"- Layer 1+2 PII redaction recall: {guard['layer1_plus_2']['pii_redaction_recall']:.1%}",
        f"- Layer 1+2 over-redaction count: {guard['layer1_plus_2']['over_redaction_count']}",
        f"- Note: {guard.get('llm_note')}",
        "",
        "| Category | Passed | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for name, stats in guard["layer1"]["per_category"].items():
        lines.append(f"| {name} | {stats['passed']} | {stats['total']} | {stats['accuracy']:.1%} |")
    lines += [
        "",
        "## Verifier",
        "",
        f"- Cases: {verifier['total_cases']}",
        f"- Accuracy: {verifier['accuracy']:.1%}",
        "",
        "| Expected \\ Actual | supported | unsupported | contradicted |",
        "|---|---|---|---|",
    ]
    for expected in ("supported", "unsupported", "contradicted"):
        row = verifier["confusion"].get(expected, {})
        lines.append(
            f"| {expected} | {row.get('supported', 0)} | {row.get('unsupported', 0)} | {row.get('contradicted', 0)} |"
        )
    failures = (
        [{"layer": "layer1", **item} for item in guard["layer1"]["failures"]]
        + [{"layer": "layer1+2", **item} for item in guard["layer1_plus_2"]["failures"]]
        + [{"layer": "verifier", **item} for item in verifier["failures"]]
    )
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    else:
        for failure in failures:
            lines.append(f"- `{failure.get('id')}` ({failure.get('category')}): {json.dumps(failure, default=str)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _guardrail_rows(label: str, guard: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (f"{label}.blocked_recall", f"{guard['blocked_recall']:.1%}"),
        (f"{label}.false_positive_rate", f"{guard['false_positive_rate']:.1%}"),
        (f"{label}.pii_redaction_recall", f"{guard['pii_redaction_recall']:.1%}"),
        (f"{label}.over_redaction_count", str(guard["over_redaction_count"])),
    ]


def _print_table(payload: dict[str, Any]) -> None:
    guard = payload["guardrails"]
    layer1 = guard["layer1"]
    layer2 = guard["layer1_plus_2"]
    verifier = payload["verifier"]
    rows = _guardrail_rows("layer1", layer1) + _guardrail_rows("layer1+2", layer2) + [
        ("verifier.accuracy", f"{verifier['accuracy']:.1%}"),
        (
            "failures",
            str(len(layer1["failures"]) + len(layer2["failures"]) + len(verifier["failures"])),
        ),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name.ljust(width)}  {value}")
    if guard.get("llm_note"):
        print(guard["llm_note"])
    print("\nGuardrail failures (layer 1):")
    for failure in layer1["failures"]:
        print(f"  {failure['id']} ({failure['category']}): {failure['detail']}")
    if not layer1["failures"]:
        print("  none")
    if layer2["failures"] != layer1["failures"]:
        print("Guardrail failures (layer 1+2, new only):")
        layer1_ids = {failure.get("id") for failure in layer1["failures"]}
        new = [failure for failure in layer2["failures"] if failure.get("id") not in layer1_ids]
        for failure in new:
            print(f"  {failure['id']} ({failure['category']}): {failure['detail']}")
        if not new:
            print("  none")
    print("Verifier failures:")
    for failure in verifier["failures"]:
        print(f"  {failure['id']} ({failure['category']}): expected {failure['expected']}, got {failure['actual']} -- {failure['reason']}")
    if not verifier["failures"]:
        print("  none")


def _classifier_key_present() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY"))


def run(use_llm: bool = False) -> dict[str, Any]:
    """Run both evaluations, write results, print a summary and return the payload."""
    guardrail_cases = _load_jsonl(ROOT / "guardrail_cases.jsonl")
    verifier_cases = json.loads((ROOT / "verifier_cases.json").read_text(encoding="utf-8"))
    layer1 = evaluate_guardrails(guardrail_cases, use_llm=False)
    llm_available = use_llm and _classifier_key_present()
    layer2 = evaluate_guardrails(guardrail_cases, use_llm=llm_available)
    if use_llm and not llm_available:
        note = "classifier unavailable, rules only"
    elif use_llm:
        note = "classifier enabled (layer 1 + layer 2)"
    else:
        note = "rules only (pass --llm to enable the classifier)"
    guardrails = {
        **layer2,
        "layer1": layer1,
        "layer1_plus_2": layer2,
        "llm_requested": use_llm,
        "llm_available": llm_available,
        "llm_note": note,
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guardrails": guardrails,
        "verifier": evaluate_verifier(verifier_cases),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, RESULTS_DIR / "latest.md")
    _print_table(payload)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Offline guardrail and verifier evaluation")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="also use the Gemini classifier (needs GOOGLE_API_KEY; otherwise rules only)",
    )
    args = parser.parse_args(argv)
    return run(use_llm=args.llm)


if __name__ == "__main__":
    main()
