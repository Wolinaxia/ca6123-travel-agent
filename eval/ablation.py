"""Verifier ablation (M1, light version).

Four sub-commands::

    python -m eval.ablation dry-run
    python -m eval.ablation collect [--limit N] [--only id1,id2]
    python -m eval.ablation sheet [--force]
    python -m eval.ablation score [--allow-unlabelled]

``dry-run`` and ``collect`` run the real pipeline (need ``GOOGLE_API_KEY``);
``sheet`` and ``score`` work offline from the raw file written by ``collect``.
See ``docs/VERIFIER_ABLATION_PLAN.md`` and ``docs/ABLATION_SPEC.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import travel_agent.core  # noqa: F401  (loads .env the same way the other eval scripts do)
from travel_agent.agents.orchestrator import (
    Conversation,
    assemble_answer,
    assess_with_rules,
    verify_results,
)
from travel_agent.agents.specialists import DEFAULT_MODEL, SPECIALIST_NAMES
from travel_agent.core.reference.airports import get as airport_by_iata
from travel_agent.tools import live
from travel_agent.verification.checks import verify_claims
from travel_agent.verification.store import add_tool_result

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REQUESTS = HERE / "ablation_requests.json"
RESULTS = HERE / "results"
RAW = RESULTS / "ablation_raw.jsonl"
LABELS = RESULTS / "ablation_labels.csv"
REPORT_JSON = RESULTS / "ablation.json"
REPORT_MD = RESULTS / "ablation.md"
REPORT_SVG = RESULTS / "ablation.svg"

STATE_KEYS = ("intent", "trip_goal", "specialist_results", "evidence_store",
              "last_verification", "final_answer")
SHEET_COLUMNS = ("claim_id", "scenario", "request", "agent", "claim", "risk",
                 "confidence", "evidence", "label", "note")
ALLOWED_LABELS = ("correct", "wrong_number", "wrong_place", "risk_downplayed",
                  "unsupported", "stale", "other_wrong")
RISK_ORDER = {"unknown": -1, "low": 0, "moderate": 1, "high": 2}
SIGNAL_AGENTS = ("flight_airport", "weather")
NON_CLAIM_INTENTS = ("out_of_scope", "blocked")
CANDIDATE_AIRPORTS = ("JFK", "LGA", "EWR", "BOS")
TIME_PHRASES = ("this evening", "this afternoon", "tonight", "today")

FAULTS: tuple[tuple[str, bool], ...] = (
    ("wrong_number", True),
    ("wrong_airport", True),
    ("fabricated_id", True),
    ("no_citation", True),
    ("risk_downplayed", True),
    ("normal_vs_program", True),
    ("wrong_time_window", False),
)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_requests() -> list[dict[str, Any]]:
    return json.loads(REQUESTS.read_text(encoding="utf-8"))


def select_requests(requests: list[dict[str, Any]], limit: Optional[int] = None,
                    only: Optional[str] = None) -> list[dict[str, Any]]:
    if only:
        wanted = [item.strip() for item in only.split(",") if item.strip()]
        requests = [item for item in requests if item["id"] in wanted]
    if limit is not None:
        requests = requests[:limit]
    return requests


def has_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY"))


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# dry-run / collect (need a model)
# --------------------------------------------------------------------------


def dry_run(requests: list[dict[str, Any]]) -> None:
    first: dict[str, dict[str, Any]] = {}
    for request in requests:
        first.setdefault(request["scenario"], request)
    for scenario, request in first.items():
        conversation = Conversation(scenario=scenario, verifier_enabled=True)
        state = conversation.send(request["text"])
        results = state.get("specialist_results") or {}
        claims = {name: len((results.get(name) or {}).get("claims") or []) for name in SPECIALIST_NAMES}
        unavailable = {name: len((results.get(name) or {}).get("unavailable") or [])
                       for name in SPECIALIST_NAMES}
        store = state.get("evidence_store") or {}
        print(f"{request['id']} {scenario}: intent={state.get('intent')} claims={claims} "
              f"evidence_store={len(store)} unavailable={unavailable}")


def collect(requests: list[dict[str, Any]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    model = str(DEFAULT_MODEL)
    records: list[dict[str, Any]] = []
    for request in requests:
        base = {
            "id": request["id"],
            "scenario": request["scenario"],
            "expected_intent": request["intent"],
            "request": request["text"],
            "model": model,
            "git_commit": commit,
            "collected_at": _now(),
        }
        try:
            conversation = Conversation(scenario=request["scenario"], verifier_enabled=True)
            state = conversation.send(request["text"])
            record = {**base, **{key: state.get(key) for key in STATE_KEYS}}
        except Exception as exc:  # a failed request must not stop the run
            record = {**base, "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        status = "error" if "error" in record else "ok"
        print(f"{request['id']} {request['scenario']}: {status}")
    with RAW.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
    print(f"wrote {len(records)} record(s) to {RAW}")


# --------------------------------------------------------------------------
# Store helpers
# --------------------------------------------------------------------------


def _normalise_store(store: Any) -> dict[str, Any]:
    if not isinstance(store, dict):
        return {}
    inner = store.get("evidence_store")
    return inner if isinstance(inner, dict) else store


def _cited_evidence(claim: dict[str, Any], store: Any) -> list[tuple[str, dict[str, Any]]]:
    normalised = _normalise_store(store)
    items: list[tuple[str, dict[str, Any]]] = []
    for evidence_id in claim.get("evidence_ids") or []:
        record = normalised.get(str(evidence_id))
        if isinstance(record, dict):
            evidence = record.get("evidence")
            if isinstance(evidence, dict):
                items.append((str(evidence_id), evidence))
    return items


def _code_in_evidence(code: str, items: list[tuple[str, dict[str, Any]]]) -> bool:
    for evidence_id, evidence in items:
        haystack = f"{evidence.get('subject') or ''} {evidence_id}".upper()
        if re.search(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", haystack):
            return True
    return False


def _iata_codes(text: str) -> list[str]:
    codes: list[str] = []
    for match in re.finditer(r"\b[A-Z]{3}\b", text):
        code = match.group(0)
        if code not in codes and airport_by_iata(code) is not None:
            codes.append(code)
    return codes


# --------------------------------------------------------------------------
# Mutations (defined without looking at verdicts)
# --------------------------------------------------------------------------


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


_NUMBER_UNIT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:min(?:ute)?s?\b|%)", re.I)


def mutate_wrong_number(claim: dict[str, Any]) -> Optional[dict[str, Any]]:
    text = str(claim.get("text") or "")
    match = _NUMBER_UNIT.search(text)
    if not match:
        return None
    value = float(match.group(1)) * 3 + 20
    new_text = text[:match.start(1)] + _format_number(value) + text[match.end(1):]
    return {**claim, "text": new_text}


def mutate_wrong_airport(claim: dict[str, Any], store: Any) -> Optional[dict[str, Any]]:
    text = str(claim.get("text") or "")
    items = _cited_evidence(claim, store)
    code = next((item for item in _iata_codes(text) if _code_in_evidence(item, items)), None)
    if not code:
        return None
    combined = (text + " " + " ".join(f"{evidence.get('subject') or ''} {evidence_id}"
                                      for evidence_id, evidence in items)).upper()
    replacement = next(
        (item for item in CANDIDATE_AIRPORTS
         if item != code and not re.search(rf"(?<![A-Z0-9]){item}(?![A-Z0-9])", combined)),
        None,
    )
    if not replacement:
        return None
    return {**claim, "text": re.sub(rf"\b{code}\b", replacement, text, count=1)}


def mutate_fabricated_id(claim: dict[str, Any], store: Any) -> Optional[dict[str, Any]]:
    evidence_ids = [str(item) for item in claim.get("evidence_ids") or []]
    if not evidence_ids:
        return None
    normalised = _normalise_store(store)
    fabricated = []
    for evidence_id in evidence_ids:
        parts = evidence_id.split(":")
        parts[-1] = "zz9"
        candidate = ":".join(parts)
        while candidate in normalised:
            parts[-1] += "9"
            candidate = ":".join(parts)
        fabricated.append(candidate)
    return {**claim, "evidence_ids": fabricated}


def mutate_no_citation(claim: dict[str, Any]) -> dict[str, Any]:
    return {**claim, "evidence_ids": []}


def mutate_risk_downplayed(claim: dict[str, Any], agent: str,
                           rule_level: Optional[str]) -> Optional[dict[str, Any]]:
    if agent not in SIGNAL_AGENTS:
        return None
    if claim.get("risk") not in ("moderate", "high"):
        return None
    if rule_level not in ("elevated", "high", "severe"):
        return None
    return {**claim, "risk": "low"}


def mutate_normal_vs_program(claim: dict[str, Any], store: Any) -> Optional[dict[str, Any]]:
    normalised = _normalise_store(store)
    for evidence_id, evidence in _cited_evidence(claim, store):
        record = normalised.get(evidence_id) or {}
        source = str(record.get("source") or evidence.get("source") or "")
        data = evidence.get("data") or {}
        if source != "faa_nas" or not data.get("program"):
            continue
        airport = str(evidence.get("subject") or "").strip()
        if not airport:
            parts = evidence_id.split(":")
            airport = parts[1] if len(parts) > 1 else ""
        if not airport:
            return None
        return {**claim, "text": f"{airport} is operating normally with no FAA delays."}
    return None


def mutate_wrong_time_window(claim: dict[str, Any]) -> Optional[dict[str, Any]]:
    text = str(claim.get("text") or "")
    lowered = text.lower()
    found = [(lowered.find(phrase), phrase) for phrase in TIME_PHRASES if lowered.find(phrase) != -1]
    if not found:
        return None
    start, phrase = min(found, key=lambda item: item[0])
    return {**claim, "text": text[:start] + "tomorrow" + text[start + len(phrase):]}


def build_mutation(fault: str, claim: dict[str, Any], agent: str, store: Any,
                   rule_level: Optional[str]) -> Optional[dict[str, Any]]:
    if fault == "wrong_number":
        return mutate_wrong_number(claim)
    if fault == "wrong_airport":
        return mutate_wrong_airport(claim, store)
    if fault == "fabricated_id":
        return mutate_fabricated_id(claim, store)
    if fault == "no_citation":
        return mutate_no_citation(claim)
    if fault == "risk_downplayed":
        return mutate_risk_downplayed(claim, agent, rule_level)
    if fault == "normal_vs_program":
        return mutate_normal_vs_program(claim, store)
    if fault == "wrong_time_window":
        return mutate_wrong_time_window(claim)
    raise ValueError(f"unknown fault: {fault}")


# --------------------------------------------------------------------------
# sheet
# --------------------------------------------------------------------------


def _observation(record: Any) -> str:
    if isinstance(record, dict):
        evidence = record.get("evidence")
        if isinstance(evidence, dict):
            return str(evidence.get("observation") or "")
    return ""


def _evidence_cell(claim: dict[str, Any], store: Any) -> str:
    normalised = _normalise_store(store)
    parts = []
    for evidence_id in claim.get("evidence_ids") or []:
        evidence_id = str(evidence_id)
        record = normalised.get(evidence_id)
        parts.append(f"{evidence_id}: {_observation(record)}" if record is not None
                     else f"{evidence_id}: MISSING")
    return " | ".join(parts)


def build_sheet_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        if record.get("error"):
            continue
        results = record.get("specialist_results") or {}
        store = record.get("evidence_store") or {}
        for agent in SPECIALIST_NAMES:
            claims = (results.get(agent) or {}).get("claims") or []
            for index, claim in enumerate(claims):
                rows.append({
                    "claim_id": f"{record.get('id')}:{agent}:{index}",
                    "scenario": str(record.get("scenario") or ""),
                    "request": str(record.get("request") or ""),
                    "agent": agent,
                    "claim": str(claim.get("text") or ""),
                    "risk": str(claim.get("risk") or ""),
                    "confidence": str(claim.get("confidence") or ""),
                    "evidence": _evidence_cell(claim, store),
                    "label": "",
                    "note": "",
                })
    return rows


def write_sheet(rows: list[dict[str, str]], path: Path = LABELS, force: bool = False) -> None:
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
        if any((row.get("label") or "").strip() for row in existing) and not force:
            raise SystemExit(f"{path} already has labelled rows; re-run with --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@contextlib.contextmanager
def active_scenario(name: Optional[str]) -> Iterator[Optional[datetime]]:
    from travel_agent.tools import replay

    if not name:
        yield None
        return
    replay.use_demo_scenario(name)
    try:
        yield replay.demo_now()
    finally:
        replay.use_demo_scenario(None)


def _status(verdicts: dict[str, Any]) -> Optional[str]:
    items = verdicts.get("verdicts") or []
    return items[0].get("status") if items else None


def store_for_agent(agent: str, store: Any, rules_result: Optional[dict[str, Any]]) -> dict[str, Any]:
    base = dict(_normalise_store(store))
    if agent in SIGNAL_AGENTS and rules_result:
        container = {"evidence_store": dict(base)}
        add_tool_result(container, rules_result)
        return container["evidence_store"]
    return base


def prepare_record(record: dict[str, Any], labels: dict[str, str],
                   engine: Callable[..., dict] = live.assess_trip_risk) -> dict[str, Any]:
    results = record.get("specialist_results") or {}
    store = record.get("evidence_store") or {}
    goal = record.get("trip_goal") or {}
    intent = record.get("intent") or None
    with active_scenario(record.get("scenario")):
        rules = assess_with_rules(goal, engine)
        verification = verify_results(results, store, rules.get("result"))
        rule_view = {key: value for key, value in rules.items() if key != "result"}
        on_answer = assemble_answer(intent, goal, results, verification, rule_view, None)
        off_answer = assemble_answer(intent, goal, results, None, rule_view, None)

    ordered = [(agent, index, claim)
               for agent in SPECIALIST_NAMES
               for index, claim in enumerate((results.get(agent) or {}).get("claims") or [])]
    verdicts = verification.get("verdicts") or []
    claims = []
    for (agent, index, claim), verdict in zip(ordered, verdicts):
        claim_id = f"{record.get('id')}:{agent}:{index}"
        claims.append({
            "claim_id": claim_id,
            "agent": agent,
            "index": index,
            "claim": claim,
            "label": labels.get(claim_id, ""),
            "status": verdict.get("status"),
        })
    return {
        "id": record.get("id"),
        "scenario": record.get("scenario"),
        "intent": intent,
        "claims": claims,
        "turn_without_claims": turn_without_claims(record, intent, len(ordered)),
        "on": on_answer,
        "off": off_answer,
        "store": store,
        "rules": rules,
        "verification": verification,
        "last_verification": record.get("last_verification"),
    }


def turn_without_claims(record: dict[str, Any], intent: Optional[str], claim_count: int) -> bool:
    """A turn with no claims cannot be scored for escalation (or Arm B)."""
    if record.get("error"):
        return True
    if not intent or intent in NON_CLAIM_INTENTS:
        return True
    return claim_count == 0


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _empty_counts() -> dict[str, int]:
    return {"shown": 0, "shown_wrong": 0, "wrong": 0, "wrong_held": 0,
            "correct": 0, "correct_held": 0, "turns": 0, "escalations": 0}


def _add_claim(counts: dict[str, int], label: str, shown: bool) -> None:
    wrong = label != "correct"
    if shown:
        counts["shown"] += 1
    if wrong:
        counts["wrong"] += 1
        if shown:
            counts["shown_wrong"] += 1
        else:
            counts["wrong_held"] += 1
    else:
        counts["correct"] += 1
        if not shown:
            counts["correct_held"] += 1


def _rates(counts: dict[str, int]) -> dict[str, Any]:
    return {
        "bad_claims_shown": {"wrong": counts["shown_wrong"], "shown": counts["shown"],
                             "rate": _ratio(counts["shown_wrong"], counts["shown"])},
        "catch_rate": {"held": counts["wrong_held"], "total": counts["wrong"],
                       "rate": _ratio(counts["wrong_held"], counts["wrong"])},
        "false_hold_rate": {"held": counts["correct_held"], "total": counts["correct"],
                            "rate": _ratio(counts["correct_held"], counts["correct"])},
        "escalation_rate": {"escalations": counts["escalations"], "turns": counts["turns"],
                            "rate": _ratio(counts["escalations"], counts["turns"])},
    }


def compute_arm_a(prepared: list[dict[str, Any]]) -> dict[str, Any]:
    on = _empty_counts()
    off = _empty_counts()
    by_scenario: dict[str, dict[str, dict[str, int]]] = {}
    headline_shift: list[dict[str, Any]] = []

    for record in prepared:
        if record.get("turn_without_claims"):
            continue
        scenario = str(record.get("scenario") or "")
        buckets = by_scenario.setdefault(scenario, {"on": _empty_counts(), "off": _empty_counts()})
        on["turns"] += 1
        off["turns"] += 1
        buckets["on"]["turns"] += 1
        buckets["off"]["turns"] += 1
        on_escalates = bool((record.get("on") or {}).get("needs_human"))
        off_escalates = bool((record.get("off") or {}).get("needs_human"))
        on["escalations"] += int(on_escalates)
        off["escalations"] += int(off_escalates)
        buckets["on"]["escalations"] += int(on_escalates)
        buckets["off"]["escalations"] += int(off_escalates)

        for item in record.get("claims") or []:
            label = item.get("label") or ""
            if not label:
                continue
            shown_on = item.get("status") == "supported"
            _add_claim(on, label, shown_on)
            _add_claim(off, label, True)
            _add_claim(buckets["on"], label, shown_on)
            _add_claim(buckets["off"], label, True)

        on_risk = (record.get("on") or {}).get("overall_risk")
        off_risk = (record.get("off") or {}).get("overall_risk")
        if on_risk in RISK_ORDER and off_risk in RISK_ORDER and RISK_ORDER[on_risk] < RISK_ORDER[off_risk]:
            held = []
            for item in record.get("claims") or []:
                if item.get("status") != "supported":
                    held.append({"claim_id": item.get("claim_id"), "label": item.get("label") or "",
                                 "was_correct": (item.get("label") or "") == "correct"})
            headline_shift.append({"id": record.get("id"), "scenario": scenario,
                                   "on_overall_risk": on_risk, "off_overall_risk": off_risk,
                                   "held_claims": held})

    scenario_report: dict[str, Any] = {}
    for scenario, buckets in by_scenario.items():
        scenario_report[scenario] = {
            "on": {
                "false_hold_rate": _ratio(buckets["on"]["correct_held"], buckets["on"]["correct"]),
                "false_holds": buckets["on"]["correct_held"],
                "correct": buckets["on"]["correct"],
                "escalation_rate": _ratio(buckets["on"]["escalations"], buckets["on"]["turns"]),
                "escalations": buckets["on"]["escalations"],
                "turns": buckets["on"]["turns"],
            },
            "off": {
                "false_hold_rate": _ratio(buckets["off"]["correct_held"], buckets["off"]["correct"]),
                "false_holds": buckets["off"]["correct_held"],
                "correct": buckets["off"]["correct"],
                "escalation_rate": _ratio(buckets["off"]["escalations"], buckets["off"]["turns"]),
                "escalations": buckets["off"]["escalations"],
                "turns": buckets["off"]["turns"],
            },
        }

    return {"on": _rates(on), "off": _rates(off), "by_scenario": scenario_report,
            "headline_shift": headline_shift}


def compute_arm_b(prepared: list[dict[str, Any]],
                  verify: Callable[..., dict] = verify_claims) -> dict[str, Any]:
    faults: dict[str, dict[str, Any]] = {
        name: {"applicable": 0, "caught": 0, "in_scope": in_scope}
        for name, in_scope in FAULTS
    }
    missed: list[dict[str, Any]] = []
    control_failures: list[str] = []
    base_claims = 0
    correct_excluded = 0
    controls_total = 0
    controls_supported = 0

    for record in prepared:
        if record.get("turn_without_claims"):
            continue
        rules = record.get("rules") or {}
        with active_scenario(record.get("scenario")):
            for item in record.get("claims") or []:
                label = item.get("label") or ""
                if label == "correct" and item.get("status") != "supported":
                    correct_excluded += 1
                if not (label == "correct" and item.get("status") == "supported"):
                    continue
                base_claims += 1
                agent = item.get("agent")
                store = store_for_agent(agent, record.get("store") or {}, rules.get("result"))
                rule_level = rules.get("engine_level")
                controls_total += 1
                if _status(verify([item["claim"]], store)) == "supported":
                    controls_supported += 1
                else:
                    control_failures.append(str(item.get("claim_id")))
                for fault, in_scope in FAULTS:
                    mutated = build_mutation(fault, item["claim"], agent, store, rule_level)
                    if mutated is None:
                        continue
                    faults[fault]["applicable"] += 1
                    status = _status(verify([mutated], store))
                    if status != "supported":
                        faults[fault]["caught"] += 1
                    elif in_scope:
                        missed.append({"claim_id": item.get("claim_id"), "fault": fault,
                                       "mutated_text": mutated.get("text")})

    if controls_supported != controls_total:
        raise RuntimeError(
            "Arm B controls are not all supported "
            f"({controls_supported}/{controls_total}); base claims are supported by construction. "
            f"Unsupported control ids: {', '.join(control_failures)}"
        )

    for details in faults.values():
        details["catch_rate"] = _ratio(details["caught"], details["applicable"])

    in_scope = [faults[name] for name, scope in FAULTS if scope]
    in_scope_caught = sum(details["caught"] for details in in_scope)
    in_scope_applicable = sum(details["applicable"] for details in in_scope)
    out_of_scope = faults["wrong_time_window"]

    return {
        "base_claims": base_claims,
        "correct_excluded_unsupported": correct_excluded,
        "faults": faults,
        "in_scope_catch_rate": {"caught": in_scope_caught, "applicable": in_scope_applicable,
                                "rate": _ratio(in_scope_caught, in_scope_applicable)},
        "out_of_scope": {"fault": "wrong_time_window", **out_of_scope},
        "controls": {"total": controls_total, "supported": controls_supported},
        "missed": missed,
    }


def all_claim_ids(records: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for record in records:
        if record.get("error"):
            continue
        results = record.get("specialist_results") or {}
        for agent in SPECIALIST_NAMES:
            claims = (results.get(agent) or {}).get("claims") or []
            for index in range(len(claims)):
                ids.append(f"{record.get('id')}:{agent}:{index}")
    return ids


def unlabelled_claim_ids(records: list[dict[str, Any]], labels: dict[str, str]) -> list[str]:
    return sorted(claim_id for claim_id in all_claim_ids(records)
                  if not (labels.get(claim_id) or "").strip())


def load_label_rows(path: Path = LABELS) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_labels(path: Path = LABELS) -> dict[str, str]:
    return {(row.get("claim_id") or "").strip(): (row.get("label") or "").strip()
            for row in load_label_rows(path) if (row.get("claim_id") or "").strip()}


def claim_texts(records: list[dict[str, Any]]) -> dict[str, str]:
    """Claim id -> claim text for every claim in the raw records."""
    texts: dict[str, str] = {}
    for record in records:
        if record.get("error"):
            continue
        results = record.get("specialist_results") or {}
        for agent in SPECIALIST_NAMES:
            claims = (results.get(agent) or {}).get("claims") or []
            for index, claim in enumerate(claims):
                texts[f"{record.get('id')}:{agent}:{index}"] = str(claim.get("text") or "")
    return texts


def label_text_mismatches(records: list[dict[str, Any]], path: Path = LABELS) -> list[str]:
    """Claim ids whose label row no longer matches the raw claim text (stale labels)."""
    texts = claim_texts(records)
    mismatched = []
    for row in load_label_rows(path):
        claim_id = (row.get("claim_id") or "").strip()
        if not claim_id:
            continue
        if claim_id not in texts or str(row.get("claim") or "") != texts[claim_id]:
            mismatched.append(claim_id)
    return sorted(set(mismatched))


def build_report(records: list[dict[str, Any]], labels: dict[str, str],
                 engine: Callable[..., dict] = live.assess_trip_risk,
                 commit: Optional[str] = None, model: Optional[str] = None,
                 allow_unlabelled: bool = False) -> dict[str, Any]:
    prepared: list[dict[str, Any]] = []
    mismatches: list[str] = []
    turns_without_claims: list[str] = []
    checked = 0
    errors = 0
    for record in records:
        if record.get("error"):
            errors += 1
            turns_without_claims.append(str(record.get("id")))
            continue
        item = prepare_record(record, labels, engine)
        if item["turn_without_claims"]:
            turns_without_claims.append(str(record.get("id")))
        if item["intent"] == "assess_trip":
            checked += 1
            saved = (record.get("last_verification") or {}).get("verdicts") or []
            recomputed = (item["verification"] or {}).get("verdicts") or []
            if [entry.get("status") for entry in saved] != [entry.get("status") for entry in recomputed]:
                mismatches.append(str(record.get("id")))
        prepared.append(item)

    claim_count = sum(len(item["claims"]) for item in prepared)
    label_counts: dict[str, int] = {}
    for item in prepared:
        for claim in item["claims"]:
            label = claim.get("label") or ""
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

    unlabelled = unlabelled_claim_ids(records, labels)
    excluded = len(unlabelled) if allow_unlabelled else 0

    return {
        "git_commit": commit or git_commit(),
        "model": model or str(DEFAULT_MODEL),
        "generated_at": _now(),
        "records": len(prepared),
        "records_with_error": errors,
        "turns_without_claims": turns_without_claims,
        "claims": claim_count,
        "labels": label_counts,
        "unlabelled_excluded": excluded,
        "consistency": {"checked_records": checked, "mismatching_records": len(mismatches),
                        "mismatching_ids": mismatches},
        "arm_a": compute_arm_a(prepared),
        "arm_b": compute_arm_b(prepared),
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.0f}%)"


def _rate_cell(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def render_markdown(report: dict[str, Any]) -> str:
    arm_a = report["arm_a"]
    arm_b = report["arm_b"]
    on, off = arm_a["on"], arm_a["off"]
    lines = [
        "# Verifier ablation (M1)",
        "",
        f"Generated: {report['generated_at']}  ",
        f"Git: `{report['git_commit']}` · model: `{report['model']}` · "
        f"records: {report['records']} · claims: {report['claims']}",
        "",
    ]
    if report.get("unlabelled_excluded"):
        lines += [f"Unlabelled claims excluded: {report['unlabelled_excluded']}.", ""]
    without = report.get("turns_without_claims") or []
    if without:
        lines += [f"Turns without claims (excluded from escalation rates): "
                  f"{', '.join(without)}.", ""]
    lines += [
        "## Headline",
        "",
        "| Measure | Verifier ON | Verifier OFF |",
        "|---|---|---|",
        f"| Bad claims shown | {_pct(on['bad_claims_shown']['wrong'], on['bad_claims_shown']['shown'])} "
        f"| {_pct(off['bad_claims_shown']['wrong'], off['bad_claims_shown']['shown'])} |",
        f"| Escalation rate | {_pct(on['escalation_rate']['escalations'], on['escalation_rate']['turns'])} "
        f"| {_pct(off['escalation_rate']['escalations'], off['escalation_rate']['turns'])} |",
        "",
        "## Arm A: real claims",
        "",
        "| Measure | Verifier ON | Verifier OFF |",
        "|---|---|---|",
        f"| Bad claims shown | {_pct(on['bad_claims_shown']['wrong'], on['bad_claims_shown']['shown'])} "
        f"| {_pct(off['bad_claims_shown']['wrong'], off['bad_claims_shown']['shown'])} |",
        f"| Catch rate (wrong claims held back) | "
        f"{_pct(on['catch_rate']['held'], on['catch_rate']['total'])} "
        f"| {_pct(off['catch_rate']['held'], off['catch_rate']['total'])} |",
        f"| False hold rate (correct claims held back) | "
        f"{_pct(on['false_hold_rate']['held'], on['false_hold_rate']['total'])} "
        f"| {_pct(off['false_hold_rate']['held'], off['false_hold_rate']['total'])} |",
        f"| Escalation rate | {_pct(on['escalation_rate']['escalations'], on['escalation_rate']['turns'])} "
        f"| {_pct(off['escalation_rate']['escalations'], off['escalation_rate']['turns'])} |",
        "",
        "### By scenario",
        "",
        "| Scenario | False holds ON | False holds OFF | Escalation ON | Escalation OFF |",
        "|---|---|---|---|---|",
    ]
    for scenario, values in arm_a["by_scenario"].items():
        lines.append(
            f"| {scenario} "
            f"| {_pct(values['on']['false_holds'], values['on']['correct'])} "
            f"| {_pct(values['off']['false_holds'], values['off']['correct'])} "
            f"| {_pct(values['on']['escalations'], values['on']['turns'])} "
            f"| {_pct(values['off']['escalations'], values['off']['turns'])} |"
        )

    lines += ["", "### Headline shift", ""]
    if not arm_a["headline_shift"]:
        lines.append("None.")
    else:
        for shift in arm_a["headline_shift"]:
            held = ", ".join(
                f"{item['claim_id']} ({item['label'] or 'unlabelled'})" for item in shift["held_claims"]
            ) or "no held claims"
            lines.append(f"- `{shift['id']}` ({shift['scenario']}): {shift['off_overall_risk']} → "
                         f"{shift['on_overall_risk']}; held: {held}")

    lines += [
        "",
        "## Arm B: injected faults",
        "",
        f"Base claims: {arm_b['base_claims']}. Correct claims excluded for not being supported: "
        f"{arm_b['correct_excluded_unsupported']}. "
        f"Controls supported: {arm_b['controls']['supported']}/{arm_b['controls']['total']}.",
        "",
        "| Fault | In scope | Applicable | Caught | Catch rate |",
        "|---|---|---|---|---|",
    ]
    for name, _scope in FAULTS:
        details = arm_b["faults"][name]
        lines.append(f"| {name} | {'yes' if details['in_scope'] else 'no'} "
                     f"| {details['applicable']} | {details['caught']} "
                     f"| {_rate_cell(details['catch_rate'])} |")
    overall = arm_b["in_scope_catch_rate"]
    lines += [
        "",
        f"Overall catch rate over the six in-scope faults: "
        f"{_pct(overall['caught'], overall['applicable'])}.",
        f"Out-of-scope (expected to get through): "
        f"{_pct(arm_b['out_of_scope']['caught'], arm_b['out_of_scope']['applicable'])}.",
        "",
        "### Missed in-scope mutations",
        "",
    ]
    if not arm_b["missed"]:
        lines.append("None.")
    else:
        for miss in arm_b["missed"]:
            lines.append(f"- `{miss['claim_id']}` {miss['fault']}: \"{miss['mutated_text']}\"")

    lines += [
        "",
        "## Limitations",
        "",
        "- Single run per request.",
        "- One labeller.",
        "- M4's two extra approval-card checks are excluded.",
        "- Three recorded scenarios on one route (New York ↔ Boston).",
        "",
    ]
    return "\n".join(lines)


def _xml_escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(arm_b: dict[str, Any]) -> str:
    width, row_height, left, right, top = 760, 30, 210, 90, 80
    extra_gap = 20
    faults = arm_b["faults"]
    ordered = [name for name, _scope in FAULTS if name in faults]
    in_scope_names = [name for name in ordered if faults[name]["in_scope"]]
    out_scope_names = [name for name in ordered if not faults[name]["in_scope"]]
    rows = [(name, False) for name in in_scope_names] + [(name, True) for name in out_scope_names]
    plot_width = width - left - right
    height = top + len(rows) * row_height + (extra_gap if out_scope_names else 0) + 50

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        "<title>Verifier catch rate by injected fault</title>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-size="18" '
        f'font-weight="bold">Verifier catch rate by injected fault</text>',
    ]

    y = top
    positions: list[tuple[str, bool, int]] = []
    for name, out_of_scope in rows:
        if out_of_scope and (not positions or not positions[-1][1]):
            y += extra_gap
        positions.append((name, out_of_scope, y))
        y += row_height
    plot_bottom = y + 6

    for percent in (0, 25, 50, 75, 100):
        x = left + plot_width * percent / 100
        parts.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{plot_bottom}" '
                     f'stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top - 14}" text-anchor="middle" '
                     f'font-size="11" fill="#4a5568">{percent}%</text>')

    parts.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{plot_bottom}" '
                 f'stroke="#4a5568" stroke-width="1"/>')

    for name, out_of_scope, row_y in positions:
        details = faults[name]
        rate = details.get("catch_rate") or 0.0
        bar_width = max(0.0, plot_width * rate)
        bar_y = row_y + 7
        fill = "#a0aec0" if out_of_scope else "#2b6cb0"
        label = f"{name} (out of scope)" if out_of_scope else name
        parts.append(f'<text x="{left - 8}" y="{row_y + 20}" text-anchor="end" '
                     f'font-size="12" fill="#1a202c">{_xml_escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{bar_y}" width="{bar_width:.1f}" height="16" '
                     f'fill="{fill}"/>')
        parts.append(f'<text x="{left + bar_width + 6:.1f}" y="{row_y + 20}" '
                     f'font-size="12" fill="#1a202c">{details["caught"]}/{details["applicable"]}</text>')

    parts.append(f'<text x="{left + plot_width / 2:.0f}" y="{height - 12}" text-anchor="middle" '
                 f'font-size="12" fill="#4a5568">Catch rate (% of applicable mutations)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run_score(raw_path: Path = RAW, labels_path: Path = LABELS, out_dir: Path = RESULTS,
              allow_unlabelled: bool = False,
              engine: Callable[..., dict] = live.assess_trip_risk) -> int:
    if not raw_path.exists():
        print(f"missing raw file: {raw_path} (run `collect` first)", file=sys.stderr)
        return 1
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = load_labels(labels_path)

    stale = label_text_mismatches(records, labels_path)
    if stale:
        print("labels do not match the raw claim text (stale labels? re-run `sheet`): "
              + ", ".join(stale), file=sys.stderr)
        return 1

    invalid = {claim_id: label for claim_id, label in labels.items()
               if label and label not in ALLOWED_LABELS}
    if invalid:
        print("unknown labels: " + ", ".join(f"{claim_id}={label}" for claim_id, label in sorted(invalid.items())),
              file=sys.stderr)
        return 1

    unlabelled = unlabelled_claim_ids(records, labels)
    if unlabelled and not allow_unlabelled:
        print("unlabelled claims: " + ", ".join(unlabelled), file=sys.stderr)
        print("fill eval/results/ablation_labels.csv or pass --allow-unlabelled", file=sys.stderr)
        return 1

    model = next((record.get("model") for record in records if record.get("model")), str(DEFAULT_MODEL))
    commit = next((record.get("git_commit") for record in records if record.get("git_commit")), None)
    report = build_report(records, labels, engine=engine, commit=commit, model=model,
                          allow_unlabelled=allow_unlabelled)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    (out_dir / "ablation.md").write_text(markdown, encoding="utf-8")
    (out_dir / "ablation.svg").write_text(render_svg(report["arm_b"]), encoding="utf-8")
    print(markdown)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.ablation",
                                     description="Verifier ablation: dry-run, collect, label, score.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser("dry-run", help="run the first request of each scenario (needs a key)")
    dry.set_defaults(func=_cmd_dry_run)

    collect_parser = subparsers.add_parser("collect", help="run every request and write the raw JSONL")
    collect_parser.add_argument("--limit", type=int, default=None, help="run at most N requests")
    collect_parser.add_argument("--only", default=None, help="comma-separated request ids")
    collect_parser.set_defaults(func=_cmd_collect)

    sheet_parser = subparsers.add_parser("sheet", help="write the labelling CSV from the raw file")
    sheet_parser.add_argument("--force", action="store_true", help="overwrite filled labels")
    sheet_parser.set_defaults(func=_cmd_sheet)

    score_parser = subparsers.add_parser("score", help="score the labelled raw file")
    score_parser.add_argument("--allow-unlabelled", action="store_true",
                              help="exclude unlabelled claims instead of failing")
    score_parser.set_defaults(func=_cmd_score)
    return parser


def _cmd_dry_run(_args: argparse.Namespace) -> int:
    if not has_key():
        print("no GOOGLE_API_KEY: dry-run needs the model", file=sys.stderr)
        return 1
    dry_run(load_requests())
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    if not has_key():
        print("no GOOGLE_API_KEY: collect needs the model", file=sys.stderr)
        return 1
    collect(select_requests(load_requests(), args.limit, args.only))
    return 0


def _cmd_sheet(args: argparse.Namespace) -> int:
    if not RAW.exists():
        print(f"missing raw file: {RAW} (run `collect` first)", file=sys.stderr)
        return 1
    records = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = build_sheet_rows(records)
    write_sheet(rows, LABELS, force=args.force)
    print(f"wrote {len(rows)} claim(s) to {LABELS}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    return run_score(allow_unlabelled=args.allow_unlabelled)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
