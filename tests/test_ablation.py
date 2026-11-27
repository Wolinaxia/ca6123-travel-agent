"""Offline tests for the Verifier ablation script (``eval/ablation.py``).

No network, no model, no key: the synthetic evidence store is hand-written and
every mutation is a pure function.
"""

from __future__ import annotations

import copy
import csv
import json
import xml.etree.ElementTree as ET

import pytest

from eval import ablation
from travel_agent.agents.orchestrator import combine_verifications, verify_results
from travel_agent.verification.checks import verify_claims

FAA_ID = "faa_nas:LGA:GDP:0"
OTHER_ID = "open_meteo:BOS:day:2026-11-22"

STORE = {
    FAA_ID: {
        "evidence": {
            "id": FAA_ID,
            "source": "faa_nas",
            "subject": "LGA",
            "observation": "Ground Delay Program at LGA, average delay 148 min",
            "data": {"program": "GDP", "avg_delay_min": 148},
        },
        "tool_ok": True,
        "source": "faa_nas",
    },
    OTHER_ID: {
        "evidence": {
            "id": OTHER_ID,
            "source": "open_meteo",
            "subject": "BOS",
            "observation": "Rain, 15 % chance of showers, wind 20 knots",
            "data": {"precip_chance_pct": 15},
        },
        "tool_ok": True,
        "source": "open_meteo",
    },
}


def _claim(text: str, ids: list[str], risk: str | None = None) -> dict:
    claim = {"text": text, "evidence_ids": ids, "confidence": "high"}
    if risk is not None:
        claim["risk"] = risk
    return claim


CLAIM_GDP = _claim("LGA has a ground delay program averaging 148 min.", [FAA_ID], "high")
CLAIM_SHOWERS = _claim("There is a 15 % chance of showers at BOS.", [OTHER_ID])
CLAIM_EVENING = _claim("Expect delays at LGA this evening.", [FAA_ID], "high")


def _record() -> dict:
    return {
        "id": "t1",
        "scenario": "",
        "intent": "plan_trip",
        "request": "test request",
        "model": "test-model",
        "git_commit": "abc1234",
        "trip_goal": {"origin_airport": "LGA", "destination_airport": "BOS",
                      "departure_date": "2026-11-22", "departure_window": "evening"},
        "specialist_results": {
            "flight_airport": {"agent": "flight_airport", "claims": [CLAIM_GDP, CLAIM_EVENING],
                               "needs_human": False, "unavailable": []},
        },
        "evidence_store": STORE,
        "last_verification": None,
    }


def _engine(*_args, **_kwargs) -> dict:
    evidence = {
        "id": "engine:LGA-BOS:2026-11-22:evening:summary",
        "source": "engine",
        "subject": "LGA-BOS",
        "observation": "",
        "fetched_at": "2026-11-22T17:00:00+00:00",
        "data": {"level": "high", "confidence": 0.8, "action": "monitor"},
    }
    return {"source": "engine", "ok": True, "evidence": [evidence]}


# ------------------------------------------------------------------ mutations


def test_wrong_number_applies_and_changes_first_minutes_number():
    mutated = ablation.mutate_wrong_number(CLAIM_GDP)
    assert mutated is not None
    assert "464 min" in mutated["text"]
    assert mutated["evidence_ids"] == CLAIM_GDP["evidence_ids"]


def test_wrong_number_does_not_apply_without_a_unit_number():
    assert ablation.mutate_wrong_number(CLAIM_EVENING) is None


def test_wrong_airport_replaces_cited_code_with_an_absent_one():
    mutated = ablation.mutate_wrong_airport(CLAIM_GDP, STORE)
    assert mutated is not None
    assert mutated["text"].startswith("JFK has")
    assert "LGA" not in mutated["text"]


def test_wrong_airport_does_not_apply_without_cited_code():
    assert ablation.mutate_wrong_airport(_claim("Delays are likely.", [FAA_ID]), STORE) is None


def test_fabricated_id_changes_last_segment_and_is_not_in_store():
    mutated = ablation.mutate_fabricated_id(CLAIM_GDP, STORE)
    assert mutated is not None
    assert mutated["evidence_ids"] == ["faa_nas:LGA:GDP:zz9"]
    assert all(item not in STORE for item in mutated["evidence_ids"])


def test_fabricated_id_keeps_appending_until_the_id_is_absent():
    store = {**STORE, "faa_nas:LGA:GDP:zz9": STORE[FAA_ID],
             "faa_nas:LGA:GDP:zz99": STORE[FAA_ID]}
    mutated = ablation.mutate_fabricated_id(CLAIM_GDP, store)
    assert mutated is not None
    assert mutated["evidence_ids"] == ["faa_nas:LGA:GDP:zz999"]
    assert mutated["evidence_ids"][0] not in store


def test_fabricated_id_does_not_apply_without_ids():
    assert ablation.mutate_fabricated_id(_claim("No citation.", []), STORE) is None


def test_no_citation_clears_ids():
    assert ablation.mutate_no_citation(CLAIM_GDP)["evidence_ids"] == []


def test_risk_downplayed_applies_only_to_signals_below_the_rule_engine():
    assert ablation.mutate_risk_downplayed(CLAIM_GDP, "flight_airport", "high")["risk"] == "low"
    assert ablation.mutate_risk_downplayed(CLAIM_GDP, "weather", "severe")["risk"] == "low"
    assert ablation.mutate_risk_downplayed(CLAIM_GDP, "flight_airport", "low") is None
    assert ablation.mutate_risk_downplayed(CLAIM_GDP, "alternatives", "high") is None
    assert ablation.mutate_risk_downplayed(_claim("LGA is calm.", [FAA_ID], "low"),
                                           "weather", "high") is None


def test_normal_vs_program_applies_only_to_a_cited_faa_program():
    mutated = ablation.mutate_normal_vs_program(CLAIM_GDP, STORE)
    assert mutated is not None
    assert mutated["text"] == "LGA is operating normally with no FAA delays."
    assert mutated["evidence_ids"] == CLAIM_GDP["evidence_ids"]
    assert mutated["risk"] == CLAIM_GDP["risk"]
    assert ablation.mutate_normal_vs_program(CLAIM_SHOWERS, STORE) is None


def test_normal_vs_program_reads_the_evidence_source_not_the_storing_tool():
    store = {
        FAA_ID: {
            "evidence": {
                "id": FAA_ID,
                "source": "faa_nas",
                "subject": "LGA",
                "observation": "Ground Delay Program at LGA, average delay 148 min",
                "data": {"program": "GDP", "avg_delay_min": 148},
            },
            "tool_ok": True,
            "source": "deadline_planner",
        },
    }
    mutated = ablation.mutate_normal_vs_program(CLAIM_GDP, store)
    assert mutated is not None
    assert mutated["text"] == "LGA is operating normally with no FAA delays."


def test_wrong_time_window_replaces_the_first_phrase():
    mutated = ablation.mutate_wrong_time_window(CLAIM_EVENING)
    assert mutated is not None
    assert mutated["text"] == "Expect delays at LGA tomorrow."
    assert ablation.mutate_wrong_time_window(CLAIM_GDP) is None


# ---------------------------------------------------------------- sheet


def test_sheet_has_exact_columns_and_no_verdict_fields(tmp_path):
    rows = ablation.build_sheet_rows([_record()])
    assert len(rows) == 2
    assert tuple(rows[0]) == ("claim_id", "scenario", "request", "agent", "claim", "risk",
                              "confidence", "cited_ids", "evidence", "label", "note")
    assert tuple(rows[0]) == ablation.SHEET_COLUMNS
    forbidden = {"verdict", "status", "reason"}
    assert not (set(rows[0]) & forbidden)
    assert rows[0]["evidence"].startswith(f"{FAA_ID}: Ground Delay Program")
    assert rows[0]["claim_id"] == "t1:flight_airport:0"


def test_sheet_shows_numbers_from_data_missing_in_observation():
    claim = _claim("LGA has a max delay of 362 minutes.", [FAA_ID], "high")
    record = _record()
    record["evidence_store"] = copy.deepcopy(record["evidence_store"])
    record["specialist_results"]["flight_airport"]["claims"] = [claim]
    record["evidence_store"][FAA_ID]["evidence"]["data"]["max_delay_min"] = 362
    rows = ablation.build_sheet_rows([record])
    assert "362" in rows[0]["evidence"]
    assert "362" not in rows[0]["evidence"].split(" || data: ")[0]


def test_sheet_omits_replay_and_scenario_bookkeeping_keys():
    claim = _claim("LGA has delays.", [FAA_ID], "high")
    record = _record()
    record["evidence_store"] = copy.deepcopy(record["evidence_store"])
    record["specialist_results"]["flight_airport"]["claims"] = [claim]
    record["evidence_store"][FAA_ID]["evidence"]["data"].update(
        {"replay": "demo", "scenario": "s1"})
    rows = ablation.build_sheet_rows([record])
    assert '"replay"' not in rows[0]["evidence"]
    assert '"scenario"' not in rows[0]["evidence"]


def test_sheet_cuts_long_data_at_1200_characters():
    claim = _claim("LGA has delays.", [FAA_ID], "high")
    record = _record()
    record["evidence_store"] = copy.deepcopy(record["evidence_store"])
    record["specialist_results"]["flight_airport"]["claims"] = [claim]
    record["evidence_store"][FAA_ID]["evidence"]["data"]["padding"] = "x" * 2000
    rows = ablation.build_sheet_rows([record])
    data_part = rows[0]["evidence"].split(" || data: ", 1)[1]
    assert len(data_part) == 1201
    assert data_part.endswith("…")


def test_sheet_omits_validity_when_both_ends_are_missing():
    claim = _claim("LGA has delays.", [FAA_ID], "high")
    record = _record()
    record["specialist_results"]["flight_airport"]["claims"] = [claim]
    rows = ablation.build_sheet_rows([record])
    assert " || valid: " not in rows[0]["evidence"]


def test_sheet_shows_validity_when_present():
    claim = _claim("LGA has delays.", [FAA_ID], "high")
    record = _record()
    record["evidence_store"] = copy.deepcopy(record["evidence_store"])
    record["specialist_results"]["flight_airport"]["claims"] = [claim]
    record["evidence_store"][FAA_ID]["evidence"].update(
        {"valid_from": "2026-11-22T17:00:00+00:00", "valid_to": "2026-11-22T19:00:00+00:00"})
    rows = ablation.build_sheet_rows([record])
    assert " || valid: 2026-11-22T17:00:00+00:00..2026-11-22T19:00:00+00:00" in rows[0]["evidence"]


def test_sheet_leaves_uncited_claim_visibly_empty():
    claim = _claim("No citation.", [])
    record = _record()
    record["specialist_results"]["flight_airport"]["claims"] = [claim]
    rows = ablation.build_sheet_rows([record])
    assert rows[0]["cited_ids"] == ""
    assert rows[0]["evidence"] == ""
    assert rows[0]["claim"] == "No citation."


def test_sheet_cited_ids_join_with_comma_space():
    claim = _claim("Both sources.", [FAA_ID, OTHER_ID])
    record = _record()
    record["specialist_results"]["flight_airport"]["claims"] = [claim]
    rows = ablation.build_sheet_rows([record])
    assert rows[0]["cited_ids"] == f"{FAA_ID}, {OTHER_ID}"


def test_sheet_marks_missing_evidence():
    claim = _claim("Unknown evidence.", ["made:up:id"])
    record = _record()
    record["specialist_results"]["flight_airport"]["claims"] = [claim]
    rows = ablation.build_sheet_rows([record])
    assert "made:up:id: MISSING" in rows[0]["evidence"]


def test_sheet_refuses_to_overwrite_filled_labels_without_force(tmp_path):
    path = tmp_path / "labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ablation.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerow({**{column: "" for column in ablation.SHEET_COLUMNS},
                         "claim_id": "t1:flight_airport:0", "label": "correct"})
    rows = ablation.build_sheet_rows([_record()])
    with pytest.raises(SystemExit):
        ablation.write_sheet(rows, path=path, force=False)
    ablation.write_sheet(rows, path=path, force=True)
    with path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [row["label"] for row in written] == ["", ""]


# ---------------------------------------------------------------- metrics


def test_arm_a_counts_false_holds_and_catches():
    prepared = [
        {
            "id": "r1", "scenario": "s1", "intent": "assess_trip",
            "claims": [
                {"claim_id": "r1:flight_airport:0", "agent": "flight_airport",
                 "label": "wrong_number", "status": "contradicted"},
                {"claim_id": "r1:flight_airport:1", "agent": "flight_airport",
                 "label": "correct", "status": "supported"},
            ],
            "on": {"overall_risk": "low", "needs_human": True},
            "off": {"overall_risk": "high", "needs_human": False},
        },
        {
            "id": "r2", "scenario": "s1", "intent": "assess_trip",
            "claims": [
                {"claim_id": "r2:weather:0", "agent": "weather",
                 "label": "correct", "status": "contradicted"},
            ],
            "on": {"overall_risk": "low", "needs_human": True},
            "off": {"overall_risk": "low", "needs_human": False},
        },
    ]
    arm = ablation.compute_arm_a(prepared)
    assert arm["on"]["bad_claims_shown"] == {"wrong": 0, "shown": 1, "rate": 0.0}
    assert arm["off"]["bad_claims_shown"] == {"wrong": 1, "shown": 3, "rate": 1 / 3}
    assert arm["on"]["catch_rate"] == {"held": 1, "total": 1, "rate": 1.0}
    assert arm["on"]["false_hold_rate"] == {"held": 1, "total": 2, "rate": 0.5}
    assert arm["off"]["false_hold_rate"] == {"held": 0, "total": 2, "rate": 0.0}
    assert arm["on"]["escalation_rate"] == {"escalations": 2, "turns": 2, "rate": 1.0}
    assert arm["off"]["escalation_rate"] == {"escalations": 0, "turns": 2, "rate": 0.0}
    assert len(arm["headline_shift"]) == 1
    assert arm["headline_shift"][0]["id"] == "r1"


def test_arm_b_catches_the_injected_faults_and_keeps_controls():
    labels = {"t1:flight_airport:0": "correct", "t1:flight_airport:1": "correct"}
    prepared = [ablation.prepare_record(_record(), labels, engine=_engine)]
    arm = ablation.compute_arm_b(prepared)
    assert arm["base_claims"] == 2
    assert arm["controls"] == {"total": 2, "supported": 2}
    assert arm["faults"]["fabricated_id"]["caught"] == 2
    assert arm["faults"]["no_citation"]["caught"] == 2
    assert arm["faults"]["wrong_number"]["applicable"] == 1
    assert arm["faults"]["wrong_number"]["caught"] == 1
    assert arm["faults"]["wrong_airport"]["caught"] == 2
    assert arm["faults"]["risk_downplayed"]["applicable"] == 2
    assert arm["faults"]["risk_downplayed"]["caught"] == 2
    # normal_vs_program depends on the Verifier's airport-status rule; only its
    # applicability is part of this ablation's contract.
    assert arm["faults"]["normal_vs_program"]["applicable"] == 2
    assert arm["out_of_scope"]["applicable"] == 1


def _scenario_record() -> dict:
    evidence = {
        **STORE[FAA_ID],
        "evidence": {**STORE[FAA_ID]["evidence"],
                     "valid_to": "2026-11-23T00:00:00+00:00"},
    }
    record = _record()
    record["id"] = "clock1"
    record["scenario"] = "nyc_rain_20261122_1708z"
    record["specialist_results"] = {
        "flight_airport": {"agent": "flight_airport", "claims": [CLAIM_GDP],
                           "needs_human": False, "unavailable": []},
    }
    record["evidence_store"] = {FAA_ID: evidence}
    return record


def test_arm_b_verifies_inside_the_scenario_clock():
    labels = {"clock1:flight_airport:0": "correct"}
    prepared = [ablation.prepare_record(_scenario_record(), labels, engine=_engine)]
    assert prepared[0]["claims"][0]["status"] == "supported"

    arm = ablation.compute_arm_b(prepared)
    assert arm["controls"] == {"total": 1, "supported": 1}
    assert arm["faults"]["no_citation"]["applicable"] == 1
    assert arm["faults"]["no_citation"]["caught"] == 1
    assert arm["base_claims"] == 1


def test_arm_b_raises_when_a_control_is_not_supported():
    prepared = [{
        "id": "r1", "scenario": "", "turn_without_claims": False, "rules": {},
        "claims": [{"claim_id": "r1:flight_airport:0", "agent": "flight_airport",
                    "claim": CLAIM_GDP, "label": "correct", "status": "supported"}],
        "store": STORE,
    }]

    def unsupported(*_args, **_kwargs):
        return {"verdicts": [{"status": "unsupported"}]}

    with pytest.raises(RuntimeError, match="controls"):
        ablation.compute_arm_b(prepared, verify=unsupported)


def test_headline_shift_treats_unknown_as_below_low():
    prepared = [{
        "id": "r1", "scenario": "s1", "intent": "assess_trip",
        "claims": [{"claim_id": "r1:flight_airport:0", "agent": "flight_airport",
                    "label": "correct", "status": "contradicted"}],
        "on": {"overall_risk": "unknown", "needs_human": True},
        "off": {"overall_risk": "high", "needs_human": False},
    }]
    arm = ablation.compute_arm_a(prepared)
    assert len(arm["headline_shift"]) == 1
    shift = arm["headline_shift"][0]
    assert shift["on_overall_risk"] == "unknown"
    assert shift["off_overall_risk"] == "high"


def test_held_claims_group_by_reason_with_numbers_normalised():
    prepared = [{
        "id": "r1", "scenario": "s1", "intent": "assess_trip",
        "turn_without_claims": False,
        "claims": [
            {"claim_id": "r1:weather:0", "agent": "weather", "label": "correct",
             "status": "contradicted",
             "reason": "states 26 mph but cited evidence does not"},
            {"claim_id": "r1:weather:1", "agent": "weather", "label": "wrong_number",
             "status": "contradicted",
             "reason": "states 25 mph but cited evidence does not"},
        ],
    }]
    rows = ablation.compute_held_by_reason(prepared)
    assert len(rows) == 1
    assert rows[0]["reason"] == "states N mph but cited evidence does not"
    assert rows[0]["correct"] == 1
    assert rows[0]["wrong"] == 1
    assert rows[0]["count"] == 2


def test_turns_without_claims_are_reported_and_excluded_from_escalation():
    no_claims = {
        "id": "c1", "scenario": "", "intent": "plan_trip",
        "specialist_results": {}, "evidence_store": {}, "trip_goal": {},
        "last_verification": None,
    }
    missing_intent = {**no_claims, "id": "c2", "intent": None}
    errored = {"id": "c3", "scenario": "", "error": "boom"}
    report = ablation.build_report([no_claims, missing_intent, errored], {},
                                   engine=_engine)
    assert set(report["turns_without_claims"]) == {"c1", "c2", "c3"}
    assert report["arm_a"]["on"]["escalation_rate"]["turns"] == 0
    assert report["arm_a"]["off"]["escalation_rate"]["turns"] == 0


def test_multi_agent_claims_line_up_with_ids_and_shown_claims():
    weather_claim = _claim("There is a 55 % chance of showers at BOS.", [OTHER_ID])
    alternative_claim = _claim("BOS has light rain showers this evening.", [OTHER_ID])
    record = _record()
    record["id"] = "m1"
    record["specialist_results"] = {
        "flight_airport": {"agent": "flight_airport", "claims": [CLAIM_GDP],
                           "needs_human": False, "unavailable": []},
        "weather": {"agent": "weather", "claims": [weather_claim],
                    "needs_human": False, "unavailable": []},
        "alternatives": {"agent": "alternatives", "claims": [alternative_claim],
                         "needs_human": False, "unavailable": []},
    }
    prepared = ablation.prepare_record(record, {}, engine=_engine)
    claims = prepared["claims"]
    assert [item["claim_id"] for item in claims] == [
        "m1:flight_airport:0", "m1:weather:0", "m1:alternatives:0",
    ]
    assert [item["status"] for item in claims] == [
        "supported", "contradicted", "supported",
    ]
    shown = sum(1 for item in claims if item["status"] == "supported")
    assert shown == len(prepared["on"]["shown_claims"])


# ---------------------------------------------------------------- score/CLI


def _write_raw(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _write_labels(path, labels, records=None, encoding="utf-8"):
    texts = ablation.claim_texts(records or [])
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ablation.SHEET_COLUMNS)
        writer.writeheader()
        for claim_id, label in labels.items():
            writer.writerow({**{column: "" for column in ablation.SHEET_COLUMNS},
                             "claim_id": claim_id, "claim": texts.get(claim_id, ""),
                             "label": label})


def test_score_fails_on_unlabelled_claims_and_allows_excluding_them(tmp_path):
    raw = tmp_path / "raw.jsonl"
    labels_path = tmp_path / "labels.csv"
    out = tmp_path / "out"
    _write_raw(raw, [_record()])
    _write_labels(labels_path, {"t1:flight_airport:0": "correct"}, records=[_record()])

    assert ablation.run_score(raw, labels_path, out, engine=_engine) == 1
    assert not (out / "ablation.json").exists()

    assert ablation.run_score(raw, labels_path, out, allow_unlabelled=True, engine=_engine) == 0
    report = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
    assert report["claims"] == 2
    assert report["unlabelled_excluded"] == 1
    assert report["arm_b"]["base_claims"] == 1

    svg = (out / "ablation.svg").read_text(encoding="utf-8")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert (out / "ablation.md").read_text(encoding="utf-8").startswith("# Verifier ablation")


def test_score_records_the_labeller_and_defaults_when_omitted(tmp_path):
    raw = tmp_path / "raw.jsonl"
    labels_path = tmp_path / "labels.csv"
    _write_raw(raw, [_record()])
    _write_labels(labels_path,
                  {"t1:flight_airport:0": "correct", "t1:flight_airport:1": "correct"},
                  records=[_record()])

    named = tmp_path / "named"
    assert ablation.run_score(raw, labels_path, named, engine=_engine, labeller="Alice") == 0
    report = json.loads((named / "ablation.json").read_text(encoding="utf-8"))
    assert report["labeller"] == "Alice"
    markdown = (named / "ablation.md").read_text(encoding="utf-8")
    assert "Labels: Alice" in markdown
    assert "One labeller (Alice);" in markdown

    default = tmp_path / "default"
    assert ablation.run_score(raw, labels_path, default, engine=_engine) == 0
    report = json.loads((default / "ablation.json").read_text(encoding="utf-8"))
    assert report["labeller"] == "not recorded"
    markdown = (default / "ablation.md").read_text(encoding="utf-8")
    assert "Labels: not recorded" in markdown
    assert "One labeller (not recorded);" in markdown


def test_zero_applicable_in_scope_fault_is_called_out():
    record = _record()
    record["id"] = "z1"
    record["specialist_results"] = {
        "flight_airport": {"agent": "flight_airport", "claims": [CLAIM_EVENING],
                           "needs_human": False, "unavailable": []},
    }
    labels = {"z1:flight_airport:0": "correct"}
    report = ablation.build_report([record], labels, engine=_engine)
    assert report["arm_b"]["faults"]["wrong_number"]["applicable"] == 0
    markdown = ablation.render_markdown(report)
    assert "No applicable base claims for: wrong_number." in markdown


def test_score_rejects_labels_whose_claim_text_is_stale(tmp_path, capsys):
    raw = tmp_path / "raw.jsonl"
    labels_path = tmp_path / "labels.csv"
    out = tmp_path / "out"
    _write_raw(raw, [_record()])
    _write_labels(labels_path,
                  {"t1:flight_airport:0": "correct", "t1:flight_airport:1": "correct"},
                  records=[_record()])
    with labels_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["claim"] = "A stale claim text from a previous collect."
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ablation.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    assert ablation.run_score(raw, labels_path, out, engine=_engine) == 1
    assert "t1:flight_airport:0" in capsys.readouterr().err
    assert not (out / "ablation.json").exists()


def test_labels_are_read_from_a_bom_prefixed_csv(tmp_path):
    raw = tmp_path / "raw.jsonl"
    labels_path = tmp_path / "labels.csv"
    out = tmp_path / "out"
    _write_raw(raw, [_record()])
    _write_labels(labels_path,
                  {"t1:flight_airport:0": "correct", "t1:flight_airport:1": "correct"},
                  records=[_record()], encoding="utf-8-sig")
    assert labels_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert set(ablation.load_labels(labels_path)) == {"t1:flight_airport:0", "t1:flight_airport:1"}
    assert ablation.run_score(raw, labels_path, out, engine=_engine) == 0


def test_sheet_values_contain_no_verifier_output(tmp_path):
    marker = "VERIFIER_ONLY_MARKER_9f3a"
    record = _record()
    record["last_verification"] = {"summary": marker,
                                   "verdicts": [{"status": "contradicted", "reason": marker}]}
    record["final_answer"] = {"text": marker, "held_back": [{"reason": marker}]}
    path = tmp_path / "labels.csv"
    ablation.write_sheet(ablation.build_sheet_rows([record]), path=path)
    written = path.read_text(encoding="utf-8")
    assert marker not in written


def test_verify_results_matches_the_finish_recipe():
    results = {
        "flight_airport": {"agent": "flight_airport", "claims": [CLAIM_GDP]},
        "weather": {"agent": "weather", "claims": []},
        "alternatives": {"agent": "alternatives", "claims": [CLAIM_SHOWERS]},
    }
    rules_result = _engine()
    with_rules = {"evidence_store": dict(STORE)}
    ablation.add_tool_result(with_rules, rules_result)
    expected = combine_verifications(
        verify_claims([CLAIM_GDP], with_rules["evidence_store"]),
        verify_claims([CLAIM_SHOWERS], STORE),
    )
    assert verify_results(results, STORE, rules_result) == expected


def test_help_and_subcommand_help_work_without_a_key(capsys):
    with pytest.raises(SystemExit) as top:
        ablation.main(["--help"])
    assert top.value.code == 0
    for command in ("dry-run", "collect", "sheet", "score"):
        with pytest.raises(SystemExit) as sub:
            ablation.main([command, "--help"])
        assert sub.value.code == 0
    assert "dry-run" in capsys.readouterr().out
