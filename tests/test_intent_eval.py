"""Guards on the intent evaluation set (eval/intent_cases.jsonl)."""

from __future__ import annotations

from collections import Counter

from eval.run_intent_eval import SLOT_CHECKS, load_cases, score_case, summarise
from travel_agent.agents.intent import FEW_SHOT, INTENTS, classify_with_rules


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def test_no_eval_case_is_a_few_shot_example():
    # Otherwise the LLM would be graded on examples it is shown in its prompt.
    shown = {_norm(request) for request, _ in FEW_SHOT}
    assert not [case["id"] for case in load_cases() if _norm(case["text"]) in shown]


def test_cases_are_balanced_and_well_formed():
    cases = load_cases()
    assert len({case["id"] for case in cases}) == len(cases)
    assert Counter(case["intent"] for case in cases) == {intent: 6 for intent in INTENTS}
    allowed = {"id", "text", "intent", *SLOT_CHECKS}
    for case in cases:
        assert set(case) <= allowed, case["id"]


def test_rules_baseline_runs_offline():
    rows = [score_case(case, classify_with_rules) for case in load_cases()]
    summary = summarise(rows)
    assert summary["cases"] == 24
    assert all(row["source"] == "rules" for row in rows)
