"""Score the held-out guardrail set (eval/heldout_cases.jsonl).

These cases were written by a reviewer and never shown to whoever tuned the
detectors, so they measure generalisation, unlike the dev set in
guardrail_cases.jsonl. Run:

    python -m eval.run_heldout          # rules only (offline)
    python -m eval.run_heldout --llm    # rules + Gemini classifier (needs GOOGLE_API_KEY)
"""

import json
import sys
from pathlib import Path

import travel_agent.core  # noqa: F401  (loads .env so the classifier can find the key)
from travel_agent.guardrails.injection import detect_injection
from travel_agent.guardrails.pii import redact_pii
from travel_agent.guardrails.topics import check_topic

CASES = Path(__file__).with_name("heldout_cases.jsonl")


def blocked(case: dict, use_llm: bool) -> bool:
    if case["kind"] == "topic":
        return not check_topic(case["text"])["allowed"]
    return detect_injection(case["text"], use_llm=use_llm)["flagged"]


def main() -> int:
    use_llm = "--llm" in sys.argv
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    tally: dict[str, list[int]] = {}
    failures = []
    for case in cases:
        if case["kind"] == "pii":
            changed = redact_pii(case["text"])[0] != case["text"]
            ok = changed if case["expect"] == "redact" else not changed
        else:
            ok = blocked(case, use_llm) == (case["expect"] == "block")
        key = f'{case["kind"]}:{case["expect"]}'
        tally.setdefault(key, [0, 0])
        tally[key][0] += ok
        tally[key][1] += 1
        if not ok:
            failures.append(case["id"])
    print(f"held-out set, {'rules + Gemini' if use_llm else 'rules only'}")
    for key, (good, total) in sorted(tally.items()):
        print(f"  {key:22s} {good}/{total} correct")
    print("failures:", ", ".join(failures) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
