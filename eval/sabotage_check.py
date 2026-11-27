"""Proof that the M4 safety tests are real: switch a protection OFF and watch its tests fail.

    python -m eval.sabotage_check

For each safety feature, the test suite is run once with that feature
disabled *inside the test process only* (no project file is changed). A real
test must FAIL when the thing it protects is removed; a fake test would keep
passing. Offline and free (no API key used).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
TESTS = ["tests/test_m4_danger.py", "tests/test_m4_hitl.py", "tests/test_m4_planner.py", "tests/test_m4_history.py"]

MODES = {
    "none": "nothing switched off (normal code)",
    "injection": "the 'looks like an instruction to the AI' check",
    "expiry": "the 15-minute approval expiry",
    "fake_booking_claim": "the 'says booked but nothing was booked' check",
    "stale_data": "the block on booking with data older than 30 min",
    "missing_faa_data": "the rule that missing FAA data is never 'comfortable'",
    "history_rule": "the lesson learned from real FAA/BTS history",
}


# ---------------------------------------------------------------- pytest plugin part
def pytest_configure(config):  # called by pytest when loaded with -p eval.sabotage_check
    mode = os.environ.get("SABOTAGE", "none")
    if mode == "injection":
        import travel_agent.hitl.approval as approval
        import travel_agent.memory.cases as cases
        import travel_agent.memory.preferences as preferences

        never = lambda text: False  # noqa: E731
        approval.looks_like_instruction = never
        cases.looks_like_instruction = never
        preferences.looks_like_instruction = never
    elif mode == "expiry":
        import travel_agent.hitl.approval as approval

        approval.APPROVAL_TTL_MIN = 10_000_000
    elif mode == "fake_booking_claim":
        import travel_agent.hitl.approval as approval

        approval._BOOKED_RE = re.compile(r"(?!x)x")  # never matches
    elif mode == "stale_data":
        import travel_agent.hitl.approval as approval

        approval.MAX_EVIDENCE_AGE_MIN = 10_000_000
    elif mode == "missing_faa_data":
        import travel_agent.tools.deadline_planner as planner

        planner._downgrade_if_delay_unknown = lambda option, notes: None
    elif mode == "history_rule":
        import travel_agent.tools.deadline_planner as planner

        planner._apply_history = lambda option, notes: None


# ---------------------------------------------------------------- runner part
def main() -> None:
    print(f"{'Safety feature switched OFF':58} | tests failed")
    print("-" * 80)
    for mode, label in MODES.items():
        env = dict(os.environ, SABOTAGE=mode, TDA_CASE_EMBEDDINGS="0", PYTHONIOENCODING="utf-8")
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "eval.sabotage_check", "-p", "no:cacheprovider", "-q", "-rf", "--color=no",
             "-p", "no:warnings", *TESTS, "-k", "not G1"],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        failed = sorted({re.sub(r"\[.*", "", line.split("::")[-1].split(" ")[0])
                         for line in out.stdout.splitlines() if line.startswith("FAILED")})
        summary = (out.stdout.strip().splitlines() or ["?"])[-1]
        count = re.search(r"(\d+) failed", summary)
        names = ", ".join(name.replace("test_", "")[:28] for name in failed[:3]) + (" ..." if len(failed) > 3 else "")
        print(f"{label:58} | {count.group(1) if count else 0}  {names}")
    print("\nA real test fails when its protection is removed; with nothing switched off, 0 should fail.")


if __name__ == "__main__":
    main()
