"""The Verifier agent.

The agent is intentionally thin: it receives the specialists' claims, calls one
deterministic tool **once**, and reports the verdicts verbatim. All checking
logic lives in ``travel_agent.verification.checks`` so it can be unit-tested
without a model or an API key. Building the agent performs no network calls and
needs no credentials.

The tool reads the evidence store from ``tool_context.state`` (populated by the
``record_and_sanitize`` callback); the agent never passes evidence ids through
the model, which keeps potentially large evidence out of the prompt.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext

from travel_agent.guardrails import callbacks as guard
from travel_agent.observability.trace import get_active_tracer
from travel_agent.verification.checks import verify_claims

VERIFIER_INSTRUCTION = """\
You are the Verifier. You receive claims from specialist travel agents. Each
claim is a dict with:
  - "text": the statement the specialist made,
  - "evidence_ids": the ids of the evidence it is based on (required),
  - "confidence": "low" | "medium" | "high",
  - "risk": "low" | "moderate" | "high" or null.

Call `verify_claims_tool` exactly once with the full list of claims. Do not
call it again and do not invent evidence ids. Report every verdict exactly as
returned: never upgrade, downgrade or reword a status. If the result has
`needs_human` true, state clearly which claims a human must check and why, using
the returned reasons. Keep the response short and factual.

The authoritative verdict is also persisted in `tool_context.state[
"last_verification"]`. The human-in-the-loop step must read the verdict from
that state key, never from this agent's prose.
"""


def verify_claims_tool(claims: list[dict], tool_context: ToolContext) -> dict:
    """Verify specialised claims against the evidence tools already returned.

    Pass every claim from the specialists in one call. Each claim must look
    like::

        {"text": "SFO arrivals are delayed about 34 min this evening",
         "evidence_ids": ["faa_nas:SFO:GDP:0"],
         "confidence": "high",
         "risk": "moderate"}

    Returns `{"verdicts": [...], "needs_human": bool, "summary": str, "counts":
    {...}}`. Each verdict has `claim`, `status`
    (`supported`/`unsupported`/`contradicted`), `reason` and the individual
    `checks` that ran. `evidence_ids` must be ids returned by a tool; unknown
    ids are reported as unsupported.
    """
    state: Any = getattr(tool_context, "state", None)
    store: Any = {}
    try:
        if state is not None:
            store = state.get("evidence_store") or {}
    except Exception:
        store = {}

    result = verify_claims(claims, store if isinstance(store, dict) else {})

    try:
        if state is not None:
            state["last_verification"] = result
    except Exception:
        pass

    try:
        tracer = get_active_tracer()
        if tracer is not None:
            tracer.log_verdicts(result["counts"], result["needs_human"])
    except Exception:
        pass
    return result


def build_verifier_agent(model: str = "gemini-2.5-flash-lite") -> LlmAgent:
    """Build the ``verifier`` ``LlmAgent`` (no API key or network required)."""
    return LlmAgent(
        name="verifier",
        model=model,
        instruction=VERIFIER_INSTRUCTION,
        tools=[verify_claims_tool],
        before_model_callback=guard.input_guard,
        after_model_callback=guard.output_guard,
    )
