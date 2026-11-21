# Guardrails

Deterministic guardrail layers plus an optional Gemini classifier and the ADK
callbacks that wire them in. Everything works offline; the classifier is only
asked when enabled and a key is present.

| Module | Purpose |
|---|---|
| `pii.py` | Replace emails, phones, Luhn-valid cards, SG NRIC/FIN and passport numbers with typed placeholders. Flight numbers, dates, times, prices and distances are left alone. |
| `injection.py` | Layer 1 rules for prompt injection (override, role hijack, prompt extraction, fake role tags, forged approval, encoded payload); `detect_injection` optionally adds layer 2. |
| `llm_classifier.py` | Optional Gemini layer 2: `classify_injection` / batched `classify_injections`. No key -> `available=False`, no call; ~5 s timeout; sha256 cache. |
| `topics.py` | Blocks harmful *intent* (make/acquire weapons, evade or conceal from security/customs, fake documents/smuggling, self-harm). Asking what the rules allow is allowed. |
| `sanitize.py` | Sanitise a `ToolResult` on a copy: every string in untrusted evidence (observation, subject, nested `data`) is checked, and PII redacted. |
| `callbacks.py` | ADK hooks: `input_guard`, `output_guard`, `tool_gate`, `record_and_sanitize`. |

## Wiring an `LlmAgent`

ADK may give one callable or a list per hook. `observability.trace.chain_callbacks`
is available when you need to combine guardrails with tracing:

```python
from travel_agent.guardrails import callbacks as guard
from travel_agent.observability.trace import chain_callbacks, trace_callbacks

trace = trace_callbacks()          # add a Tracer to write JSONL
agent = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    tools=ALL_TOOLS,
    before_model_callback=chain_callbacks(guard.input_guard, trace["before_model"]),
    after_model_callback=chain_callbacks(guard.output_guard, trace["after_model"]),
    before_tool_callback=chain_callbacks(guard.tool_gate, trace["before_tool"]),
    after_tool_callback=chain_callbacks(guard.record_and_sanitize, trace["after_tool"]),
)
```

ADK invokes callbacks by keyword, and every function here uses ADK's parameter
names. Failure policy: input/output guards and the tool gate fail **closed**;
`record_and_sanitize` drops untrusted evidence and keeps trusted evidence if the
sanitizer raises. Call `guard.assert_guarded(agent)` in tests.

Tool gate: tool names are split on `_`, `-` and camelCase and matched as whole
words, so `paypal_lookup` and `display_hold_music` are not gated while
`pay_fare`, `hold_fare`, `rebook_flight`, `reserve_seat`, `purchase_ticket` and
`change_flight` are (a leading/trailing side-effect word such as
`book|reserve|purchase|pay|ticket|cancel|hold`; `rebook` counts as `book`).
Explicit `GATED_TOOLS` entries always need an exact-args approval in
`approved_actions`; approvals are single-use. Layer 2 is off unless
`GUARDRAIL_LLM=1` (or `use_llm=True`); set `GUARDRAIL_MODEL` to override
`gemini-3.5-flash-lite`.

## State keys

| Key | Shape | Written by |
|---|---|---|
| `evidence_store` | `{evidence_id: {"evidence": {...}, "tool_ok": bool, "source": str}}` | `record_and_sanitize` |
| `approved_actions` | `list[{"tool": str, "args": dict}]` (single-use) | the HITL / approval step (M4) |
| `completed_actions` | `list[{"tool": str, "args": dict, "ok": True}]` | `record_and_sanitize` |
| `guardrail_events` | `list[{"kind", "decision", "reasons"}]` | the guardrail callbacks |

The same keys are documented in `docs/INTERFACES.md`.
