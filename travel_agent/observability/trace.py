"""JSON Lines tracing for agent runs.

One ``Tracer`` writes one file per run: ``<log_dir>/<run_id>.jsonl``. Every line
is a self-contained JSON object with a ``ts`` and the event type, so the log can
be tailed, grepped, or loaded by ``summarize_run`` without a schema registry.

Why JSONL instead of the built-in ADK telemetry? It survives process crashes
part-way through a line, is trivial to parse in the demo notebook, and keeps
PII handling under our control: tool arguments are redacted *before* they are
written.

The module also exposes ``set_active_tracer`` / ``get_active_tracer``. Guardrail
callbacks and the verifier tool call these helpers, so they log when a tracer is
configured and are silent no-ops otherwise. That keeps the guardrail modules
free of global state of their own.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from travel_agent.guardrails.pii import redact_pii
from travel_agent.guardrails.sanitize import sanitize_tool_result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    """Deterministic JSON for dict keys (argument timing lookups)."""
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _redact_value(value: Any) -> Any:
    """Recursively PII-redact string values inside tool arguments."""
    if isinstance(value, str):
        return redact_pii(value)[0]
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


class Tracer:
    """Append-only JSONL tracer for a single agent run."""

    def __init__(self, log_dir: str | Path = "logs", run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{self.run_id}.jsonl"
        self._lock = threading.Lock()
        self._timing_lock = threading.Lock()
        # A key may map to several starts (parallel identical calls); we pop
        # FIFO so each after-hook gets its own before-hook.
        self._tool_starts: dict[Any, list[float]] = {}
        self._model_starts: dict[tuple[str, str], float] = {}

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event and return the record that was written."""
        record = {"ts": _now_iso(), "run_id": self.run_id, "event": event, **fields}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return record

    def log_tool_call(
        self,
        *,
        agent: str | None,
        tool: str,
        args: Any,
        latency_ms: float | None,
        ok: bool,
        evidence_ids: list[str] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Log one finished tool call (arguments are PII-redacted first)."""
        return self.log(
            "tool_call",
            agent=agent,
            tool=tool,
            args=_redact_value(args),
            latency_ms=None if latency_ms is None else round(latency_ms, 2),
            ok=bool(ok),
            evidence_ids=evidence_ids or [],
            error=error,
        )

    def log_guardrail(self, kind: str, decision: str, reasons: list[str] | None = None) -> dict[str, Any]:
        """Log a guardrail decision (``kind`` is e.g. injection/pii/tool_gate)."""
        return self.log("guardrail", kind=kind, decision=decision, reasons=reasons or [])

    def log_verdicts(self, counts: dict[str, int], needs_human: bool) -> dict[str, Any]:
        """Log the status counts and the human-review flag from a Verifier run."""
        return self.log("verdicts", counts=counts, needs_human=bool(needs_human))

    def log_model_call(self, agent: str | None, latency_ms: float | None = None) -> dict[str, Any]:
        """Log one model invocation (latency is ``None`` when it could not be timed)."""
        return self.log("model_call", agent=agent, latency_ms=None if latency_ms is None else round(latency_ms, 2))

    def _tool_key(self, tool: str, args: Any, key: str | None = None) -> Any:
        """Prefer ADK's ``function_call_id``; otherwise fall back to stable args."""
        if key:
            return ("fid", self.run_id, key)
        return ("args", self.run_id, tool, _stable_json(args))

    def start_tool(self, tool: str, args: Any, key: str | None = None) -> None:
        """Remember a start time keyed by call id, else ``(run_id, tool, args)``."""
        marker = self._tool_key(tool, args, key)
        with self._timing_lock:
            self._tool_starts.setdefault(marker, []).append(time.perf_counter())

    def pop_tool_latency(self, tool: str, args: Any, key: str | None = None) -> float | None:
        """Return and forget one elapsed time (FIFO for duplicate keys)."""
        marker = self._tool_key(tool, args, key)
        with self._timing_lock:
            starts = self._tool_starts.get(marker)
            if not starts:
                return None
            started = starts.pop(0)
            if not starts:
                self._tool_starts.pop(marker, None)
        return (time.perf_counter() - started) * 1000.0

    def start_model(self, agent: str | None) -> None:
        """Remember the start time of the current model call for ``agent``."""
        self._model_starts[(self.run_id, agent or "")] = time.perf_counter()

    def pop_model_latency(self, agent: str | None) -> float | None:
        """Return and forget the elapsed milliseconds for a model call."""
        started = self._model_starts.pop((self.run_id, agent or ""), None)
        return None if started is None else (time.perf_counter() - started) * 1000.0


_active_tracer: Tracer | None = None


def set_active_tracer(tracer: Tracer | None) -> None:
    """Set (or clear) the process-wide tracer used by guardrails and the verifier."""
    global _active_tracer
    _active_tracer = tracer


def get_active_tracer() -> Tracer | None:
    """The currently configured tracer, or None."""
    return _active_tracer


def _agent_name(ctx: Any) -> str | None:
    for attr in ("agent_name",):
        value = getattr(ctx, attr, None)
        if value:
            return str(value)
    return None


def _bind_run_id(ctx: Any, tracer: Tracer) -> None:
    """Publish the trace id in ``ctx.state`` for other components to reuse."""
    try:
        state = getattr(ctx, "state", None)
        if state is not None:
            state["trace_run_id"] = tracer.run_id
    except Exception:
        pass


def trace_callbacks(tracer: Tracer | None = None) -> dict[str, Callable[..., Any]]:
    """Return ``before_tool``/``after_tool``/``before_model``/``after_model`` callbacks.

    Tool latency is keyed by ``(run_id, tool.name, stable json of args)``, exactly
    as required for matching a before/after pair even when tools interleave.
    Callbacks never raise: a tracing failure must not break the run.
    """
    active = tracer

    def _tracer() -> Tracer | None:
        return active or _active_tracer

    def before_tool(tool: Any, args: dict[str, Any], tool_context: Any) -> None:
        try:
            tracer_ = _tracer()
            if tracer_ is not None:
                _bind_run_id(tool_context, tracer_)
                tracer_.start_tool(
                    str(getattr(tool, "name", "") or "unknown"),
                    args,
                    key=getattr(tool_context, "function_call_id", None),
                )
        except Exception:  # tracing must never break the agent
            return None
        return None

    def after_tool(tool: Any, args: dict[str, Any], tool_context: Any, tool_response: dict[str, Any]) -> None:
        try:
            tracer_ = _tracer()
            if tracer_ is None:
                return None
            name = str(getattr(tool, "name", "") or "unknown")
            latency = tracer_.pop_tool_latency(
                name, args, key=getattr(tool_context, "function_call_id", None)
            )
            response = tool_response if isinstance(tool_response, dict) else {}
            try:
                response, _ = sanitize_tool_result(response)
            except Exception:
                pass
            evidence_ids = [
                str(item.get("id"))
                for item in (response.get("evidence") or [])
                if isinstance(item, dict) and item.get("id")
            ]
            error = response.get("error")
            if isinstance(error, str):
                error = redact_pii(error)[0]
            tracer_.log_tool_call(
                agent=_agent_name(tool_context),
                tool=name,
                args=args,
                latency_ms=latency,
                ok=bool(response.get("ok", False)),
                evidence_ids=evidence_ids,
                error=error,
            )
        except Exception:
            return None
        return None

    def before_model(callback_context: Any, llm_request: Any) -> None:
        try:
            tracer_ = _tracer()
            if tracer_ is not None:
                _bind_run_id(callback_context, tracer_)
                tracer_.start_model(_agent_name(callback_context))
        except Exception:
            return None
        return None

    def after_model(callback_context: Any, llm_response: Any) -> None:
        try:
            tracer_ = _tracer()
            if tracer_ is not None:
                tracer_.log_model_call(
                    _agent_name(callback_context),
                    tracer_.pop_model_latency(_agent_name(callback_context)),
                )
        except Exception:
            return None
        return None

    return {
        "before_tool": before_tool,
        "after_tool": after_tool,
        "before_model": before_model,
        "after_model": after_model,
    }


def _fail_closed_result(kwargs: dict[str, Any]) -> Any:
    """The safe answer when a ``__fail_closed__`` guard raises."""
    if "llm_request" in kwargs or "llm_response" in kwargs:
        return LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text="I can't help with that request.")]
            )
        )
    if "tool_context" in kwargs and "tool_response" not in kwargs:
        return {
            "ok": False,
            "error": "blocked: guardrail error",
            "evidence": [],
            "source": "guardrail",
        }
    return None


def chain_callbacks(*fns: Callable[..., Any]) -> Callable[..., Any]:
    """Compose callbacks of the same kind; the first non-None return wins.

    Positional **and** keyword arguments are forwarded unchanged, which is how
    ADK invokes callbacks (``callback_context=``, ``llm_request=``, ``tool=``,
    ``args=``, ``tool_context=``, ``tool_response=``). All callbacks run (so
    tracing side effects still happen); the first non-None result propagates.

    A callback marked ``__fail_closed__`` (all guardrails are) makes the chain
    fail closed if it raises: before-model/before-tool hooks return a refusal or
    blocked result instead of ``None``. Other exceptions are swallowed, as
    before.

    Async callbacks are rejected with a clear ``TypeError`` when the chain is
    built: ADK already accepts a list of callbacks, so an async hook should be
    wired as a list rather than hidden inside this synchronous chain.
    """
    for fn in fns:
        if inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"chain_callbacks does not support async callbacks ({getattr(fn, '__name__', fn)}); "
                "pass async callbacks to ADK as a list instead"
            )

    def _chain(*args: Any, **kwargs: Any) -> Any:
        result: Any = None
        for fn in fns:
            try:
                value = fn(*args, **kwargs)
            except Exception:
                if getattr(fn, "__fail_closed__", False):
                    fallback = _fail_closed_result(kwargs)
                    if fallback is not None:
                        return fallback
                continue
            if inspect.isawaitable(value):
                close = getattr(value, "close", None)
                if close is not None:
                    close()
                raise TypeError(
                    "chain_callbacks does not support callbacks that return awaitables; "
                    "pass async callbacks to ADK as a list instead"
                )
            if result is None and value is not None:
                result = value
        return result

    _chain.__chain_callbacks__ = fns
    return _chain


def summarize_run(path: str | Path) -> dict[str, Any]:
    """Aggregate one JSONL run into the numbers shown in the demo.

    Returns tool call counts, failure rate, mean latency, guardrail blocks and
    verdict counts. Malformed lines are skipped rather than failing the summary.
    """
    path = Path(path)
    tools: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    failures = 0
    call_count = 0
    guardrail_blocks = 0
    verdicts: dict[str, int] = {}
    needs_human = 0
    events = 0

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            events += 1
            kind = record.get("event")
            if kind == "tool_call":
                name = str(record.get("tool") or "unknown")
                entry = tools.setdefault(name, {"calls": 0, "failures": 0, "latencies": []})
                entry["calls"] += 1
                call_count += 1
                raw_latency = record.get("latency_ms")
                if raw_latency is not None:
                    latency = float(raw_latency)
                    entry["latencies"].append(latency)
                    latencies.append(latency)
                if record.get("ok") is False:
                    entry["failures"] += 1
                    failures += 1
            elif kind == "guardrail":
                if str(record.get("decision")) in {"block", "refuse"}:
                    guardrail_blocks += 1
            elif kind == "verdicts":
                for status, count in (record.get("counts") or {}).items():
                    verdicts[str(status)] = verdicts.get(str(status), 0) + int(count)
                if record.get("needs_human"):
                    needs_human += 1

    for entry in tools.values():
        values = entry.pop("latencies")
        entry["mean_latency_ms"] = round(sum(values) / len(values), 2) if values else 0.0

    return {
        "run_id": path.stem,
        "path": str(path),
        "events": events,
        "tool_calls": call_count,
        "tool_failures": failures,
        "failure_rate": round(failures / call_count, 3) if call_count else 0.0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "tools": tools,
        "guardrail_blocks": guardrail_blocks,
        "verdicts": verdicts,
        "needs_human": needs_human,
    }
