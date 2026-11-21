"""Evidence store shared through ``ctx.state["evidence_store"]``.

The store is a plain JSON-serialisable dict::

    {evidence_id: {"evidence": <Evidence as dict>,
                   "tool_ok": bool,
                   "source": str}}

Keeping it a dict (rather than a class) is deliberate: ADK copies
``ctx.state`` between callbacks and tools, and every value crosses that boundary
as JSON. The Verifier and the tool callbacks therefore see exactly the same
evidence, with no shared mutable objects and no hidden global state.
"""

from __future__ import annotations

from typing import Any


def _store(state: Any) -> dict[str, dict[str, Any]]:
    """Get (and create) the evidence dict inside a dict-like ADK state.

    ADK's ``State`` is mapping-like but is not a ``MutableMapping`` subclass, so
    we duck-type ``get``/``__setitem__`` rather than checking the ABC.
    """
    if state is None:
        return {}
    try:
        store = state.get("evidence_store")
    except Exception:
        return {}
    if not isinstance(store, dict):
        store = {}
        try:
            state["evidence_store"] = store
        except Exception:
            return {}
    return store


def add_tool_result(state: Any, result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Add every evidence item in a ``ToolResult`` dict to the state store.

    ``tool_ok`` and ``source`` are recorded alongside the evidence so the
    Verifier can tell "the tool failed" from "the tool returned nothing".
    Returns the store itself for convenient chaining in tests.
    """
    store = _store(state)
    if not isinstance(result, dict):
        return store
    source = str(result.get("source") or "")
    tool_ok = bool(result.get("ok", False))
    for item in result.get("evidence") or []:
        if isinstance(item, dict) and item.get("id"):
            store[str(item["id"])] = {
                "evidence": item,
                "tool_ok": tool_ok,
                "source": source,
            }
    return store


def get(state: Any, evidence_id: str) -> dict[str, Any] | None:
    """Look up one stored evidence record (``None`` if it was never returned)."""
    return _store(state).get(evidence_id)


def all_ids(state: Any) -> list[str]:
    """All evidence ids currently in the store."""
    return sorted(_store(state).keys())
