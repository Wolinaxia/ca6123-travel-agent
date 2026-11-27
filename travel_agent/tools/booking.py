"""Mock booking tool (M4). Nothing real is ever booked or paid.

Safety layers, from outside in:

1. The name starts with ``book``, so ``guardrails.callbacks.tool_gate`` blocks
   every call unless ``state["approved_actions"]`` holds a single-use approval
   for exactly this tool and these arguments (M1's deny-by-default gate).
2. The tool itself refuses an ``option_id`` that the planner never produced in
   this session (it must exist in ``state["evidence_store"]``), and refuses an
   option the planner judged ``likely_miss``.
3. Every result says ``mock=True`` and the confirmation code starts with
   ``MOCK-`` so it can never be mistaken for a real ticket.

The Alternatives agent does **not** get this tool: it can only propose. The
HITL step (``travel_agent.hitl.approval.execute_approved``) runs it after a
human approves, through the same ``tool_gate``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from travel_agent.schemas import Evidence, ToolResult

SOURCE = "mock_booking"


def _state(tool_context: Any) -> Any:
    return getattr(tool_context, "state", None)


def book_trip_option(option_id: str, tool_context: Any = None) -> dict:
    """MOCK-book one planned option (no real ticket, no payment).

    `option_id` must be a `plan:<mode>:<route>` id returned by
    `plan_trip_options` in this session. This tool is blocked by a guardrail
    until a human has approved exactly this call.
    """
    try:
        state = _state(tool_context)
        store = state.get("evidence_store") if state is not None else None
        record = (store or {}).get(option_id) if isinstance(store, dict) else None
        if not record:
            raise ValueError(f"unknown option '{option_id}': only options planned in this session can be booked")
        evidence = record.get("evidence") or {}
        data = evidence.get("data") or {}
        if not str(option_id).startswith("plan:") or not data.get("mode"):
            raise ValueError(f"'{option_id}' is not a bookable planned option")
        if data.get("verdict") == "likely_miss":
            raise ValueError("refusing to book an option that is expected to miss the deadline")
        booked_at = datetime.now(timezone.utc).isoformat()
        code = "MOCK-" + hashlib.sha1(f"{option_id}|{booked_at}".encode("utf-8")).hexdigest()[:6].upper()
        route = str(data.get("route") or evidence.get("subject") or "")
        result = Evidence(
            id=f"booking:{code}",
            source=SOURCE,
            subject=route,
            observation=f"MOCK booking {code} for {data.get('mode')} {route} (demo only: no real ticket, no payment)",
            data={
                "mock": True,
                "confirmation": code,
                "option_id": option_id,
                "mode": data.get("mode"),
                "route": route,
                "latest_departure_local": data.get("latest_departure_local"),
                "booked_at": booked_at,
            },
            fetched_at=booked_at,
        )
        return ToolResult(source=SOURCE, ok=True, evidence=[result]).to_dict()
    except Exception as exc:
        return ToolResult(source=SOURCE, ok=False, evidence=[], error=str(exc)).to_dict()


BOOKING_TOOLS = [book_trip_option]
