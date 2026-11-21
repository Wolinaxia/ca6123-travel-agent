"""Sanitise untrusted tool output before it reaches the model.

News headlines are the main untrusted channel: a headline is attacker-authored
text, so it may carry a prompt injection. Every ``Evidence`` that is marked
``data["untrusted"] is True`` is treated as hostile in **every** free-text field:
``observation``, ``subject`` and every nested string inside ``data``. A flagged
string is replaced by a fixed marker and the evidence is tagged with
``injection_removed`` so downstream code and the trace can see what happened.
PII in the same strings is redacted as well.

The caller's dict is never mutated: we work on a deep copy and return the copy
plus a list of events. The events are the audit trail for the trace.

Injection detection is rules-first. When layer 2 is enabled (``GUARDRAIL_LLM=1``
or ``use_llm=True``), all unflagged strings of one tool result are batched into
a **single** classifier call; a failed classifier leaves the text unchanged
(server-side blocking still applies, and the result is logged).
"""

from __future__ import annotations

import copy
from typing import Any

from .injection import _llm_enabled, detect_injection_rules
from .pii import redact_pii

REMOVED_MARKER = "[removed: possible prompt injection]"


def _iter_strings(value: Any, path: tuple[str, ...] = ()):
    """Yield ``(path, string)`` for every string inside ``value``."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(item, path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_strings(item, path + (str(index),))


def _set_path(container: Any, path: tuple[str, ...], replacement: str) -> None:
    target = container
    for key in path[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    last = path[-1]
    if isinstance(target, list):
        target[int(last)] = replacement
    else:
        target[last] = replacement


def _flag_llm(texts: list[str]) -> list[bool]:
    """Ask layer 2 once for all unflagged strings of one result."""
    if not texts:
        return []
    from .llm_classifier import classify_injections

    try:
        verdicts = classify_injections(texts)
    except Exception:
        return [False] * len(texts)
    return [
        bool(verdict.get("available") and verdict.get("injection"))
        for verdict in verdicts
    ]


def sanitize_tool_result(
    result: dict[str, Any], use_llm: bool | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(sanitized_result, events)`` for a ``ToolResult``-shaped dict.

    Only evidence with ``data.untrusted`` is injection-checked; every string in
    that evidence is covered. ``data.text`` is PII-redacted for trusted evidence
    too, matching the old behaviour. Events look like
    ``{"evidence_id", "kind", "detail"}`` and are safe to log.
    """
    sanitized = copy.deepcopy(result)
    events: list[dict[str, Any]] = []
    if not isinstance(sanitized, dict):
        return sanitized, events

    for item in sanitized.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        evidence_id = item.get("id")

        if data.get("untrusted") is True:
            # Collect every free-text field: subject, observation and data.
            targets: list[tuple[dict[str, Any], tuple[str, ...]]] = []
            for field in ("subject", "observation"):
                if isinstance(item.get(field), str) and item.get(field):
                    targets.append((item, (field,)))
            for path, text in _iter_strings(data):
                if text:
                    targets.append((data, path))

            flagged: list[tuple[dict[str, Any], tuple[str, ...]]] = []
            unflagged_texts: list[str] = []
            unflagged_targets: list[tuple[dict[str, Any], tuple[str, ...]]] = []
            for container, path in targets:
                text = container[path[0]] if len(path) == 1 else _get_path(container, path)
                verdict = detect_injection_rules(text)
                if verdict["flagged"]:
                    flagged.append((container, path))
                else:
                    unflagged_texts.append(text)
                    unflagged_targets.append((container, path))

            if _llm_enabled(use_llm) and unflagged_texts:
                for (container, path), is_injection in zip(
                    unflagged_targets, _flag_llm(unflagged_texts)
                ):
                    if is_injection:
                        flagged.append((container, path))

            for container, path in flagged:
                _set_path(container, path, REMOVED_MARKER)
            if flagged:
                data["injection_removed"] = True
                events.append(
                    {
                        "evidence_id": evidence_id,
                        "kind": "injection",
                        "detail": ["removed_flagged_text"],
                    }
                )

            for container, path in targets:
                text = container[path[0]] if len(path) == 1 else _get_path(container, path)
                if not isinstance(text, str) or not text:
                    continue
                redacted, findings = redact_pii(text)
                if findings:
                    _set_path(container, path, redacted)
                    events.append(
                        {
                            "evidence_id": evidence_id,
                            "kind": "pii",
                            "detail": [finding["type"] for finding in findings],
                        }
                    )
        else:
            text = data.get("text")
            if isinstance(text, str) and text:
                redacted, findings = redact_pii(text)
                if findings:
                    data["text"] = redacted
                    events.append(
                        {
                            "evidence_id": evidence_id,
                            "kind": "pii",
                            "detail": [finding["type"] for finding in findings],
                        }
                    )

    return sanitized, events


def _get_path(container: Any, path: tuple[str, ...]) -> Any:
    target = container
    for key in path:
        target = target[int(key)] if isinstance(target, list) else target[key]
    return target
