"""Shared data contract between tools, agents, and the Verifier.

See docs/INTERFACES.md for the reasoning behind each field.
Plain dataclasses on purpose: no extra dependency, easy to turn into dicts for ADK.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

Risk = Literal["low", "moderate", "high"]
Confidence = Literal["low", "medium", "high"]
VerdictStatus = Literal["supported", "unsupported", "contradicted"]


@dataclass
class Evidence:
    """One normalized observation produced by a tool."""

    id: str
    source: str
    subject: str
    observation: str
    fetched_at: str
    data: dict[str, Any] = field(default_factory=dict)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ToolResult:
    """What every tool returns. On failure: ok=False, empty evidence, error set."""

    source: str
    ok: bool
    evidence: list[Evidence] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    """A statement made by an agent. Must cite the evidence it is based on."""

    text: str
    evidence_ids: list[str]
    confidence: Confidence
    risk: Optional[Risk] = None


@dataclass
class AgentResult:
    agent: str
    claims: list[Claim] = field(default_factory=list)
    needs_human: bool = False


@dataclass
class Verdict:
    claim: Claim
    status: VerdictStatus
    reason: str
