"""Human-in-the-loop approval for proposed actions (M4)."""

from .approval import (
    APPROVAL_TTL_MIN,
    MAX_EVIDENCE_AGE_MIN,
    Proposal,
    ProposalError,
    approve,
    build_proposal,
    execute_approved,
    explain,
    reject,
    render_card_html,
    render_card_text,
    request_change,
)

__all__ = [
    "APPROVAL_TTL_MIN",
    "MAX_EVIDENCE_AGE_MIN",
    "Proposal",
    "ProposalError",
    "approve",
    "build_proposal",
    "execute_approved",
    "explain",
    "reject",
    "render_card_html",
    "render_card_text",
    "request_change",
]
