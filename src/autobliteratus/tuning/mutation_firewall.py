"""Fail-closed authorization checks for in-memory model mutation."""

from __future__ import annotations

from typing import Any

MUTATING_STAGE_IDS = frozenset(
    {
        "tuning.refusal_direction_ablation",
        "tuning.calibrated_quantization",
    }
)


class MutationAuthorizationError(PermissionError):
    """Raised when a mutating stage is not inside an authorized transaction."""


def require_mutation_authorized(stage_id: str, context: dict[str, Any]) -> None:
    """Require explicit opt-in, an armed gate, and an active transaction.

    This helper is intentionally called by each mutating stage rather than only
    by the top-level pipeline so direct stage invocation cannot silently bypass
    the runtime firewall.
    """
    if context.get("allow_mutation") is not True:
        raise MutationAuthorizationError(
            f"{stage_id} is mutation-capable and requires allow_mutation=True."
        )
    gate = context.get("mutation_gate")
    if not isinstance(gate, dict) or gate.get("status") != "armed":
        raise MutationAuthorizationError(
            f"{stage_id} requires an armed mutation gate; no mutation was applied."
        )
    transaction = context.get("mutation_transaction")
    if transaction is None or not bool(getattr(transaction, "active", False)):
        raise MutationAuthorizationError(
            f"{stage_id} requires an active transactional checkpoint; no mutation was applied."
        )
    proposal = context.get("topology_proposal_report")
    if proposal is not None and proposal.get("validated") is not True:
        raise MutationAuthorizationError(
            f"{stage_id} received an invalid topology proposal; no mutation was applied."
        )


def is_mutating_stage(stage_id: str) -> bool:
    """Return whether a runtime stage can change model state."""
    return stage_id in MUTATING_STAGE_IDS
