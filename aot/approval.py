"""Human approval gate.

A proposed action can only move forward when a named operator types the exact
confirmation phrase for that action id. `execute` refuses anything that is not
`approved`, and even then it is a dry run: this repository contains no code that
acts on an external system.

This is a process guard, not a security boundary: anyone with write access to the
database can append events.
"""

from __future__ import annotations

from .store import Store


class ApprovalError(PermissionError):
    pass


def confirmation_phrase(action_id: str) -> str:
    return f"approve {action_id}"


def decide(store: Store, action_id: str, operator: str, typed: str) -> str:
    """Record the operator's decision. Returns 'approved' or 'rejected'."""
    if not operator.strip():
        raise ApprovalError("an operator name is required")
    status = store.status(action_id)
    if status != "proposed":
        raise ApprovalError(f"action {action_id} is {status or 'unknown'}, not awaiting a decision")
    event = "approved" if typed.strip() == confirmation_phrase(action_id) else "rejected"
    store.log_event(action_id, event, f"human:{operator.strip()}", {"typed": typed.strip()[:80]})
    return event


def execute(store: Store, action_id: str) -> str:
    status = store.status(action_id)
    if status != "approved":
        raise ApprovalError(f"action {action_id} is {status or 'unknown'}: refused, human approval is required")
    p = store.proposal(action_id)
    message = f"DRY RUN - would schedule the flexible load from {p['start_utc']} to {p['end_utc']}. Nothing was sent."
    store.log_event(action_id, "executed_dry_run", "system", {"message": message})
    return message
