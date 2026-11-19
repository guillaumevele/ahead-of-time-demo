"""Deterministic gate and approval receipt, delegated to agent-proof-kit.

The action log is exported as an agent-proof "agent run" document. Two uses:

- `gate`: before `execute`, the run is rendered as it would look *after* execution and
  checked with `agent-proof verify`. Under `policies/aot-policy.json` a `write` action that
  completes without `approval: granted` is a critical finding, so the check fails.
  Approval is read from the raw events and counts only if a `human:` actor recorded it.
- `write_receipt`: after a human approves, the run is bundled and attested
  (`agent-proof bundle` + `sign-bundle`). `verify_receipt` checks it back.

If the CLI is missing, both fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .store import ROOT, Store

CLI = ROOT / "node_modules" / ".bin" / "agent-proof"
POLICY = ROOT / "policies" / "aot-policy.json"
RECEIPTS = ROOT / "receipts"


class GateError(RuntimeError):
    pass


def human_approved(store: Store, action_id: str) -> bool:
    ev = store.events(action_id)
    return bool(((ev.event == "approved") & ev.actor.str.startswith("human:")).any())


def agent_run_document(store: Store, action_id: str, outcome: str | None = None) -> dict:
    """Export one action as an agent-proof run. `outcome` overrides the recorded outcome
    of the write action (used by the gate to test the state execution would create)."""
    p = store.proposal(action_id)
    if p is None:
        raise GateError(f"no proposal with id {action_id}")
    ev = store.events(action_id)
    executed = bool((ev.event == "executed_dry_run").any())
    approved = human_approved(store, action_id)
    approver = ev[ev.event == "approved"].actor.iloc[-1] if approved else None
    return {
        "schemaVersion": 1,
        "runId": f"aot-{action_id}",
        "subject": "Flexible load scheduling proposal from a 24 h demand forecast",
        "synthetic": False,
        "generatedAt": ev.ts.iloc[0].strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "agent": {"name": "aot prescribe-flexible-load", "provider": "mistral", "model": p.get("llm_model", "unknown")},
        "objectives": ["Propose when to run one flexible 4-hour load. Never execute it."],
        "actions": [
            {"id": "a1", "type": "read", "target": "clickhouse: aot.raw (fr_load)",
             "approval": "not_required", "outcome": "completed"},
            {"id": "a2", "type": "read", "target": f"forecast: {p.get('forecast_model', 'unknown')}",
             "approval": "not_required", "outcome": "completed"},
            {"id": "a3", "type": "write",
             "target": f"schedule flexible load {p['start_utc']} -> {p['end_utc']}",
             "approval": "granted" if approved else "missing",
             "outcome": outcome or ("completed" if executed else "blocked"),
             "note": f"approved by {approver}" if approved else "no human approval on record"},
        ],
        "outputs": [
            {"id": "o1", "channel": "final", "content": p["rationale"],
             "claims": [{"text": "The proposed window was compared with the lowest-demand window of the forecast.",
                         "evidence": "reference-window-check"}]},
        ],
        "evidence": [
            {"id": "reference-window-check", "kind": "command", "command": "aot.agent.validate", "result": "pass",
             "excerpt": f"reference_start_utc={p['reference_start_utc']} "
                        f"excess_vs_reference_mw={p['excess_vs_reference_mw']}"},
        ],
    }


def _cli(*args: str) -> subprocess.CompletedProcess:
    if not CLI.exists():
        raise GateError("agent-proof CLI not installed (run `npm ci`); refusing by default")
    return subprocess.run([str(CLI), *args], capture_output=True, text=True, timeout=120)


def verify_document(doc: dict, workdir: Path) -> tuple[bool, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "agent-run.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    r = _cli("verify", "--input", str(path), "--policy", str(POLICY), "--format", "text")
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def gate(store: Store, action_id: str, workdir: Path | None = None) -> None:
    """Raise GateError unless the post-execution state passes the policy."""
    doc = agent_run_document(store, action_id, outcome="completed")
    ok, report = verify_document(doc, workdir or RECEIPTS / action_id / "gate")
    if not ok:
        raise GateError(f"agent-proof gate refused action {action_id}:\n{report}")


def write_receipt(store: Store, action_id: str, out_dir: Path | None = None) -> Path:
    """Bundle and attest the approved run. Returns the attestation path.

    Digest-only by default (tamper evidence). If AOT_SIGNING_KEY points to an RSA private
    key in PEM format, the attestation also carries an RSA-SHA256 signature.
    """
    out = out_dir or RECEIPTS / action_id
    ok, report = verify_document(agent_run_document(store, action_id), out)
    if not ok:
        raise GateError(f"run does not pass the policy, no receipt written:\n{report}")
    bundle, attestation = out / "proof-bundle.json", out / "attestation.json"
    steps = [
        ["bundle", "--input", str(out / "agent-run.json"), "--policy", str(POLICY),
         "--scan-path", str(out), "--out", str(bundle)],
        ["sign-bundle", "--bundle", str(bundle), "--out", str(attestation)]
        + (["--private-key", os.environ["AOT_SIGNING_KEY"]] if os.environ.get("AOT_SIGNING_KEY") else []),
    ]
    for step in steps:
        r = _cli(*step)
        if r.returncode != 0:
            raise GateError(f"agent-proof {step[0]} failed:\n{r.stdout}{r.stderr}")
    # The bundle holds agent-proof's evaluation, not the run itself: bind the two.
    (out / "receipt.json").write_text(json.dumps({
        "action_id": action_id,
        "agent_run_sha256": _sha256(out / "agent-run.json"),
        "bundle_digest": json.loads(attestation.read_text())["digest"]["value"],
    }, indent=2) + "\n")
    return attestation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_receipt(out_dir: Path, public_key: str | None = None) -> tuple[bool, str]:
    """Three checks: the attestation matches the bundle (agent-proof), the run document
    is the one the receipt was written for, and that document still passes the policy."""
    args = ["verify-bundle-signature", "--bundle", str(out_dir / "proof-bundle.json"),
            "--signature", str(out_dir / "attestation.json")]
    r = _cli(*args, *(["--public-key", public_key] if public_key else []))
    report = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        return False, report
    receipt = json.loads((out_dir / "receipt.json").read_text())
    if receipt["bundle_digest"] != json.loads((out_dir / "attestation.json").read_text())["digest"]["value"]:
        return False, "receipt does not match the attestation"
    if receipt["agent_run_sha256"] != _sha256(out_dir / "agent-run.json"):
        return False, "agent-run.json was modified after the receipt was written"
    v = _cli("verify", "--input", str(out_dir / "agent-run.json"), "--policy", str(POLICY), "--format", "text")
    return v.returncode == 0, f"{report} | {(v.stdout + v.stderr).strip()}"
