"""Fast tests: no network, no model weights. Timestral and Mistral are replaced by stubs."""

import json

import numpy as np
import pytest

from aot import agent, approval, backtest, forecast as F, metrics as M, proof, tracing
from aot.store import ROOT, Store


@pytest.fixture()
def store():
    s = Store()
    s.load_series()
    yield s
    s.close()


@pytest.fixture()
def stub_timestral(monkeypatch):
    """Deterministic stand-in: demand is lowest at hours 2..5 of the horizon."""

    def fake(context, horizon):
        levels = np.array([0.1, 0.5, 0.9])
        median = 50_000 + 5_000 * np.abs(np.arange(horizon) - 3.5)
        return F.QuantileForecast(levels, np.stack([median - 1000, median, median + 1000], axis=1))

    monkeypatch.setattr(F, "timestral_forecast", fake)


def llm_answering(start_offset_hours=2, **extra):
    def llm(messages):
        first = messages[1]["content"].splitlines()[2].split(",")[0]  # first forecast hour
        import pandas as pd

        start = (pd.Timestamp(first) + pd.Timedelta(hours=start_offset_hours)).strftime("%Y-%m-%dT%H:%MZ")
        return json.dumps({"start_utc": start, "rationale": "lowest block", "confidence": "medium"} | extra), {
            "input": 10,
            "output": 5,
        }

    return llm


# --- ClickHouse -------------------------------------------------------------


def test_snapshot_loads_and_gaps_stay_nan(store):
    y = store.hourly()
    assert store.load_series() == 7580  # idempotent
    assert len(y) == 1896 and int(y.isna().sum()) == 1
    assert store.load_series("paris_bike") == 1822
    b = store.hourly("paris_bike")
    assert len(b) == 1896 and int(b.isna().sum()) == 74
    assert (np.diff(y.index.asi8) == np.diff(y.index.asi8)[0]).all()  # gap-free hourly grid


# --- forecasting and metrics ---------------------------------------------------


def test_seasonal_naive_repeats_last_season_and_skips_nan():
    ctx = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0])
    assert F.seasonal_naive(ctx, 4, season=3).tolist() == [4.0, 2.0, 6.0, 4.0]


def test_metrics():
    y = np.array([10.0, 20.0, np.nan, 40.0])
    yhat = np.array([12.0, 18.0, 30.0, 44.0])
    assert M.mae(y, yhat) == pytest.approx(8 / 3)
    levels = np.array([0.1, 0.5, 0.9])
    point = np.repeat(yhat[:, None], 3, axis=1)
    assert M.wql(y, point, levels) == pytest.approx(8 / 70)  # sum|err| / sum|y|
    assert M.seasonal_naive_scale(np.array([1.0, 2.0, 4.0, 2.0]), 2) == pytest.approx(1.5)
    assert M.coverage(y, yhat - 3, yhat + 3) == pytest.approx(2 / 3)


# --- agent -----------------------------------------------------------------------


def test_untraced_run_is_refused(store, stub_timestral):
    with pytest.raises(tracing.UntracedRunError):
        agent.propose(store, tracing.Tracer(None), llm=llm_answering())
    assert store.events().empty


def test_agent_only_proposes(store, stub_timestral):
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(2), allow_untraced=True)
    assert r["status"] == "proposed" and r["excess_vs_reference_mw"] == 0
    assert store.status(r["id"]) == "proposed"
    assert list(store.events().event) == ["proposed"]


@pytest.mark.parametrize(
    "llm, reason",
    [
        (lambda m: ("not json", {}), "not JSON"),
        (llm_answering(22), "does not fit"),
        (llm_answering(48), "outside the forecast horizon"),
        (llm_answering(2, confidence="certain"), "confidence"),
    ],
)
def test_guardrail_rejects_bad_proposals(store, stub_timestral, tmp_path, llm, reason):
    r = agent.propose(store, tracing.Tracer(None), llm=llm, allow_untraced=True)
    assert r["status"] == "invalid" and reason in r["reason"]
    with pytest.raises(approval.ApprovalError):
        approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"]), tmp_path)


def test_agent_run_emits_langfuse_observations(store, stub_timestral):
    """Real Langfuse SDK, spans captured in memory instead of being sent to a server."""
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="test-public", secret_key="test-secret", base_url="http://127.0.0.1:9", span_exporter=exporter
    )
    r = agent.propose(store, tracing.Tracer(client), llm=llm_answering())
    names = {s.name for s in exporter.get_finished_spans()}
    assert {
        "prescribe-flexible-load",
        "clickhouse.hourly_load",
        "timestral.forecast",
        "mistral.chat",
        "validate-proposal",
    } <= names
    assert r["traced"] and r["trace_id"]


# --- approval gate ---------------------------------------------------------------

needs_cli = pytest.mark.skipif(not proof.CLI.exists(), reason="agent-proof CLI missing, run `npm ci`")


def test_nothing_executes_without_human_approval(store, stub_timestral, tmp_path):
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    with pytest.raises(approval.ApprovalError):
        approval.execute(store, r["id"], tmp_path)
    with pytest.raises(approval.ApprovalError):
        approval.decide(store, r["id"], " ", approval.confirmation_phrase(r["id"]), tmp_path)
    assert approval.decide(store, r["id"], "alice", "yes", tmp_path) == "rejected"
    with pytest.raises(approval.ApprovalError):
        approval.execute(store, r["id"], tmp_path)
    assert "executed_dry_run" not in set(store.events().event)


@needs_cli
def test_approved_action_is_a_dry_run_with_a_verifiable_receipt(store, stub_timestral, tmp_path):
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    assert approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"]), tmp_path) == "approved"
    ok, report = proof.verify_receipt(tmp_path / r["id"])
    assert ok and "digest=match" in report
    assert approval.execute(store, r["id"], tmp_path).startswith("DRY RUN")
    assert list(store.events(r["id"]).event) == ["proposed", "approved", "executed_dry_run"]
    with pytest.raises(approval.ApprovalError, match="executed_dry_run"):  # one execution per approval
        approval.execute(store, r["id"], tmp_path)
    with pytest.raises(approval.ApprovalError):  # one decision per proposal
        approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"]), tmp_path)


@needs_cli
@pytest.mark.parametrize(
    "filename, old, new",
    [
        ("agent-run.json", "human:alice", "human:mallory"),  # who approved
        ("proof-bundle.json", '"score": 100', '"score": 99'),  # what the gate concluded
    ],
)
def test_tampered_receipt_is_detected(store, stub_timestral, tmp_path, filename, old, new):
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"]), tmp_path)
    assert proof.verify_receipt(tmp_path / r["id"])[0]
    target = tmp_path / r["id"] / filename
    assert old in target.read_text()
    target.write_text(target.read_text().replace(old, new))
    assert not proof.verify_receipt(tmp_path / r["id"])[0]


@needs_cli
def test_receipt_can_carry_an_rsa_signature(store, stub_timestral, tmp_path, monkeypatch):
    import subprocess

    key, pub = tmp_path / "key.pem", tmp_path / "pub.pem"
    subprocess.run(["openssl", "genrsa", "-out", str(key), "2048"], check=True, capture_output=True)
    subprocess.run(["openssl", "rsa", "-in", str(key), "-pubout", "-out", str(pub)], check=True, capture_output=True)
    monkeypatch.setenv("AOT_SIGNING_KEY", str(key))
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"]), tmp_path / "receipts")
    ok, report = proof.verify_receipt(tmp_path / "receipts" / r["id"], public_key=str(pub))
    assert ok and "signature=valid" in report


@needs_cli
def test_agent_proof_gate_fails_an_unapproved_execution_and_passes_an_approved_one(store, stub_timestral, tmp_path):
    """Red then green on the external gate itself, same document, only the approval differs."""
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    executed = lambda: proof.agent_run_document(store, r["id"], outcome="completed")  # noqa: E731
    ok, report = proof.verify_document(executed(), tmp_path / "red")
    assert not ok and "FAIL" in report
    approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"]), tmp_path)
    ok, report = proof.verify_document(executed(), tmp_path / "green")
    assert ok and "PASS" in report


@needs_cli
def test_gate_refuses_an_approval_not_recorded_by_a_human(store, stub_timestral, tmp_path):
    """The status check alone would let this through: the log says `approved`,
    but the event was appended by the agent. The agent-proof gate refuses it."""
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    store.log_event(r["id"], "approved", "agent:self", {})
    assert store.status(r["id"]) == "approved"
    with pytest.raises(approval.ApprovalError, match="agent-proof gate refused"):
        approval.execute(store, r["id"], tmp_path)
    assert "executed_dry_run" not in set(store.events().event)


def test_gate_fails_closed_without_the_cli(store, stub_timestral, tmp_path, monkeypatch):
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    store.log_event(r["id"], "approved", "human:alice", {})
    monkeypatch.setattr(proof, "CLI", tmp_path / "missing")
    with pytest.raises(approval.ApprovalError, match="not installed"):
        approval.execute(store, r["id"], tmp_path)


# --- backtest helpers ----------------------------------------------------------------


def test_relative_mae_interval_and_verdict():
    model, base = np.full(34, 50.0), np.full(34, 100.0)
    rel = backtest.relative_mae_ci(model, base)
    assert rel["point"] == pytest.approx(0.5) and rel["ci95"] == pytest.approx([0.5, 0.5])
    assert backtest.verdict({"a": rel, "b": rel}) == "Timestral better"
    worse = {"point": 1.2, "ci95": [1.05, 1.4]}
    assert backtest.verdict({"a": rel, "b": worse}) == "Timestral worse"
    assert backtest.verdict({"a": rel, "b": {"point": 0.9, "ci95": [0.8, 1.1]}}) == "inconclusive"


def test_check_flags_drift_beyond_tolerance():
    recorded = json.loads(backtest.RESULTS.read_text())
    assert backtest.check(recorded, recorded) == []
    drifted = json.loads(json.dumps(recorded))
    drifted["series"]["fr_load"]["results"]["timestral"]["mae"] *= 1.05
    assert len(backtest.check(drifted, recorded)) == 1


# --- README stays in sync with the recorded backtest ------------------------------


def test_readme_quotes_the_recorded_backtest():
    recorded = json.loads((ROOT / "results" / "backtest.json").read_text())
    assert backtest.render(recorded) in (ROOT / "README.md").read_text()
