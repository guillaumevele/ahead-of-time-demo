"""Fast tests: no network, no model weights. Timestral and Mistral are replaced by stubs."""

import json

import numpy as np
import pytest

from aot import agent, approval, backtest, forecast as F, metrics as M, tracing
from aot.store import ROOT, Store


@pytest.fixture()
def store():
    s = Store()
    s.load_snapshot()
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
    assert store.load_snapshot() == 7580  # idempotent
    assert len(y) == 1896 and int(y.isna().sum()) == 1
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
def test_guardrail_rejects_bad_proposals(store, stub_timestral, llm, reason):
    r = agent.propose(store, tracing.Tracer(None), llm=llm, allow_untraced=True)
    assert r["status"] == "invalid" and reason in r["reason"]
    with pytest.raises(approval.ApprovalError):
        approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"]))


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


def test_nothing_executes_without_human_approval(store, stub_timestral):
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    with pytest.raises(approval.ApprovalError):
        approval.execute(store, r["id"])
    with pytest.raises(approval.ApprovalError):
        approval.decide(store, r["id"], " ", approval.confirmation_phrase(r["id"]))
    assert approval.decide(store, r["id"], "alice", "yes") == "rejected"
    with pytest.raises(approval.ApprovalError):
        approval.execute(store, r["id"])


def test_approved_action_is_a_dry_run(store, stub_timestral):
    r = agent.propose(store, tracing.Tracer(None), llm=llm_answering(), allow_untraced=True)
    assert approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"])) == "approved"
    assert approval.execute(store, r["id"]).startswith("DRY RUN")
    assert list(store.events(r["id"]).event) == ["proposed", "approved", "executed_dry_run"]
    with pytest.raises(approval.ApprovalError):  # one decision per proposal
        approval.decide(store, r["id"], "alice", approval.confirmation_phrase(r["id"]))


# --- README stays in sync with the recorded backtest ------------------------------


def test_readme_quotes_the_recorded_backtest():
    recorded = json.loads((ROOT / "results" / "backtest.json").read_text())
    assert backtest.render(recorded) in (ROOT / "README.md").read_text()
