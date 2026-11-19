"""Prescription step: Mistral Large reads the forecast and proposes one action.

The agent can only *propose*. Its output is validated, written to the ClickHouse
action log with status `proposed`, and stops there. See `approval.py` for the gate.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Callable

import numpy as np
import pandas as pd

from . import forecast as F
from .store import Store
from .tracing import Tracer, require

LLM_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-large-4")
HORIZON = 24
DURATION = 4

SYSTEM = (
    "You advise the operator of one flexible electrical load that must run for exactly "
    f"{DURATION} consecutive hours within the next {HORIZON} hours. The goal is to run it when French "
    "national electricity demand is forecast to be lowest. You receive a probabilistic hourly "
    "forecast (q10, median, q90, in MW). Propose one start hour. You cannot execute anything: a "
    "human operator reviews your proposal. Answer with a single JSON object with keys "
    '"start_utc" (ISO 8601, one of the listed hours), "rationale" (two sentences at most, '
    'mention the uncertainty), "confidence" ("low", "medium" or "high").'
)

LLM = Callable[[list[dict]], tuple[str, dict]]


def mistral_llm(messages: list[dict]) -> tuple[str, dict]:
    from mistralai.client import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    r = client.chat.complete(
        model=LLM_MODEL, messages=messages, temperature=0, top_p=1, response_format={"type": "json_object"}
    )
    usage = {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens}
    content = r.choices[0].message.content
    if not isinstance(content, str):  # chunked content: keep the text chunks only
        content = "".join(getattr(c, "text", "") for c in content if getattr(c, "type", "") == "text")
    return content, usage


def forecast_table(index: pd.DatetimeIndex, fc: F.QuantileForecast) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_utc": index.strftime("%Y-%m-%dT%H:%MZ"),
            "q10": np.round(fc.quantile(0.1)).astype(int),
            "median": np.round(fc.median).astype(int),
            "q90": np.round(fc.quantile(0.9)).astype(int),
        }
    )


def reference_start(table: pd.DataFrame) -> int:
    """Deterministic answer: start index of the window with the lowest mean median forecast."""
    means = table["median"].rolling(DURATION).mean().shift(-(DURATION - 1))
    return int(means.idxmin())


def validate(raw: str, table: pd.DataFrame) -> tuple[dict | None, str | None]:
    """Guardrail. Returns (proposal, None) or (None, reason)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"not JSON: {e}"
    if not isinstance(data, dict):
        return None, "not a JSON object"
    try:
        start = pd.Timestamp(data.get("start_utc"))
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    except (ValueError, TypeError):
        return None, "start_utc is not a timestamp"
    hours = list(table.hour_utc)
    key = start.strftime("%Y-%m-%dT%H:%MZ")
    if key not in hours:
        return None, "start_utc is outside the forecast horizon"
    i = hours.index(key)
    if i + DURATION > len(hours):
        return None, "the window does not fit inside the forecast horizon"
    if data.get("confidence") not in ("low", "medium", "high"):
        return None, "confidence must be low, medium or high"
    ref = reference_start(table)
    window_mean = float(table["median"].iloc[i : i + DURATION].mean())
    ref_mean = float(table["median"].iloc[ref : ref + DURATION].mean())
    return {
        "action": "schedule_flexible_load",
        "start_utc": key,
        "end_utc": (start + pd.Timedelta(hours=DURATION)).strftime("%Y-%m-%dT%H:%MZ"),
        "rationale": str(data.get("rationale", ""))[:600],
        "confidence": data["confidence"],
        "window_mean_median_mw": round(window_mean),
        "reference_start_utc": hours[ref],
        "excess_vs_reference_mw": round(window_mean - ref_mean),
    }, None


def propose(store: Store, tracer: Tracer, llm: LLM = mistral_llm, allow_untraced: bool = False) -> dict:
    """One agent run. Returns {"id", "status", ...}. Never executes the action."""
    require(tracer, allow_untraced)
    action_id = uuid.uuid4().hex[:8]
    with tracer.observe("agent", "prescribe-flexible-load", input={"action_id": action_id}) as root:
        with tracer.observe("tool", "clickhouse.hourly_load") as obs:
            y = store.hourly()
            obs.update(output={"points": len(y), "last_hour": str(y.index[-1])})
        with tracer.observe("tool", "timestral.forecast", input={"model": F.MODEL_ID, "horizon": HORIZON}) as obs:
            fc = F.timestral_forecast(y.to_numpy(), HORIZON)
            index = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
            table = forecast_table(index, fc)
            obs.update(output=table.to_dict("records"))
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Forecast:\n" + table.to_csv(index=False)},
        ]
        with tracer.observe("generation", "mistral.chat", model=LLM_MODEL, input=messages) as gen:
            raw, usage = llm(messages)
            gen.update(output=raw, usage_details=usage)
        with tracer.observe("guardrail", "validate-proposal", input=raw) as obs:
            proposal, reason = validate(raw, table)
            obs.update(output=proposal or {"rejected": reason})
        trace_id = tracer.trace_id()
        if proposal is None:
            store.log_event(action_id, "invalid", "guardrail", {"reason": reason, "raw": raw[:600], "trace_id": trace_id})
            result = {"id": action_id, "status": "invalid", "reason": reason}
        else:
            proposal |= {"llm_model": LLM_MODEL, "forecast_model": F.MODEL_ID, "trace_id": trace_id}
            store.log_event(action_id, "proposed", f"agent:{LLM_MODEL}", proposal)
            result = {"id": action_id, "status": "proposed", **proposal}
        root.update(output=result)
    tracer.flush()
    return result | {"traced": tracer.enabled}
