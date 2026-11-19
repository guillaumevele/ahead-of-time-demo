"""Rolling-origin backtest of Timestral against seasonal-naive baselines.

    uv run python -m aot.backtest

Protocol (fixed before the first run, see README):
- series: hourly mean of French national electricity load (RTE eCO2mix), read from ClickHouse;
- one forecast origin per day at 00:00 UTC, horizon 24 hours;
- every scored target hour is dated after the Timestral weights were published
  (first origin: 2026-10-16 00:00 UTC; weights committed 2026-10-15 15:22 UTC);
- context: all history available before the origin, no covariates, model defaults;
- baselines: seasonal naive with a weekly (168 h) and a daily (24 h) season.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from . import forecast as F
from . import metrics as M
from .store import ROOT, SNAPSHOT, Store

FIRST_ORIGIN = pd.Timestamp("2026-10-16 00:00", tz="UTC")
HORIZON = 24
WEEK, DAY = 168, 24


def run() -> dict:
    store = Store()
    n_rows = store.load_snapshot()
    y = store.hourly()
    store.close()

    origins = pd.date_range(FIRST_ORIGIN, y.index[-1] - pd.Timedelta(hours=HORIZON - 1), freq="D")
    names = ("timestral", "naive_week", "naive_day")
    abs_err = {n: [] for n in names}
    scaled_err = {n: [] for n in names}
    per_window_mae = {n: [] for n in names}
    targets, q_all, levels = [], [], None
    naive_pts = {"naive_week": [], "naive_day": []}

    t0 = time.time()
    for origin in origins:
        ctx = y[y.index < origin].to_numpy()
        tgt = y[(y.index >= origin) & (y.index < origin + pd.Timedelta(hours=HORIZON))].to_numpy()
        fc = F.timestral_forecast(ctx, HORIZON)
        levels = fc.levels
        preds = {
            "timestral": fc.median,
            "naive_week": F.seasonal_naive(ctx, HORIZON, WEEK),
            "naive_day": F.seasonal_naive(ctx, HORIZON, DAY),
        }
        scale = M.seasonal_naive_scale(ctx, WEEK)
        for n in names:
            e = np.abs(tgt - preds[n])
            abs_err[n].append(e)
            scaled_err[n].append(e / scale)
            per_window_mae[n].append(float(np.nanmean(e)))
        targets.append(tgt)
        q_all.append(fc.quantiles)
        for n in naive_pts:
            naive_pts[n].append(preds[n])
    elapsed = time.time() - t0

    tgt = np.concatenate(targets)
    q = np.concatenate(q_all)
    scored = int((~np.isnan(tgt)).sum())
    res = {}
    for n in names:
        quant = q if n == "timestral" else np.repeat(np.concatenate(naive_pts[n])[:, None], len(levels), axis=1)
        res[n] = {
            "mae_mw": float(np.nanmean(np.concatenate(abs_err[n]))),
            "mase": float(np.nanmean(np.concatenate(scaled_err[n]))),
            "wql": M.wql(tgt, quant, levels),
        }
    i10, i90 = int(np.argmin(np.abs(levels - 0.1))), int(np.argmin(np.abs(levels - 0.9)))
    wins = {
        b: int(sum(t < v for t, v in zip(per_window_mae["timestral"], per_window_mae[b])))
        for b in ("naive_week", "naive_day")
    }
    return {
        "data_sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        "raw_rows": n_rows,
        "hourly_points": int(len(y)),
        "missing_hours": int(y.isna().sum()),
        "model": F.MODEL_ID,
        "model_revision": F.MODEL_REVISION,
        "versions": {p: version(p) for p in ("timestral", "torch", "chdb")},
        "first_origin": str(origins[0]),
        "last_origin": str(origins[-1]),
        "windows": int(len(origins)),
        "scored_hours": scored,
        "results": res,
        "coverage_q10_q90": M.coverage(tgt, q[:, i10], q[:, i90]),
        "windows_won_by_timestral": wins,
        "timestral_seconds_total": round(elapsed, 1),
    }


def render(r: dict) -> str:
    lines = [
        f"data      : {SNAPSHOT.relative_to(ROOT)}  sha256={r['data_sha256'][:16]}...",
        f"            {r['raw_rows']} quarter-hour rows -> {r['hourly_points']} hourly points "
        f"({r['missing_hours']} missing, kept as NaN)",
        f"model     : {r['model']} @ {r['model_revision'][:12]}  "
        f"(timestral {r['versions']['timestral']}, torch {r['versions']['torch']}, chdb {r['versions']['chdb']}, CPU)",
        f"protocol  : {r['windows']} daily origins, {r['first_origin'][:16]} -> {r['last_origin'][:16]} UTC, "
        f"horizon 24 h, {r['scored_hours']} scored hours",
        "",
        f"{'forecaster':<26}{'MAE (MW)':>10}{'MASE':>8}{'WQL':>9}",
    ]
    labels = {
        "timestral": "Timestral-123M (median)",
        "naive_week": "seasonal naive, 168 h",
        "naive_day": "seasonal naive, 24 h",
    }
    for n, lab in labels.items():
        m = r["results"][n]
        lines.append(f"{lab:<26}{m['mae_mw']:>10.0f}{m['mase']:>8.3f}{m['wql']:>9.4f}")
    w = r["windows_won_by_timestral"]
    lines += [
        "",
        f"Timestral q10-q90 interval covers {r['coverage_q10_q90']:.1%} of scored hours (nominal 80%)",
        f"Timestral has the lower daily MAE in {w['naive_week']}/{r['windows']} windows vs weekly naive, "
        f"{w['naive_day']}/{r['windows']} vs daily naive",
        f"Timestral inference time: {r['timestral_seconds_total']} s for {r['windows']} forecasts",
    ]
    return "\n".join(lines)


def main() -> None:
    r = run()
    print(render(r))
    if "--json" in sys.argv:
        out = ROOT / "results" / "backtest.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
