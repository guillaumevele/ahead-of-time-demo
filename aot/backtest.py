"""Rolling-origin backtest of Timestral against seasonal-naive baselines.

    uv run python -m aot.backtest            # run and print
    uv run python -m aot.backtest --json     # also rewrite results/backtest.json
    uv run python -m aot.backtest --check    # run and compare with results/backtest.json

The protocol is frozen in PREREG.md, committed before this analysis was written.
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
from .store import ROOT, SERIES, Store

FIRST_ORIGIN = pd.Timestamp("2026-10-16 00:00", tz="UTC")
LAST_ORIGIN = pd.Timestamp("2026-11-18 00:00", tz="UTC")
HORIZON = 24
WEEK, DAY = 168, 24
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261119
RESULTS = ROOT / "results" / "backtest.json"
CHECK_TOLERANCE = 0.01  # relative; float kernels differ slightly across CPUs

LABELS = {
    "timestral": "Timestral-123M (median)",
    "naive_week": "seasonal naive, 168 h",
    "naive_day": "seasonal naive, 24 h",
}
TITLES = {
    "fr_load": "Series A - French national electricity load (MW), replication",
    "paris_bike": "Series B - Paris bike counter, Sebastopol S-N (bikes/h), confirmatory",
}


def relative_mae_ci(model_sums: np.ndarray, base_sums: np.ndarray) -> dict:
    """Relative MAE with a paired bootstrap over windows (percentile 95% interval)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(model_sums)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    ratios = model_sums[idx].sum(axis=1) / base_sums[idx].sum(axis=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {"point": float(model_sums.sum() / base_sums.sum()), "ci95": [float(lo), float(hi)]}


def verdict(rel: dict) -> str:
    if all(r["ci95"][1] < 1 for r in rel.values()):
        return "Timestral better"
    if any(r["ci95"][0] > 1 for r in rel.values()):
        return "Timestral worse"
    return "inconclusive"


def run_series(y: pd.Series) -> dict:
    origins = pd.date_range(FIRST_ORIGIN, LAST_ORIGIN, freq="D")
    names = tuple(LABELS)
    abs_err = {n: [] for n in names}
    scaled_err = {n: [] for n in names}
    targets, q_all, levels = [], [], None
    points = {"naive_week": [], "naive_day": []}

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
        targets.append(tgt)
        q_all.append(fc.quantiles)
        for n in points:
            points[n].append(preds[n])
    elapsed = time.time() - t0

    tgt = np.concatenate(targets)
    q = np.concatenate(q_all)
    window_sums = {n: np.array([np.nansum(e) for e in abs_err[n]]) for n in names}
    window_mae = {n: np.array([np.nanmean(e) for e in abs_err[n]]) for n in names}
    res = {}
    for n in names:
        quant = q if n == "timestral" else np.repeat(np.concatenate(points[n])[:, None], len(levels), axis=1)
        q1, q2, q3 = np.percentile(window_mae[n], [25, 50, 75])
        res[n] = {
            "mae": float(np.nanmean(np.concatenate(abs_err[n]))),
            "mase": float(np.nanmean(np.concatenate(scaled_err[n]))),
            "wql": M.wql(tgt, quant, levels),
            "window_mae_quartiles": [float(q1), float(q2), float(q3)],
        }
    rel = {b: relative_mae_ci(window_sums["timestral"], window_sums[b]) for b in ("naive_week", "naive_day")}
    i10, i90 = int(np.argmin(np.abs(levels - 0.1))), int(np.argmin(np.abs(levels - 0.9)))
    return {
        "hourly_points": int(len(y)),
        "missing_hours": int(y.isna().sum()),
        "windows": int(len(origins)),
        "scored_hours": int((~np.isnan(tgt)).sum()),
        "results": res,
        "relative_mae": rel,
        "verdict": verdict(rel),
        "coverage_q10_q90": M.coverage(tgt, q[:, i10], q[:, i90]),
        "windows_won_by_timestral": {
            b: int((window_mae["timestral"] < window_mae[b]).sum()) for b in ("naive_week", "naive_day")
        },
        "timestral_seconds": round(elapsed, 1),
    }


def run() -> dict:
    store = Store()
    out = {
        "model": F.MODEL_ID,
        "model_revision": F.MODEL_REVISION,
        "versions": {p: version(p) for p in ("timestral", "torch", "chdb")},
        "first_origin": str(FIRST_ORIGIN),
        "last_origin": str(LAST_ORIGIN),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "series": {},
    }
    for name, (path, *_rest) in SERIES.items():
        rows = store.load_series(name)
        r = run_series(store.hourly(name))
        out["series"][name] = {
            "file": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_rows": rows,
            **r,
        }
    store.close()
    return out


def render(r: dict) -> str:
    v = r["versions"]
    lines = [
        f"model     : {r['model']} @ {r['model_revision'][:12]}  "
        f"(timestral {v['timestral']}, torch {v['torch']}, chdb {v['chdb']}, CPU)",
        f"protocol  : PREREG.md - daily origins {r['first_origin'][:10]} -> {r['last_origin'][:10]} at 00:00 UTC, horizon 24 h",
        f"bootstrap : paired over windows, {r['bootstrap']['resamples']} resamples, seed {r['bootstrap']['seed']}",
    ]
    for name, s in r["series"].items():
        lines += [
            "",
            TITLES[name],
            f"  data: {s['file']}  sha256={s['sha256'][:16]}...  {s['raw_rows']} rows -> "
            f"{s['hourly_points']} hourly points ({s['missing_hours']} missing, kept as NaN)",
            f"  {s['windows']} windows, {s['scored_hours']} scored hours",
            "",
            f"  {'forecaster':<26}{'MAE':>9}{'MASE':>8}{'WQL':>9}   per-window MAE q1 / median / q3",
        ]
        for n, lab in LABELS.items():
            m = s["results"][n]
            a, b, c = m["window_mae_quartiles"]
            lines.append(f"  {lab:<26}{m['mae']:>9.1f}{m['mase']:>8.3f}{m['wql']:>9.4f}   {a:.1f} / {b:.1f} / {c:.1f}")
        lines.append("")
        for b in ("naive_week", "naive_day"):
            x = s["relative_mae"][b]
            lines.append(
                f"  relative MAE vs {LABELS[b]:<22}: {x['point']:.3f}  95% CI [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
                f"  (lower MAE in {s['windows_won_by_timestral'][b]}/{s['windows']} windows)"
            )
        lines += [
            f"  preregistered verdict: {s['verdict']}",
            f"  q10-q90 interval covers {s['coverage_q10_q90']:.1%} of scored hours (nominal 80%)",
        ]
    return "\n".join(lines)


def _numbers(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("timestral_seconds", "versions"):
                yield from _numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}[{i}]")
    else:
        yield path, obj


def check(new: dict, recorded: dict) -> list[str]:
    """Differences between a rerun and the recorded results, beyond CHECK_TOLERANCE."""
    old = dict(_numbers(recorded))
    diffs = []
    for path, value in _numbers(new):
        ref = old.get(path)
        if isinstance(value, float) and isinstance(ref, (int, float)):
            if abs(value - ref) > CHECK_TOLERANCE * max(abs(ref), 1e-9):
                diffs.append(f"{path}: {value} != recorded {ref}")
        elif value != ref:
            diffs.append(f"{path}: {value!r} != recorded {ref!r}")
    return diffs


def main() -> int:
    r = run()
    print(render(r))
    seconds = sum(s["timestral_seconds"] for s in r["series"].values())
    print(f"\nTimestral inference time: {seconds:.1f} s")
    if "--json" in sys.argv:
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(r, indent=2) + "\n")
        print(f"wrote {RESULTS.relative_to(ROOT)}")
    if "--check" in sys.argv:
        diffs = check(r, json.loads(RESULTS.read_text()))
        print(f"check against {RESULTS.relative_to(ROOT)} (tolerance {CHECK_TOLERANCE:.0%}): "
              + ("OK" if not diffs else f"{len(diffs)} difference(s)"))
        for d in diffs:
            print("  " + d)
        return 1 if diffs else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
