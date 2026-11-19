"""Forecast accuracy metrics. All functions ignore NaN targets."""

from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y - yhat)))


def seasonal_naive_scale(context: np.ndarray, season: int) -> float:
    """In-sample MAE of the seasonal-naive forecast, the denominator of MASE."""
    context = np.asarray(context, dtype=float)
    return float(np.nanmean(np.abs(context[season:] - context[:-season])))


def pinball(y: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    diff = y - q
    return np.maximum(level * diff, (level - 1) * diff)


def wql(y: np.ndarray, quantiles: np.ndarray, levels: np.ndarray) -> float:
    """Weighted quantile loss: 2 * sum of pinball losses / sum |y|, averaged over levels.

    For a point forecast repeated at every level this equals sum|y - yhat| / sum|y|.
    """
    ok = ~np.isnan(y)
    y = y[ok]
    losses = [2 * pinball(y, quantiles[ok, i], lv).sum() / np.abs(y).sum() for i, lv in enumerate(levels)]
    return float(np.mean(losses))


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    ok = ~np.isnan(y)
    return float(np.mean((y[ok] >= lower[ok]) & (y[ok] <= upper[ok])))
