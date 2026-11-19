"""Forecasters: Timestral (open weights, CPU) and the seasonal-naive baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MODEL_ID = os.environ.get("TIMESTRAL_MODEL", "mistralai/Timestral-123M")
# Weights revision pinned so the backtest replays against the same checkpoint.
MODEL_REVISION = os.environ.get("TIMESTRAL_REVISION", "5a147287addda85f6acd8da3fb8858fe9af7e8f7")


@dataclass
class QuantileForecast:
    levels: np.ndarray  # (Q,)
    quantiles: np.ndarray  # (H, Q)

    def quantile(self, level: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(self.levels - level)))
        if abs(self.levels[idx] - level) > 1e-6:
            raise ValueError(f"quantile level {level} not produced by the model")
        return self.quantiles[:, idx]

    @property
    def median(self) -> np.ndarray:
        return self.quantile(0.5)


def seasonal_naive(context: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """y_hat[t+k] = last observed value at the same position in the season.

    If that value is missing, look one more season back, and so on.
    """
    context = np.asarray(context, dtype=float)
    n = len(context)
    out = np.full(horizon, np.nan)
    for k in range(horizon):
        i = n + k - season
        while i >= n:
            i -= season
        while i >= 0 and np.isnan(context[i]):
            i -= season
        if i >= 0:
            out[k] = context[i]
    return out


@lru_cache(maxsize=1)
def _load_timestral():
    import torch
    from timestral import TimestralForecaster

    torch.manual_seed(0)
    return TimestralForecaster.from_pretrained(MODEL_ID, device="cpu", revision=MODEL_REVISION)


def timestral_forecast(context: np.ndarray, horizon: int) -> QuantileForecast:
    """Zero-shot univariate forecast. NaN in `context` marks a missing value."""
    model = _load_timestral()
    fc = model.forecast(np.asarray(context, dtype=np.float32), prediction_length=horizon)
    q = fc.quantiles.detach().cpu().numpy().astype(float).reshape(horizon, -1)  # (1, 1, H, Q) -> (H, Q)
    levels = np.round(np.asarray(fc.quantile_levels, dtype=float), 2)
    return QuantileForecast(levels=levels, quantiles=q)
