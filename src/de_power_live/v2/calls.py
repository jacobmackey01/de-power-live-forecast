"""The rules that turn a point forecast into calls B and C.

Pure functions, used unchanged by the live seal, the training script and the
replay, and inside the frozen source hash. Everything here uses only numbers
known at the seal.

Call C. The interval for a target day is the forecast plus the spread of the
last POOL_DAYS days of v2's own forecast errors, measured in units of the price
scale. v1 fixed its interval width once, on a summer of data, and coverage fell
from 0.81 in August to 0.50 in October as the market changed.

Call B. A spike hour is one whose price is above the 90th percentile of the
hourly prices of the last THRESHOLD_DAYS delivery days. Its probability comes
from the same error pool. The baseline is how often that wall-clock hour was
above the same threshold over the same days.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
POOL_DAYS = 28
MIN_POOL_ERRORS = 200
THRESHOLD_DAYS = 28
MIN_THRESHOLD_DAYS = 21
SPIKE_QUANTILE = 0.90
# Phase 1's call B thresholds, carried as descriptive results only.
DESCRIPTIVE_THRESHOLDS = {"negative": 0.0, "below_10": 10.0}


class CallError(RuntimeError):
    """Raised when a call cannot be made from the inputs available."""


def error_pool(
    target_day: pd.Timestamp,
    live_errors: dict[pd.Timestamp, np.ndarray],
    warm_errors: dict[pd.Timestamp, np.ndarray],
) -> tuple[np.ndarray, dict]:
    """Scaled errors for the POOL_DAYS days before ``target_day``.

    A day v2 sealed live contributes its live errors. A day before the first
    live seal contributes the warm-start errors frozen with the model. A day
    with neither (a missed day) contributes nothing. Live always wins.
    """
    target_day = pd.Timestamp(target_day).normalize()
    parts, live_days, warm_days, absent = [], [], [], []
    for back in range(POOL_DAYS, 0, -1):
        day = target_day - pd.Timedelta(days=back)
        if day in live_errors:
            parts.append(np.asarray(live_errors[day], dtype=float))
            live_days.append(day)
        elif day in warm_errors:
            parts.append(np.asarray(warm_errors[day], dtype=float))
            warm_days.append(day)
        else:
            absent.append(day)
    pool = np.concatenate(parts) if parts else np.array([], dtype=float)
    pool = pool[~np.isnan(pool)]
    if len(pool) < MIN_POOL_ERRORS:
        raise CallError(
            f"only {len(pool)} forecast errors in the {POOL_DAYS} days before "
            f"{target_day.date()}; need {MIN_POOL_ERRORS} to set an interval"
        )
    report = {
        "n_errors": int(len(pool)),
        "live_days": len(live_days),
        "warm_start_days": len(warm_days),
        "days_without_errors": [str(d.date()) for d in absent],
        "first_day": str((target_day - pd.Timedelta(days=POOL_DAYS)).date()),
        "last_day": str((target_day - pd.Timedelta(days=1)).date()),
        "scaled_quantiles": {str(q): round(float(v), 6) for q, v in zip(QUANTILES, np.quantile(pool, QUANTILES))},
        "sha256": hashlib.sha256(np.round(pool, 6).tobytes()).hexdigest(),
    }
    return pool, report


def interval_quantiles(pred: np.ndarray, scale: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Price quantiles per hour, shape (hours, len(QUANTILES)), non-decreasing."""
    spread = np.quantile(pool, QUANTILES)
    return np.asarray(pred, dtype=float)[:, None] + np.asarray(scale, dtype=float)[:, None] * spread[None, :]


def event_probability(
    pred: np.ndarray, scale: np.ndarray, pool: np.ndarray, threshold: float, above: bool
) -> np.ndarray:
    """Share of the error pool that would put each hour past ``threshold``."""
    scenarios = np.asarray(pred, dtype=float)[:, None] + np.asarray(scale, dtype=float)[:, None] * pool[None, :]
    hits = scenarios > threshold if above else scenarios < threshold
    return hits.mean(axis=1)


def _threshold_window(price_grid: pd.DataFrame, target_day: pd.Timestamp) -> pd.DataFrame:
    target_day = pd.Timestamp(target_day).normalize()
    window = price_grid.loc[
        target_day - pd.Timedelta(days=THRESHOLD_DAYS) : target_day - pd.Timedelta(days=1)
    ].dropna(how="any")
    if len(window) < MIN_THRESHOLD_DAYS:
        raise CallError(
            f"only {len(window)} complete price days in the {THRESHOLD_DAYS} before "
            f"{target_day.date()}; need {MIN_THRESHOLD_DAYS} to set the spike threshold"
        )
    return window


def spike_threshold(price_grid: pd.DataFrame, target_day: pd.Timestamp) -> float:
    """The SPIKE_QUANTILE of hourly prices over the days before ``target_day``."""
    window = _threshold_window(price_grid, target_day)
    return float(np.quantile(window.to_numpy(dtype=float), SPIKE_QUANTILE))


def spike_baseline(price_grid: pd.DataFrame, target_day: pd.Timestamp, threshold: float) -> pd.Series:
    """For each wall-clock hour, how often it was above ``threshold`` in the window."""
    window = _threshold_window(price_grid, target_day)
    return (window > threshold).mean(axis=0)
