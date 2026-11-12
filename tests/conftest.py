"""Shared fixtures: synthetic history and rehearsed seals for the v2 tests.

Synthetic data, so the tests need no network and no cache. The rehearsals run
the real seal and scoring path over three stretches that contain every
clock-change day from the start of phase 2 to the end of 2027.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from de_power_live.v2.rehearse import rehearse
from de_power_live.v2.train import freeze

TZ = "Europe/Berlin"
FAST = dict(
    n_estimators=20, learning_rate=0.2, max_depth=3, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_lambda=1.0, objective="reg:absoluteerror", random_state=7, n_jobs=1,
)
# Each block: train through the first date, then seal every day of the range.
# The ranges cover 2026-10-25 (25h), 2027-03-28 (23h) and 2027-10-31 (25h), and
# the day after each, whose "yesterday" is the odd-length day.
BLOCKS = {
    "autumn_2026": ("2026-10-12", "2026-10-14", "2026-10-27"),
    "spring_2027": ("2027-03-16", "2027-03-18", "2027-03-30"),
    "autumn_2027": ("2027-10-19", "2027-10-21", "2027-11-02"),
}
CLOCK_CHANGE_DAYS = {"2026-10-25": 25, "2027-03-28": 23, "2027-10-31": 25}


@pytest.fixture(scope="session")
def history():
    rng = np.random.default_rng(11)
    index = pd.date_range("2026-01-01", "2027-11-10", freq="h", tz="UTC")
    local = index.tz_convert(TZ)
    hours = np.arange(len(index))
    level = 100 + 80 * (index > pd.Timestamp("2026-09-20", tz="UTC"))  # a step, like autumn 2026
    wind = 0.35 + 0.25 * np.sin(hours / 53.0) + 0.1 * rng.standard_normal(len(index))
    solar = np.clip(np.sin((local.hour - 6) / 12 * np.pi), 0, None) * 400
    prices = pd.Series(
        level * (1 + 0.25 * np.sin((local.hour - 8) / 24 * 2 * np.pi) - 0.4 * (wind - 0.35))
        - 0.05 * solar - 15 * (local.dayofweek >= 5) + 6 * rng.standard_normal(len(index)),
        index=index,
    )
    weather = pd.DataFrame(
        {
            "wind_cf": np.clip(wind, 0, 1),
            "wind_speed_100m": 6 + 8 * np.clip(wind, 0, 1),
            "wind_gust_10m": 9 + 10 * np.clip(wind, 0, 1),
            "solar_radiation": solar,
            "solar_direct": solar * 0.6,
            "cloud_cover": 50 + 20 * np.sin(hours / 31.0),
            "temperature": 10 + 8 * np.sin(hours / (24 * 58.0)) + 3 * np.sin((local.hour - 9) / 24 * 2 * np.pi),
            "surface_pressure": 1013 + 5 * np.sin(hours / 97.0),
        },
        index=index,
    )
    weather["wind_cf_ramp"] = weather["wind_cf"].diff()
    weather["solar_radiation_ramp"] = weather["solar_radiation"].diff()
    return prices, weather


@pytest.fixture(scope="session")
def rehearsals(history, tmp_path_factory):
    """Freeze a small model and rehearse every block. Returns per-block results."""
    prices, weather = history
    out = {}
    for name, (train_last, first, last) in BLOCKS.items():
        root = tmp_path_factory.mktemp(name)
        model_dir, work_dir = root / "model", root / "predictions"
        work_dir.mkdir()
        manifest = freeze(prices, weather, pd.Timestamp(train_last), model_dir, params=FAST)
        ledger = rehearse(prices, weather, pd.date_range(first, last, freq="D"), model_dir, work_dir)
        out[name] = {"model_dir": model_dir, "work_dir": work_dir, "ledger": ledger, "manifest": manifest}
    return out
