"""Tests for the v2 features and call rules.

Two properties carry the phase-2 record. A forecast for a day may not read a
price from that day or later, and every lag must be a lag on the local
calendar. Both are tested by changing the inputs and watching what moves,
rather than by reading the code and agreeing with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from de_power_live.v2 import calls
from de_power_live.v2.features import (
    SCALE_FLOOR,
    assert_no_leakage,
    build_dataset,
    local_grid,
    slot_index,
    LeakageError,
)

TZ = "Europe/Berlin"


def _history(start="2026-08-01", end="2026-11-20", seed=3):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="h", tz="UTC")
    prices = pd.Series(120 + 20 * np.sin(np.arange(len(index)) / 24 * 2 * np.pi) + rng.standard_normal(len(index)), index=index)
    weather = pd.DataFrame(
        {col: rng.random(len(index)) for col in (
            "wind_cf", "wind_speed_100m", "wind_gust_10m", "solar_radiation", "solar_direct",
            "cloud_cover", "temperature", "surface_pressure", "wind_cf_ramp", "solar_radiation_ramp")},
        index=index,
    )
    return prices, weather


def _delivery_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(TZ).normalize().tz_localize(None)


# ---- The local calendar ---------------------------------------------------


def test_a_25_hour_day_averages_its_two_0200s():
    prices, _ = _history()
    grid = local_grid(prices)
    day = pd.Timestamp("2026-10-25")
    local = prices.index.tz_convert(TZ)
    both = prices[(local.normalize().tz_localize(None) == day) & (local.hour == 2)]
    assert len(both) == 2
    assert grid.loc[day, 2] == pytest.approx(both.mean())
    assert grid.loc[day].notna().all()


def test_a_23_hour_day_fills_0200_from_its_neighbours():
    index = pd.date_range("2027-03-20", "2027-04-05", freq="h", tz="UTC")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    grid = local_grid(series)
    day = pd.Timestamp("2027-03-28")
    assert grid.loc[day, 2] == pytest.approx((grid.loc[day, 1] + grid.loc[day, 3]) / 2)
    assert grid.loc[day].notna().all()


def test_both_0200s_share_one_slot():
    hours = pd.date_range("2026-10-24 22:00", periods=25, freq="h", tz="UTC")
    slots = slot_index(hours)
    assert len(hours) == 25 and len(set(slots)) == 24
    assert list(slots).count((pd.Timestamp("2026-10-25"), 2)) == 2


def test_yesterday_means_the_same_wall_clock_hour_across_a_clock_change():
    """The property v1 lacked. On the day after a 25-hour day, the 23:00 anchor
    is 23:00 the day before, which is 25 elapsed hours earlier, not 24."""
    prices, weather = _history()
    ds = build_dataset(prices, weather)
    target = (pd.Timestamp("2026-10-26"), 23)
    expected = prices[pd.Timestamp("2026-10-25 23:00", tz=TZ).tz_convert("UTC")]
    assert ds.anchor[target] == pytest.approx(expected)
    # And on the 25-hour day itself, 23:00 looks back to 23:00 on the 24th.
    expected = prices[pd.Timestamp("2026-10-24 23:00", tz=TZ).tz_convert("UTC")]
    assert ds.anchor[(pd.Timestamp("2026-10-25"), 23)] == pytest.approx(expected)


# ---- The information rule ----------------------------------------------------


@pytest.mark.parametrize("target", ["2026-10-14", "2026-10-25", "2026-10-26", "2026-11-03"])
def test_a_days_features_ignore_every_price_from_that_day_onward(target):
    prices, weather = _history()
    target = pd.Timestamp(target)
    changed = prices.copy()
    future = _delivery_day(changed.index) >= target
    changed[future] = changed[future] * 3 + 500

    a, b = build_dataset(prices, weather), build_dataset(changed, weather)
    rows = a.X.index.get_level_values("day") == target
    pd.testing.assert_frame_equal(a.X[rows], b.X[rows])
    pd.testing.assert_series_equal(a.anchor[rows], b.anchor[rows])
    pd.testing.assert_series_equal(a.scale[rows], b.scale[rows])
    pd.testing.assert_series_equal(a.last_week[rows], b.last_week[rows])
    # The check can fail: the realised price for the day did change.
    assert not np.allclose(a.y[rows], b.y[rows])


def test_a_days_features_ignore_weather_beyond_that_day():
    prices, weather = _history()
    target = pd.Timestamp("2026-10-20")
    changed = weather.copy()
    changed[_delivery_day(changed.index) > target] += 100.0
    a, b = build_dataset(prices, weather), build_dataset(prices, changed)
    rows = a.X.index.get_level_values("day") == target
    pd.testing.assert_frame_equal(a.X[rows], b.X[rows])


def test_the_live_row_is_the_training_row():
    """One function builds both. A seal with prices only up to the seal day must
    produce exactly the row that training later sees for the same day."""
    prices, weather = _history()
    target = pd.Timestamp("2026-10-20")
    live = build_dataset(prices[_delivery_day(prices.index) < target], weather, last_target_day=target)
    later = build_dataset(prices, weather)
    rows = later.X.index.get_level_values("day") == target
    live_rows = live.X.index.get_level_values("day") == target
    pd.testing.assert_frame_equal(live.X[live_rows], later.X[rows])
    assert live.y[live_rows].isna().all()


def test_no_feature_is_a_price_level():
    """Every price column must be unchanged when the whole price history moves
    by a constant. That is what 'level-free' means, and it is what v1 was not."""
    prices, weather = _history()
    a, b = build_dataset(prices, weather), build_dataset(prices + 200.0, weather)
    pd.testing.assert_frame_equal(a.X, b.X)
    assert (b.anchor - a.anchor).dropna().round(6).eq(200.0).all()


def test_the_scale_is_floored():
    prices, weather = _history()
    ds = build_dataset(prices * 0.01, weather)
    assert ds.scale.dropna().eq(SCALE_FLOOR).all()


def test_leakage_guard_accepts_a_real_seal_and_refuses_the_rest():
    prices, _ = _history()
    seal = pd.Timestamp("2026-10-19 07:40", tz=TZ).tz_convert("UTC")
    known = prices[_delivery_day(prices.index) <= pd.Timestamp("2026-10-19")]
    assert assert_no_leakage(known, seal, pd.Timestamp("2026-10-20"))["ok"]
    with pytest.raises(LeakageError, match="not the day after"):
        assert_no_leakage(known, seal, pd.Timestamp("2026-10-21"))
    with pytest.raises(LeakageError, match="on or after the target"):
        assert_no_leakage(prices, seal, pd.Timestamp("2026-10-20"))
    with pytest.raises(LeakageError, match="anchor"):
        assert_no_leakage(known[:-48], seal, pd.Timestamp("2026-10-20"))
    with pytest.raises(ValueError):
        assert_no_leakage(known, seal.tz_localize(None), pd.Timestamp("2026-10-20"))


# ---- Call rules ------------------------------------------------------------------


def _errors(days, value):
    return {pd.Timestamp(d): np.full(24, float(value)) for d in days}


def test_the_pool_prefers_live_errors_and_skips_missed_days():
    target = pd.Timestamp("2026-12-10")
    window = pd.date_range(target - pd.Timedelta(days=calls.POOL_DAYS), target - pd.Timedelta(days=1))
    warm = _errors(window[:20], 1.0)
    live = _errors(list(window[15:22]) + list(window[23:]), 2.0)  # overlaps warm; day 22 missed
    pool, report = calls.error_pool(target, live, warm)
    assert report["live_days"] == 12 and report["warm_start_days"] == 15
    assert report["days_without_errors"] == [str(window[22].date())]
    assert (pool == 2.0).sum() == 12 * 24 and (pool == 1.0).sum() == 15 * 24


def test_the_pool_never_reaches_the_target_day_or_beyond_the_window():
    target = pd.Timestamp("2026-12-10")
    live = _errors(pd.date_range("2026-10-01", "2026-12-20"), 0.5)
    live[target] = np.full(24, 99.0)
    pool, report = calls.error_pool(target, live, {})
    assert len(pool) == calls.POOL_DAYS * 24 and pool.max() == 0.5
    assert report["first_day"] == "2026-11-12" and report["last_day"] == "2026-12-09"


def test_too_few_errors_is_an_error_not_a_narrow_interval():
    target = pd.Timestamp("2026-12-10")
    with pytest.raises(calls.CallError, match="forecast errors"):
        calls.error_pool(target, _errors(["2026-12-09"], 0.1), {})


def test_intervals_scale_with_the_price_level():
    pool = np.linspace(-1, 1, 201)
    narrow = calls.interval_quantiles(np.array([100.0]), np.array([50.0]), pool)[0]
    wide = calls.interval_quantiles(np.array([100.0]), np.array([150.0]), pool)[0]
    assert list(narrow) == sorted(narrow)
    assert (wide[4] - wide[0]) == pytest.approx(3 * (narrow[4] - narrow[0]))
    assert narrow[2] == pytest.approx(100.0)


def test_spike_threshold_and_baseline_use_only_the_days_before():
    prices, _ = _history()
    grid = local_grid(prices)
    target = pd.Timestamp("2026-10-20")
    threshold = calls.spike_threshold(grid, target)
    window = grid.loc["2026-09-22":"2026-10-19"]
    assert len(window) == calls.THRESHOLD_DAYS
    assert threshold == pytest.approx(np.quantile(window.to_numpy(), calls.SPIKE_QUANTILE))
    changed = grid.copy()
    changed.loc[target:] += 1000.0
    assert calls.spike_threshold(changed, target) == threshold
    baseline = calls.spike_baseline(grid, target, threshold)
    assert len(baseline) == 24 and baseline.between(0, 1).all()
    assert baseline.mean() == pytest.approx(1 - calls.SPIKE_QUANTILE, abs=0.01)


def test_spike_probability_moves_the_right_way():
    pool = np.linspace(-0.5, 0.5, 101)
    low, high = calls.event_probability(np.array([100.0, 200.0]), np.array([100.0, 100.0]), pool, 180.0, above=True)
    assert low == 0.0 and 0.5 < high <= 1.0
    below = calls.event_probability(np.array([100.0]), np.array([100.0]), pool, 180.0, above=False)[0]
    assert below == 1.0
