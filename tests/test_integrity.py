"""Tests for the properties the record depends on.

These deliberately target the failure modes that would not announce themselves:
silent unit changes, DST day lengths, look-ahead leakage, and late seals. A
broken model is obvious; a leaking one is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from de_power_live.dataset import seal_time_for, target_hours_for
from de_power_live.features import (
    LeakageError,
    assert_no_leakage,
    build_features,
    easter_sunday,
    german_public_holidays,
)
from de_power_live.predict import SealError, auction_close_utc, seal
from de_power_live.weather import (
    WeatherError,
    assert_declared_units,
    assert_plausible_units,
    turbine_power_curve,
)


# ---- power curve --------------------------------------------------------


def test_power_curve_shape():
    assert turbine_power_curve(np.array([0.0]))[0] == 0.0
    assert turbine_power_curve(np.array([2.9]))[0] == 0.0  # below cut-in
    assert turbine_power_curve(np.array([12.0]))[0] == pytest.approx(1.0)
    assert turbine_power_curve(np.array([20.0]))[0] == pytest.approx(1.0)


def test_power_curve_cuts_out_in_a_storm():
    """A storm reduces output. A linear wind feature would predict the opposite
    at exactly the hours prices move most."""
    assert turbine_power_curve(np.array([24.9]))[0] == pytest.approx(1.0)
    assert turbine_power_curve(np.array([25.0]))[0] == 0.0
    assert turbine_power_curve(np.array([40.0]))[0] == 0.0


def test_power_curve_monotonic_on_ramp():
    ramp = turbine_power_curve(np.arange(3.0, 12.0, 0.5))
    assert np.all(np.diff(ramp) > 0)


def test_power_curve_preserves_nan():
    assert np.isnan(turbine_power_curve(np.array([np.nan]))[0])


# ---- unit guards --------------------------------------------------------


def test_declared_units_reject_kmh():
    """The real bug this guards: Open-Meteo returns km/h unless asked otherwise,
    and km/h fed to an m/s power curve reads ~3.6x too windy."""
    with pytest.raises(WeatherError, match="km/h"):
        assert_declared_units({"wind_speed_100m": "km/h"}, "test-site")


def test_declared_units_accept_ms():
    assert_declared_units(
        {"wind_speed_100m": "m/s", "wind_gusts_10m": "m/s", "temperature_2m": "°C"},
        "test-site",
    )


def test_plausible_units_catch_a_silent_rescale():
    kmh_values = pd.DataFrame({"wind_speed_100m": [45.0, 62.0, 80.0]})
    with pytest.raises(WeatherError, match="plausible range"):
        assert_plausible_units(kmh_values, "test-site")


def test_plausible_units_pass_on_real_ms_values():
    assert_plausible_units(pd.DataFrame({"wind_speed_100m": [0.4, 8.2, 17.9]}), "ok")


# ---- DST ----------------------------------------------------------------


def test_spring_forward_is_23_hours():
    assert len(target_hours_for(pd.Timestamp("2026-03-29"))) == 23


def test_autumn_back_is_25_hours():
    """Inside the evaluation window; hard-coding 24 would silently drop an hour."""
    assert len(target_hours_for(pd.Timestamp("2026-10-25"))) == 25


def test_ordinary_day_is_24_hours():
    assert len(target_hours_for(pd.Timestamp("2026-08-02"))) == 24


def test_target_hours_are_unique_and_ordered():
    hours = target_hours_for(pd.Timestamp("2026-10-25"))
    assert hours.is_monotonic_increasing
    assert len(set(hours)) == len(hours)


def test_seal_time_tracks_dst():
    """The seal is anchored to local market time, so its UTC hour must shift."""
    summer = seal_time_for(pd.Timestamp("2026-08-02"))
    winter = seal_time_for(pd.Timestamp("2026-12-02"))
    assert summer.hour == 9  # 11:00 CEST
    assert winter.hour == 10  # 11:00 CET


def test_auction_close_is_the_day_before():
    close = auction_close_utc(pd.Timestamp("2026-08-02"))
    assert close.tz_convert("Europe/Berlin").date() == pd.Timestamp("2026-08-01").date()
    assert close.tz_convert("Europe/Berlin").hour == 12


# ---- calendar -----------------------------------------------------------


def test_easter_known_values():
    assert easter_sunday(2026) == pd.Timestamp("2026-04-05")
    assert easter_sunday(2024) == pd.Timestamp("2024-03-31")


def test_german_holidays_include_unity_day():
    assert pd.Timestamp("2026-10-03") in german_public_holidays(2026)
    assert pd.Timestamp("2026-01-01") in german_public_holidays(2026)


# ---- leakage ------------------------------------------------------------


def _toy_prices(end: pd.Timestamp, days: int = 30) -> pd.Series:
    index = pd.date_range(end - pd.Timedelta(days=days), end, freq="h", tz="UTC")
    return pd.Series(np.linspace(20, 120, len(index)), index=index)


def test_leakage_check_rejects_target_before_cutoff():
    cutoff = pd.Timestamp("2026-08-01T09:00:00Z")
    past = pd.date_range("2026-07-31", periods=24, freq="h", tz="UTC")
    with pytest.raises(LeakageError, match="precedes cutoff"):
        assert_no_leakage(_toy_prices(cutoff), None, cutoff, past)


def test_leakage_check_passes_for_a_proper_next_day_target():
    """The lag may land on an hour later than the cutoff, provided that hour's
    delivery day had already been auctioned."""
    cutoff = pd.Timestamp("2026-08-01T09:00:00Z")
    hours = target_hours_for(pd.Timestamp("2026-08-02"))
    report = assert_no_leakage(_toy_prices(cutoff), None, cutoff, hours)
    assert report["ok"]
    assert report["latest_price_lag_delivery_day"] <= report["cutoff_local_day"]


def test_leakage_check_rejects_a_two_day_ahead_target():
    """A D+2 target would need D+1 prices, whose auction has not run."""
    cutoff = pd.Timestamp("2026-08-01T09:00:00Z")
    hours = target_hours_for(pd.Timestamp("2026-08-03"))
    with pytest.raises(LeakageError, match="had not cleared"):
        assert_no_leakage(_toy_prices(cutoff), None, cutoff, hours)


def _toy_weather() -> pd.DataFrame:
    return pd.DataFrame(
        1.0,
        index=pd.date_range("2026-06-20", "2026-08-05", freq="h", tz="UTC"),
        columns=[
            "wind_cf", "wind_speed_100m", "wind_gust_10m", "solar_radiation",
            "solar_direct", "cloud_cover", "temperature", "surface_pressure",
            "wind_cf_ramp", "solar_radiation_ramp",
        ],
    )


def test_features_ignore_delivery_days_not_yet_auctioned():
    """Poison every price for delivery days after the cutoff's local day.

    Those auctions had not cleared at seal time. If any of it reaches the
    features, the frames differ.
    """
    cutoff = pd.Timestamp("2026-08-01T09:00:00Z")
    hours = target_hours_for(pd.Timestamp("2026-08-02"))

    clean = _toy_prices(cutoff + pd.Timedelta(days=4), days=45)
    poisoned = clean.copy()
    local_day = pd.DatetimeIndex(poisoned.index.tz_convert("Europe/Berlin")).normalize()
    unpublished = local_day > cutoff.tz_convert("Europe/Berlin").normalize()
    poisoned[unpublished] = 9999.0

    a = build_features(_toy_weather(), clean, None, cutoff, hours)
    b = build_features(_toy_weather(), poisoned, None, cutoff, hours)
    pd.testing.assert_frame_equal(a, b)


def test_features_do_use_already_auctioned_hours_of_today():
    """The counterpart: prices for today's later hours are known at seal time and
    must actually be used. Dropping them would cost the strongest lag features."""
    cutoff = pd.Timestamp("2026-08-01T09:00:00Z")
    hours = target_hours_for(pd.Timestamp("2026-08-02"))

    base = _toy_prices(cutoff + pd.Timedelta(days=4), days=45)
    shifted = base.copy()
    local = pd.DatetimeIndex(shifted.index.tz_convert("Europe/Berlin"))
    today_evening = (
        pd.DatetimeIndex(local).normalize() == cutoff.tz_convert("Europe/Berlin").normalize()
    ) & (local.hour >= 12)
    shifted[today_evening] = shifted[today_evening] + 50.0

    a = build_features(_toy_weather(), base, None, cutoff, hours)
    b = build_features(_toy_weather(), shifted, None, cutoff, hours)
    assert not a["price_lag_24"].equals(b["price_lag_24"]), (
        "today's already-auctioned evening prices must reach the lag features"
    )


# ---- sealing ------------------------------------------------------------


# ---- conformal calibration ----------------------------------------------


def test_conformal_offset_widens_an_overconfident_interval():
    """Truth consistently outside a too-narrow interval must produce a positive
    offset, i.e. the interval gets wider."""
    from de_power_live.model import _conformal_offsets

    n = 500
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 10, n)
    # Interval of +/-1 around zero: far too narrow for sd-10 noise.
    q = np.column_stack([
        np.full(n, -1.0), np.full(n, -0.5), np.zeros(n),
        np.full(n, 0.5), np.full(n, 1.0),
    ])
    offsets = _conformal_offsets(q, truth)
    assert offsets["80"] > 0
    assert offsets["50"] > 0
    assert offsets["80"] > offsets["50"], "80% interval needs the larger widening"


def test_conformal_offset_narrows_an_overwide_interval():
    """The score is negative when truth sits comfortably inside, so a needlessly
    wide interval is tightened rather than left alone."""
    from de_power_live.model import _conformal_offsets

    n = 500
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 1, n)
    q = np.column_stack([
        np.full(n, -100.0), np.full(n, -50.0), np.zeros(n),
        np.full(n, 50.0), np.full(n, 100.0),
    ])
    offsets = _conformal_offsets(q, truth)
    assert offsets["80"] < 0
    assert offsets["50"] < 0


def test_conformal_offset_restores_nominal_coverage():
    """The point of the exercise: applying the offset to fresh data from the same
    distribution should land coverage near nominal."""
    from de_power_live.model import _conformal_offsets

    rng = np.random.default_rng(2)
    cal_truth = rng.normal(0, 10, 2000)
    fresh_truth = rng.normal(0, 10, 2000)

    def band(n):
        return np.column_stack([
            np.full(n, -4.0), np.full(n, -2.0), np.zeros(n),
            np.full(n, 2.0), np.full(n, 4.0),
        ])

    offsets = _conformal_offsets(band(2000), cal_truth)
    q = band(2000)
    lo = q[:, 0] - offsets["80"]
    hi = q[:, 4] + offsets["80"]
    coverage = float(np.mean((fresh_truth >= lo) & (fresh_truth <= hi)))
    assert 0.75 <= coverage <= 0.85, f"coverage {coverage:.3f} not near nominal 0.80"


# ---- sealing ------------------------------------------------------------


def test_seal_refuses_to_overwrite(tmp_path: Path):
    payload = {"delivery_date_local": "2026-08-02", "predictions": []}
    seal(payload, directory=tmp_path)
    with pytest.raises(SealError, match="write-once"):
        seal(payload, directory=tmp_path)


def test_sealed_file_is_valid_json(tmp_path: Path):
    payload = {"delivery_date_local": "2026-08-03", "predictions": [], "model_version": "v1"}
    path = seal(payload, directory=tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["model_version"] == "v1"
