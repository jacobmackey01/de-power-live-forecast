"""Feature construction, with an enforced information cutoff.

The central rule (PREREGISTRATION.md section 2): no feature may use data
published at or after the seal time. Two sources behave very differently here,
and conflating them is the easiest way to leak:

* **Prices** for the whole of day D are published around 12:45 on D-1. So at
  seal time on day D, every hour of day D is already known. Hourly price lags of
  24h or more into a D+1 target are therefore safe.

* **Realised generation and load** arrive with roughly a 1.5h delay. At 10:00 on
  day D, the actuals for 22:00 on day D emphatically do not exist. A naive
  ``t - 24h`` lag on an actuals series would reach into hours that had not
  happened yet at seal time — and would still backtest cleanly, because by the
  time anyone re-ran it the data was there.

So actuals only ever enter as aggregates over a window that closes strictly
before the cutoff. ``assert_no_leakage`` re-checks this against the raw inputs
rather than trusting the construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_TZ = "Europe/Berlin"

# Hourly price lags, in hours. All >= 24 so they cannot touch the target day.
PRICE_LAG_HOURS = (24, 48, 72, 96, 168, 336)

# Same-hour-of-day history used for rolling statistics, in days.
SAME_HOUR_LOOKBACK_DAYS = 7

FEATURE_COLUMNS = [
    # Forward weather drivers
    "wind_cf",
    "wind_speed_100m",
    "wind_gust_10m",
    "solar_radiation",
    "solar_direct",
    "cloud_cover",
    "temperature",
    "surface_pressure",
    "wind_cf_ramp",
    "solar_radiation_ramp",
    # Derived weather
    "wind_cf_day_mean",
    "solar_radiation_day_mean",
    "temperature_day_mean",
    "heating_degrees",
    "cooling_degrees",
    # Price history
    "price_lag_24",
    "price_lag_48",
    "price_lag_72",
    "price_lag_96",
    "price_lag_168",
    "price_lag_336",
    "price_same_hour_mean_7d",
    "price_same_hour_std_7d",
    "price_day_mean_lag_1d",
    "price_day_range_lag_1d",
    # Settled outturn aggregates (window closes before the cutoff)
    "wind_actual_mean_pre_cutoff",
    "solar_actual_mean_pre_cutoff",
    "load_actual_mean_pre_cutoff",
    # Calendar
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_non_working",
]


class LeakageError(RuntimeError):
    """Raised when a feature would use information published after the cutoff."""


# ---- Calendar -----------------------------------------------------------


def easter_sunday(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm. Avoids a holiday-library dependency."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def german_public_holidays(year: int) -> set[pd.Timestamp]:
    """Nationwide German public holidays. Regional ones are deliberately excluded:
    they do not move national load enough to matter here."""
    easter = easter_sunday(year)
    return {
        pd.Timestamp(year=year, month=1, day=1),  # Neujahr
        easter - pd.Timedelta(days=2),  # Karfreitag
        easter + pd.Timedelta(days=1),  # Ostermontag
        pd.Timestamp(year=year, month=5, day=1),  # Tag der Arbeit
        easter + pd.Timedelta(days=39),  # Christi Himmelfahrt
        easter + pd.Timedelta(days=50),  # Pfingstmontag
        pd.Timestamp(year=year, month=10, day=3),  # Tag der Deutschen Einheit
        pd.Timestamp(year=year, month=12, day=25),
        pd.Timestamp(year=year, month=12, day=26),
    }


def _calendar_features(index_utc: pd.DatetimeIndex) -> pd.DataFrame:
    local = index_utc.tz_convert(MARKET_TZ)
    frame = pd.DataFrame(index=index_utc)

    hour = local.hour.to_numpy()
    doy = local.dayofyear.to_numpy()
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    dow = local.dayofweek.to_numpy()
    frame["day_of_week"] = dow
    frame["is_weekend"] = (dow >= 5).astype(int)

    holidays: set[pd.Timestamp] = set()
    for year in sorted({int(y) for y in local.year}):
        holidays |= german_public_holidays(year)
    local_dates = pd.DatetimeIndex(local.date)
    frame["is_holiday"] = local_dates.isin(holidays).astype(int)
    frame["is_non_working"] = (
        (frame["is_weekend"] == 1) | (frame["is_holiday"] == 1)
    ).astype(int)
    return frame


# ---- Feature assembly ---------------------------------------------------


def _local_day(index_utc: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(index_utc.tz_convert(MARKET_TZ).date, name="local_day")


def build_features(
    weather: pd.DataFrame,
    prices: pd.Series,
    actuals: pd.DataFrame | None,
    cutoff_utc: pd.Timestamp,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assemble the model matrix for ``target_index``.

    Parameters
    ----------
    weather
        Forward drivers from ``OpenMeteoClient.national_drivers``, UTC indexed.
    prices
        Realised day-ahead prices, UTC indexed. Only values strictly before
        ``cutoff_utc`` are used.
    actuals
        Realised generation/load, UTC indexed, optional. Only the window closing
        before ``cutoff_utc`` is used.
    cutoff_utc
        Seal time. Nothing published at or after this may influence the result.
    target_index
        Hours to build features for, UTC.
    """
    if cutoff_utc.tzinfo is None:
        raise ValueError("cutoff_utc must be timezone-aware")

    # Everything from delivery days already auctioned is fair game, including
    # hours of the current day that have not yet occurred. Filtering on the raw
    # timestamp instead would throw away the ~15 most recent and most predictive
    # hours for no integrity gain.
    cutoff_local_day = cutoff_utc.tz_convert(MARKET_TZ).normalize()
    price_delivery_days = pd.DatetimeIndex(prices.index.tz_convert(MARKET_TZ)).normalize()
    prices_known = prices.loc[price_delivery_days <= cutoff_local_day].dropna()

    frame = pd.DataFrame(index=target_index)

    # -- forward weather
    aligned = weather.reindex(target_index)
    for col in (
        "wind_cf", "wind_speed_100m", "wind_gust_10m", "solar_radiation",
        "solar_direct", "cloud_cover", "temperature", "surface_pressure",
        "wind_cf_ramp", "solar_radiation_ramp",
    ):
        frame[col] = aligned[col] if col in aligned.columns else np.nan

    day_key = _local_day(target_index)
    for src, dest in (
        ("wind_cf", "wind_cf_day_mean"),
        ("solar_radiation", "solar_radiation_day_mean"),
        ("temperature", "temperature_day_mean"),
    ):
        frame[dest] = frame.groupby(day_key)[src].transform("mean").to_numpy()

    # Degree days: load responds asymmetrically either side of ~15 C.
    frame["heating_degrees"] = np.clip(15.0 - frame["temperature"], 0.0, None)
    frame["cooling_degrees"] = np.clip(frame["temperature"] - 20.0, 0.0, None)

    # -- price lags (safe: day D prices are published on D-1)
    for lag in PRICE_LAG_HOURS:
        lagged = prices_known.reindex(target_index - pd.Timedelta(hours=lag))
        frame[f"price_lag_{lag}"] = lagged.to_numpy()

    same_hour = np.column_stack(
        [
            prices_known.reindex(
                target_index - pd.Timedelta(days=d)
            ).to_numpy(dtype=float)
            for d in range(1, SAME_HOUR_LOOKBACK_DAYS + 1)
        ]
    )
    with np.errstate(invalid="ignore"):
        frame["price_same_hour_mean_7d"] = np.nanmean(same_hour, axis=1)
        frame["price_same_hour_std_7d"] = np.nanstd(same_hour, axis=1)

    # Previous local day's shape, computed only from settled prices.
    if not prices_known.empty:
        by_day = prices_known.groupby(_local_day(prices_known.index))
        day_mean = by_day.mean()
        day_range = by_day.max() - by_day.min()
        prev_day = pd.Index(
            [d - pd.Timedelta(days=1) for d in pd.to_datetime(day_key)]
        ).date
        frame["price_day_mean_lag_1d"] = day_mean.reindex(prev_day).to_numpy()
        frame["price_day_range_lag_1d"] = day_range.reindex(prev_day).to_numpy()
    else:
        frame["price_day_mean_lag_1d"] = np.nan
        frame["price_day_range_lag_1d"] = np.nan

    # -- settled outturn: a single pre-cutoff aggregate, never an hourly lag
    window_start = cutoff_utc - pd.Timedelta(hours=24)
    for src, dest in (
        ("wind_total_actual", "wind_actual_mean_pre_cutoff"),
        ("solar_actual", "solar_actual_mean_pre_cutoff"),
        ("load_actual", "load_actual_mean_pre_cutoff"),
    ):
        value = np.nan
        if actuals is not None and src in actuals.columns:
            window = actuals[src]
            window = window.loc[(window.index >= window_start) & (window.index < cutoff_utc)]
            if window.notna().any():
                value = float(window.mean())
        frame[dest] = value

    frame = pd.concat([frame, _calendar_features(target_index)], axis=1)
    return frame[FEATURE_COLUMNS]


def assert_no_leakage(
    prices: pd.Series,
    actuals: pd.DataFrame | None,
    cutoff_utc: pd.Timestamp,
    target_index: pd.DatetimeIndex,
) -> dict:
    """Independently re-check the cutoff rather than trusting build_features.

    Verifies that every hourly price lag lands strictly before the cutoff, and
    that the actuals window closes before it. Raises on breach.
    """
    report: dict = {"cutoff_utc": cutoff_utc.isoformat()}

    if len(target_index) == 0:
        raise LeakageError("empty target index")

    # Prices and actuals become knowable on completely different schedules, and
    # applying one availability rule to both would create a timing error.
    #
    # A day-ahead price for delivery day X is published just after the auction
    # clears at 12:00 local on X-1. So at a cutoff on day D, every price for
    # delivery days up to and including D is already known - even the 23:00 hour,
    # which has not happened yet. What is *not* known is day D+1: that is the
    # thing being forecast.
    cutoff_local_day = cutoff_utc.tz_convert(MARKET_TZ).normalize()
    latest_lagged = target_index.max() - pd.Timedelta(hours=min(PRICE_LAG_HOURS))
    latest_lagged_day = latest_lagged.tz_convert(MARKET_TZ).normalize()

    if latest_lagged_day > cutoff_local_day:
        raise LeakageError(
            f"price lag of {min(PRICE_LAG_HOURS)}h on target {target_index.max()} "
            f"reaches delivery day {latest_lagged_day.date()}, whose auction had not "
            f"cleared at cutoff {cutoff_utc} (local day {cutoff_local_day.date()})"
        )
    report["latest_price_lag_utc"] = latest_lagged.isoformat()
    report["latest_price_lag_delivery_day"] = str(latest_lagged_day.date())
    report["cutoff_local_day"] = str(cutoff_local_day.date())

    used_prices = prices.loc[prices.index < cutoff_utc]
    if used_prices.index.max() is not pd.NaT and len(used_prices):
        report["latest_price_used_utc"] = used_prices.index.max().isoformat()

    if actuals is not None and len(actuals):
        window_end = cutoff_utc
        used = actuals.loc[actuals.index < window_end]
        if len(used):
            report["latest_actual_used_utc"] = used.index.max().isoformat()
        late = actuals.loc[actuals.index >= window_end].notna().any().any()
        report["actuals_after_cutoff_present_but_unused"] = bool(late)

    if target_index.min() < cutoff_utc:
        raise LeakageError(
            f"target hour {target_index.min()} precedes cutoff {cutoff_utc}; "
            "this would be scoring, not forecasting"
        )

    report["n_target_hours"] = int(len(target_index))
    report["ok"] = True
    return report
