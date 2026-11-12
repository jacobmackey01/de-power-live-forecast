"""Level-free features on the local calendar.

One function builds the model matrix for training, for the replay and for the
live seal, so there is no second code path for train/serve skew to hide in.

The information rule (PREREGISTRATION_PHASE2.md section 2). The row for target
day T is sealed on S = T-1 and may use:

* prices for delivery days up to and including S. Day S cleared at noon on S-1,
  so at the seal on S all of it is known. Nothing from day T.
* the weather forecast for S and T.

Realised generation and load are deliberately not used. On 2026-11-11 a live
fetch at 09:00 Berlin found solar and onshore-wind outturns published only up
to 08:00 the previous morning, a day behind, while training history has them
complete. A model trained on a 24-hour outturn mean would have been served a
blank at every live seal: in the replay that costs about a quarter of the skill
over a winter, and the inputs add nothing measurable when present. v1 has this
skew (docs/v1-lessons.md section 8). v2 avoids it by not using the series.

Every lag is "the same wall-clock hour k days earlier", taken from a grid with
one row per local day and 24 wall-clock columns. v1 used fixed UTC offsets, and
on the 25-hour day 2026-10-25 a 24-hour lag landed inside the delivery day and
the forecast could not be sealed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import MARKET_TZ
from ..features import german_public_holidays

# The live seal window, Europe/Berlin. No input depends on the clock, so this
# only fixes when a rehearsal pretends to seal.
SEAL_LOCAL = pd.Timedelta(hours=7, minutes=40)

SCALE_DAYS = 7
SCALE_FLOOR = 30.0  # EUR/MWh; keeps the scaled target finite in a near-zero week

WEATHER_COLS = (
    "wind_cf", "wind_speed_100m", "wind_gust_10m", "solar_radiation", "solar_direct",
    "cloud_cover", "temperature", "surface_pressure", "wind_cf_ramp", "solar_radiation_ramp",
)
DELTA_COLS = ("wind_cf", "solar_radiation", "temperature", "cloud_cover")
HOURS = range(24)


class LeakageError(RuntimeError):
    """Raised when an input would let a forecast see its own delivery day."""


# ---- The local calendar ---------------------------------------------------


def local_grid(series: pd.Series) -> pd.DataFrame:
    """Hourly UTC series -> one row per local day, 24 wall-clock columns.

    A 25-hour day has two 02:00s: the slot holds their mean. A 23-hour day has
    no 02:00: the slot holds the mean of 01:00 and 03:00. Days are contiguous,
    so shifting the grid by k rows is a lag of k local days.
    """
    series = series.dropna()
    local = series.index.tz_convert(MARKET_TZ)
    frame = pd.DataFrame(
        {
            "day": local.normalize().tz_localize(None),
            "hour": local.hour,
            "value": series.to_numpy(dtype=float),
        }
    )
    grid = frame.groupby(["day", "hour"])["value"].mean().unstack("hour").reindex(columns=HOURS)
    gap = grid[2].isna() & grid[1].notna() & grid[3].notna()
    grid.loc[gap, 2] = (grid.loc[gap, 1] + grid.loc[gap, 3]) / 2.0
    return grid.reindex(pd.date_range(grid.index.min(), grid.index.max(), freq="D"))


def slot_index(index_utc: pd.DatetimeIndex) -> pd.MultiIndex:
    """(local day, wall-clock hour) for each UTC hour. Both 02:00s of a 25-hour
    day map to the same slot."""
    local = index_utc.tz_convert(MARKET_TZ)
    return pd.MultiIndex.from_arrays(
        [local.normalize().tz_localize(None), local.hour], names=["day", "hour"]
    )


def calendar_flags(days: pd.DatetimeIndex) -> pd.DataFrame:
    holidays: set[pd.Timestamp] = set()
    for year in sorted(set(days.year)):
        holidays |= german_public_holidays(int(year))
    frame = pd.DataFrame(index=days)
    frame["dow"] = days.dayofweek
    frame["is_weekend"] = (frame["dow"] >= 5).astype(int)
    frame["is_holiday"] = days.isin(list(holidays)).astype(int)
    # 24 and 31 December are working days on paper and holidays on the grid,
    # and the days between behave like neither.
    frame["is_year_end"] = (
        ((days.month == 12) & (days.day >= 24)) | ((days.month == 1) & (days.day <= 1))
    ).astype(int)
    frame["is_non_working"] = ((frame["is_weekend"] == 1) | (frame["is_holiday"] == 1)).astype(int)
    return frame


def nominal_seal_utc(target_day: pd.Timestamp) -> pd.Timestamp:
    """The seal window for ``target_day``: SEAL_LOCAL on the day before."""
    day_before = pd.Timestamp(target_day).normalize() - pd.Timedelta(days=1)
    return (day_before.tz_localize(MARKET_TZ) + SEAL_LOCAL).tz_convert("UTC")


# ---- The dataset ------------------------------------------------------------


@dataclass
class Dataset:
    """Everything indexed by (target day, wall-clock hour)."""

    X: pd.DataFrame  # level-free features; price columns in EUR/MWh differences
    price_columns: list[str]  # the columns of X that are divided by the scale
    y: pd.Series  # realised price, NaN where not yet known
    anchor: pd.Series  # same hour on the seal day: the "yesterday" baseline
    last_week: pd.Series  # same hour seven days before the target day
    scale: pd.Series  # trailing price level known at the seal, floored

    def design(self) -> tuple[pd.DataFrame, pd.Series]:
        """Model matrix and target, with price differences in units of the scale."""
        X = self.X.copy()
        X[self.price_columns] = X[self.price_columns].div(self.scale, axis=0)
        return X, (self.y - self.anchor) / self.scale

    def usable(self) -> pd.Series:
        """Rows a forecast can be made for: an anchor and a scale both exist."""
        return self.anchor.notna() & self.scale.notna()


def _stack(grid: pd.DataFrame, name: str) -> pd.Series:
    out = grid.stack(future_stack=True)
    out.index.names = ["day", "hour"]
    return out.rename(name)


def build_dataset(
    prices: pd.Series,
    weather: pd.DataFrame,
    last_target_day: pd.Timestamp | None = None,
) -> Dataset:
    """Features for every target day from the first price day to ``last_target_day``.

    ``last_target_day`` defaults to the last day with prices; pass the day after
    it to get the row a live seal needs.
    """
    P = local_grid(prices)
    last = pd.Timestamp(last_target_day) if last_target_day is not None else P.index.max()
    days = pd.date_range(P.index.min(), last, freq="D")
    P = P.reindex(days)

    def lag(grid: pd.DataFrame, k: int) -> pd.DataFrame:
        return grid.shift(k)

    def per_day(series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            np.repeat(series.to_numpy(dtype=float)[:, None], 24, axis=1), index=days, columns=HOURS
        )

    anchor = lag(P, 1)
    day_mean = P.mean(axis=1)
    day_range = P.max(axis=1) - P.min(axis=1)
    # Average of the seven daily means ending on the seal day.
    level = day_mean.rolling(SCALE_DAYS, min_periods=SCALE_DAYS).mean().shift(1)
    scale = level.clip(lower=SCALE_FLOOR)

    same_hour = np.stack([lag(P, k).to_numpy() for k in range(1, 8)])
    complete = ~np.isnan(same_hour).any(axis=0)
    same_hour_mean = pd.DataFrame(
        np.where(complete, same_hour.mean(axis=0), np.nan), index=days, columns=HOURS
    )
    same_hour_std = pd.DataFrame(
        np.where(complete, same_hour.std(axis=0), np.nan), index=days, columns=HOURS
    )

    grids: dict[str, pd.DataFrame] = {}

    # Price history: differences from the anchor or from a day's own mean.
    grids["d_lag2"] = lag(P, 2) - anchor
    grids["d_lag3"] = lag(P, 3) - anchor
    grids["d_lag7"] = lag(P, 7) - anchor  # same weekday as the target
    grids["d_lag8"] = lag(P, 8) - lag(P, 7)  # what the same transition did last week
    grids["d_lag14"] = lag(P, 14) - anchor
    grids["d_same_hour_mean_7d"] = same_hour_mean - anchor
    grids["same_hour_std_7d"] = same_hour_std
    grids["anchor_vs_day_mean"] = anchor.sub(day_mean.shift(1), axis=0)
    grids["lag7_vs_day_mean"] = lag(P, 7).sub(day_mean.shift(7), axis=0)
    grids["day_mean_change_1d"] = per_day(day_mean.shift(1) - day_mean.shift(2))
    grids["day_mean_vs_level"] = per_day(day_mean.shift(1) - level)
    grids["day_range_lag1"] = per_day(day_range.shift(1))
    grids["day_range_lag7"] = per_day(day_range.shift(7))
    grids["day_mean_change_lag7"] = per_day(day_mean.shift(7) - day_mean.shift(8))
    price_columns = list(grids)

    # Forward weather for the target day and its change from the seal day: the
    # target is a change, so the change in its drivers is the natural input.
    for col in WEATHER_COLS:
        grid = local_grid(weather[col]).reindex(days)
        grids[col] = grid
        if col in DELTA_COLS:
            mean = grid.mean(axis=1, skipna=False)
            grids[f"d_{col}"] = grid - lag(grid, 1)
            grids[f"{col}_day_mean"] = per_day(mean)
            grids[f"d_{col}_day_mean"] = per_day(mean - mean.shift(1))
    heating = (15.0 - grids["temperature"]).clip(lower=0.0)
    grids["heating_degrees"] = heating
    grids["d_heating_degrees"] = heating - lag(heating, 1)
    grids["cooling_degrees"] = (grids["temperature"] - 20.0).clip(lower=0.0)

    # Calendar, for the target day and for the seal day it is a change from.
    target_cal = calendar_flags(days)
    seal_cal = calendar_flags(days - pd.Timedelta(days=1))
    seal_cal.index = days
    for col in ("dow", "is_weekend", "is_holiday", "is_year_end", "is_non_working"):
        grids[col] = per_day(target_cal[col])
    grids["seal_day_non_working"] = per_day(seal_cal["is_non_working"])
    grids["seal_day_year_end"] = per_day(seal_cal["is_year_end"])
    doy = pd.Series(days.dayofyear, index=days)
    grids["doy_sin"] = per_day(np.sin(2 * np.pi * doy / 365.25))
    grids["doy_cos"] = per_day(np.cos(2 * np.pi * doy / 365.25))

    X = pd.concat([_stack(grid, name) for name, grid in grids.items()], axis=1)
    hour = X.index.get_level_values("hour").to_numpy()
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    return Dataset(
        X=X,
        price_columns=price_columns,
        y=_stack(P, "price"),
        anchor=_stack(anchor, "anchor"),
        last_week=_stack(lag(P, 7), "last_week"),
        scale=_stack(per_day(scale), "scale"),
    )


def assert_no_leakage(prices: pd.Series, cutoff_utc: pd.Timestamp, target_day: pd.Timestamp) -> dict:
    """Refuse inputs that could not have existed at the seal.

    The target must be the local day after the seal, and the price series must
    hold nothing for the target day or later. At a genuine seal it cannot: day
    T is published only after the gate has closed. If such a price is present,
    this run is not a forecast.
    """
    if cutoff_utc.tzinfo is None:
        raise ValueError("cutoff_utc must be timezone-aware")
    target_day = pd.Timestamp(target_day).normalize()
    seal_day = cutoff_utc.tz_convert(MARKET_TZ).normalize().tz_localize(None)
    if target_day != seal_day + pd.Timedelta(days=1):
        raise LeakageError(
            f"target {target_day.date()} is not the day after the seal day {seal_day.date()}"
        )
    delivery_day = prices.dropna().index.tz_convert(MARKET_TZ).normalize().tz_localize(None)
    newest = delivery_day.max()
    if newest >= target_day:
        raise LeakageError(
            f"price series already holds delivery day {newest.date()}, which is on or after "
            f"the target {target_day.date()}: its auction had not cleared at a real seal"
        )
    if newest < seal_day:
        raise LeakageError(
            f"newest price is for {newest.date()}, but the seal day {seal_day.date()} cleared "
            "yesterday and is the anchor of every forecast"
        )
    return {
        "cutoff_utc": cutoff_utc.isoformat(),
        "seal_day_local": str(seal_day.date()),
        "target_day_local": str(target_day.date()),
        "newest_price_delivery_day": str(newest.date()),
        "ok": True,
    }
