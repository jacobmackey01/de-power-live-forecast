"""Research library for the v2 candidate: level-free features and a pseudo-live replay.

Nothing here is a result. The replay window (2026-08-03 onward) is development
data: it was looked at while the candidates were being designed, so numbers from
it motivate a design and cannot stand as evidence for it. The only evidence v2
will have is the live record it seals after its pre-registration is frozen.

Design, in one paragraph. v1 predicted price minus last week's price, but it
also saw raw price levels and learned that a high price falls back to the level
of its training years. v2 removes the level: the target is the change from the
same local hour yesterday, every price feature is a difference, and (in the
scaled variants) both are divided by a trailing price level so that a change
means the same thing at 90 EUR/MWh as at 250. All lags are on the local
calendar, never a UTC offset, so 23- and 25-hour days need no special case
downstream.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from de_power_live.features import german_public_holidays

REPO = Path(__file__).resolve().parents[2]
TZ = "Europe/Berlin"

# v2 seals at the live window, and trains at the same hour. v1 trained at 11:00
# and sealed between 07:30 and 11:22.
SEAL_LOCAL = pd.Timedelta(hours=7, minutes=30)
# SMARD outturns arrive roughly 1.5h late. The outturn window closes this long
# before the seal so that training never averages over hours that would not yet
# have been published live.
OUTTURN_PUBLICATION_LAG = pd.Timedelta(hours=2)

SCALE_FLOOR = 30.0  # EUR/MWh; keeps the scaled target finite in a near-zero week
SCALE_DAYS = 7

WEATHER_COLS = [
    "wind_cf", "wind_speed_100m", "wind_gust_10m", "solar_radiation", "solar_direct",
    "cloud_cover", "temperature", "surface_pressure", "wind_cf_ramp", "solar_radiation_ramp",
]
DELTA_COLS = ["wind_cf", "solar_radiation", "temperature", "cloud_cover"]


def load_history() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    paths = sorted((REPO / "cache").glob("history_2.3y_*.pkl"))
    if not paths:
        raise FileNotFoundError("run research/v2/fetch.py first")
    with paths[-1].open("rb") as handle:
        prices, weather, actuals, _ = pickle.load(handle)
    return prices, weather, actuals


# ---- The local calendar ---------------------------------------------------


def local_grid(series: pd.Series) -> pd.DataFrame:
    """Hourly UTC series -> one row per local day, 24 columns of wall-clock hours.

    A 25-hour day has two 02:00s: the slot holds their mean. A 23-hour day has
    no 02:00: the slot holds the mean of 01:00 and 03:00. Every lag taken from
    this grid is therefore "the same wall-clock hour k days earlier", which is
    the rule the 2026-10-25 miss showed v1 needed.
    """
    local = series.index.tz_convert(TZ)
    frame = pd.DataFrame(
        {"day": local.normalize().tz_localize(None), "hour": local.hour, "v": series.to_numpy(dtype=float)}
    )
    grid = frame.groupby(["day", "hour"])["v"].mean().unstack("hour").reindex(columns=range(24))
    gap = grid[2].isna() & grid[1].notna() & grid[3].notna()
    grid.loc[gap, 2] = (grid.loc[gap, 1] + grid.loc[gap, 3]) / 2.0
    full = pd.date_range(grid.index.min(), grid.index.max(), freq="D")
    return grid.reindex(full)


def stack_grid(grid: pd.DataFrame, name: str) -> pd.Series:
    out = grid.stack(future_stack=True)
    out.index.names = ["day", "hour"]
    return out.rename(name)


def calendar_flags(days: pd.DatetimeIndex) -> pd.DataFrame:
    holidays: set[pd.Timestamp] = set()
    for year in sorted(set(days.year)):
        holidays |= german_public_holidays(int(year))
    frame = pd.DataFrame(index=days)
    frame["dow"] = days.dayofweek
    frame["is_weekend"] = (frame["dow"] >= 5).astype(int)
    frame["is_holiday"] = days.isin(list(holidays)).astype(int)
    # 24 and 31 December are working days on paper and holidays on the grid, and
    # the days between behave like neither. v1 had no way to say so.
    frame["is_year_end"] = (
        ((days.month == 12) & (days.day >= 24)) | ((days.month == 1) & (days.day <= 1))
    ).astype(int)
    frame["is_non_working"] = ((frame["is_weekend"] == 1) | (frame["is_holiday"] == 1)).astype(int)
    return frame


# ---- Features ---------------------------------------------------------------


@dataclass
class Dataset:
    X: pd.DataFrame  # index: (target day, hour); level-free features
    y: pd.Series  # realised price, EUR/MWh
    anchor: pd.Series  # same wall-clock hour on the seal day
    scale: pd.Series  # trailing price level known at the seal
    d7: pd.Series  # same wall-clock hour, same weekday last week


def build_dataset(prices: pd.Series, weather: pd.DataFrame, actuals: pd.DataFrame) -> Dataset:
    """Features for every target day, each using only what its seal could see.

    For target day T sealed on S = T-1: prices for delivery days <= S (day S
    cleared at noon on S-1), the weather forecast for S and T, and outturns
    published before the seal. Row (T, h) never reads a price from day T.
    """
    P = local_grid(prices)
    days = P.index

    def lag(grid: pd.DataFrame, k: int) -> pd.DataFrame:
        return grid.shift(k, freq="D").reindex(days)

    anchor = lag(P, 1)  # day S
    day_mean = P.mean(axis=1)
    day_min, day_max = P.min(axis=1), P.max(axis=1)

    # Trailing level through day S: the scale a change is measured against.
    level = day_mean.rolling(SCALE_DAYS, min_periods=SCALE_DAYS).mean().shift(1)
    scale = level.clip(lower=SCALE_FLOOR)

    same_hour = np.stack([lag(P, k).to_numpy() for k in range(1, 8)])
    same_hour_mean = pd.DataFrame(np.nanmean(same_hour, axis=0), index=days, columns=P.columns)
    same_hour_std = pd.DataFrame(np.nanstd(same_hour, axis=0), index=days, columns=P.columns)

    parts: dict[str, pd.Series] = {}

    def add(name: str, grid: pd.DataFrame) -> None:
        parts[name] = stack_grid(grid, name)

    def per_day(name: str, series: pd.Series) -> None:
        add(name, pd.DataFrame(np.repeat(series.to_numpy()[:, None], 24, axis=1), index=days, columns=P.columns))

    # Price history, all as differences from the anchor or from a day mean.
    add("d_lag2", lag(P, 2) - anchor)
    add("d_lag3", lag(P, 3) - anchor)
    add("d_lag7", lag(P, 7) - anchor)  # same weekday as the target
    add("d_lag8", lag(P, 8) - lag(P, 7))  # what last week's same transition did
    add("d_lag14", lag(P, 14) - anchor)
    add("d_same_hour_mean_7d", same_hour_mean - anchor)
    add("same_hour_std_7d", same_hour_std)
    add("anchor_vs_day_mean", anchor.sub(day_mean.shift(1), axis=0))
    add("lag7_vs_day_mean", lag(P, 7).sub(day_mean.shift(7), axis=0))
    per_day("day_mean_change_1d", day_mean.shift(1) - day_mean.shift(2))
    per_day("day_mean_vs_level", day_mean.shift(1) - level)
    per_day("day_range_lag1", (day_max - day_min).shift(1))
    per_day("day_range_lag7", (day_max - day_min).shift(7))
    per_day("day_mean_change_lag7", day_mean.shift(7) - day_mean.shift(8))

    price_cols = list(parts)

    # Forward weather for the target day, and its change from the seal day: the
    # target is a change, so the change in its drivers is the natural input.
    for col in WEATHER_COLS:
        grid = local_grid(weather[col])
        grid = grid.reindex(days)
        add(col, grid)
        if col in DELTA_COLS:
            add(f"d_{col}", grid - lag(grid, 1))
            per_day(f"{col}_day_mean", grid.mean(axis=1))
            per_day(f"d_{col}_day_mean", grid.mean(axis=1) - grid.mean(axis=1).shift(1))
    temp = local_grid(weather["temperature"]).reindex(days)
    add("heating_degrees", (15.0 - temp).clip(lower=0.0))
    add("d_heating_degrees", (15.0 - temp).clip(lower=0.0) - (15.0 - lag(temp, 1)).clip(lower=0.0))
    add("cooling_degrees", (temp - 20.0).clip(lower=0.0))

    # Settled outturn: 24h mean closing before the seal, less the publication lag.
    for src, dest in (
        ("wind_total_actual", "wind_actual_pre_seal"),
        ("solar_actual", "solar_actual_pre_seal"),
        ("load_actual", "load_actual_pre_seal"),
    ):
        hourly = actuals[src]
        values = []
        for day in days:
            seal = ((day - pd.Timedelta(days=1)).tz_localize(TZ) + SEAL_LOCAL).tz_convert("UTC")
            end = seal - OUTTURN_PUBLICATION_LAG
            window = hourly.loc[(hourly.index >= end - pd.Timedelta(hours=24)) & (hourly.index < end)]
            values.append(float(window.mean()) if window.notna().sum() >= 12 else np.nan)
        per_day(dest, pd.Series(values, index=days))

    # Calendar, for the target day and for the seal day it is a change from.
    cal_t = calendar_flags(days)
    cal_s = calendar_flags(days - pd.Timedelta(days=1))
    cal_s.index = days
    for col in ("dow", "is_weekend", "is_holiday", "is_year_end", "is_non_working"):
        per_day(col, cal_t[col])
    per_day("seal_day_non_working", cal_s["is_non_working"])
    per_day("seal_day_year_end", cal_s["is_year_end"])
    doy = pd.Series(days.dayofyear, index=days)
    per_day("doy_sin", np.sin(2 * np.pi * doy / 365.25))
    per_day("doy_cos", np.cos(2 * np.pi * doy / 365.25))

    X = pd.concat(parts.values(), axis=1)
    hour = X.index.get_level_values("hour").to_numpy()
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X.attrs["price_cols"] = price_cols

    return Dataset(
        X=X,
        y=stack_grid(P, "price"),
        anchor=stack_grid(anchor, "anchor"),
        scale=stack_grid(pd.DataFrame(np.repeat(scale.to_numpy()[:, None], 24, axis=1), index=days, columns=P.columns), "scale"),
        d7=stack_grid(lag(P, 7), "d7"),
    )


# ---- Model -------------------------------------------------------------------

PRICE_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_lambda=1.0, objective="reg:absoluteerror", random_state=20261110,
    n_jobs=1,  # one thread, so a refit can be rebuilt bit for bit
)


def day_mask(index: pd.MultiIndex, first: pd.Timestamp | None, last: pd.Timestamp) -> np.ndarray:
    day = index.get_level_values("day")
    mask = day <= last
    if first is not None:
        mask &= day >= first
    return np.asarray(mask)


def design(ds: Dataset, scaled: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Model matrix and target. Scaled: price differences in units of the level."""
    X = ds.X.copy()
    target = ds.y - ds.anchor
    if scaled:
        for col in ds.X.attrs["price_cols"]:
            X[col] = X[col] / ds.scale
        target = target / ds.scale
    return X, target


def fit_predict(
    ds: Dataset,
    scaled: bool,
    train_last: pd.Timestamp,
    predict_days: pd.DatetimeIndex,
    train_first: pd.Timestamp | None = None,
    params: dict | None = None,
) -> pd.Series:
    """Fit on target days <= train_last, predict price for ``predict_days``."""
    X, target = design(ds, scaled)
    ok = target.notna() & ds.anchor.notna() & ds.scale.notna()
    train = ok.to_numpy() & day_mask(X.index, train_first, train_last)
    model = XGBRegressor(**(params or PRICE_PARAMS)).fit(X[train], target[train])

    rows = X.index.get_level_values("day").isin(predict_days) & (ds.anchor.notna() & ds.scale.notna()).to_numpy()
    change = pd.Series(model.predict(X[rows]), index=X.index[rows])
    if scaled:
        change = change * ds.scale[rows]
    return (ds.anchor[rows] + change).rename("pred")


def replay(
    ds: Dataset,
    scaled: bool,
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
    refit_every: int | None,
    train_window_days: int | None = None,
    params: dict | None = None,
) -> pd.Series:
    """Walk forward over target days, refitting every ``refit_every`` days.

    A model that predicts target day T was fitted on target days <= T - 2: at
    the seal on T-1, the newest realised day is T-1 itself, whose prices were
    published at noon on T-2 and are therefore known, but the refit is run the
    evening before the seal, when T-1 is the last complete day and T-2 the last
    day whose own forecast error is settled. Using T-2 keeps the replay on the
    safe side of that line. ``refit_every=None`` fits once, before ``first_day``.
    """
    days = pd.date_range(first_day, last_day, freq="D")
    step = refit_every or len(days)
    out = []
    for start in range(0, len(days), step):
        block = days[start : start + step]
        train_last = block[0] - pd.Timedelta(days=2)
        train_first = None if train_window_days is None else train_last - pd.Timedelta(days=train_window_days)
        out.append(fit_predict(ds, scaled, train_last, block, train_first, params))
    return pd.concat(out)


# ---- Scoring -------------------------------------------------------------------


def score_table(frame: pd.DataFrame, models: list[str], by: pd.Series | None = None) -> pd.DataFrame:
    """Pooled MAE per column and skill against yesterday (d1) and last week (d7)."""
    def one(group: pd.DataFrame) -> pd.Series:
        row = {"hours": len(group)}
        mae = {m: float((group[m] - group["y"]).abs().mean()) for m in ["d1", "d7", *models]}
        row.update({f"mae_{m}": mae[m] for m in mae})
        for m in models:
            row[f"skill_d1_{m}"] = 1 - mae[m] / mae["d1"]
        return pd.Series(row)

    if by is None:
        return one(frame).to_frame().T
    table = frame.groupby(by).apply(one)
    table.loc["all"] = one(frame)
    return table


def block_bootstrap_skill(frame: pd.DataFrame, model: str, base: str, draws: int = 5000, seed: int = 1) -> tuple[float, float]:
    """95% CI for pooled MAE skill, resampling whole delivery days."""
    day = frame.index.get_level_values("day")
    err_m = (frame[model] - frame["y"]).abs().groupby(day).sum().to_numpy()
    err_b = (frame[base] - frame["y"]).abs().groupby(day).sum().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err_m), size=(draws, len(err_m)))
    skill = 1 - err_m[idx].sum(axis=1) / err_b[idx].sum(axis=1)
    return float(np.quantile(skill, 0.025)), float(np.quantile(skill, 0.975))
