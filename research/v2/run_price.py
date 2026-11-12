"""Replay the price-level candidates over the live period and score them.

    python research/v2/run_price.py

Writes research/v2/out/price_replay.parquet-free CSV of hourly predictions so the
interval and call B experiments reuse the same replay instead of refitting.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import lib  # noqa: E402

from de_power_live.dataset import build_training_set  # noqa: E402
from de_power_live.model import ANCHOR_FEATURE, PRICE_PARAMS as V1_PARAMS, DayAheadModel  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
FIRST = pd.Timestamp("2026-08-03")


def to_grid_index(index_utc: pd.DatetimeIndex) -> pd.MultiIndex:
    local = index_utc.tz_convert(lib.TZ)
    return pd.MultiIndex.from_arrays([local.normalize().tz_localize(None), local.hour], names=["day", "hour"])


def live_v1() -> pd.Series:
    rows = {}
    for path in sorted(glob.glob(str(lib.REPO / "predictions" / "2026-*.json"))):
        payload = json.load(open(path))
        day = pd.Timestamp(payload["delivery_date_local"])
        for p in payload["predictions"]:
            hour = pd.Timestamp(p["hour_local"]).hour
            rows[(day, hour)] = p["call_a_price_eur_mwh"]
    out = pd.Series(rows, name="v1_live")
    out.index.names = ["day", "hour"]
    return out


def v1_spec_replays(prices, weather, actuals, last: pd.Timestamp) -> pd.DataFrame:
    """v1's own features and target, (a) frozen at 1 Aug on replay weather and
    (b) refitted weekly. (a) against the live record measures how much the
    replay flatters; (b) shows what refitting alone would have bought."""
    first_day = pd.Timestamp(weather.index.min().tz_convert(lib.TZ).date()) + pd.Timedelta(days=15)
    data = build_training_set(prices, weather, actuals, first_day, last)
    X, y = data.features, data.prices
    idx = to_grid_index(X.index)
    day = idx.get_level_values("day")
    anchor = X[ANCHOR_FEATURE].to_numpy(dtype=float)
    ok = y.notna().to_numpy() & ~np.isnan(anchor)

    frozen = DayAheadModel.load(lib.REPO / "models" / "v1")
    replay_rows = (day >= FIRST) & ~np.isnan(anchor)
    out = pd.DataFrame(index=idx[replay_rows])
    out["v1_frozen_replay"] = frozen.predict(X[replay_rows]).price

    params = dict(V1_PARAMS, n_jobs=1)
    days = pd.date_range(FIRST, last, freq="D")
    preds = []
    for start in range(0, len(days), 7):
        block = days[start : start + 7]
        train = ok & (day <= block[0] - pd.Timedelta(days=2))
        model = XGBRegressor(**params).fit(X[train], (y.to_numpy(dtype=float) - anchor)[train])
        rows = day.isin(block) & ~np.isnan(anchor)
        preds.append(pd.Series(model.predict(X[rows]) + anchor[rows], index=idx[rows]))
    out["v1_spec_weekly"] = pd.concat(preds).groupby(level=[0, 1]).first().reindex(out.index)
    return out.groupby(level=[0, 1]).first()


def main() -> None:
    prices, weather, actuals = lib.load_history()
    t0 = time.time()
    ds = lib.build_dataset(prices, weather, actuals)
    full_days = ds.y.groupby(level="day").count()
    last = full_days[full_days >= 23].index.max()
    print(f"dataset {ds.X.shape}, {time.time() - t0:.0f}s; price days {ds.y.dropna().index[0][0].date()} .. {last.date()}")
    lvl = ds.y.groupby(level="day").mean()
    print("mean price by half-year:\n", lvl.groupby([lvl.index.year, (lvl.index.month - 1) // 6]).agg(["mean", "max"]).round(1).to_string())

    candidates = {
        "add_frozen": dict(scaled=False, refit_every=None),
        "add_weekly": dict(scaled=False, refit_every=7),
        "scaled_frozen": dict(scaled=True, refit_every=None),
        "scaled_weekly": dict(scaled=True, refit_every=7),
        "scaled_weekly_1y": dict(scaled=True, refit_every=7, train_window_days=365),
    }
    frame = pd.DataFrame({"y": ds.y, "d1": ds.anchor, "d7": ds.d7, "scale": ds.scale})
    frame = frame[lib.day_mask(frame.index, FIRST, last)]
    for name, kw in candidates.items():
        t0 = time.time()
        frame[name] = lib.replay(ds, first_day=FIRST, last_day=last, **kw)
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    frame = frame.join(v1_spec_replays(prices, weather, actuals, last))
    frame = frame.join(live_v1())
    print(f"v1 replays: {time.time() - t0:.0f}s")
    frame.to_csv(OUT / "price_replay.csv")

    models = list(candidates) + ["v1_spec_weekly", "v1_frozen_replay"]
    complete = frame.dropna(subset=["y", "d1", "d7", *models])
    month = pd.Series(complete.index.get_level_values("day").strftime("%Y-%m"), index=complete.index)
    print(f"\n== All replay days with every model ({complete.index.get_level_values('day').nunique()} days) ==")
    print(lib.score_table(complete, models, month).round(3).T.to_string())

    live = complete.dropna(subset=["v1_live"])
    month = pd.Series(live.index.get_level_values("day").strftime("%Y-%m"), index=live.index)
    print(f"\n== Days v1 sealed live ({live.index.get_level_values('day').nunique()} days) ==")
    print(lib.score_table(live, models + ["v1_live"], month).round(3).T.to_string())

    print("\nreplay flattery check, v1 frozen on replay weather vs v1 as sealed live:")
    print("  mean |replay - live| = %.2f EUR/MWh" % (live["v1_frozen_replay"] - live["v1_live"]).abs().mean())

    print("\n95% CI of pooled skill vs yesterday (day-block bootstrap), all replay days:")
    for m in models:
        lo, hi = lib.block_bootstrap_skill(complete, m, "d1")
        print(f"  {m:20s} {1 - (complete[m]-complete['y']).abs().mean()/(complete['d1']-complete['y']).abs().mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nbias (forecast minus outturn) by yesterday's price level, EUR/MWh:")
    bins = pd.cut(complete["d1"], [-500, 50, 100, 150, 200, 250, 1000])
    print(complete.groupby(bins, observed=True).apply(
        lambda g: pd.Series({"hours": len(g), **{m: (g[m] - g["y"]).mean() for m in ["scaled_weekly", "add_weekly", "v1_spec_weekly", "v1_frozen_replay"]}})
    ).round(1).to_string())


if __name__ == "__main__":
    main()
