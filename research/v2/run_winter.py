"""The same v2 design over last winter, the calendar span phase 2 will cover.

    python research/v2/run_winter.py

A check on a period the design was not built on: the features and variants were
fixed from the Aug-Nov 2026 replay and are run here once, unchanged. Less
training history than the live model will have (from Aug 2024 only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import lib  # noqa: E402
import run_calls  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40)
FIRST, LAST = pd.Timestamp("2025-11-21"), pd.Timestamp("2026-02-28")
WARM = pd.Timestamp("2025-09-20")


def main() -> None:
    prices, weather, actuals = lib.load_history()
    ds = lib.build_dataset(prices, weather, actuals)
    frame = pd.DataFrame({"y": ds.y, "d1": ds.anchor, "d7": ds.d7, "scale": ds.scale})
    frame = frame[lib.day_mask(frame.index, WARM, LAST)]

    frame["scaled_weekly"] = lib.replay(ds, scaled=True, first_day=WARM, last_day=LAST, refit_every=7)
    frozen = lib.fit_predict(ds, True, FIRST - pd.Timedelta(days=2), pd.date_range(FIRST, LAST, freq="D"))
    frame["scaled_frozen"] = frozen
    frame["add_weekly"] = lib.replay(ds, scaled=False, first_day=FIRST, last_day=LAST, refit_every=7)

    models = ["scaled_weekly", "scaled_frozen", "add_weekly"]
    window = frame[frame.index.get_level_values("day") >= FIRST].dropna(subset=["y", "d1", "d7", *models])
    month = pd.Series(window.index.get_level_values("day").strftime("%Y-%m"), index=window.index)
    print(f"== Last winter, {window.index.get_level_values('day').nunique()} days ==")
    print(lib.score_table(window, models, month).round(3).T.to_string())
    for m in models:
        lo, hi = lib.block_bootstrap_skill(window, m, "d1")
        skill = 1 - (window[m] - window["y"]).abs().mean() / (window["d1"] - window["y"]).abs().mean()
        print(f"  {m:15s} skill vs yesterday {skill:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    run_calls.FIRST = FIRST
    centre = frame.assign(pred=frame["scaled_weekly"])
    for n_days in (28, 56):
        print(f"\nintervals, N={n_days} pooled")
        print(run_calls.coverage_table(centre, run_calls.intervals(centre, n_days, False)).to_string())

    # Spike call, same definition as in run_calls.
    day_all = centre.index.get_level_values("day")
    r_all = ((centre["y"] - centre["pred"]) / centre["scale"]).to_numpy()
    y_all = centre["y"].to_numpy()
    grid = centre["y"].unstack("hour")
    rows = []
    for target in pd.date_range(FIRST, LAST, freq="D"):
        sel = np.flatnonzero((day_all == target) & centre["pred"].notna().to_numpy() & ~np.isnan(y_all))
        if len(sel) < 23:
            continue
        hist = y_all[(day_all >= target - pd.Timedelta(days=28)) & (day_all < target) & ~np.isnan(y_all)]
        pool = r_all[(day_all >= target - pd.Timedelta(days=56)) & (day_all < target) & ~np.isnan(r_all)]
        thr = np.quantile(hist, 0.90)
        past_grid = grid.loc[target - pd.Timedelta(days=28) : target - pd.Timedelta(days=1)]
        hours = centre.index.get_level_values("hour")[sel]
        pred, scale = centre["pred"].to_numpy()[sel], centre["scale"].to_numpy()[sel]
        for i in range(len(sel)):
            rows.append({"day": target, "spike": y_all[sel][i] > thr,
                         "p": ((pred[i] + scale[i] * pool) > thr).mean(),
                         "clim": (past_grid[hours[i]] > thr).mean()})
    ev = pd.DataFrame(rows)
    print(f"\nspike call: {int(ev['spike'].sum())} events in {ev['day'].nunique()} days, by month:",
          ev.groupby(ev['day'].dt.strftime('%Y-%m'))['spike'].sum().to_dict())
    print("  ", run_calls.pr_auc_with_ci(ev["spike"].to_numpy().astype(int), ev["p"].to_numpy(), ev["clim"].to_numpy(), ev["day"].to_numpy()))


if __name__ == "__main__":
    main()
