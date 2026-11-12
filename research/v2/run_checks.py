"""Two follow-up checks asked for by the independent audit of 2026-11-12.

    python research/v2/run_checks.py > research/v2/out/followup_checks.log

1. Is a weekly refit different from a single fit? A paired comparison of daily
   errors, not a comparison of two rounded skill figures.
2. What do the outturn inputs add, and what happens if a model trained on them
   is served a blank, as it would be live? The README quoted these figures
   before the experiment was committed; this is the experiment.

Uses the research features in lib.py, which still include the outturn inputs.
The production features do not. Development data, as everything in this folder.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
import lib  # noqa: E402

OUTTURN = ["wind_actual_pre_seal", "solar_actual_pre_seal", "load_actual_pre_seal"]
PERIODS = {
    "3 Aug to 10 Nov 2026": ("2026-08-03", "2026-11-10", "2026-08-01"),
    "21 Nov 2025 to 28 Feb 2026": ("2025-11-21", "2026-02-28", "2025-11-19"),
}


def fit_predict(ds: lib.Dataset, train_last: str, first: str, last: str, blank: list[str] | None = None) -> pd.Series:
    X, target = lib.design(ds, scaled=True)
    ok = (target.notna() & ds.anchor.notna() & ds.scale.notna()).to_numpy()
    day = X.index.get_level_values("day")
    model = XGBRegressor(**lib.PRICE_PARAMS).fit(X[ok & (day <= pd.Timestamp(train_last))], target[ok & (day <= pd.Timestamp(train_last))])
    rows = (day >= pd.Timestamp(first)) & (day <= pd.Timestamp(last)) & (ds.anchor.notna() & ds.scale.notna()).to_numpy()
    served = X[rows].copy()
    if blank:
        served[blank] = np.nan
    return ds.anchor[rows] + pd.Series(model.predict(served), index=X.index[rows]) * ds.scale[rows]


def paired(frame: pd.DataFrame, a: str, b: str, draws: int = 10_000) -> str:
    day = frame.index.get_level_values("day")
    diff = ((frame[a] - frame["y"]).abs() - (frame[b] - frame["y"]).abs()).groupby(day).sum().to_numpy()
    hours = frame.groupby(day).size().to_numpy()
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    lo, hi = np.quantile(diff[idx].sum(axis=1) / hours[idx].sum(axis=1), [0.025, 0.975])
    return f"{diff.sum() / hours.sum():+.3f} EUR/MWh, 95% CI [{lo:+.3f}, {hi:+.3f}], {len(diff)} days"


def skill(frame: pd.DataFrame, col: str) -> float:
    return 1 - (frame[col] - frame["y"]).abs().mean() / (frame["d1"] - frame["y"]).abs().mean()


def main() -> None:
    prices, weather, actuals = lib.load_history()
    ds = lib.build_dataset(prices, weather, actuals)
    without = copy.copy(ds)
    without.X = ds.X.drop(columns=OUTTURN)
    without.X.attrs = dict(ds.X.attrs)

    for label, (first, last, train_last) in PERIODS.items():
        print(f"== {label} (fitted once on target days up to {train_last})")
        frame = pd.DataFrame({"y": ds.y, "d1": ds.anchor})
        frame["with_outturn"] = fit_predict(ds, train_last, first, last)
        frame["trained_with_served_blank"] = fit_predict(ds, train_last, first, last, blank=OUTTURN[:2])
        frame["without_outturn"] = fit_predict(without, train_last, first, last)
        frame["weekly"] = lib.replay(ds, scaled=True, first_day=pd.Timestamp(first), last_day=pd.Timestamp(last), refit_every=7)
        frame = frame.dropna()
        for col in ("with_outturn", "trained_with_served_blank", "without_outturn", "weekly"):
            print(f"  skill vs yesterday, {col:26s} {skill(frame, col):+.3f}   MAE {(frame[col] - frame['y']).abs().mean():.3f}")
        print("  paired MAE difference, weekly refit minus single fit:      ", paired(frame, "weekly", "with_outturn"))
        print("  paired MAE difference, without outturn minus with:          ", paired(frame, "without_outturn", "with_outturn"))
        print("  paired MAE difference, served blank minus served complete:  ", paired(frame, "trained_with_served_blank", "with_outturn"))
        print()
    print("Blank = solar and onshore-wind outturn inputs set to missing at prediction time, which is")
    print("what a live seal would have had: on 2026-11-11 SMARD had published them only up to the")
    print("previous morning. Load was available and is left in place.")


if __name__ == "__main__":
    main()
