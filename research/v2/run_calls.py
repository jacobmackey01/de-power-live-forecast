"""Intervals (call C) and event calls (call B) on top of the scaled v2 centre.

    python research/v2/run_calls.py

Everything is walk-forward: the interval for target day T is built from the
forecast errors of days T-N .. T-1, all of which are known at the seal on T-1
(day T-1 cleared at noon on T-2). The centre is replayed from 2026-06-01 with a
weekly refit so that the errors used to warm up the first days of the window are
themselves out of sample.

Development data, like the price replay: these numbers choose a design, they do
not test it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
import lib  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40)
OUT = Path(__file__).parent / "out"
WARM = pd.Timestamp("2026-06-01")
FIRST = pd.Timestamp("2026-08-03")
BLOCKS = {**{h: 0 for h in range(0, 6)}, **{h: 1 for h in range(6, 10)}, **{h: 2 for h in range(10, 16)}, **{h: 3 for h in range(16, 24)}}
LEVELS = (0.10, 0.25, 0.75, 0.90)


def month_of(index: pd.MultiIndex) -> pd.Series:
    return pd.Series(index.get_level_values("day").strftime("%Y-%m"), index=index)


def intervals(frame: pd.DataFrame, n_days: int, by_block: bool) -> pd.DataFrame:
    """Rolling empirical quantiles of scaled errors -> price quantiles per hour."""
    day = frame.index.get_level_values("day")
    hour = frame.index.get_level_values("hour")
    block = np.array([BLOCKS[h] for h in hour]) if by_block else np.zeros(len(frame), dtype=int)
    r = ((frame["y"] - frame["pred"]) / frame["scale"]).to_numpy()
    out = np.full((len(frame), len(LEVELS)), np.nan)
    for target in pd.date_range(FIRST, day.max(), freq="D"):
        past = (day >= target - pd.Timedelta(days=n_days)) & (day < target) & ~np.isnan(r)
        rows = day == target
        for b in np.unique(block[rows]):
            pool = r[past & (block == b)]
            if len(pool) < 40:
                continue
            sel = rows & (block == b)
            out[sel] = np.quantile(pool, LEVELS)[None, :]
    q = frame["pred"].to_numpy()[:, None] + frame["scale"].to_numpy()[:, None] * out
    return pd.DataFrame(q, index=frame.index, columns=[f"q{int(l * 100)}" for l in LEVELS])


def coverage_table(frame: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    ok = q["q10"].notna() & frame["y"].notna() & (frame.index.get_level_values("day") >= FIRST)
    y, q = frame.loc[ok, "y"], q[ok]
    stats = pd.DataFrame({
        "cov80": (y >= q["q10"]) & (y <= q["q90"]),
        "cov50": (y >= q["q25"]) & (y <= q["q75"]),
        "width80": q["q90"] - q["q10"],
        "above_q90": y > q["q90"],
        "below_q10": y < q["q10"],
    }).astype(float)
    table = stats.groupby(month_of(stats.index)).mean()
    table.loc["all"] = stats.mean()
    return table.round(3)


def pr_auc_with_ci(y: np.ndarray, score: np.ndarray, base: np.ndarray, day: np.ndarray, draws: int = 2000) -> str:
    """Average precision of score and baseline, and a day-block CI on the gap."""
    ap_m, ap_b = average_precision_score(y, score), average_precision_score(y, base)
    days = np.unique(day)
    groups = [np.flatnonzero(day == d) for d in days]
    rng = np.random.default_rng(7)
    diffs = []
    for _ in range(draws):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))])
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], score[idx]) - average_precision_score(y[idx], base[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return f"model {ap_m:.3f}  baseline {ap_b:.3f}  diff {ap_m - ap_b:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main() -> None:
    prices, weather, actuals = lib.load_history()
    ds = lib.build_dataset(prices, weather, actuals)
    full = ds.y.groupby(level="day").count()
    last = full[full >= 23].index.max()

    frame = pd.DataFrame({"y": ds.y, "d1": ds.anchor, "d7": ds.d7, "scale": ds.scale})
    frame = frame[lib.day_mask(frame.index, WARM - pd.Timedelta(days=60), last)]
    frame["pred"] = lib.replay(ds, scaled=True, first_day=WARM, last_day=last, refit_every=7)
    frame.to_csv(OUT / "calls_replay.csv")
    window = frame[frame.index.get_level_values("day") >= FIRST].dropna(subset=["y", "pred"])
    print("centre: scaled, weekly refit. window MAE %.2f vs yesterday %.2f" % (
        (window["pred"] - window["y"]).abs().mean(), (window["d1"] - window["y"]).abs().mean()))

    # ---- Call C ------------------------------------------------------------
    print("\n=== Call C: rolling intervals from the last N days of scaled errors ===")
    chosen = None
    for n_days in (14, 28, 56):
        for by_block in (False, True):
            q = intervals(frame, n_days, by_block)
            table = coverage_table(frame, q)
            print(f"\nN={n_days} days, {'by time-of-day block' if by_block else 'pooled'}")
            print(table.to_string())
            if n_days == 28 and by_block:
                chosen = q
    # Same idea with yesterday's price as the centre, to show how much of the
    # interval quality comes from the centre.
    naive = frame.assign(pred=frame["d1"])
    print("\nN=28 by block, centred on yesterday's price instead of v2")
    print(coverage_table(naive, intervals(naive, 28, True)).to_string())

    # ---- Call B ------------------------------------------------------------
    print("\n=== Call B candidates ===")
    day_all = frame.index.get_level_values("day")
    hour_all = frame.index.get_level_values("hour")
    r_all = ((frame["y"] - frame["pred"]) / frame["scale"]).to_numpy()
    y_all = frame["y"].to_numpy()
    grid = frame["y"].unstack("hour")

    rows = []
    for target in pd.date_range(FIRST, last, freq="D"):
        sel = np.flatnonzero((day_all == target) & frame["pred"].notna().to_numpy() & frame["y"].notna().to_numpy())
        if len(sel) < 23:
            continue
        past28 = (day_all >= target - pd.Timedelta(days=28)) & (day_all < target)
        pool = r_all[(day_all >= target - pd.Timedelta(days=56)) & (day_all < target) & ~np.isnan(r_all)]
        hist = y_all[past28 & ~np.isnan(y_all)]
        spike_thr, cheap_thr = np.quantile(hist, 0.90), np.quantile(hist, 0.10)
        pred, scale, truth = frame["pred"].to_numpy()[sel], frame["scale"].to_numpy()[sel], y_all[sel]
        sims = pred[:, None] + scale[:, None] * pool[None, :]  # hours x scenarios
        past_grid = grid.loc[target - pd.Timedelta(days=28) : target - pd.Timedelta(days=1)]
        hours = hour_all[sel]
        d1 = frame["d1"].to_numpy()[sel]
        # 3 cheapest hours of the day: rank within each simulated day, using the
        # same error draw for every hour is wrong (errors are per hour), so rank
        # on the point forecast and convert to a smooth score.
        rank_pred = pd.Series(pred).rank(method="first").to_numpy()
        rank_d1 = pd.Series(d1).rank(method="first").to_numpy()
        rank_true = pd.Series(truth).rank(method="first").to_numpy()
        cheap_clim = past_grid.rank(axis=1, method="first").le(3).mean(axis=0).reindex(hours).to_numpy()
        for i in range(len(sel)):
            rows.append({
                "day": target, "hour": hours[i],
                "spike": truth[i] > spike_thr, "p_spike": (sims[i] > spike_thr).mean(),
                "spike_clim": (past_grid[hours[i]] > spike_thr).mean(), "spike_d1": float(d1[i] > spike_thr),
                "low": truth[i] < cheap_thr, "p_low": (sims[i] < cheap_thr).mean(),
                "low_clim": (past_grid[hours[i]] < cheap_thr).mean(), "low_d1": float(d1[i] < cheap_thr),
                "cheap3": rank_true[i] <= 3, "s_cheap3": -rank_pred[i], "cheap3_d1": -rank_d1[i], "cheap3_clim": cheap_clim[i],
                "neg": truth[i] < 0, "below10": truth[i] < 10,
            })
    ev = pd.DataFrame(rows)
    day = ev["day"].to_numpy()
    month = ev["day"].dt.strftime("%Y-%m")
    print("events by month:\n", ev.groupby(month)[["spike", "low", "cheap3", "neg", "below10"]].sum().to_string())
    for label, y, score, bases in (
        ("spike: price above the trailing 28-day 90th percentile", "spike", "p_spike", {"hour-of-day frequency, last 28 days": "spike_clim", "yesterday same hour": "spike_d1"}),
        ("low: price below the trailing 28-day 10th percentile", "low", "p_low", {"hour-of-day frequency, last 28 days": "low_clim", "yesterday same hour": "low_d1"}),
        ("3 cheapest hours of the day", "cheap3", "s_cheap3", {"hour-of-day frequency, last 28 days": "cheap3_clim", "yesterday's ranking": "cheap3_d1"}),
    ):
        print(f"\n{label}: {int(ev[y].sum())} events in {ev['day'].nunique()} days")
        for name, col in bases.items():
            print(f"  vs {name:38s} {pr_auc_with_ci(ev[y].to_numpy().astype(int), ev[score].to_numpy(), ev[col].to_numpy(), day)}")
    # Reliability of the spike probability.
    bins = pd.cut(ev["p_spike"], [-0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
    print("\nspike probability reliability:\n", ev.groupby(bins, observed=True).agg(hours=("spike", "size"), forecast=("p_spike", "mean"), observed=("spike", "mean")).round(3).to_string())
    bins = pd.cut(ev["p_low"], [-0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
    print("\nlow probability reliability:\n", ev.groupby(bins, observed=True).agg(hours=("low", "size"), forecast=("p_low", "mean"), observed=("low", "mean")).round(3).to_string())

    # How often did the phase-1 targets occur in past winters?
    p = ds.y.dropna()
    d = p.index.get_level_values("day")
    for a, b in (("2024-11-21", "2025-02-28"), ("2025-11-21", "2026-02-28")):
        s = p[(d >= a) & (d <= b)]
        print(f"\n{a}..{b}: {len(s)} hours, mean {s.mean():.0f}, below 0: {(s < 0).sum()}, below 10: {(s < 10).sum()}, below 30: {(s < 30).sum()}")


if __name__ == "__main__":
    main()
