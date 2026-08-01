"""Fit and freeze a model version.

Reports a walk-forward sanity check before freezing. That check is **not** a
result: it is a backtest, it is exactly the thing this project exists to stop
relying on, and it appears nowhere in the pre-registered claims. Its only job is
to catch a model that is broken before it starts making live calls.

    python -m de_power_live.train --version v1 --years 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import build_training_set, fetch_history
from .model import DayAheadModel

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def walk_forward_check(
    features: pd.DataFrame,
    prices: pd.Series,
    n_folds: int = 4,
    fold_days: int = 30,
) -> dict:
    """Expanding-window check against the pre-registered B1 baseline.

    Chronological only - no random splits. A random split on time series leaks
    the future into the training set through the lag features and produces
    numbers that cannot be reproduced live.
    """
    days = pd.Index(features.index.tz_convert("Europe/Berlin").date).unique()
    days = pd.DatetimeIndex(sorted(days))
    if len(days) < fold_days * (n_folds + 2):
        return {"skipped": f"only {len(days)} days available"}

    folds = []
    for k in range(n_folds, 0, -1):
        test_end = days[-1] - pd.Timedelta(days=fold_days * (k - 1))
        test_start = test_end - pd.Timedelta(days=fold_days - 1)

        local_day = pd.DatetimeIndex(features.index.tz_convert("Europe/Berlin").date)
        train_mask = local_day < test_start
        test_mask = (local_day >= test_start) & (local_day <= test_end)
        if train_mask.sum() < 24 * 90 or test_mask.sum() == 0:
            continue

        model = DayAheadModel().fit(features[train_mask], prices[train_mask])

        test_features = features[test_mask]
        actual = prices[test_mask]
        usable = actual.notna() & test_features["price_lag_168"].notna()
        if usable.sum() == 0:
            continue
        test_features, actual = test_features[usable], actual[usable]

        pred = model.predict(test_features)
        baseline = test_features["price_lag_168"].to_numpy(dtype=float)
        truth = actual.to_numpy(dtype=float)

        mae_model = float(np.mean(np.abs(pred.price - truth)))
        mae_base = float(np.mean(np.abs(baseline - truth)))

        lo, hi = pred.quantiles[:, 0], pred.quantiles[:, -1]
        inner_lo, inner_hi = pred.quantiles[:, 1], pred.quantiles[:, -2]

        folds.append(
            {
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "n_hours": int(len(truth)),
                "mae_model": round(mae_model, 3),
                "mae_baseline_b1": round(mae_base, 3),
                "skill_vs_b1": round(1 - mae_model / mae_base, 4) if mae_base else None,
                "coverage_80": round(float(np.mean((truth >= lo) & (truth <= hi))), 4),
                "coverage_50": round(float(np.mean((truth >= inner_lo) & (truth <= inner_hi))), 4),
                "negative_hours_in_test": int((truth < 0).sum()),
            }
        )

    if not folds:
        return {"skipped": "no fold had enough training history"}

    return {
        "folds": folds,
        "pooled_skill_vs_b1": round(
            float(np.mean([f["skill_vs_b1"] for f in folds if f["skill_vs_b1"] is not None])), 4
        ),
        "pooled_coverage_80": round(float(np.mean([f["coverage_80"] for f in folds])), 4),
        "pooled_coverage_50": round(float(np.mean([f["coverage_50"] for f in folds])), 4),
        "note": (
            "Backtest only. Not a pre-registered result and not evidence of edge; "
            "it exists to catch a broken model before it goes live."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and freeze a model version")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    print(f"fetching {args.years}y of history ...")
    prices, weather, actuals, provenance = fetch_history(years=args.years)
    print(f"  prices  : {provenance['n_prices']} hours  {provenance['price_span_utc'][0]} -> {provenance['price_span_utc'][1]}")
    print(f"  weather : {provenance['weather_coverage']['n_hours']} hours, "
          f"{provenance['weather_coverage']['sites_returned']}/{provenance['weather_coverage']['sites_requested']} sites")

    first_day = pd.Timestamp(weather.index.min().tz_convert("Europe/Berlin").date()) + pd.Timedelta(days=15)
    last_day = pd.Timestamp(prices.index.max().tz_convert("Europe/Berlin").date()) - pd.Timedelta(days=1)
    print(f"building features {first_day.date()} -> {last_day.date()} ...")

    data = build_training_set(prices, weather, actuals, first_day, last_day)
    print(f"  {data.days_built} days built, {len(data.days_skipped)} skipped, "
          f"{len(data.features)} rows")
    if data.days_skipped:
        reasons: dict[str, int] = {}
        for reason in data.days_skipped.values():
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        print(f"  skip reasons: {reasons}")

    check: dict = {"skipped": "requested"}
    if not args.skip_check:
        print("walk-forward sanity check (backtest, not a result) ...")
        check = walk_forward_check(data.features, data.prices)
        for fold in check.get("folds", []):
            print(f"  {fold['test_start']}..{fold['test_end']}  "
                  f"MAE {fold['mae_model']:>7.2f} vs B1 {fold['mae_baseline_b1']:>7.2f}  "
                  f"skill {fold['skill_vs_b1']:+.3f}  "
                  f"cov80 {fold['coverage_80']:.2f}  neg {fold['negative_hours_in_test']}")
        if "pooled_skill_vs_b1" in check:
            print(f"  pooled skill vs B1: {check['pooled_skill_vs_b1']:+.4f}   "
                  f"cov80 {check['pooled_coverage_80']:.3f}  cov50 {check['pooled_coverage_50']:.3f}")

    print("fitting final model on all data ...")
    model = DayAheadModel(version=args.version).fit(data.features, data.prices)

    out = MODELS_DIR / args.version
    manifest = model.save(out)
    manifest["training_days"] = data.days_built
    manifest["days_skipped"] = len(data.days_skipped)
    manifest["walk_forward_check"] = check
    manifest["provenance"] = provenance
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nfrozen -> {out}")
    print(f"  source_sha256    : {manifest['source_sha256'][:16]}...")
    print(f"  training rows    : {manifest['training_rows']}")
    print(f"  negative base    : {manifest['negative_base_rate']:.5f}")
    print(f"  negative fitted  : {manifest['negative_model_fitted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
