"""Fit and freeze model v2, with the warm-start errors its intervals begin from.

    python -m de_power_live.v2.train --train-last 2026-11-18

Two fits. The frozen model is fitted on every target day up to ``--train-last``.
Its intervals need forecast errors from the days before the first live seal,
and a model's errors on its own training days are too small to stand in for
them. So a second, earlier model is fitted on data ending before the warm-start
days and its out-of-sample errors on those days are frozen alongside.

The replay figures printed here are a sanity check that the model is not
broken. They are not a result and appear in no pre-registered claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import fetch_history
from ..model import build_climatology
from . import MODELS_DIR, PREDICTIONS_DIR
from .calls import DESCRIPTIVE_THRESHOLDS, POOL_DAYS
from .features import Dataset, build_dataset
from .model import WARM_START, V2Model

# The warm-start model stops this many days before the first warm-start day.
WARM_GAP_DAYS = 2


def warm_start_errors(ds: Dataset, train_last: pd.Timestamp, params: dict | None = None) -> dict:
    """Out-of-sample scaled errors for the POOL_DAYS days ending at ``train_last``."""
    warm_days = pd.date_range(train_last - pd.Timedelta(days=POOL_DAYS - 1), train_last, freq="D")
    earlier = V2Model(**({"params": params} if params else {})).fit(
        ds, warm_days[0] - pd.Timedelta(days=WARM_GAP_DAYS)
    )
    pred = earlier.predict(ds, warm_days)
    scaled = ((ds.y - pred) / ds.scale).reindex(pred.index).dropna()
    return {
        "note": (
            "Scaled forecast errors of a model fitted on target days up to "
            f"{(warm_days[0] - pd.Timedelta(days=WARM_GAP_DAYS)).date()}, on the days below. "
            "Used for intervals until v2 has live errors of its own."
        ),
        "fitted_through": str((warm_days[0] - pd.Timedelta(days=WARM_GAP_DAYS)).date()),
        "scaled_errors": {
            str(day.date()): [round(float(v), 6) for v in scaled.xs(day, level="day").to_numpy()]
            for day in warm_days
            if day in scaled.index.get_level_values("day")
        },
    }


def freeze(
    prices: pd.Series,
    weather: pd.DataFrame,
    train_last: pd.Timestamp,
    out: Path,
    params: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    train_last = pd.Timestamp(train_last).normalize()
    ds = build_dataset(prices, weather)
    if ds.y.xs(train_last, level="day").notna().sum() < 23:
        raise ValueError(f"prices for {train_last.date()} are not complete; cannot train through it")

    out.mkdir(parents=True, exist_ok=True)
    warm = warm_start_errors(ds, train_last, params)
    (out / WARM_START).write_text(json.dumps(warm, indent=2), encoding="utf-8")

    model = V2Model(**({"params": params} if params else {})).fit(ds, train_last)

    known = prices[prices.index.tz_convert("Europe/Berlin").normalize().tz_localize(None) <= train_last]
    pooled = np.concatenate([np.array(v) for v in warm["scaled_errors"].values()])
    extra = {
        "train_last": str(train_last.date()),
        "warm_start": {
            "days": list(warm["scaled_errors"]),
            "fitted_through": warm["fitted_through"],
            "n_errors": int(len(pooled)),
            "mean_abs_scaled_error": round(float(np.mean(np.abs(pooled))), 5),
        },
        "climatology": {
            name: build_climatology(known, threshold) for name, threshold in DESCRIPTIVE_THRESHOLDS.items()
        },
        "provenance": provenance or {},
        "note": "Frozen. Any change to a hashed source or artefact is a new model version.",
    }
    return model.save(out, extra)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and freeze model v2")
    parser.add_argument("--train-last", required=True, help="last target day to train on, YYYY-MM-DD")
    parser.add_argument("--years", type=float, default=2.3)
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    if (args.out / "MANIFEST.json").exists():
        sealed = sorted(PREDICTIONS_DIR.glob("????-??-??.json")) if PREDICTIONS_DIR.exists() else []
        if sealed and args.out == MODELS_DIR:
            raise SystemExit(
                f"models/v2 is frozen and {len(sealed)} v2 prediction(s) exist. "
                "Train a new version instead of replacing this one."
            )

    prices, weather, _, provenance = fetch_history(years=args.years)
    provenance = {k: v for k, v in provenance.items() if k != "identity_check"}
    manifest = freeze(prices, weather, pd.Timestamp(args.train_last), args.out, provenance=provenance)
    print(f"frozen -> {args.out}")
    print(f"  trained on target days {manifest['training_days'][0]} .. {manifest['training_days'][1]}"
          f" ({manifest['training_rows']} rows)")
    print(f"  warm start: {len(manifest['warm_start']['days'])} days, {manifest['warm_start']['n_errors']} errors")
    print(f"  source {manifest['source_sha256'][:16]}...  artefact {manifest['artefact_sha256']['price.ubj'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
