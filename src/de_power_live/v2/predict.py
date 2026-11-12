"""Seal a v2 day-ahead forecast. Write-once, and never after gate closure.

    python -m de_power_live.v2.predict            # seal tomorrow
    python -m de_power_live.v2.predict --dry-run  # print without writing

``build_prediction`` takes its inputs and its clock as arguments, so the whole
seal path, leakage guard included, can be rehearsed on recorded or synthetic
inputs for any day. v1 lost 2026-10-25 because nothing ran a 25-hour day
through the real path before the day arrived; tests/test_v2_pipeline.py does
that for every clock-change day through 2027.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import BIDDING_ZONE, MARKET_TZ
from ..dataset import target_hours_for
from ..model import climatology_probability
from ..predict import SealError, auction_close_utc, seal
from . import MODEL_VERSION, MODELS_DIR, PREDICTIONS_DIR
from .calls import (
    DESCRIPTIVE_THRESHOLDS,
    POOL_DAYS,
    QUANTILES,
    error_pool,
    event_probability,
    interval_quantiles,
    spike_baseline,
    spike_threshold,
)
from .features import assert_no_leakage, build_dataset, local_grid, slot_index
from .inputs import LiveInputs, fetch_live_inputs
from .model import V2Model, load_warm_start

SCHEMA_VERSION = 2


def live_errors(
    predictions_dir: Path, prices: pd.Series, target_day: pd.Timestamp
) -> dict[pd.Timestamp, np.ndarray]:
    """Scaled errors of v2's own sealed forecasts in the pool window.

    Each error uses the forecast and the scale exactly as sealed, and the
    realised price for that hour. A sealed day whose prices are not all
    published yet contributes the hours that are.
    """
    out: dict[pd.Timestamp, np.ndarray] = {}
    for back in range(POOL_DAYS, 0, -1):
        day = pd.Timestamp(target_day).normalize() - pd.Timedelta(days=back)
        path = Path(predictions_dir) / f"{day.date()}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scale = float(payload["scale_eur_mwh"])
        hours = pd.DatetimeIndex([pd.Timestamp(p["hour_utc"]) for p in payload["predictions"]])
        pred = np.array([p["call_a_price_eur_mwh"] for p in payload["predictions"]], dtype=float)
        truth = prices.reindex(hours).to_numpy(dtype=float)
        out[day] = (truth - pred) / scale
    return out


def build_prediction(
    inputs: LiveInputs,
    now_utc: pd.Timestamp,
    model_dir: Path = MODELS_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
    target_day: pd.Timestamp | None = None,
    sealed_at_utc: pd.Timestamp | None = None,
) -> dict:
    """Assemble the sealed payload for the local day after ``now_utc``.

    ``now_utc`` is the information cutoff. ``sealed_at_utc`` defaults to the
    real clock when the payload is finished; a rehearsal passes its own.
    """
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    seal_day = now_utc.tz_convert(MARKET_TZ).normalize().tz_localize(None)
    target = seal_day + pd.Timedelta(days=1) if target_day is None else pd.Timestamp(target_day).normalize()

    close = auction_close_utc(target)
    if now_utc >= close:
        raise SealError(
            f"auction for {target.date()} closed at {close.isoformat()}; it is now "
            f"{now_utc.isoformat()}. Refusing to seal a forecast after gate closure."
        )

    leakage = assert_no_leakage(inputs.prices, now_utc, target)

    model = V2Model.load(model_dir)  # verifies artefacts and source against the manifest
    ds = build_dataset(inputs.prices, inputs.weather, last_target_day=target)

    slots = pd.MultiIndex.from_product([[target], range(24)], names=["day", "hour"])
    if not ds.usable().reindex(slots).fillna(False).all():
        raise SealError(
            f"no anchor or scale for some hour of {target.date()}: the seal day's prices or the "
            "seven days before them are incomplete"
        )
    X, _ = ds.design()
    weather_gaps = X.loc[slots, ["wind_cf", "solar_radiation", "temperature"]].isna().sum().sum()
    if weather_gaps:
        raise SealError(f"weather forecast does not cover every hour of {target.date()}")

    pred = model.predict(ds, [target]).reindex(slots)
    scale = ds.scale.reindex(slots)
    anchor = ds.anchor.reindex(slots)
    last_week = ds.last_week.reindex(slots)

    pool, pool_report = error_pool(
        target, live_errors(predictions_dir, inputs.prices, target), load_warm_start(model_dir)
    )
    quantiles = pd.DataFrame(interval_quantiles(pred, scale, pool), index=slots)

    price_grid = local_grid(inputs.prices)
    threshold = spike_threshold(price_grid, target)
    baseline_spike = spike_baseline(price_grid, target, threshold)
    p_spike = pd.Series(event_probability(pred, scale, pool, threshold, above=True), index=slots)
    p_low = {
        name: pd.Series(event_probability(pred, scale, pool, level, above=False), index=slots)
        for name, level in DESCRIPTIVE_THRESHOLDS.items()
    }
    climatology = model.manifest["climatology"]

    hours = target_hours_for(target)
    hour_slots = slot_index(hours)
    sealed_at = (sealed_at_utc or datetime.now(timezone.utc)).isoformat()

    def number(value: float, digits: int = 3):
        return None if pd.isna(value) else round(float(value), digits)

    calls = []
    for utc_ts, local_ts, slot in zip(hours, hours.tz_convert(MARKET_TZ), hour_slots):
        q = quantiles.loc[slot].to_numpy()
        calls.append(
            {
                "hour_utc": utc_ts.isoformat(),
                "hour_local": local_ts.isoformat(),
                "call_a_price_eur_mwh": number(pred.loc[slot]),
                "baseline_same_hour_yesterday": number(anchor.loc[slot], 2),
                "baseline_same_hour_last_week": number(last_week.loc[slot], 2),
                "call_b_prob_spike": number(p_spike.loc[slot], 5),
                "baseline_spike": number(baseline_spike.loc[slot[1]], 5),
                "call_b_prob_negative": number(p_low["negative"].loc[slot], 5),
                "call_b_prob_below_10": number(p_low["below_10"].loc[slot], 5),
                "baseline_negative": climatology_probability(climatology["negative"], utc_ts),
                "baseline_below_10": climatology_probability(climatology["below_10"], utc_ts),
                "call_c_quantiles": {str(level): number(value) for level, value in zip(QUANTILES, q)},
                "call_c_interval_50": [number(q[1]), number(q[3])],
                "call_c_interval_80": [number(q[0]), number(q[4])],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "bidding_zone": BIDDING_ZONE,
        "delivery_date_local": str(target.date()),
        "n_hours": int(len(hours)),
        "dst_note": {23: "23-hour day (spring forward)", 25: "25-hour day (autumn back)"}.get(
            len(hours), "standard 24-hour day"
        ),
        "sealed_at_utc": sealed_at,
        "auction_closes_utc": close.isoformat(),
        "minutes_before_close": round((close - pd.Timestamp(sealed_at)).total_seconds() / 60, 1),
        "model_version": MODEL_VERSION,
        "model_source_sha256": model.manifest["source_sha256"],
        "model_artefact_sha256": model.manifest["artefact_sha256"],
        "model_trained_at_utc": model.manifest["trained_at_utc"],
        "information_cutoff_utc": now_utc.isoformat(),
        "leakage_check": leakage,
        "price_check": inputs.price_check,
        "weather_coverage": inputs.weather_coverage,
        "scale_eur_mwh": number(scale.iloc[0], 4),
        "spike_threshold_eur_mwh": number(threshold, 3),
        "error_pool": pool_report,
        "predictions": calls,
        "input_provenance": inputs.provenance,
        "disclaimer": (
            "Forecast only. No tradeable edge is claimed and no P&L is implied. "
            "See PREREGISTRATION_PHASE2.md for the calls being made and what counts as a null."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal a v2 day-ahead forecast")
    parser.add_argument("--dry-run", action="store_true", help="print without writing")
    args = parser.parse_args()

    inputs = fetch_live_inputs()
    payload = build_prediction(inputs, pd.Timestamp.now(tz="UTC"))

    prices = np.array([p["call_a_price_eur_mwh"] for p in payload["predictions"]])
    width = np.array([p["call_c_interval_80"][1] - p["call_c_interval_80"][0] for p in payload["predictions"]])
    spike = np.array([p["call_b_prob_spike"] for p in payload["predictions"]])
    print(f"delivery {payload['delivery_date_local']}  ({payload['dst_note']})")
    print(f"sealed   {payload['sealed_at_utc']}  ({payload['minutes_before_close']:.0f} min before gate closure)")
    print(f"model    {payload['model_version']}  src {payload['model_source_sha256'][:12]}...")
    print(f"scale    {payload['scale_eur_mwh']:.1f} EUR/MWh   spike threshold {payload['spike_threshold_eur_mwh']:.1f}")
    pool = payload["error_pool"]
    print(f"errors   {pool['n_errors']} from {pool['live_days']} live + {pool['warm_start_days']} warm-start days")
    print(f"  call A price   : mean {prices.mean():7.2f}  min {prices.min():7.2f}  max {prices.max():7.2f}")
    print(f"  call B P(spike): mean {spike.mean():7.4f}  hours>0.5: {(spike > 0.5).sum()}")
    print(f"  call C 80% wid : mean {width.mean():7.2f}")

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0
    path = seal(payload, PREDICTIONS_DIR)
    print(f"\nsealed -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
