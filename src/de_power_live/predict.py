"""Produce and seal one day's forecast.

Refuses to write if the auction has already closed. A prediction sealed after
gate closure is not a prediction, and quietly writing one anyway would be the
single most damaging thing this system could do to its own record.

    python -m de_power_live.predict                 # tomorrow
    python -m de_power_live.predict --dry-run       # print, write nothing
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import AUCTION_CLOSE_LOCAL_HOUR, BIDDING_ZONE, MARKET_TZ
from .dataset import target_hours_for
from .features import assert_no_leakage, build_features
from .model import QUANTILES, DayAheadModel, source_hash
from .smard import SmardClient
from .weather import OpenMeteoClient

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = REPO_ROOT / "predictions"
MODELS_DIR = REPO_ROOT / "models"


class SealError(RuntimeError):
    """Raised when a prediction cannot be honestly sealed."""


def auction_close_utc(target_day: pd.Timestamp) -> pd.Timestamp:
    """Gate closure for delivery on ``target_day``: 12:00 local on the day before."""
    day_before = pd.Timestamp(target_day).tz_localize(None).normalize() - pd.Timedelta(days=1)
    local = day_before.tz_localize(MARKET_TZ) + pd.Timedelta(hours=AUCTION_CLOSE_LOCAL_HOUR)
    return local.tz_convert("UTC")


def build_prediction(target_day: pd.Timestamp, model_version: str = "v1") -> dict:
    """Assemble a sealed prediction payload for ``target_day``."""
    now = pd.Timestamp.now(tz="UTC")
    close = auction_close_utc(target_day)
    if now >= close:
        raise SealError(
            f"auction for {pd.Timestamp(target_day).date()} closed at {close.isoformat()}; "
            f"it is now {now.isoformat()}. Refusing to seal a forecast after gate closure."
        )

    hours = target_hours_for(target_day)

    smard = SmardClient()
    market = smard.fetch_frame(
        [
            "price_da",
            "load_actual",
            "wind_onshore_actual",
            "wind_offshore_actual",
            "solar_actual",
            "load_fc",
            "renewables_fc",
            "residual_load_fc",
            "wind_onshore_fc",
            "wind_offshore_fc",
            "solar_fc",
        ],
        n_blocks=8,
    )

    identities = smard.verify_filter_identities(market)
    if not identities.get("all_ok"):
        raise SealError(
            f"SMARD filter identities failed: {identities}. Aborting rather than "
            "modelling on a possibly relabelled series."
        )

    prices = market["price_da"].dropna()
    actuals = pd.DataFrame(
        {
            "wind_total_actual": market["wind_onshore_actual"].fillna(0)
            + market["wind_offshore_actual"].fillna(0),
            "solar_actual": market["solar_actual"],
            "load_actual": market["load_actual"],
        }
    )

    meteo = OpenMeteoClient()
    weather, coverage = meteo.national_drivers()

    missing_hours = [h for h in hours if h not in weather.index]
    if missing_hours:
        raise SealError(
            f"weather covers only {len(hours) - len(missing_hours)}/{len(hours)} "
            f"hours of {pd.Timestamp(target_day).date()}"
        )

    cutoff = pd.Timestamp.now(tz="UTC")
    leakage = assert_no_leakage(prices, actuals, cutoff, hours)

    features = build_features(
        weather=weather, prices=prices, actuals=actuals,
        cutoff_utc=cutoff, target_index=hours,
    )

    model = DayAheadModel.load(MODELS_DIR / model_version)
    result = model.predict(features)

    sealed_at = datetime.now(timezone.utc).isoformat()
    local_hours = hours.tz_convert(MARKET_TZ)

    calls = []
    for i, (utc_ts, local_ts) in enumerate(zip(hours, local_hours)):
        q = result.quantiles[i]
        calls.append(
            {
                "hour_utc": utc_ts.isoformat(),
                "hour_local": local_ts.isoformat(),
                "call_a_price_eur_mwh": round(float(result.price[i]), 3),
                "call_b_prob_negative": round(float(result.prob_negative[i]), 5),
                "call_c_quantiles": {
                    str(level): round(float(value), 3)
                    for level, value in zip(QUANTILES, q)
                },
                "call_c_interval_50": [round(float(q[1]), 3), round(float(q[3]), 3)],
                "call_c_interval_80": [round(float(q[0]), 3), round(float(q[4]), 3)],
            }
        )

    return {
        "schema_version": 1,
        "bidding_zone": BIDDING_ZONE,
        "delivery_date_local": str(pd.Timestamp(target_day).date()),
        "n_hours": len(hours),
        "dst_note": (
            "23-hour short day" if len(hours) == 23
            else "25-hour long day" if len(hours) == 25
            else "standard 24-hour day"
        ),
        "sealed_at_utc": sealed_at,
        "auction_closes_utc": close.isoformat(),
        "minutes_before_close": round((close - pd.Timestamp(sealed_at)).total_seconds() / 60, 1),
        "model_version": model.version,
        "model_source_sha256": source_hash(),
        "model_trained_at_utc": model.trained_at_utc,
        "information_cutoff_utc": cutoff.isoformat(),
        "leakage_check": leakage,
        "identity_check": identities,
        "weather_coverage": coverage,
        "predictions": calls,
        "input_provenance": {
            "smard": smard.provenance(),
            "open_meteo": meteo.provenance(),
        },
        "disclaimer": (
            "Forecast only. No tradeable edge is claimed and no P&L is implied. "
            "See PREREGISTRATION.md for the calls being made and what counts as a null."
        ),
    }


def seal(payload: dict, directory: Path = PREDICTIONS_DIR) -> Path:
    """Write the payload write-once. Refuses to overwrite an existing seal."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['delivery_date_local']}.json"
    if path.exists():
        raise SealError(
            f"{path.name} already exists. Predictions are write-once; a second "
            "seal for the same delivery day is never legitimate."
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal a day-ahead forecast")
    parser.add_argument("--date", help="delivery date YYYY-MM-DD (default: tomorrow local)")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--dry-run", action="store_true", help="print without writing")
    args = parser.parse_args()

    if args.date:
        target = pd.Timestamp(args.date)
    else:
        target = pd.Timestamp.now(tz=MARKET_TZ).normalize().tz_localize(None) + pd.Timedelta(days=1)

    payload = build_prediction(target, model_version=args.model_version)

    prices = np.array([p["call_a_price_eur_mwh"] for p in payload["predictions"]])
    probs = np.array([p["call_b_prob_negative"] for p in payload["predictions"]])
    width80 = np.array(
        [p["call_c_interval_80"][1] - p["call_c_interval_80"][0] for p in payload["predictions"]]
    )

    print(f"delivery {payload['delivery_date_local']}  ({payload['dst_note']})")
    print(f"sealed   {payload['sealed_at_utc']}  "
          f"({payload['minutes_before_close']:.0f} min before gate closure)")
    print(f"model    {payload['model_version']}  src {payload['model_source_sha256'][:12]}...")
    print(f"inputs   {len(payload['input_provenance']['smard'])} SMARD + "
          f"{len(payload['input_provenance']['open_meteo'])} Open-Meteo payloads hashed")
    print()
    print(f"  call A price   : mean {prices.mean():7.2f}  min {prices.min():7.2f}  max {prices.max():7.2f}")
    print(f"  call B P(neg)  : mean {probs.mean():7.4f}  max {probs.max():7.4f}  "
          f"hours>0.5: {(probs > 0.5).sum()}")
    print(f"  call C 80% wid : mean {width80.mean():7.2f}  min {width80.min():7.2f}  max {width80.max():7.2f}")
    print()
    print("  hour   price    P(neg)   80% interval")
    for row in payload["predictions"]:
        lo, hi = row["call_c_interval_80"]
        print(f"  {row['hour_local'][11:16]}  {row['call_a_price_eur_mwh']:7.2f}  "
              f"{row['call_b_prob_negative']:7.4f}   [{lo:7.2f}, {hi:7.2f}]")

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    path = seal(payload)
    print(f"\nsealed -> {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
