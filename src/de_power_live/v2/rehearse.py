"""Run the real seal and scoring path over past days, as each seal would have seen them.

    python -m de_power_live.v2.rehearse --train-last 2026-10-18 --first 2026-10-20 --last 2026-11-10

For each target day the inputs are cut to what existed at that day's seal
window: prices up to the seal day and the archived forecast. ``build_prediction`` then runs exactly as it does live,
leakage guard and gate check included, and writes to a scratch directory. The
errors of each rehearsed day feed the next day's intervals, as they will live.

This is how a clock-change day gets exercised before it arrives. It writes
nothing under predictions/ or results/.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from .. import MARKET_TZ
from ..dataset import fetch_history
from ..predict import seal
from .features import SEAL_LOCAL
from .final_report import build_report, render_markdown
from .inputs import LiveInputs
from .predict import build_prediction
from .score import score_day
from .train import freeze


def inputs_as_of(prices: pd.Series, weather: pd.DataFrame, now_utc: pd.Timestamp) -> LiveInputs:
    """The slice of history a seal at ``now_utc`` could have fetched."""
    seal_day = now_utc.tz_convert(MARKET_TZ).normalize().tz_localize(None)
    delivery_day = prices.index.tz_convert(MARKET_TZ).normalize().tz_localize(None)
    horizon = (seal_day + pd.Timedelta(days=3)).tz_localize(MARKET_TZ).tz_convert("UTC")
    return LiveInputs(
        prices=prices[delivery_day <= seal_day],
        weather=weather[weather.index < horizon],
        price_check={"ok": True, "rehearsal": True},
        weather_coverage={"rehearsal": True},
        provenance={"rehearsal": True},
    )


def rehearse(
    prices: pd.Series,
    weather: pd.DataFrame,
    days: pd.DatetimeIndex,
    model_dir: Path,
    work_dir: Path,
) -> list[dict]:
    """Seal and score every day in ``days``; return the ledger rows."""
    ledger = []
    for target in days:
        # The seal window on the wall clock of the day before the target.
        now = (target - pd.Timedelta(days=1) + SEAL_LOCAL).tz_localize(MARKET_TZ).tz_convert("UTC")
        payload = build_prediction(
            inputs_as_of(prices, weather, now), now,
            model_dir=model_dir, predictions_dir=work_dir, sealed_at_utc=now,
        )
        seal(payload, work_dir)
        ledger.append(score_day(payload, prices, check_model=False))
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description="Rehearse v2 seals over past days")
    parser.add_argument("--train-last", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--keep", type=Path, help="directory to keep the rehearsed files in")
    args = parser.parse_args()

    prices, weather, _, _ = fetch_history(years=2.3)
    with tempfile.TemporaryDirectory() as scratch:
        root = args.keep or Path(scratch)
        model_dir, work_dir = root / "model", root / "predictions"
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest = freeze(prices, weather, pd.Timestamp(args.train_last), model_dir)
        print(f"rehearsal model trained through {manifest['train_last']} ({manifest['training_rows']} rows)")
        ledger = rehearse(
            prices, weather, pd.date_range(args.first, args.last, freq="D"), model_dir, work_dir
        )
        for row in ledger:
            d = row["day"]
            print(f"{row['delivery_date_local']}  {row['n_hours']}h  MAE {d['mae_model']:6.2f}  "
                  f"yesterday {d['mae_same_hour_yesterday']:6.2f}  cov80 {d['coverage_80']:.2f}  spikes {d['spike_hours']}")
        if args.keep:
            (root / "ledger.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in ledger) + "\n")
        print()
        report = build_report(
            ledger, provisional=True, draws=2000,
            first=pd.Timestamp(args.first).date(), end=pd.Timestamp(args.last).date(),
        )
        print("REHEARSAL ON PAST DATA. Not a live record: nothing below is a result, and any PASS is "
              "the report's logic applied to development data.\n")
        print(render_markdown(report, title="Rehearsal"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
