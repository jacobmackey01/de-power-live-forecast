"""Score settled v2 forecasts into the phase-2 ledger.

    python -m de_power_live.v2.score
    python -m de_power_live.v2.score --record-miss 2026-12-01 --reason "..."

Never writes to predictions/. Each SCORED row carries the sealed calls and the
realised price for every hour, so the final report is computed from the ledger
alone and anyone can recompute it from that one file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import MARKET_TZ
from ..smard import SmardClient
from . import LEDGER_PATH, MODEL_VERSION, PREDICTIONS_DIR, RESULTS_DIR
from .calls import DESCRIPTIVE_THRESHOLDS, QUANTILES

# Same settle margin as phase 1: a day is scored once it ended this long ago.
SETTLE_AFTER_HOURS = 30
SUMMARY_PATH = RESULTS_DIR / "SUMMARY.md"


class IncompleteOutturn(RuntimeError):
    """Raised when a day's realised prices are not all published yet."""


def load_ledger(path: Path = LEDGER_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_ledger(entry: dict, path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def score_day(payload: dict, prices: pd.Series) -> dict:
    """One SCORED ledger row from a sealed payload and realised prices."""
    hours = pd.DatetimeIndex([pd.Timestamp(p["hour_utc"]) for p in payload["predictions"]])
    truth = prices.reindex(hours).to_numpy(dtype=float)
    if np.isnan(truth).any():
        raise IncompleteOutturn(
            f"{int(np.isnan(truth).sum())}/{len(hours)} realised prices still missing"
        )
    threshold = float(payload["spike_threshold_eur_mwh"])

    rows = []
    for call, price in zip(payload["predictions"], truth):
        q = [call["call_c_quantiles"][str(level)] for level in QUANTILES]
        rows.append(
            {
                "hour_utc": call["hour_utc"],
                "price": round(float(price), 2),
                "pred": call["call_a_price_eur_mwh"],
                "d1": call["baseline_same_hour_yesterday"],
                "d7": call["baseline_same_hour_last_week"],
                "q": q,
                "spike": int(price > threshold),
                "p_spike": call["call_b_prob_spike"],
                "base_spike": call["baseline_spike"],
                **{
                    key: value
                    for name, level in DESCRIPTIVE_THRESHOLDS.items()
                    for key, value in (
                        (name, int(price < level)),
                        (f"p_{name}", call[f"call_b_prob_{name}"]),
                        (f"base_{name}", call[f"baseline_{name}"]),
                    )
                },
            }
        )

    pred = np.array([r["pred"] for r in rows], dtype=float)
    d1 = np.array([r["d1"] for r in rows], dtype=float)
    q = np.array([r["q"] for r in rows], dtype=float)
    mae = float(np.mean(np.abs(pred - truth)))
    mae_d1 = float(np.mean(np.abs(d1 - truth)))
    return {
        "delivery_date_local": payload["delivery_date_local"],
        "status": "SCORED",
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_at_utc": payload["sealed_at_utc"],
        "minutes_before_close": payload["minutes_before_close"],
        "model_version": payload["model_version"],
        "model_source_sha256": payload["model_source_sha256"],
        "n_hours": len(rows),
        "spike_threshold_eur_mwh": threshold,
        "scale_eur_mwh": payload["scale_eur_mwh"],
        # Day-level figures for reading the ledger by eye. The report recomputes
        # everything from "hours".
        "day": {
            "mae_model": round(mae, 4),
            "mae_same_hour_yesterday": round(mae_d1, 4),
            "skill_vs_yesterday": round(1 - mae / mae_d1, 5) if mae_d1 else None,
            "bias_model": round(float(np.mean(pred - truth)), 4),
            "coverage_80": round(float(np.mean((truth >= q[:, 0]) & (truth <= q[:, 4]))), 5),
            "coverage_50": round(float(np.mean((truth >= q[:, 1]) & (truth <= q[:, 3]))), 5),
            "spike_hours": int(sum(r["spike"] for r in rows)),
        },
        "hours": rows,
    }


def miss_entry(day: str, reason: str) -> dict:
    return {
        "delivery_date_local": day,
        "status": "MISSED",
        "model_version": MODEL_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "note": (
            "No v2 forecast was sealed before gate closure. Per PREREGISTRATION_PHASE2.md "
            "section 5 this day cannot be backfilled and counts against the record."
        ),
    }


def write_summary(ledger: list[dict]) -> None:
    # Imported here: the report module imports this one for the ledger reader.
    from .final_report import build_report, render_markdown

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(ledger, provisional=True)
    SUMMARY_PATH.write_text(render_markdown(report, title="Running record, phase 2"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score settled v2 predictions")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-miss", metavar="YYYY-MM-DD")
    parser.add_argument("--reason", default="unspecified")
    args = parser.parse_args()

    ledger = load_ledger()
    already = {e["delivery_date_local"] for e in ledger}

    if args.record_miss:
        day = args.record_miss
        if day in already:
            print(f"{day} already in the v2 ledger; leaving it alone")
            return 0
        if (PREDICTIONS_DIR / f"{day}.json").exists():
            print(f"{day} was in fact sealed; refusing to record a miss")
            return 0
        entry = miss_entry(day, args.reason)
        append_ledger(entry)
        write_summary([*ledger, entry])
        print(f"recorded MISSED for {day}: {args.reason}")
        return 0

    files = sorted(PREDICTIONS_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    pending = [p for p in files if p.stem not in already]
    if not pending:
        print("nothing new to score")
        return 0

    prices = SmardClient().fetch_frame(["price_da"], n_blocks=6)["price_da"].dropna()
    now = pd.Timestamp.now(tz="UTC")
    scored = 0
    for path in pending:
        day = pd.Timestamp(path.stem)
        day_end = (day + pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
        if now < day_end + pd.Timedelta(hours=SETTLE_AFTER_HOURS):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            entry = score_day(payload, prices)
        except IncompleteOutturn as exc:
            print(f"{path.stem}: {exc}; will retry on the next run")
            continue
        print(f"{path.stem}: MAE {entry['day']['mae_model']:.2f} vs yesterday "
              f"{entry['day']['mae_same_hour_yesterday']:.2f}, cov80 {entry['day']['coverage_80']:.2f}")
        if not args.dry_run:
            append_ledger(entry)
        ledger.append(entry)
        scored += 1

    if scored and not args.dry_run:
        write_summary(ledger)
    print(f"scored {scored} day(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
