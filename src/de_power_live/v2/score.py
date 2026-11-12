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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import MARKET_TZ
from ..smard import SmardClient
from . import FIRST_DELIVERY_DAY, LEDGER_PATH, MODEL_VERSION, MODELS_DIR, PREDICTIONS_DIR, RESULTS_DIR
from .audit import check_ledger, check_seal, days_due
from .calls import DESCRIPTIVE_THRESHOLDS, QUANTILES

# Same settle margin as phase 1: a day is scored once it ended this long ago.
SETTLE_AFTER_HOURS = 30
SUMMARY_PATH = RESULTS_DIR / "SUMMARY.md"
# A day with no forecast is entered as MISSED once its gate closed this long
# ago. The margin is far longer than any push takes, so a forecast that was
# sealed in time cannot still be on its way.
RECONCILE_GRACE = timedelta(minutes=45)
RECONCILED_REASON = (
    "No v2 forecast was published before gate closure, and the seal job recorded no miss at the "
    "time. Entered afterwards by the scoring job, which checks every due delivery day."
)
# SMARD serves prices in weekly blocks. Fetch far enough back for the oldest
# unscored day, however long scoring has been down.
MIN_PRICE_BLOCKS = 6
MAX_PRICE_BLOCKS = 60


class InvalidSeal(RuntimeError):
    """Raised when a sealed file is not something the record can rest on."""


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


def score_day(payload: dict, prices: pd.Series, check_model: bool = True) -> dict:
    """One SCORED ledger row from a sealed payload and realised prices.

    ``check_model`` holds the file to the frozen model in models/v2. A rehearsal
    seals with a scratch model and turns it off; nothing else should.
    """
    # Scoring is where a sealed file starts to count, so it is where the file is
    # held to what a seal must be: the right hours, once each, sealed before its
    # own gate, by the frozen model. A file that fails is not scored at all.
    manifest_path = MODELS_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and check_model else None
    problems = check_seal(payload, payload.get("delivery_date_local", ""), manifest)
    if problems:
        raise InvalidSeal("; ".join(problems))

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


def reconcile(ledger: list[dict], predictions_dir: Path, now: datetime, first: date = FIRST_DELIVERY_DAY) -> list[dict]:
    """MISSED rows for due days that have neither a forecast nor a ledger row.

    The seal job records a miss when its seal step fails. It cannot when the
    job never ran, was cancelled, or failed somewhere else, and until this was
    added such a day left no trace at all. This looks at the calendar instead:
    every delivery day from the first one, once its gate has closed.

    It invents nothing. It records that no forecast exists, which is the same
    thing a miss recorded at the time would say.
    """
    have = {row["delivery_date_local"] for row in ledger}
    out = []
    for day in days_due(now, first, RECONCILE_GRACE):
        key = day.isoformat()
        if key in have or (Path(predictions_dir) / f"{key}.json").exists():
            continue
        out.append(miss_entry(key, RECONCILED_REASON))
    return out


def price_blocks_for(oldest_pending: date, today: date) -> int:
    weeks = (today - oldest_pending).days // 7 + 3
    return max(MIN_PRICE_BLOCKS, min(MAX_PRICE_BLOCKS, weeks))


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
    check_ledger(ledger)  # a repeated day is refused before anything is added to it
    already = {e["delivery_date_local"] for e in ledger}

    if args.record_miss:
        # Appends one row and nothing else. The summary is rewritten by the
        # scoring job, so that this path and that one never edit the same file
        # and a rebase between them cannot conflict.
        day = args.record_miss
        if day in already:
            print(f"{day} already in the v2 ledger; leaving it alone")
            return 0
        if (PREDICTIONS_DIR / f"{day}.json").exists():
            print(f"{day} was in fact sealed; refusing to record a miss")
            return 0
        append_ledger(miss_entry(day, args.reason))
        print(f"recorded MISSED for {day}: {args.reason}")
        return 0

    now = pd.Timestamp.now(tz="UTC")
    changed = False
    for entry in reconcile(ledger, PREDICTIONS_DIR, now.to_pydatetime()):
        print(f"{entry['delivery_date_local']}: no forecast and no miss on record; entering MISSED")
        if not args.dry_run:
            append_ledger(entry)
        ledger.append(entry)
        already.add(entry["delivery_date_local"])
        changed = True

    files = sorted(PREDICTIONS_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    pending = [p for p in files if p.stem not in already]
    settled = []
    for path in pending:
        day_end = (pd.Timestamp(path.stem) + pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
        if now >= day_end + pd.Timedelta(hours=SETTLE_AFTER_HOURS):
            settled.append(path)

    invalid: list[str] = []
    scored = 0
    if settled:
        blocks = price_blocks_for(date.fromisoformat(settled[0].stem), now.date())
        prices = SmardClient().fetch_frame(["price_da"], n_blocks=blocks)["price_da"].dropna()
        for path in settled:
            payload = json.loads(path.read_text(encoding="utf-8"))
            try:
                entry = score_day(payload, prices)
            except IncompleteOutturn as exc:
                print(f"{path.stem}: {exc}; will retry on the next run")
                continue
            except InvalidSeal as exc:
                print(f"::error::{path.stem}: sealed file is not valid and was not scored: {exc}")
                invalid.append(path.stem)
                continue
            print(f"{path.stem}: MAE {entry['day']['mae_model']:.2f} vs yesterday "
                  f"{entry['day']['mae_same_hour_yesterday']:.2f}, cov80 {entry['day']['coverage_80']:.2f}")
            # Appended day by day, so a failure on a later day cannot take an
            # earlier day's score down with it.
            if not args.dry_run:
                append_ledger(entry)
            ledger.append(entry)
            scored += 1
            changed = True

    # Rewritten whenever there is a ledger, not only when this run added to it:
    # a miss recorded by the seal job changes the record without passing here.
    if ledger and not args.dry_run:
        write_summary(ledger)
    print(f"scored {scored} day(s)" + (", ledger changed" if changed else ""))
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
