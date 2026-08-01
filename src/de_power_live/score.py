"""Score settled predictions and append to the ledger.

This job never writes to ``predictions/``. It only appends to ``results/``.
Keeping the two separable is what makes "the forecast was fixed before the
outcome was known" checkable by anyone with a clone, rather than a claim.

    python -m de_power_live.score
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import MARKET_TZ
from .model import QUANTILES, climatology_probability
from .smard import SmardClient

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = REPO_ROOT / "predictions"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
LEDGER_PATH = RESULTS_DIR / "ledger.jsonl"
SUMMARY_PATH = RESULTS_DIR / "SUMMARY.md"

# Settle delay: outturn prices publish shortly after the auction, but leave room
# for late revisions before treating a day as final.
SETTLE_AFTER_HOURS = 30


def load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    rows = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_ledger(entry: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def pinball_loss(truth: np.ndarray, pred: np.ndarray, level: float) -> float:
    delta = truth - pred
    return float(np.mean(np.maximum(level * delta, (level - 1) * delta)))


class IncompleteOutturn(RuntimeError):
    """Raised when a delivery day cannot yet be scored completely.

    Deliberately distinct from a scoring failure: the caller must skip the day
    and retry later rather than append a partial entry, because an appended day
    is final and would never be revisited.
    """


def load_climatology(model_version: str) -> tuple[dict, dict]:
    """Frozen call B baselines for a model version, plus its thresholds."""
    manifest_path = MODELS_DIR / model_version / "MANIFEST.json"
    if not manifest_path.exists():
        raise IncompleteOutturn(f"no manifest for model {model_version}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    climatology = manifest.get("climatology")
    if not climatology:
        raise IncompleteOutturn(
            f"model {model_version} has no frozen climatology; the call B baseline "
            "must be fixed before the window, not estimated after it"
        )
    return climatology, manifest.get("call_b_thresholds", {})


def score_day(payload: dict, prices: pd.Series) -> dict:
    """Score one sealed prediction against realised prices.

    Requires the delivery day to be *completely* settled: every predicted hour
    must have a realised price, and every hour must have both pre-registered
    baselines available. A partial day appended to the ledger becomes permanent
    and silently reduces the sample the record is judged on.
    """
    rows = payload["predictions"]
    hours = pd.DatetimeIndex([pd.Timestamp(r["hour_utc"]) for r in rows])

    truth = prices.reindex(hours).to_numpy(dtype=float)
    b1 = prices.reindex(hours - pd.Timedelta(hours=168)).to_numpy(dtype=float)
    b2 = prices.reindex(hours - pd.Timedelta(hours=24)).to_numpy(dtype=float)

    missing_truth = int(np.isnan(truth).sum())
    missing_b1 = int(np.isnan(b1).sum())
    missing_b2 = int(np.isnan(b2).sum())
    if missing_truth or missing_b1 or missing_b2:
        raise IncompleteOutturn(
            f"{missing_truth}/{len(hours)} realised prices, {missing_b1} B1 and "
            f"{missing_b2} B2 baseline values still missing"
        )

    climatology, thresholds = load_climatology(payload["model_version"])

    model_price = np.array([r["call_a_price_eur_mwh"] for r in rows], dtype=float)
    prob_neg = np.array([r["call_b_prob_negative"] for r in rows], dtype=float)
    q = np.array(
        [[r["call_c_quantiles"][str(level)] for level in QUANTILES] for r in rows],
        dtype=float,
    )

    # Every metric below is computed over the identical, complete set of hours,
    # so the skill ratios compare like with like.
    t, m = truth, model_price

    mae_model = float(np.mean(np.abs(m - t)))
    mae_b1 = float(np.mean(np.abs(b1 - t)))
    mae_b2 = float(np.mean(np.abs(b2 - t)))

    is_neg = (t < 0).astype(int)
    lo80, hi80 = q[:, 0], q[:, 4]
    lo50, hi50 = q[:, 1], q[:, 3]

    # Retain, per hour, everything the pre-registered call B tests need: the
    # model score, the frozen baseline, and the outcome at BOTH declared
    # thresholds. Without the second outcome the fallback target would be
    # impossible to evaluate from the ledger at all.
    call_b_hours = []
    for i, ts in enumerate(hours):
        entry = {
            "hour_utc": ts.isoformat(),
            "model_score": round(float(prob_neg[i]), 6),
            "price": round(float(t[i]), 3),
        }
        for name, threshold in thresholds.items():
            entry[f"outcome_{name}"] = int(t[i] < threshold)
            entry[f"baseline_{name}"] = round(
                climatology_probability(climatology[name], ts), 6
            )
        call_b_hours.append(entry)

    return {
        "delivery_date_local": payload["delivery_date_local"],
        "status": "SCORED",
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_at_utc": payload["sealed_at_utc"],
        "minutes_before_close": payload.get("minutes_before_close"),
        "model_version": payload["model_version"],
        "model_source_sha256": payload["model_source_sha256"],
        "n_hours_predicted": len(rows),
        "n_hours_scored": len(hours),
        "call_a": {
            "mae_model": round(mae_model, 4),
            "mae_baseline_b1_lag168": round(mae_b1, 4),
            "mae_baseline_b2_lag24": round(mae_b2, 4),
            "skill_vs_b1": round(1 - mae_model / mae_b1, 5) if mae_b1 else None,
            "skill_vs_b2": round(1 - mae_model / mae_b2, 5) if mae_b2 else None,
            "rmse_model": round(float(np.sqrt(np.mean((m - t) ** 2))), 4),
            "bias_model": round(float(np.mean(m - t)), 4),
        },
        "call_b": {
            "n_negative_hours": int(is_neg.sum()),
            "n_below_10_hours": int((t < 10.0).sum()),
            "brier": round(float(np.mean((prob_neg - is_neg) ** 2)), 6),
            "mean_prob": round(float(np.mean(prob_neg)), 6),
            "climatology_model_version": payload["model_version"],
            # Everything the pre-registered tests need at window close, without
            # ever re-reading the sealed files: model score, frozen baseline and
            # outcome at both declared thresholds.
            "hours": call_b_hours,
        },
        "call_c": {
            "coverage_80": round(float(np.mean((t >= lo80) & (t <= hi80))), 5),
            "coverage_50": round(float(np.mean((t >= lo50) & (t <= hi50))), 5),
            "mean_width_80": round(float(np.mean(hi80 - lo80)), 4),
            "mean_width_50": round(float(np.mean(hi50 - lo50)), 4),
            "pinball": {
                str(level): round(pinball_loss(t, q[:, i], level), 5)
                for i, level in enumerate(QUANTILES)
            },
        },
        "realised": {
            "mean_price": round(float(np.mean(t)), 3),
            "min_price": round(float(np.min(t)), 3),
            "max_price": round(float(np.max(t)), 3),
        },
    }


def write_summary(ledger: list[dict]) -> None:
    scored = [e for e in ledger if e.get("status") == "SCORED"]
    missed = [e for e in ledger if e.get("status") == "MISSED"]

    lines = [
        "# Running scoring ledger",
        "",
        "Descriptive only. The formal assessment against the pre-registered success",
        "criteria happens once, at the close of the evaluation window on 2026-10-31.",
        "Reading these numbers and stopping early is precluded by the pre-registration.",
        "",
        f"*Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}*",
        "",
        f"- Days scored: **{len(scored)}**",
        f"- Days missed: **{len(missed)}**"
        + (f" ({', '.join(e['delivery_date_local'] for e in missed)})" if missed else ""),
        "",
    ]

    if scored:
        skills = [e["call_a"]["skill_vs_b1"] for e in scored if e["call_a"]["skill_vs_b1"] is not None]
        cov80 = [e["call_c"]["coverage_80"] for e in scored]
        cov50 = [e["call_c"]["coverage_50"] for e in scored]
        neg = sum(e["call_b"]["n_negative_hours"] for e in scored)
        below10 = sum(e["call_b"].get("n_below_10_hours", 0) for e in scored)

        lines += [
            "## Running aggregates",
            "",
            "| Call | Metric | Value |",
            "|---|---|---|",
            f"| A | mean daily MAE | {np.mean([e['call_a']['mae_model'] for e in scored]):.2f} EUR/MWh |",
            f"| A | mean skill vs B1 | {np.mean(skills):+.4f} |" if skills else "| A | mean skill vs B1 | n/a |",
            f"| B | negative hours observed | {neg} |",
            f"| B | powered (needs 30) | {'yes' if neg >= 30 else 'not yet'} |",
            f"| B | fallback (<10 EUR/MWh) hours | {below10} |",
            f"| C | empirical 80% coverage | {np.mean(cov80):.3f} (nominal 0.800) |",
            f"| C | empirical 50% coverage | {np.mean(cov50):.3f} (nominal 0.500) |",
            "",
            "## Per-day",
            "",
            "| Delivery | Sealed (min before close) | MAE | B1 MAE | Skill | Cov80 | Neg hrs |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in sorted(scored, key=lambda x: x["delivery_date_local"], reverse=True):
            a, c = e["call_a"], e["call_c"]
            skill = f"{a['skill_vs_b1']:+.3f}" if a["skill_vs_b1"] is not None else "n/a"
            b1 = f"{a['mae_baseline_b1_lag168']:.2f}" if a["mae_baseline_b1_lag168"] else "n/a"
            lines.append(
                f"| {e['delivery_date_local']} | {e.get('minutes_before_close', 0):.0f} | "
                f"{a['mae_model']:.2f} | {b1} | {skill} | {c['coverage_80']:.2f} | "
                f"{e['call_b']['n_negative_hours']} |"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score settled predictions")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--record-miss",
        metavar="YYYY-MM-DD",
        help="record a delivery day the predict job failed to seal",
    )
    parser.add_argument("--reason", default="unspecified")
    args = parser.parse_args()

    ledger = load_ledger()
    already = {e["delivery_date_local"] for e in ledger}

    if args.record_miss:
        day = args.record_miss
        if day in already:
            print(f"{day} already in the ledger; leaving it alone")
            return 0
        if (PREDICTIONS_DIR / f"{day}.json").exists():
            print(f"{day} was in fact sealed; refusing to record a miss")
            return 0
        entry = {
            "delivery_date_local": day,
            "status": "MISSED",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": args.reason,
            "note": (
                "No forecast was sealed before gate closure. Per PREREGISTRATION.md "
                "section 5 this day cannot be backfilled and counts against the record."
            ),
        }
        append_ledger(entry)
        ledger.append(entry)
        write_summary(ledger)
        print(f"recorded MISSED for {day}: {args.reason}")
        return 0

    files = sorted(PREDICTIONS_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    if not files:
        print("no sealed predictions yet")
        return 0

    smard = SmardClient()
    prices = smard.fetch_frame(["price_da"], n_blocks=6)["price_da"].dropna()
    now = pd.Timestamp.now(tz="UTC")

    scored_count = 0
    for path in files:
        day = path.stem
        if day in already:
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        end_local = pd.Timestamp(day).tz_localize(MARKET_TZ) + pd.Timedelta(days=1)
        if now < end_local.tz_convert("UTC") + pd.Timedelta(hours=SETTLE_AFTER_HOURS):
            print(f"{day}: not settled yet, skipping")
            continue

        try:
            entry = score_day(payload, prices)
        except IncompleteOutturn as exc:
            # Skip without appending. An appended day is final, so a partial
            # one would permanently shrink the evaluated sample.
            print(f"{day}: incomplete, will retry ({exc})")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{day}: cannot score ({exc})")
            continue

        a, c = entry["call_a"], entry["call_c"]
        print(
            f"{day}: MAE {a['mae_model']:.2f} vs B1 {a['mae_baseline_b1_lag168']}, "
            f"skill {a['skill_vs_b1']}, cov80 {c['coverage_80']:.2f}, "
            f"neg {entry['call_b']['n_negative_hours']}"
        )
        if not args.dry_run:
            append_ledger(entry)
            ledger.append(entry)
        scored_count += 1

    if not args.dry_run:
        write_summary(ledger)
        print(f"\n{scored_count} newly scored; summary -> {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
