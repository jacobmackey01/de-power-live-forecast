"""Warn while there is still margin, rather than after it is gone.

A red ``predict`` run is not an early warning. By the time ``SealError`` fires
the gate has already closed and the delivery day is already lost - the failure
email is a post-mortem, not an alarm.

The quantity that degrades first is ``minutes_before_close``, recorded inside
every sealed prediction. Scheduled Actions are queued rather than guaranteed,
so that number drifts down for days before it ever reaches zero. Watching it
turns eight consecutive missed days into one warning email on the first day the
slack got thin.

Thresholds are deliberately generous. The observed queue delay has reached 12h,
so a margin that looks comfortable in isolation can still be one bad morning
from a miss.

    python -m de_power_live.gate_margin              # check the newest seal
    python -m de_power_live.gate_margin --json       # machine-readable

Exits 0 when the margin is healthy, 1 when it is below ``--warn-minutes``. It
reads and reports; it never writes, and it never touches the seal path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = REPO_ROOT / "predictions"

# Below this, the next queue spike plausibly pushes a seal past the gate.
DEFAULT_WARN_MINUTES = 90.0

# Below this, treat it as effectively a coin flip and say so louder.
CRITICAL_MINUTES = 45.0

# How many recent seals to read when reporting the trend. The point is to catch
# a slow slide, which a single reading cannot show.
TREND_SEALS = 5


def sealed_files(directory: Path = PREDICTIONS_DIR) -> list[Path]:
    """Sealed prediction files, oldest first. Delivery date sorts as the name."""
    return sorted(p for p in directory.glob("*.json") if p.stem[:2].isdigit())


def read_margin(path: Path) -> tuple[str, float]:
    """Return ``(delivery_date, minutes_before_close)`` for one sealed file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    delivery = payload["delivery_date_local"]
    margin = payload.get("minutes_before_close")
    if margin is None:
        raise KeyError(f"{path.name} has no minutes_before_close")
    return delivery, float(margin)


def collect(directory: Path = PREDICTIONS_DIR, n: int = TREND_SEALS) -> list[tuple[str, float]]:
    """The ``n`` most recent seals as ``(delivery_date, margin)``, oldest first."""
    return [read_margin(p) for p in sealed_files(directory)[-n:]]


def assess(readings: list[tuple[str, float]], warn_minutes: float) -> dict:
    """Judge the newest reading, with the recent trend as context."""
    if not readings:
        return {
            "ok": False,
            "reason": "no sealed predictions found",
            "readings": [],
        }

    delivery, margin = readings[-1]
    # A negative slope over the window matters more than any single reading:
    # it is the difference between a noisy morning and a schedule that is
    # steadily losing to the queue.
    trend = None
    if len(readings) >= 2:
        trend = round(readings[-1][1] - readings[0][1], 1)

    level = "ok"
    if margin < CRITICAL_MINUTES:
        level = "critical"
    elif margin < warn_minutes:
        level = "warn"

    return {
        "ok": level == "ok",
        "level": level,
        "delivery_date_local": delivery,
        "minutes_before_close": margin,
        "warn_minutes": warn_minutes,
        "critical_minutes": CRITICAL_MINUTES,
        "trend_minutes_over_window": trend,
        "readings": [{"delivery_date_local": d, "minutes_before_close": m} for d, m in readings],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check remaining margin to gate closure")
    parser.add_argument(
        "--warn-minutes",
        type=float,
        default=DEFAULT_WARN_MINUTES,
        help=f"warn below this many minutes of margin (default {DEFAULT_WARN_MINUTES:.0f})",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    report = assess(collect(), args.warn_minutes)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    if not report["readings"]:
        print("no sealed predictions found; nothing to check")
        return 1

    print("margin to gate closure, most recent seals:")
    for row in report["readings"]:
        print(f"  {row['delivery_date_local']}  {row['minutes_before_close']:7.1f} min")

    trend = report["trend_minutes_over_window"]
    if trend is not None:
        direction = "shrinking" if trend < 0 else "growing"
        print(f"\ntrend over {len(report['readings'])} seals: {trend:+.1f} min ({direction})")

    margin = report["minutes_before_close"]
    if report["level"] == "critical":
        print(
            f"\nCRITICAL: {margin:.1f} min of margin on {report['delivery_date_local']}, "
            f"below {CRITICAL_MINUTES:.0f}. The next queue spike misses the gate. "
            "Seal manually with workflow_dispatch today and move the cron earlier."
        )
        return 1
    if report["level"] == "warn":
        print(
            f"\nWARNING: {margin:.1f} min of margin on {report['delivery_date_local']}, "
            f"below {args.warn_minutes:.0f}. Scheduled runs have been queued up to 12h "
            "late on this platform; this is the point to act, not after a MISSED row."
        )
        return 1

    print(f"\nOK: {margin:.1f} min of margin on {report['delivery_date_local']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
