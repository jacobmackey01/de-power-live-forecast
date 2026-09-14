"""Decide whether a delivery day still needs sealing, and refuse to guess.

The predict workflow skips its work when tomorrow is already sealed. That skip
is only safe if "already sealed" means a complete, coherent seal for the day in
question. A truncated write, a half-committed file, or a file whose contents
belong to a different delivery day would otherwise be read as success: the run
would exit green, no forecast would exist, and no MISSED row would be written
either. The day would vanish from the record rather than count against it,
which is the one outcome the ledger exists to prevent.

So this module never infers a seal from a filename. It parses the file, checks
the fields the record actually depends on, and reports one of three states:

``absent``   nothing there; proceed and seal.
``sealed``   a valid seal for this day; skip, there is nothing left to do.
``invalid``  something is there and it is wrong. Stop and say so.

A fourth case sits outside the file entirely: the ledger may already carry a row
for the day. ``recorded_in_ledger`` reports that separately, because sealing
over a day already recorded MISSED produces exactly the contradiction audit.py
rejects.

``invalid`` is deliberately not routed into the miss path. A malformed file is
an operator problem, not a closed gate, and manufacturing a MISSED row from it
would write a false reason into an append-only ledger. The repair path for a
day genuinely lost this way is record-miss.yml, exactly as PREREGISTRATION.md
section 5 intends.

Standard library only, so the check cannot fail for reasons unrelated to the
file it is checking.

    python -m de_power_live.seal_guard 2026-09-09
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = REPO_ROOT / "predictions"
LEDGER_PATH = REPO_ROOT / "results" / "ledger.jsonl"

# Fields without which a sealed file cannot support the claims made about it:
# which day it is for, when it was sealed, that it beat the gate, which frozen
# model produced it, and the calls themselves.
REQUIRED_FIELDS = (
    "delivery_date_local",
    "sealed_at_utc",
    "auction_closes_utc",
    "minutes_before_close",
    "model_version",
    "model_source_sha256",
    "n_hours",
    "predictions",
)

# Every hour must carry all three pre-registered calls (section 3).
REQUIRED_CALL_FIELDS = ("hour_utc", "call_a_price_eur_mwh", "call_b_prob_negative")

ABSENT = "absent"
SEALED = "sealed"
INVALID = "invalid"


def recorded_in_ledger(target: str, ledger_path: Path = LEDGER_PATH) -> bool:
    """Whether the ledger already holds a row for ``target``.

    Parsed as JSON rather than grepped. A substring search over the raw file
    would match a date appearing in some other field - a reason string, say -
    and suppress a legitimate seal on the strength of a coincidence.
    """
    if not ledger_path.exists():
        return False
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Malformed ledger lines are audit.py's business, not this guard's.
            continue
        if isinstance(row, dict) and row.get("delivery_date_local") == target:
            return True
    return False


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def inspect(target: str, directory: Path = PREDICTIONS_DIR) -> tuple[str, list[str]]:
    """Classify the seal file for ``target`` as absent, sealed or invalid."""
    path = directory / f"{target}.json"
    if not path.exists():
        return ABSENT, []

    problems: list[str] = []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return INVALID, [f"{path.name} could not be read: {exc}"]

    if not raw.strip():
        return INVALID, [f"{path.name} is empty"]

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return INVALID, [f"{path.name} is not valid JSON: {exc}"]

    if not isinstance(payload, dict):
        return INVALID, [f"{path.name} is {type(payload).__name__}, not an object"]

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        problems.append(f"{path.name} is missing required field(s): {', '.join(missing)}")
        # Without these there is nothing coherent left to check.
        return INVALID, problems

    # A file named for one day but sealed for another is the most dangerous
    # case here, because every downstream lookup is by filename.
    declared = payload["delivery_date_local"]
    if declared != target:
        problems.append(
            f"{path.name} declares delivery_date_local {declared!r}, not {target!r}"
        )

    for field in ("sealed_at_utc", "auction_closes_utc"):
        try:
            _parse_utc(payload[field])
        except (TypeError, ValueError) as exc:
            problems.append(f"{path.name} has an unparseable {field}: {exc}")

    margin = payload["minutes_before_close"]
    if not isinstance(margin, (int, float)) or isinstance(margin, bool):
        problems.append(
            f"{path.name} has a non-numeric minutes_before_close: {margin!r}"
        )
    elif margin <= 0:
        problems.append(
            f"{path.name} records {margin} minutes before close, so it was sealed "
            "at or after gate closure and is not a prediction"
        )

    calls = payload["predictions"]
    if not isinstance(calls, list) or not calls:
        problems.append(f"{path.name} has no predictions")
    else:
        n_hours = payload["n_hours"]
        if isinstance(n_hours, int) and len(calls) != n_hours:
            problems.append(
                f"{path.name} claims {n_hours} hours but carries {len(calls)}"
            )
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                problems.append(f"{path.name} hour {index} is not an object")
                break
            absent = [f for f in REQUIRED_CALL_FIELDS if f not in call]
            if absent:
                problems.append(
                    f"{path.name} hour {index} is missing: {', '.join(absent)}"
                )
                break

    if problems:
        return INVALID, problems
    return SEALED, []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report whether a delivery day is already validly sealed"
    )
    parser.add_argument("target", help="delivery date YYYY-MM-DD")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="directory holding sealed predictions",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help="ledger to check for an existing row for this day",
    )
    args = parser.parse_args()

    state, problems = inspect(args.target, args.predictions_dir)

    # Checked before the benign outcomes: a broken file must surface whatever
    # else happens to be true about the day.
    if state == INVALID:
        print(f"predictions/{args.target}.json exists but is not a usable seal:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nRefusing to treat this as a sealed day and refusing to overwrite it: "
            "predictions are write-once. No MISSED row is being invented from it "
            "either, because a malformed file is not a closed gate. Inspect the file "
            "by hand. If the day was genuinely lost, record it through record-miss.yml.",
            file=sys.stderr,
        )
        return 1

    if state == SEALED:
        print(f"{args.target} is already validly sealed; nothing to do.")
        print("skip=true")
        return 0

    if recorded_in_ledger(args.target, args.ledger):
        print(f"{args.target} is already recorded in the ledger; nothing to do.")
        print("skip=true")
        return 0

    print(f"{args.target} is unsealed; proceeding.")
    print("skip=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
