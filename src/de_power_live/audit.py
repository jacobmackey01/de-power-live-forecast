"""Read-only audit of the published record.

This module answers one question: does what the repository publishes still
follow from the ledger? It reports and exits non-zero. It never writes, never
repairs, and never regenerates a committed artefact - a discrepancy is a thing
to look at, not a thing to paper over, and an auditor that silently fixes what
it finds destroys the evidence it exists to preserve.

It deliberately imports nothing beyond the standard library and
:mod:`de_power_live.track_record`, so it cannot fail for reasons unrelated to
the record it is auditing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from de_power_live.track_record import LEDGER_PATH, OUTPUT_PATH, load_record, render_svg

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = REPO_ROOT / "predictions"
SUMMARY_PATH = REPO_ROOT / "results" / "SUMMARY.md"

VALID_STATUSES = {"SCORED", "MISSED"}

SCORED_RE = re.compile(r"^- Days scored: \*\*(\d+)\*\*", re.MULTILINE)
MISSED_RE = re.compile(r"^- Days missed: \*\*(\d+)\*\*(?: \(([^)]*)\))?", re.MULTILINE)


def read_ledger_rows(path: Path) -> tuple[list[dict], list[str]]:
    """Parse the ledger, reporting rather than raising on a malformed line."""
    problems: list[str] = []
    rows: list[dict] = []
    if not path.exists():
        return rows, [f"{path} does not exist"]

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"ledger line {number} is not valid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            problems.append(f"ledger line {number} is not a JSON object")
            continue
        rows.append(row)
    return rows, problems


def check_ledger_shape(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: dict[str, int] = {}

    for number, row in enumerate(rows, start=1):
        day = row.get("delivery_date_local")
        status = row.get("status")

        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day or ""):
            problems.append(f"ledger row {number} has no usable delivery_date_local")
            continue
        if status not in VALID_STATUSES:
            problems.append(f"{day}: status {status!r} is not one of {sorted(VALID_STATUSES)}")
        if day in seen:
            problems.append(f"{day}: duplicated in the ledger (rows {seen[day]} and {number})")
        else:
            seen[day] = number

    return problems


def check_against_sealed_predictions(rows: list[dict], predictions_dir: Path) -> list[str]:
    """A SCORED day must have its sealed file; a MISSED day must not."""
    problems: list[str] = []
    for row in rows:
        day = row.get("delivery_date_local")
        status = row.get("status")
        if not isinstance(day, str):
            continue
        sealed = (predictions_dir / f"{day}.json").exists()
        if status == "SCORED" and not sealed:
            problems.append(f"{day}: SCORED but predictions/{day}.json is missing")
        if status == "MISSED" and sealed:
            problems.append(
                f"{day}: MISSED but predictions/{day}.json exists - "
                "a sealed day must never be recorded as missed"
            )
    return problems


def check_summary(rows: list[dict], summary_path: Path) -> list[str]:
    """The published counts must follow from the ledger.

    This re-derives the claims instead of re-running write_summary, because
    write_summary stamps a wall-clock time and would never compare equal.
    """
    if not summary_path.exists():
        return [f"{summary_path} does not exist"]

    problems: list[str] = []
    summary = summary_path.read_text(encoding="utf-8")

    scored = [r for r in rows if r.get("status") == "SCORED"]
    missed = [r for r in rows if r.get("status") == "MISSED"]

    scored_match = SCORED_RE.search(summary)
    if scored_match is None:
        problems.append("SUMMARY.md has no 'Days scored' line")
    elif int(scored_match.group(1)) != len(scored):
        problems.append(
            f"SUMMARY.md claims {scored_match.group(1)} days scored; "
            f"the ledger holds {len(scored)}"
        )

    missed_match = MISSED_RE.search(summary)
    if missed_match is None:
        problems.append("SUMMARY.md has no 'Days missed' line")
        return problems

    if int(missed_match.group(1)) != len(missed):
        problems.append(
            f"SUMMARY.md claims {missed_match.group(1)} days missed; "
            f"the ledger holds {len(missed)}"
        )

    listed = [d.strip() for d in (missed_match.group(2) or "").split(",") if d.strip()]
    expected = [r["delivery_date_local"] for r in missed if isinstance(r.get("delivery_date_local"), str)]
    if listed != expected:
        problems.append(
            "SUMMARY.md lists missed days "
            f"{listed or '[]'} but the ledger holds {expected or '[]'}"
        )

    return problems


def check_svg(ledger_path: Path, svg_path: Path) -> list[str]:
    """The committed figure must be exactly what the ledger renders to."""
    if not svg_path.exists():
        return [f"{svg_path} does not exist"]
    try:
        expected = render_svg(load_record(ledger_path))
    except Exception as exc:  # noqa: BLE001 - any failure here is a finding
        return [f"cannot render the figure from the ledger: {exc}"]

    if svg_path.read_text(encoding="utf-8") != expected:
        return [
            "the committed track-record SVG is not what the current ledger renders to; "
            "it is stale or was edited by hand"
        ]
    return []


def audit(
    ledger_path: Path = LEDGER_PATH,
    summary_path: Path = SUMMARY_PATH,
    svg_path: Path = OUTPUT_PATH,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> list[str]:
    rows, problems = read_ledger_rows(ledger_path)
    problems += check_ledger_shape(rows)
    problems += check_against_sealed_predictions(rows, predictions_dir)
    problems += check_summary(rows, summary_path)
    problems += check_svg(ledger_path, svg_path)
    return problems


def main(argv: list[str] | None = None) -> int:
    problems = audit()
    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        print(f"\n{len(problems)} discrepancy(ies) found. Nothing has been changed.", file=sys.stderr)
        return 1
    print("OK: ledger, SUMMARY.md and the track-record figure agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
