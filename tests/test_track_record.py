"""Regression tests for the ledger-derived prospective track-record figure."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from de_power_live.track_record import (
    TrackRecordError,
    cumulative_points,
    load_record,
    render_svg,
    write_svg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "results" / "ledger.jsonl"
SVG_PATH = REPO_ROOT / "results" / "prospective_track_record.svg"
PREDICT_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "predict.yml"


def _scored(
    delivery_date: str,
    hours: int = 24,
    *,
    model_mae: float = 10.0,
    b1_mae: float = 20.0,
    b2_mae: float = 15.0,
    negative_indices: set[int] | None = None,
    prices: list[float] | None = None,
    model_scores: list[float] | None = None,
    baseline_scores: list[float] | None = None,
    coverage_50_hits: int | None = None,
    coverage_80_hits: int | None = None,
    model_version: str = "v1",
) -> dict:
    negative_indices = negative_indices or set()
    if prices is None:
        prices = [-5.0 if i in negative_indices else 50.0 for i in range(hours)]
    assert len(prices) == hours
    model_scores = model_scores or [0.1] * hours
    baseline_scores = baseline_scores or [0.2] * hours
    assert len(model_scores) == len(baseline_scores) == hours

    hourly = []
    for i, price in enumerate(prices):
        outcome_negative = int(price < 0)
        hourly.append(
            {
                "hour_utc": f"{delivery_date}T{i:02d}:00:00+00:00",
                "model_score": model_scores[i],
                "price": price,
                "outcome_negative": outcome_negative,
                "baseline_negative": baseline_scores[i],
                "outcome_below_10": int(price < 10),
            }
        )

    model_brier = sum(
        (item["model_score"] - item["outcome_negative"]) ** 2 for item in hourly
    ) / hours
    negative_hours = sum(item["outcome_negative"] for item in hourly)
    below_10_hours = sum(item["outcome_below_10"] for item in hourly)
    coverage_50_hits = hours // 2 if coverage_50_hits is None else coverage_50_hits
    coverage_80_hits = round(hours * 0.8) if coverage_80_hits is None else coverage_80_hits

    return {
        "delivery_date_local": delivery_date,
        "status": "SCORED",
        "model_version": model_version,
        "n_hours_predicted": hours,
        "n_hours_scored": hours,
        "call_a": {
            "mae_model": model_mae,
            "mae_baseline_b1_lag168": b1_mae,
            "mae_baseline_b2_lag24": b2_mae,
        },
        "call_b": {
            "n_negative_hours": negative_hours,
            "n_below_10_hours": below_10_hours,
            "brier": round(model_brier, 6),
            "hours": hourly,
        },
        "call_c": {
            "coverage_50": round(coverage_50_hits / hours, 5),
            "coverage_80": round(coverage_80_hits / hours, 5),
        },
    }


def _write_ledger(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_generation_is_byte_identical_and_does_not_mutate_ledger(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    output = tmp_path / "figure.svg"
    _write_ledger(ledger, [_scored("2026-08-03")])
    before = ledger.read_bytes()

    write_svg(ledger, output)
    first = output.read_bytes()
    write_svg(ledger, output)
    second = output.read_bytes()

    assert first == second
    assert ledger.read_bytes() == before


def test_pending_and_missed_rows_are_excluded_from_metrics(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    output = tmp_path / "figure.svg"
    _write_ledger(
        ledger,
        [
            _scored("2026-08-03", hours=24),
            {"delivery_date_local": "2026-08-04", "status": "PENDING"},
            {"delivery_date_local": "2026-08-05", "status": "MISSED"},
        ],
    )

    record = write_svg(ledger, output)
    assert len(record.days) == 1
    assert record.scored_hours == 24
    assert record.pending_count == 1
    assert record.missed_count == 1
    assert "1 pending day" in output.read_text(encoding="utf-8")


def test_scored_rows_are_sorted_chronologically(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [_scored("2026-08-03"), _scored("2026-08-01")],
    )

    record = load_record(ledger)
    assert [day.delivery_date.isoformat() for day in record.days] == [
        "2026-08-01",
        "2026-08-03",
    ]


def test_mae_is_weighted_by_unequal_23_24_25_hour_denominators(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            _scored("2026-03-29", hours=23, model_mae=10, b1_mae=30, b2_mae=20),
            _scored("2026-08-01", hours=24, model_mae=20, b1_mae=40, b2_mae=30),
            _scored("2026-10-25", hours=25, model_mae=30, b1_mae=50, b2_mae=40),
        ],
    )

    endpoint = cumulative_points(load_record(ledger))[-1]
    assert endpoint["hours"] == 72
    assert endpoint["mae_model"] == pytest.approx((10 * 23 + 20 * 24 + 30 * 25) / 72)
    assert endpoint["mae_b1"] == pytest.approx((30 * 23 + 40 * 24 + 50 * 25) / 72)
    assert endpoint["mae_b2"] == pytest.approx((20 * 23 + 30 * 24 + 40 * 25) / 72)


def test_model_and_baseline_brier_scores_are_reconstructed_from_hourly_records(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            _scored(
                "2026-08-01",
                hours=4,
                prices=[50.0, -5.0, -5.0, 50.0],
                model_scores=[0.0, 0.8, 0.2, 0.4],
                baseline_scores=[0.1, 0.5, 0.5, 0.1],
            )
        ],
    )

    endpoint = cumulative_points(load_record(ledger))[-1]
    assert endpoint["brier_model"] == pytest.approx(0.21)
    assert endpoint["brier_baseline"] == pytest.approx(0.13)
    assert endpoint["negative_hours"] == 2


def test_interval_coverage_uses_recovered_hourly_hit_counts(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            _scored(
                "2026-08-01",
                hours=3,
                coverage_50_hits=1,
                coverage_80_hits=2,
            ),
            _scored(
                "2026-08-02",
                hours=5,
                coverage_50_hits=4,
                coverage_80_hits=4,
            ),
        ],
    )

    endpoint = cumulative_points(load_record(ledger))[-1]
    assert endpoint["coverage_50"] == pytest.approx(5 / 8)
    assert endpoint["coverage_80"] == pytest.approx(6 / 8)


def test_duplicate_scored_delivery_dates_are_rejected(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [_scored("2026-08-01"), _scored("2026-08-01")])

    with pytest.raises(TrackRecordError, match="duplicate scored delivery date"):
        load_record(ledger)


def test_malformed_and_semantically_inconsistent_rows_are_rejected(tmp_path: Path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"status":"SCORED"\n', encoding="utf-8")
    with pytest.raises(TrackRecordError, match="malformed JSON"):
        load_record(malformed)

    inconsistent = tmp_path / "inconsistent.jsonl"
    row = _scored("2026-08-01", hours=24)
    row["n_hours_scored"] = 23
    _write_ledger(inconsistent, [row])
    with pytest.raises(TrackRecordError, match="n_hours_scored"):
        load_record(inconsistent)


def test_required_boundary_wording_is_present(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [_scored("2026-08-01")])
    svg = render_svg(load_record(ledger))

    for wording in (
        "Prospective DE-LU forecast track record",
        "Sealed before auction close",
        "scored after settlement",
        "lower is better",
        "SCORED rows only",
        "no final claim",
    ):
        assert wording in svg


def test_readme_references_the_committed_figure_and_results_explanation():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "[![Prospective DE-LU forecast record" in readme
    assert "results/prospective_track_record.svg" in readme
    assert "](results/README.md)" in readme
    assert SVG_PATH.exists()


def test_committed_figure_matches_the_current_ledger():
    assert LEDGER_PATH.exists()
    assert SVG_PATH.exists()
    expected = render_svg(load_record(LEDGER_PATH))
    assert SVG_PATH.read_text(encoding="utf-8") == expected


def test_predict_workflow_has_truthful_success_and_miss_release_paths():
    workflow = PREDICT_WORKFLOW_PATH.read_text(encoding="utf-8")

    miss_start = workflow.index("      - name: Record a miss if sealing failed")
    generate_start = workflow.index(
        "      - name: Generate track-record figure for a missed forecast"
    )
    commit_start = workflow.index("      - name: Commit", generate_start)
    miss_block = workflow[miss_start:generate_start]
    generate_block = workflow[generate_start:commit_start]
    commit_block = workflow[commit_start:]

    assert "id: miss" in miss_block
    assert "if: failure() && steps.seal.outcome == 'failure'" in miss_block
    assert "id: figure" in generate_block
    # The bug this pins: an "if:" with no status function gets an implicit
    # success() from Actions, so on the failure path this step - and the Commit
    # step that keys off its outcome - were silently skipped, and the recorded
    # miss died with the runner instead of reaching the ledger.
    assert "if: always() && steps.miss.outcome == 'success'" in generate_block
    assert "python -m de_power_live.track_record" in generate_block

    assert (
        "if: always() && (steps.seal.outcome == 'success' || steps.figure.outcome == 'success')"
        in commit_block
    )
    assert 'if [ "${{ steps.seal.outcome }}" = "success" ]; then' in commit_block
    assert 'git commit -m "Seal forecast ' in commit_block
    assert 'git commit -m "Record missed forecast ' in commit_block
    assert "No forecast was sealed before gate closure" in commit_block
    assert "Sealed by scheduled run" in commit_block


def test_every_failure_path_step_carries_a_status_function():
    """Steps on the miss path must opt out of the implicit success() gate.

    GitHub Actions ANDs a bare ``if:`` with ``success()``. Once the seal step
    fails the job is in a failed state, so any later step whose condition omits
    ``always()``/``failure()`` is skipped - which is precisely how five recorded
    misses (2026-08-30 to 2026-09-03) were computed on the runner and then
    thrown away instead of being committed.
    """
    workflow = PREDICT_WORKFLOW_PATH.read_text(encoding="utf-8")

    conditions = re.findall(r"^\s+if: (.+)$", workflow, flags=re.MULTILINE)
    failure_path = [c for c in conditions if "steps.miss" in c or "steps.figure" in c]
    assert failure_path, "no failure-path conditions found; step names changed?"

    for condition in failure_path:
        assert any(fn in condition for fn in ("always()", "failure()")), (
            f"{condition!r} has no status function, so Actions ANDs it with "
            "success() and skips the step whenever the seal has failed"
        )


def test_predict_is_scheduled_with_real_slack_before_the_gate():
    """The cron must leave room for Actions queueing delay, and sit off the hour.

    The summer gate is 10:00 UTC. At "0 6" the platform delivered every run
    between 4h30 and 12h late from 2026-08-27 onward, so four hours of slack on
    the most contended minute of the hour was not enough.
    """
    workflow = PREDICT_WORKFLOW_PATH.read_text(encoding="utf-8")

    match = re.search(r'cron: "(\d+) (\d+) \* \* \*"', workflow)
    assert match, "predict cron not found or no longer a simple daily schedule"
    minute, hour = int(match.group(1)), int(match.group(2))

    summer_gate_utc = 10 * 60
    slack_minutes = summer_gate_utc - (hour * 60 + minute)
    assert slack_minutes >= 6 * 60, (
        f"only {slack_minutes} min of slack before the 10:00 UTC summer gate"
    )
    assert minute != 0, "on-the-hour crons sit in the most delayed Actions slot"
