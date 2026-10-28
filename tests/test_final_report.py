"""Tests for the formal assessment of the evaluation window."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from de_power_live import final_report
from de_power_live.final_report import (
    WindowIncomplete,
    build_report,
    local_day_hours,
    render_markdown,
)
from de_power_live.track_record import EVALUATION_END, load_record

BERLIN = ZoneInfo("Europe/Berlin")
FIRST = date(2026, 10, 20)


@pytest.fixture(autouse=True)
def _fewer_bootstrap_draws(monkeypatch):
    # The verdict logic does not depend on the draw count, and 10,000 draws per
    # report made this file the slowest in the suite by a wide margin.
    monkeypatch.setattr(final_report, "BOOTSTRAP_DRAWS", 200)


def _hours_utc(day: date) -> list[str]:
    start = datetime.combine(day, time(), BERLIN).astimezone(timezone.utc)
    return [
        (start + timedelta(hours=i)).isoformat()
        for i in range(local_day_hours(day))
    ]


def _day(
    day: date,
    *,
    model_mae: float = 10.0,
    b1_mae: float = 20.0,
    negative: set[int] = frozenset(),
    model_hits: bool = True,
    inside_80: float = 0.8,
    inside_50: float = 0.5,
) -> tuple[dict, dict]:
    """One SCORED ledger row and the sealed prediction it was scored from."""
    hours = _hours_utc(day)
    n = len(hours)
    n80, n50 = round(n * inside_80), round(n * inside_50)
    predictions, hourly = [], []
    for i, ts in enumerate(hours):
        price = -5.0 if i in negative else 50.0
        # Shift the quantiles so the outturn lands inside the 50% band for the
        # first n50 hours, inside only the 80% band up to n80, and outside after.
        offset = 0.0 if i < n50 else 30.0 if i < n80 else 100.0
        q = [price + offset + d for d in (-40, -20, 0, 20, 40)]
        predictions.append(
            {
                "hour_utc": ts,
                "call_c_quantiles": {str(k): v for k, v in zip(final_report.QUANTILES, q)},
            }
        )
        is_neg = int(price < 0)
        hourly.append(
            {
                "hour_utc": ts,
                "model_score": (0.9 if is_neg else 0.05) if model_hits else 0.1,
                "price": price,
                "outcome_negative": is_neg,
                "baseline_negative": 0.1,
                "outcome_below_10": int(price < 10),
                "baseline_below_10": 0.1,
            }
        )
    brier = sum((h["model_score"] - h["outcome_negative"]) ** 2 for h in hourly) / n
    row = {
        "delivery_date_local": day.isoformat(),
        "status": "SCORED",
        "minutes_before_close": 120.0,
        "model_version": "v1",
        "model_source_sha256": "a" * 64,
        "n_hours_predicted": n,
        "n_hours_scored": n,
        "call_a": {
            "mae_model": model_mae,
            "mae_baseline_b1_lag168": b1_mae,
            "mae_baseline_b2_lag24": 15.0,
            "rmse_model": model_mae * 1.2,
            "bias_model": -1.0,
        },
        "call_b": {
            "n_negative_hours": sum(h["outcome_negative"] for h in hourly),
            "n_below_10_hours": sum(h["outcome_below_10"] for h in hourly),
            "brier": round(brier, 6),
            "hours": hourly,
        },
        "call_c": {"coverage_50": round(n50 / n, 5), "coverage_80": round(n80 / n, 5)},
    }
    return row, {"delivery_date_local": day.isoformat(), "predictions": predictions}


def _missed(day: date) -> dict:
    return {"delivery_date_local": day.isoformat(), "status": "MISSED", "reason": "predict job failed"}


def _write(tmp_path: Path, days: list[tuple[dict, dict | None]]) -> tuple[Path, Path]:
    ledger = tmp_path / "ledger.jsonl"
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    ledger.write_text("".join(json.dumps(row) + "\n" for row, _ in days), encoding="utf-8")
    for row, payload in days:
        if payload is not None:
            (predictions / f"{row['delivery_date_local']}.json").write_text(json.dumps(payload))
    return ledger, predictions


def _window(**kwargs) -> list[tuple[dict, dict | None]]:
    """FIRST through EVALUATION_END, the 25-hour day missed, the rest scored."""
    days, day = [], FIRST
    while day <= EVALUATION_END:
        days.append((_missed(day), None) if day == date(2026, 10, 25) else _day(day, **kwargs))
        day += timedelta(days=1)
    return days


def test_day_lengths_follow_the_market_clock():
    assert local_day_hours(date(2026, 10, 25)) == 25
    assert local_day_hours(date(2026, 3, 29)) == 23
    assert local_day_hours(date(2026, 10, 26)) == 24


def test_an_incomplete_window_refuses_to_report(tmp_path):
    ledger, predictions = _write(tmp_path, _window()[:-1])
    with pytest.raises(WindowIncomplete, match="2026-10-31"):
        build_report(ledger, predictions)


def test_a_clear_improvement_passes_call_a_and_b(tmp_path):
    ledger, predictions = _write(tmp_path, _window(negative={10, 11, 12}))
    report = build_report(ledger, predictions)["pooled"]
    assert report["call_a"]["verdict"] == "PASS"
    assert report["call_a"]["vs_b1"]["skill"] == pytest.approx(0.5)
    assert report["call_b"]["tested_target"] == "negative"
    assert report["call_b"]["verdict"] == "PASS"


def test_no_better_than_the_baseline_is_null(tmp_path):
    ledger, predictions = _write(tmp_path, _window(model_mae=20.0, negative={10, 11, 12}))
    assert build_report(ledger, predictions)["pooled"]["call_a"]["verdict"] == "NULL"


def test_too_few_events_at_both_thresholds_is_underpowered(tmp_path):
    ledger, predictions = _write(tmp_path, _window(negative={10}))
    call_b = build_report(ledger, predictions)["pooled"]["call_b"]
    assert call_b["targets"]["negative"]["events"] < 30
    assert call_b["tested_target"] is None
    assert call_b["verdict"] == "UNDERPOWERED"


def test_coverage_inside_and_outside_the_band(tmp_path):
    ledger, predictions = _write(tmp_path, _window(negative={10, 11, 12}))
    assert build_report(ledger, predictions)["pooled"]["call_c"]["verdict"] == "PASS"

    narrow = tmp_path / "narrow"
    narrow.mkdir()
    ledger, predictions = _write(narrow, _window(negative={10, 11, 12}, inside_80=0.6))
    call_c = build_report(ledger, predictions)["pooled"]["call_c"]
    assert call_c["levels"]["80"]["within_tolerance"] is False
    assert call_c["verdict"] == "NULL"


def test_ledger_coverage_must_match_the_sealed_quantiles(tmp_path):
    days = _window(negative={10, 11, 12})
    days[0][0]["call_c"]["coverage_80"] = 1.0
    ledger, predictions = _write(tmp_path, days)
    with pytest.raises(ValueError, match="coverage_80"):
        build_report(ledger, predictions)


def test_days_after_the_window_change_nothing(tmp_path):
    inside = _window(negative={10, 11, 12})
    ledger, predictions = _write(tmp_path, inside)
    before = build_report(ledger, predictions)

    later = tmp_path / "later"
    later.mkdir()
    after = [_day(EVALUATION_END + timedelta(days=i), model_mae=99.0) for i in (1, 2)]
    ledger, predictions = _write(later, inside + after)
    assert build_report(ledger, predictions) == before
    assert load_record(ledger).latest_date == EVALUATION_END
    assert load_record(ledger, through=None).latest_date == EVALUATION_END + timedelta(days=2)


def test_the_report_is_deterministic_and_names_the_clock_change_day(tmp_path):
    ledger, predictions = _write(tmp_path, _window(negative={10, 11, 12}))
    first = render_markdown(build_report(ledger, predictions))
    assert first == render_markdown(build_report(ledger, predictions))
    assert "| 2026-10-25 | 25 | **MISSED**: predict job failed |" in first
    assert "docs/v1-lessons.md" in first
