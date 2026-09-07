"""Tests for the gate-margin alarm.

An alarm that cannot fire is decoration, so most of these prove it bites. The
last one pins the real invariant: the alarm must trip on the exact margins that
preceded the 2026-08-28..09-04 gap, not merely on a hand-picked toy value.
"""

from __future__ import annotations

import json
from pathlib import Path

from de_power_live.gate_margin import (
    CRITICAL_MINUTES,
    DEFAULT_WARN_MINUTES,
    assess,
    collect,
    read_margin,
    sealed_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _seal(directory: Path, delivery: str, margin: float | None) -> Path:
    payload: dict = {"delivery_date_local": delivery}
    if margin is not None:
        payload["minutes_before_close"] = margin
    path = directory / f"{delivery}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_healthy_margin_passes(tmp_path: Path):
    _seal(tmp_path, "2026-09-08", 240.0)
    report = assess(collect(tmp_path), DEFAULT_WARN_MINUTES)
    assert report["ok"]
    assert report["level"] == "ok"


def test_thin_margin_warns(tmp_path: Path):
    _seal(tmp_path, "2026-09-08", 75.0)
    report = assess(collect(tmp_path), DEFAULT_WARN_MINUTES)
    assert not report["ok"]
    assert report["level"] == "warn"


def test_very_thin_margin_is_critical(tmp_path: Path):
    _seal(tmp_path, "2026-09-08", 20.0)
    report = assess(collect(tmp_path), DEFAULT_WARN_MINUTES)
    assert report["level"] == "critical"
    assert not report["ok"]


def test_no_seals_is_not_silently_ok(tmp_path: Path):
    report = assess(collect(tmp_path), DEFAULT_WARN_MINUTES)
    assert not report["ok"]


def test_non_date_json_is_not_mistaken_for_a_seal(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("not a seal", encoding="utf-8")
    _seal(tmp_path, "2026-09-08", 240.0)
    assert [p.name for p in sealed_files(tmp_path)] == ["2026-09-08.json"]


def test_trend_reports_the_slide_not_just_the_last_reading(tmp_path: Path):
    for delivery, margin in (
        ("2026-09-05", 121.4),
        ("2026-09-06", 139.5),
        ("2026-09-07", 125.4),
        ("2026-09-08", 103.2),
    ):
        _seal(tmp_path, delivery, margin)
    report = assess(collect(tmp_path), DEFAULT_WARN_MINUTES)
    assert report["trend_minutes_over_window"] < 0
    assert report["delivery_date_local"] == "2026-09-08"


def test_margin_is_read_as_a_float(tmp_path: Path):
    _seal(tmp_path, "2026-09-08", 103)
    assert read_margin(tmp_path / "2026-09-08.json") == ("2026-09-08", 103.0)


def test_thresholds_are_ordered():
    assert CRITICAL_MINUTES < DEFAULT_WARN_MINUTES


def test_every_committed_seal_records_its_margin():
    """The alarm depends on this field. If a seal ever lacks it, fail here."""
    for path in sealed_files(REPO_ROOT / "predictions"):
        delivery, margin = read_margin(path)
        assert margin > 0, f"{delivery} was sealed at or after gate closure"
