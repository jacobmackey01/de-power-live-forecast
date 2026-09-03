"""Tests for the read-only record audit.

The headline test is the first one: the record as committed must audit clean.
The rest prove the audit actually bites, since an auditor that cannot fail is
worse than no auditor at all.
"""

from __future__ import annotations

from pathlib import Path

from de_power_live.audit import (
    audit,
    check_against_sealed_predictions,
    check_ledger_shape,
    check_summary,
    check_svg,
    read_ledger_rows,
)
from de_power_live.track_record import LEDGER_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]


def _summary(scored: int, missed_days: list[str]) -> str:
    listed = f" ({', '.join(missed_days)})" if missed_days else ""
    return (
        "# Running scoring ledger\n\n"
        f"- Days scored: **{scored}**\n"
        f"- Days missed: **{len(missed_days)}**{listed}\n"
    )


def test_the_committed_record_audits_clean():
    assert audit() == []


# ---- the ledger itself ---------------------------------------------------


def test_malformed_json_is_reported_not_raised(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"delivery_date_local": "2026-08-01", "status": "MISSED"}\nnot json\n')
    rows, problems = read_ledger_rows(ledger)
    assert len(rows) == 1
    assert any("line 2" in p and "not valid JSON" in p for p in problems)


def test_missing_ledger_is_reported(tmp_path):
    rows, problems = read_ledger_rows(tmp_path / "absent.jsonl")
    assert rows == []
    assert any("does not exist" in p for p in problems)


def test_duplicate_delivery_day_is_reported():
    rows = [
        {"delivery_date_local": "2026-08-01", "status": "MISSED"},
        {"delivery_date_local": "2026-08-01", "status": "SCORED"},
    ]
    assert any("duplicated" in p for p in check_ledger_shape(rows))


def test_unknown_status_is_reported():
    rows = [{"delivery_date_local": "2026-08-01", "status": "PENDING"}]
    assert any("PENDING" in p for p in check_ledger_shape(rows))


def test_a_clean_ledger_shape_reports_nothing():
    rows = [
        {"delivery_date_local": "2026-08-01", "status": "SCORED"},
        {"delivery_date_local": "2026-08-02", "status": "MISSED"},
    ]
    assert check_ledger_shape(rows) == []


# ---- ledger against the sealed files -------------------------------------


def test_missed_day_with_a_sealed_prediction_is_reported(tmp_path):
    (tmp_path / "2026-08-01.json").write_text("{}")
    rows = [{"delivery_date_local": "2026-08-01", "status": "MISSED"}]
    problems = check_against_sealed_predictions(rows, tmp_path)
    assert any("must never be recorded as missed" in p for p in problems)


def test_scored_day_without_a_sealed_prediction_is_reported(tmp_path):
    rows = [{"delivery_date_local": "2026-08-01", "status": "SCORED"}]
    problems = check_against_sealed_predictions(rows, tmp_path)
    assert any("is missing" in p for p in problems)


# ---- ledger against SUMMARY.md -------------------------------------------


def test_summary_matching_the_ledger_reports_nothing(tmp_path):
    rows = [
        {"delivery_date_local": "2026-08-01", "status": "SCORED"},
        {"delivery_date_local": "2026-08-02", "status": "MISSED"},
    ]
    path = tmp_path / "SUMMARY.md"
    path.write_text(_summary(1, ["2026-08-02"]))
    assert check_summary(rows, path) == []


def test_understated_miss_count_is_reported(tmp_path):
    """The exact failure this audit exists to catch."""
    rows = [{"delivery_date_local": f"2026-08-0{n}", "status": "MISSED"} for n in range(1, 6)]
    path = tmp_path / "SUMMARY.md"
    path.write_text(_summary(0, ["2026-08-01", "2026-08-02"]))

    problems = check_summary(rows, path)
    assert any("claims 2 days missed" in p and "holds 5" in p for p in problems)


def test_miss_list_disagreeing_with_the_ledger_is_reported(tmp_path):
    rows = [
        {"delivery_date_local": "2026-08-01", "status": "MISSED"},
        {"delivery_date_local": "2026-08-02", "status": "MISSED"},
    ]
    path = tmp_path / "SUMMARY.md"
    path.write_text(_summary(0, ["2026-08-01", "2026-08-09"]))
    assert any("lists missed days" in p for p in check_summary(rows, path))


def test_wrong_scored_count_is_reported(tmp_path):
    rows = [{"delivery_date_local": "2026-08-01", "status": "SCORED"}]
    path = tmp_path / "SUMMARY.md"
    path.write_text(_summary(99, []))
    assert any("claims 99 days scored" in p for p in check_summary(rows, path))


def test_missing_summary_is_reported(tmp_path):
    assert any("does not exist" in p for p in check_summary([], tmp_path / "gone.md"))


# ---- ledger against the figure -------------------------------------------


def test_stale_figure_is_reported(tmp_path):
    stale = tmp_path / "track_record.svg"
    stale.write_text("<svg><!-- not what the ledger renders to --></svg>")
    problems = check_svg(LEDGER_PATH, stale)
    assert any("stale or was edited by hand" in p for p in problems)


def test_missing_figure_is_reported(tmp_path):
    assert any("does not exist" in p for p in check_svg(LEDGER_PATH, tmp_path / "gone.svg"))


def test_unreadable_ledger_is_reported_not_raised(tmp_path):
    broken = tmp_path / "ledger.jsonl"
    broken.write_text("{not json}\n")
    present = tmp_path / "any.svg"
    present.write_text("<svg/>")
    problems = check_svg(broken, present)
    assert problems and all(isinstance(p, str) for p in problems)


# ---- the audit never writes ----------------------------------------------


def test_audit_leaves_the_published_artefacts_untouched():
    summary = REPO_ROOT / "results" / "SUMMARY.md"
    svg = REPO_ROOT / "results" / "prospective_track_record.svg"
    ledger = REPO_ROOT / "results" / "ledger.jsonl"
    before = {p: p.read_bytes() for p in (summary, svg, ledger)}

    audit()

    for path, content in before.items():
        assert path.read_bytes() == content, f"the audit modified {path.name}"
