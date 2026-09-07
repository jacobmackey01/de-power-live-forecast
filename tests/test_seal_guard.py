"""Tests for the pre-seal guard.

The failure this guards against is specific and quiet: a file exists at
predictions/<day>.json, the workflow reads its presence as success, and the run
exits green having sealed nothing and recorded nothing. The day then leaves the
record entirely rather than counting against it, which is worse than a MISSED
row - a miss is honest, a disappearance is not.

So most of these tests feed the guard files that a naive existence check would
accept, and require it to refuse them.
"""

from __future__ import annotations

import json
from pathlib import Path

from de_power_live.seal_guard import (
    ABSENT,
    INVALID,
    SEALED,
    inspect,
    recorded_in_ledger,
)

TARGET = "2026-09-09"


def _valid_payload(delivery: str = TARGET, n_hours: int = 2) -> dict:
    return {
        "delivery_date_local": delivery,
        "sealed_at_utc": "2026-09-08T05:30:00+00:00",
        "auction_closes_utc": "2026-09-08T10:00:00+00:00",
        "minutes_before_close": 270.0,
        "model_version": "v1",
        "model_source_sha256": "0" * 64,
        "n_hours": n_hours,
        "predictions": [
            {
                "hour_utc": f"2026-09-08T{22 + i:02d}:00:00+00:00",
                "call_a_price_eur_mwh": 100.0,
                "call_b_prob_negative": 0.001,
            }
            for i in range(n_hours)
        ],
    }


def _write(directory: Path, name: str, payload) -> Path:
    path = directory / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---- The three benign outcomes ------------------------------------------


def test_absent_file_means_proceed(tmp_path: Path):
    state, problems = inspect(TARGET, tmp_path)
    assert state == ABSENT
    assert problems == []


def test_a_complete_seal_is_recognised(tmp_path: Path):
    _write(tmp_path, f"{TARGET}.json", _valid_payload())
    state, problems = inspect(TARGET, tmp_path)
    assert state == SEALED, problems
    assert problems == []


def test_the_committed_seals_all_pass_the_guard():
    """The real files must satisfy the same check the workflow applies."""
    predictions = Path(__file__).resolve().parents[1] / "predictions"
    for path in sorted(predictions.glob("*.json")):
        state, problems = inspect(path.stem, predictions)
        assert state == SEALED, f"{path.name}: {problems}"


# ---- Files a bare existence check would have accepted --------------------


def test_empty_file_is_invalid(tmp_path: Path):
    _write(tmp_path, f"{TARGET}.json", "")
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "empty" in problems[0]


def test_truncated_json_is_invalid(tmp_path: Path):
    """The realistic shape of an interrupted write."""
    body = json.dumps(_valid_payload())[: len(json.dumps(_valid_payload())) // 2]
    _write(tmp_path, f"{TARGET}.json", body)
    state, _ = inspect(TARGET, tmp_path)
    assert state == INVALID


def test_json_that_is_not_an_object_is_invalid(tmp_path: Path):
    _write(tmp_path, f"{TARGET}.json", [1, 2, 3])
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "not an object" in problems[0]


def test_missing_required_fields_are_named(tmp_path: Path):
    payload = _valid_payload()
    del payload["model_source_sha256"]
    del payload["minutes_before_close"]
    _write(tmp_path, f"{TARGET}.json", payload)
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "model_source_sha256" in problems[0]
    assert "minutes_before_close" in problems[0]


def test_a_file_sealed_for_another_day_is_invalid(tmp_path: Path):
    """The most dangerous case: every downstream lookup is by filename."""
    _write(tmp_path, f"{TARGET}.json", _valid_payload(delivery="2026-09-01"))
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "2026-09-01" in problems[0]


def test_a_seal_after_gate_closure_is_invalid(tmp_path: Path):
    """A non-positive margin means it was not a prediction when it was written."""
    payload = _valid_payload()
    payload["minutes_before_close"] = -12.0
    _write(tmp_path, f"{TARGET}.json", payload)
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "gate closure" in problems[0]


def test_a_non_numeric_margin_is_invalid(tmp_path: Path):
    payload = _valid_payload()
    payload["minutes_before_close"] = "lots"
    _write(tmp_path, f"{TARGET}.json", payload)
    state, _ = inspect(TARGET, tmp_path)
    assert state == INVALID


def test_a_boolean_margin_is_not_mistaken_for_a_number(tmp_path: Path):
    """bool is an int in Python; True would otherwise read as one minute."""
    payload = _valid_payload()
    payload["minutes_before_close"] = True
    _write(tmp_path, f"{TARGET}.json", payload)
    state, _ = inspect(TARGET, tmp_path)
    assert state == INVALID


def test_unparseable_timestamps_are_invalid(tmp_path: Path):
    payload = _valid_payload()
    payload["sealed_at_utc"] = "yesterday morning"
    _write(tmp_path, f"{TARGET}.json", payload)
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "sealed_at_utc" in problems[0]


def test_an_empty_prediction_list_is_invalid(tmp_path: Path):
    payload = _valid_payload()
    payload["predictions"] = []
    _write(tmp_path, f"{TARGET}.json", payload)
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "no predictions" in problems[0]


def test_an_hour_count_mismatch_is_invalid(tmp_path: Path):
    """A short write that still parses: 24 claimed, fewer present."""
    payload = _valid_payload(n_hours=2)
    payload["n_hours"] = 24
    _write(tmp_path, f"{TARGET}.json", payload)
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "24 hours" in problems[0]


def test_an_hour_missing_a_call_is_invalid(tmp_path: Path):
    """All three calls are pre-registered; a seal missing one is incomplete."""
    payload = _valid_payload()
    del payload["predictions"][1]["call_b_prob_negative"]
    _write(tmp_path, f"{TARGET}.json", payload)
    state, problems = inspect(TARGET, tmp_path)
    assert state == INVALID
    assert "call_b_prob_negative" in problems[0]


# ---- The ledger check ---------------------------------------------------


def test_ledger_row_is_found(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"delivery_date_local": TARGET, "status": "MISSED"}) + "\n",
        encoding="utf-8",
    )
    assert recorded_in_ledger(TARGET, ledger)


def test_absent_ledger_is_not_an_error(tmp_path: Path):
    assert not recorded_in_ledger(TARGET, tmp_path / "nothing.jsonl")


def test_a_date_mentioned_only_in_a_reason_does_not_count(tmp_path: Path):
    """A substring grep would match here and suppress a legitimate seal."""
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "delivery_date_local": "2026-09-01",
                "status": "MISSED",
                "reason": f"queue delay; see also {TARGET}",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert not recorded_in_ledger(TARGET, ledger)


def test_a_malformed_ledger_line_does_not_stop_the_check(tmp_path: Path):
    """Malformed lines are audit.py's business; the guard must still answer."""
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "{not json\n"
        + json.dumps({"delivery_date_local": TARGET, "status": "SCORED"})
        + "\n",
        encoding="utf-8",
    )
    assert recorded_in_ledger(TARGET, ledger)
