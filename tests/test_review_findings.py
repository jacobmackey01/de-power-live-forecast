"""Regression tests for the four code-review findings.

Each test fails against the code as it stood before the fix, so they pin the
behaviour rather than merely exercising it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from de_power_live.model import (
    CALL_B_THRESHOLDS,
    ModelIntegrityError,
    build_climatology,
    climatology_probability,
    verify_frozen_model,
)
from de_power_live.score import IncompleteOutturn, load_climatology, score_day

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


# ---- helpers ------------------------------------------------------------


def _payload(hours: pd.DatetimeIndex, price: float = 50.0) -> dict:
    rows = []
    for ts in hours:
        rows.append(
            {
                "hour_utc": ts.isoformat(),
                "hour_local": ts.tz_convert("Europe/Berlin").isoformat(),
                "call_a_price_eur_mwh": price,
                "call_b_prob_negative": 0.1,
                "call_c_quantiles": {
                    str(level): price + offset
                    for level, offset in zip(QUANTILE_LEVELS, (-20, -10, 0, 10, 20))
                },
            }
        )
    return {
        "delivery_date_local": str(hours[0].tz_convert("Europe/Berlin").date()),
        "sealed_at_utc": "2026-08-01T09:00:00+00:00",
        "minutes_before_close": 60.0,
        "model_version": "test",
        "model_source_sha256": "deadbeef",
        "predictions": rows,
    }


def _prices(hours: pd.DatetimeIndex, value: float = 55.0) -> pd.Series:
    """Realised prices covering the target day and both baseline lags.

    Varies on a 5-hour cycle. 168 and 24 are both divisible by neither 5 nor
    each other's cycle, so the lag-168 and lag-24 baselines genuinely differ
    from the target hour - a flat series would make MAE(B1) exactly zero and
    the skill ratio undefined.
    """
    full = pd.date_range(
        hours[0] - pd.Timedelta(hours=200), hours[-1] + pd.Timedelta(hours=1),
        freq="h", tz="UTC",
    )
    offsets = (np.arange(len(full)) % 5) * 3.0
    return pd.Series(value + offsets, index=full)


def _fake_climatology(monkeypatch):
    table = {
        name: {str(m): {str(h): 0.05 for h in range(24)} for m in range(1, 13)}
        for name in CALL_B_THRESHOLDS
    }
    monkeypatch.setattr(
        "de_power_live.score.load_climatology",
        lambda version: (table, CALL_B_THRESHOLDS),
    )


# ---- Finding 1: call B baseline frozen and scored ------------------------


def test_climatology_is_12x24_and_bounded():
    index = pd.date_range("2025-01-01", "2026-01-01", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    prices = pd.Series(rng.normal(50, 40, len(index)), index=index)

    table = build_climatology(prices, threshold=0.0)
    assert len(table) == 12
    assert all(len(hours) == 24 for hours in table.values())
    assert all(0.0 <= v <= 1.0 for hours in table.values() for v in hours.values())


def test_climatology_survives_json_round_trip():
    """Keys must be strings or the manifest silently reshapes the table."""
    index = pd.date_range("2025-01-01", "2025-06-01", freq="h", tz="UTC")
    prices = pd.Series(np.linspace(-20, 90, len(index)), index=index)
    table = build_climatology(prices, threshold=0.0)
    assert json.loads(json.dumps(table)) == table


def test_climatology_lookup_uses_local_month_and_hour():
    table = {str(m): {str(h): m * 100 + h for h in range(24)} for m in range(1, 13)}
    # 23:00 UTC in August is 01:00 the next day in Berlin (CEST, UTC+2).
    ts = pd.Timestamp("2026-08-01T23:00:00Z")
    assert climatology_probability(table, ts) == 8 * 100 + 1


def test_scoring_records_both_thresholds_and_baselines(monkeypatch):
    """The pre-declared fallback target must be evaluable from the ledger alone."""
    _fake_climatology(monkeypatch)
    hours = pd.date_range("2026-08-01T22:00:00Z", periods=24, freq="h", tz="UTC")
    prices = _prices(hours, 55.0)
    prices.loc[hours[:3]] = -5.0  # negative, and therefore also below 10
    prices.loc[hours[3:6]] = 4.0  # below 10 but not negative

    entry = score_day(_payload(hours), prices)

    assert entry["call_b"]["n_negative_hours"] == 3
    assert entry["call_b"]["n_below_10_hours"] == 6

    recorded = entry["call_b"]["hours"]
    assert len(recorded) == 24
    for row in recorded:
        assert "outcome_negative" in row and "outcome_below_10" in row
        assert "baseline_negative" in row and "baseline_below_10" in row
        assert "model_score" in row
    assert sum(r["outcome_negative"] for r in recorded) == 3
    assert sum(r["outcome_below_10"] for r in recorded) == 6


def test_scoring_refuses_a_model_without_frozen_climatology(tmp_path, monkeypatch):
    monkeypatch.setattr("de_power_live.score.MODELS_DIR", tmp_path)
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "MANIFEST.json").write_text(
        json.dumps({"model_version": "test"}), encoding="utf-8"
    )
    with pytest.raises(IncompleteOutturn, match="no frozen climatology"):
        load_climatology("test")


# ---- Finding 2: partial days must not become final -----------------------


def test_scoring_rejects_a_day_with_missing_realised_prices(monkeypatch):
    _fake_climatology(monkeypatch)
    hours = pd.date_range("2026-08-01T22:00:00Z", periods=24, freq="h", tz="UTC")
    prices = _prices(hours)
    prices.loc[hours[5:]] = np.nan  # only 5 of 24 hours settled

    with pytest.raises(IncompleteOutturn, match="realised prices"):
        score_day(_payload(hours), prices)


def test_scoring_rejects_a_day_with_missing_baseline(monkeypatch):
    """B1 gaps previously produced a skill ratio over mismatched hour sets."""
    _fake_climatology(monkeypatch)
    hours = pd.date_range("2026-08-01T22:00:00Z", periods=24, freq="h", tz="UTC")
    prices = _prices(hours)
    prices.loc[hours[:4] - pd.Timedelta(hours=168)] = np.nan

    with pytest.raises(IncompleteOutturn, match="B1"):
        score_day(_payload(hours), prices)


def test_skill_ratio_uses_one_consistent_hour_set(monkeypatch):
    """MAE and baseline MAE must be computed over identical hours, so the
    reported skill is reconstructible from the two numbers beside it."""
    _fake_climatology(monkeypatch)
    hours = pd.date_range("2026-08-01T22:00:00Z", periods=24, freq="h", tz="UTC")
    prices = _prices(hours, 55.0)

    entry = score_day(_payload(hours, price=50.0), prices)
    a = entry["call_a"]
    assert a["mae_baseline_b1_lag168"] > 0
    assert a["mae_baseline_b2_lag24"] > 0
    # Ledger values are stored rounded to 5dp, so compare at that precision.
    assert a["skill_vs_b1"] == pytest.approx(
        1 - a["mae_model"] / a["mae_baseline_b1_lag168"], abs=1e-5
    )
    assert a["skill_vs_b2"] == pytest.approx(
        1 - a["mae_model"] / a["mae_baseline_b2_lag24"], abs=1e-5
    )
    assert entry["n_hours_scored"] == 24


def test_complete_day_scores_successfully(monkeypatch):
    _fake_climatology(monkeypatch)
    hours = pd.date_range("2026-08-01T22:00:00Z", periods=24, freq="h", tz="UTC")
    entry = score_day(_payload(hours), _prices(hours))
    assert entry["status"] == "SCORED"
    assert entry["n_hours_scored"] == entry["n_hours_predicted"] == 24


# ---- Finding 3: frozen artefacts must be verified ------------------------


def _frozen_dir(tmp_path: Path) -> tuple[Path, dict]:
    directory = tmp_path / "vtest"
    directory.mkdir()
    payload = b"not really a model, but it hashes"
    (directory / "price.ubj").write_bytes(payload)
    manifest = {
        "model_version": "vtest",
        "artefact_sha256": {"price": hashlib.sha256(payload).hexdigest()},
    }
    (directory / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, manifest


def test_verify_accepts_an_untampered_model(tmp_path):
    directory, manifest = _frozen_dir(tmp_path)
    report = verify_frozen_model(directory, manifest)
    assert report["artefacts_verified"] == ["price"]


def test_verify_rejects_a_swapped_artefact(tmp_path):
    directory, manifest = _frozen_dir(tmp_path)
    (directory / "price.ubj").write_bytes(b"a different model entirely")
    with pytest.raises(ModelIntegrityError, match="does not match its manifest"):
        verify_frozen_model(directory, manifest)


def test_verify_rejects_a_missing_artefact(tmp_path):
    directory, manifest = _frozen_dir(tmp_path)
    (directory / "price.ubj").unlink()
    with pytest.raises(ModelIntegrityError, match="missing"):
        verify_frozen_model(directory, manifest)


def test_verify_rejects_a_manifest_without_hashes(tmp_path):
    directory, _ = _frozen_dir(tmp_path)
    with pytest.raises(ModelIntegrityError, match="no artefact_sha256"):
        verify_frozen_model(directory, {"model_version": "vtest"})


def test_verify_rejects_drifted_source(tmp_path):
    """A code change must surface as a new version, not a silent edit."""
    directory, manifest = _frozen_dir(tmp_path)
    manifest["source_sha256"] = "0" * 64
    with pytest.raises(ModelIntegrityError, match="NEW model version"):
        verify_frozen_model(directory, manifest)


def test_shipped_v1_verifies_against_its_manifest():
    """The model that will actually make live predictions must pass its own check."""
    directory = REPO_ROOT / "models" / "v1"
    if not (directory / "MANIFEST.json").exists():
        pytest.skip("v1 not frozen in this checkout")
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    report = verify_frozen_model(directory, manifest)
    assert set(report["artefacts_verified"]) == {"negative", "price", "quantile"}
    assert manifest["climatology"]["negative"]
    assert manifest["climatology"]["below_10"]


# ---- Finding 4: CI protects more than predictions ------------------------


def test_integrity_workflow_protects_every_frozen_path():
    workflow = (REPO_ROOT / ".github" / "workflows" / "integrity.yml").read_text(
        encoding="utf-8"
    )
    for guarded in ("predictions/", "PREREGISTRATION.md", "models/", "results/ledger.jsonl"):
        assert guarded in workflow, f"CI no longer guards {guarded}"
    assert "--diff-filter=MD" in workflow
