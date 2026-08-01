"""Refresh a frozen model's manifest without retraining it.

Narrow, pre-window-only tool. It recomputes the manifest fields that describe
the code and the frozen call B baselines, and leaves the .ubj artefacts
byte-identical - so the model that makes predictions is provably unchanged.

Why this exists rather than a retrain: XGBoost is not bit-reproducible across
runs (two identical-seed runs here differed by ~0.015 pooled skill), so
retraining to fix a bookkeeping problem would silently swap in a *different*
model than the one already validated, for no methodological reason.

Refuses to run once any prediction has been sealed. From the first seal,
PREREGISTRATION.md section 4 governs: code changes take effect as a new model
version, never as an edit to an existing one.

    python -m de_power_live.refreeze --version v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .dataset import build_training_set, fetch_history
from .model import CALL_B_THRESHOLDS, build_climatology, source_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
PREDICTIONS_DIR = REPO_ROOT / "predictions"


class RefreezeBlocked(RuntimeError):
    """Raised when refreezing would touch a model the record already depends on."""


def sealed_predictions() -> list[Path]:
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(PREDICTIONS_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))


def refreeze(version: str, years: float = 2.0) -> dict:
    sealed = sealed_predictions()
    if sealed:
        raise RefreezeBlocked(
            f"{len(sealed)} prediction(s) already sealed (first {sealed[0].name}). "
            "The evaluation window has opened, so this model is part of the record. "
            "Train a new version instead of refreshing this one."
        )

    directory = MODELS_DIR / version
    manifest_path = directory / "MANIFEST.json"
    if not manifest_path.exists():
        raise RefreezeBlocked(f"no manifest at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Artefacts must not change. Record their hashes from disk and confirm they
    # still match what the manifest claimed.
    artefacts: dict[str, str] = {}
    for name in ("price", "quantile", "negative"):
        path = directory / f"{name}.ubj"
        if path.exists():
            artefacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    previous = manifest.get("artefact_sha256") or {}
    changed = [n for n, h in artefacts.items() if previous.get(n) and previous[n] != h]
    if changed:
        raise RefreezeBlocked(
            f"artefacts {changed} differ from the manifest. Refreeze only refreshes "
            "bookkeeping; it must never be used to bless a changed model."
        )

    print(f"rebuilding training data for the frozen call B baselines ({years}y) ...")
    prices, weather, actuals, _ = fetch_history(years=years)
    first_day = pd.Timestamp(weather.index.min().tz_convert("Europe/Berlin").date()) + pd.Timedelta(days=15)
    last_day = pd.Timestamp(prices.index.max().tz_convert("Europe/Berlin").date()) - pd.Timedelta(days=1)
    data = build_training_set(prices, weather, actuals, first_day, last_day)

    target = data.prices.reindex(data.features.index).dropna()
    climatology = {
        name: build_climatology(target, threshold)
        for name, threshold in CALL_B_THRESHOLDS.items()
    }

    history = manifest.get("refreeze_history", [])
    history.append(
        {
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_source_sha256": manifest.get("source_sha256"),
            "new_source_sha256": source_hash(),
            "reason": (
                "Added frozen call B climatology baselines and model-artefact hash "
                "verification. Artefacts unchanged; no retraining performed."
            ),
        }
    )

    manifest["artefact_sha256"] = artefacts
    manifest["source_sha256"] = source_hash()
    manifest["call_b_thresholds"] = CALL_B_THRESHOLDS
    manifest["climatology"] = climatology
    manifest["climatology_rows"] = int(len(target))
    manifest["refreeze_history"] = history

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, table in climatology.items():
        pooled = sum(sum(h.values()) for h in table.values()) / (12 * 24)
        print(f"  climatology[{name}] pooled mean {pooled:.5f} over {len(target)} rows")
    print(f"  source_sha256 -> {manifest['source_sha256'][:16]}...")
    print(f"  artefacts unchanged: {sorted(artefacts)}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh a frozen model manifest")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--years", type=float, default=2.0)
    args = parser.parse_args()

    manifest = refreeze(args.version, years=args.years)
    print(f"\nrefroze {args.version} -> {MODELS_DIR / args.version / 'MANIFEST.json'}")
    print(f"  refreeze events recorded: {len(manifest['refreeze_history'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
