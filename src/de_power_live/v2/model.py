"""Model v2: one gradient-boosted regressor on a level-free target.

The model predicts ``(price - same hour yesterday) / scale`` from features that
contain no price level (features.py). The forecast is the anchor plus the
predicted change times the scale, so the level enters only through two numbers
that are observed at the seal and never have to be extrapolated over.

Fitted once and frozen. In the replay a weekly refit scored the same as a single
fit to two decimals (research/v2/README.md), and one frozen artefact keeps the
integrity check phase 1 used: a hash in the manifest, re-verified at every seal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import MODEL_VERSION
from .features import Dataset

PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="reg:absoluteerror",  # matches the MAE criterion in call A
    random_state=20261110,
    n_jobs=1,  # one thread, so the frozen artefact can be rebuilt bit for bit
)

# What determines a forecast. Not the HTTP plumbing: in phase 1 a retry fix to
# a data client could not be made mid-window because the client sat inside the
# hash. inputs.py, predict.py and score.py are deliberately outside it.
HASHED_SOURCES = ("features.py", "model.py", "calls.py")
ARTEFACT = "price.ubj"
WARM_START = "warm_start.json"


class ModelIntegrityError(RuntimeError):
    """Raised when the frozen model on disk does not match its manifest."""


def source_hash() -> str:
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for name in sorted(HASHED_SOURCES):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class V2Model:
    booster: XGBRegressor | None = None
    feature_columns: list[str] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    params: dict = field(default_factory=lambda: dict(PARAMS))

    def fit(self, ds: Dataset, train_last: pd.Timestamp, train_first: pd.Timestamp | None = None) -> "V2Model":
        """Fit on target days up to and including ``train_last``."""
        X, target = ds.design()
        day = X.index.get_level_values("day")
        rows = (target.notna() & ds.usable()).to_numpy() & np.asarray(day <= pd.Timestamp(train_last))
        if train_first is not None:
            rows &= np.asarray(day >= pd.Timestamp(train_first))
        if rows.sum() < 24 * 180:
            raise ValueError(f"only {int(rows.sum())} training rows up to {train_last}")
        self.feature_columns = list(X.columns)
        self.booster = XGBRegressor(**self.params).fit(X.loc[rows, self.feature_columns], target[rows])
        fitted_days = day[rows]
        self.manifest = {
            "model_version": MODEL_VERSION,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_rows": int(rows.sum()),
            "training_days": [str(fitted_days.min().date()), str(fitted_days.max().date())],
            "feature_columns": self.feature_columns,
            "price_columns": list(ds.price_columns),
            "hyperparameters": dict(self.params),
        }
        return self

    def predict(self, ds: Dataset, days: pd.DatetimeIndex | list[pd.Timestamp]) -> pd.Series:
        """Price forecast for every usable (day, hour) slot of ``days``."""
        if self.booster is None:
            raise ValueError("model is not fitted")
        X, _ = ds.design()
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"features missing columns: {missing}")
        rows = np.asarray(X.index.get_level_values("day").isin(pd.DatetimeIndex(days))) & ds.usable().to_numpy()
        change = self.booster.predict(X.loc[rows, self.feature_columns])
        return (ds.anchor[rows] + change * ds.scale[rows]).rename("pred")

    # ---- persistence -------------------------------------------------------

    def save(self, directory: Path, extra: dict | None = None) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / ARTEFACT))
        manifest = dict(self.manifest)
        manifest.update(extra or {})
        manifest["artefact_sha256"] = {ARTEFACT: file_hash(directory / ARTEFACT)}
        if (directory / WARM_START).exists():
            manifest["artefact_sha256"][WARM_START] = file_hash(directory / WARM_START)
        manifest["source_sha256"] = source_hash()
        manifest["hashed_sources"] = list(HASHED_SOURCES)
        (directory / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.manifest = manifest
        return manifest

    @classmethod
    def load(cls, directory: Path, verify: bool = True) -> "V2Model":
        directory = Path(directory)
        manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
        if verify:
            verify_frozen(directory, manifest)
        booster = XGBRegressor()
        booster.load_model(str(directory / ARTEFACT))
        return cls(booster=booster, feature_columns=list(manifest["feature_columns"]), manifest=manifest)


def verify_frozen(directory: Path, manifest: dict) -> None:
    """The artefacts and the forecast-determining source must match the manifest."""
    problems = []
    for name, want in manifest.get("artefact_sha256", {}).items():
        path = Path(directory) / name
        if not path.exists():
            problems.append(f"{name} is missing")
        elif file_hash(path) != want:
            problems.append(f"{name} does not match its manifest hash")
    if manifest.get("source_sha256") != source_hash():
        problems.append(
            "forecast-determining source (" + ", ".join(HASHED_SOURCES) + ") has changed since "
            "the model was frozen; a change takes effect as a new model version, never as an "
            "edit to this one"
        )
    if problems:
        raise ModelIntegrityError("frozen model v2 failed verification: " + "; ".join(problems))


def load_warm_start(directory: Path) -> dict[pd.Timestamp, np.ndarray]:
    """Scaled errors for the days before the first live seal, frozen with the model."""
    path = Path(directory) / WARM_START
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {pd.Timestamp(day): np.array(errors, dtype=float) for day, errors in payload["scaled_errors"].items()}
