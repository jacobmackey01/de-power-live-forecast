"""The three pre-registered calls, as three fitted models.

Call A  price level          XGBRegressor on the residual against lag-168h
Call B  negative-price hour  XGBClassifier, imbalance-weighted
Call C  prediction intervals XGBRegressor with a multi-quantile objective

Why residual form for A and C
-----------------------------
Gradient-boosted trees cannot extrapolate: a model trained through 2025 will
never predict a price above its training maximum, however extreme the drivers.
German power prices are not stationary in level, so predicting ``price`` in
levels bakes in a ceiling that only bites during exactly the hours that matter.
Predicting ``price - price_lag_168`` and adding the lag back moves the level
into a feature the model does not have to extrapolate over, and makes the
learned quantity directly comparable to the pre-registered B1 baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from .features import FEATURE_COLUMNS

# Central 50% and 80% intervals, per PREREGISTRATION.md call C.
QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

ANCHOR_FEATURE = "price_lag_168"

PRICE_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="reg:absoluteerror",  # matches the MAE criterion in call A
    random_state=20260801,
    n_jobs=4,
)

NEGATIVE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=20260801,
    n_jobs=4,
)

QUANTILE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=1.0,
    objective="reg:quantileerror",
    quantile_alpha=np.array(QUANTILES),
    random_state=20260801,
    n_jobs=4,
)


class ModelError(RuntimeError):
    """Raised when a model cannot be fitted or applied."""


@dataclass
class Prediction:
    """One model's output for a set of target hours."""

    price: np.ndarray  # EUR/MWh, call A
    prob_negative: np.ndarray  # P(price < 0), call B
    quantiles: np.ndarray  # (n_hours, len(QUANTILES)), call C
    quantile_levels: tuple[float, ...] = QUANTILES


@dataclass
class DayAheadModel:
    """Fits and applies all three pre-registered calls together."""

    version: str = "v1"
    price_model: XGBRegressor | None = None
    negative_model: XGBClassifier | None = None
    quantile_model: XGBRegressor | None = None
    trained_at_utc: str | None = None
    training_rows: int = 0
    training_span: tuple[str, str] | None = None
    negative_base_rate: float | None = None
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    # ---- fit ------------------------------------------------------------

    def fit(self, features: pd.DataFrame, prices: pd.Series) -> "DayAheadModel":
        matrix, target, anchor = self._align(features, prices)

        residual = target - anchor
        self.price_model = XGBRegressor(**PRICE_PARAMS).fit(matrix, residual)
        self.quantile_model = XGBRegressor(**QUANTILE_PARAMS).fit(matrix, residual)

        is_negative = (target < 0).astype(int)
        n_pos = int(is_negative.sum())
        self.negative_base_rate = float(is_negative.mean())
        if n_pos == 0:
            # No negative hours in training. Rather than fit a degenerate
            # classifier, record the fact; predict_proba falls back to the base
            # rate and call B is reported as underpowered per the pre-registration.
            self.negative_model = None
        else:
            params = dict(NEGATIVE_PARAMS)
            params["scale_pos_weight"] = float((len(is_negative) - n_pos) / n_pos)
            self.negative_model = XGBClassifier(**params).fit(matrix, is_negative)

        self.trained_at_utc = datetime.now(timezone.utc).isoformat()
        self.training_rows = int(len(matrix))
        self.training_span = (
            matrix.index.min().isoformat(),
            matrix.index.max().isoformat(),
        )
        return self

    def _align(
        self, features: pd.DataFrame, prices: pd.Series
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ModelError(f"features missing columns: {missing}")

        matrix = features[self.feature_columns]
        target = prices.reindex(matrix.index)

        # The anchor must exist for the residual form to be invertible.
        usable = target.notna() & matrix[ANCHOR_FEATURE].notna()
        if usable.sum() == 0:
            raise ModelError("no rows with both a target price and a lag-168 anchor")

        matrix = matrix.loc[usable]
        target_arr = target.loc[usable].to_numpy(dtype=float)
        anchor = matrix[ANCHOR_FEATURE].to_numpy(dtype=float)
        return matrix, target_arr, anchor

    # ---- predict --------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> Prediction:
        if self.price_model is None or self.quantile_model is None:
            raise ModelError("model is not fitted")

        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ModelError(f"features missing columns: {missing}")

        matrix = features[self.feature_columns]
        anchor = matrix[ANCHOR_FEATURE].to_numpy(dtype=float)
        if np.isnan(anchor).any():
            raise ModelError(
                f"{int(np.isnan(anchor).sum())} target hours have no {ANCHOR_FEATURE}; "
                "cannot invert the residual form"
            )

        price = self.price_model.predict(matrix) + anchor

        quantiles = self.quantile_model.predict(matrix)
        if quantiles.ndim == 1:
            quantiles = quantiles.reshape(-1, 1)
        quantiles = quantiles + anchor[:, None]
        # The multi-quantile objective does not guarantee monotonicity; crossed
        # quantiles would produce intervals with negative width. Sorting is the
        # standard repair and cannot make calibration worse.
        quantiles = np.sort(quantiles, axis=1)

        if self.negative_model is None:
            prob = np.full(len(matrix), self.negative_base_rate or 0.0, dtype=float)
        else:
            prob = self.negative_model.predict_proba(matrix)[:, 1]

        return Prediction(price=price, prob_negative=prob, quantiles=quantiles)

    # ---- persistence ----------------------------------------------------

    def save(self, directory: Path) -> dict:
        """Freeze the model to disk and return a manifest describing it."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        artefacts: dict[str, str] = {}
        for name, model in (
            ("price", self.price_model),
            ("quantile", self.quantile_model),
            ("negative", self.negative_model),
        ):
            if model is None:
                continue
            path = directory / f"{name}.ubj"
            model.save_model(str(path))
            artefacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()

        manifest = {
            "model_version": self.version,
            "trained_at_utc": self.trained_at_utc,
            "training_rows": self.training_rows,
            "training_span_utc": self.training_span,
            "negative_base_rate": self.negative_base_rate,
            "negative_model_fitted": self.negative_model is not None,
            "feature_columns": self.feature_columns,
            "quantile_levels": list(QUANTILES),
            "anchor_feature": ANCHOR_FEATURE,
            "artefact_sha256": artefacts,
            "source_sha256": source_hash(),
            "hyperparameters": {
                "price": _jsonable(PRICE_PARAMS),
                "negative": _jsonable(NEGATIVE_PARAMS),
                "quantile": _jsonable(QUANTILE_PARAMS),
            },
        }
        (directory / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    @classmethod
    def load(cls, directory: Path) -> "DayAheadModel":
        directory = Path(directory)
        manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))

        model = cls(version=manifest["model_version"])
        model.trained_at_utc = manifest.get("trained_at_utc")
        model.training_rows = manifest.get("training_rows", 0)
        span = manifest.get("training_span_utc")
        model.training_span = tuple(span) if span else None
        model.negative_base_rate = manifest.get("negative_base_rate")
        model.feature_columns = manifest["feature_columns"]

        model.price_model = XGBRegressor()
        model.price_model.load_model(str(directory / "price.ubj"))
        model.quantile_model = XGBRegressor()
        model.quantile_model.load_model(str(directory / "quantile.ubj"))

        if manifest.get("negative_model_fitted"):
            model.negative_model = XGBClassifier()
            model.negative_model.load_model(str(directory / "negative.ubj"))

        return model


def _jsonable(params: dict) -> dict:
    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()
    }


def source_hash() -> str:
    """SHA-256 over the modules that determine a prediction.

    Recorded in every sealed prediction so that 'the model did not change' is
    checkable rather than asserted.
    """
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for name in sorted(("features.py", "model.py", "weather.py", "smard.py")):
        path = here / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()
