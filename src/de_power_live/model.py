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

# Share of the (chronologically last) training data held out to calibrate the
# intervals. Never used to fit the quantile model.
CALIBRATION_FRACTION = 0.2

# Call B thresholds from PREREGISTRATION.md section 3: the primary target and
# the pre-declared fallback used only if the primary is underpowered. Both are
# fixed in advance; neither may be re-tuned once the window opens.
CALL_B_THRESHOLDS = {"negative": 0.0, "below_10": 10.0}


class ModelIntegrityError(RuntimeError):
    """Raised when a frozen model on disk does not match its manifest."""


def build_climatology(prices: pd.Series, threshold: float) -> dict[str, dict[str, float]]:
    """P(price < threshold) by (month, hour of day), from training data only.

    This is the call B baseline. It has to be frozen into the manifest now,
    before the window opens: re-estimating a baseline in November, after the
    outcomes are known, would reintroduce exactly the degree of freedom the
    pre-registration exists to remove. Keys are strings so the table survives a
    JSON round trip unchanged.
    """
    local = prices.index.tz_convert("Europe/Berlin")
    frame = pd.DataFrame(
        {
            "month": local.month,
            "hour": local.hour,
            "hit": (prices.to_numpy(dtype=float) < threshold).astype(float),
        }
    )
    grouped = frame.groupby(["month", "hour"])["hit"].mean()
    overall = float(frame["hit"].mean())

    table: dict[str, dict[str, float]] = {}
    for month in range(1, 13):
        table[str(month)] = {}
        for hour in range(24):
            key = (month, hour)
            # Months absent from training fall back to the pooled rate rather
            # than to zero, which would make the baseline trivially beatable.
            value = float(grouped[key]) if key in grouped.index else overall
            table[str(month)][str(hour)] = round(value, 6)
    return table


def climatology_probability(
    table: dict[str, dict[str, float]], timestamp: pd.Timestamp
) -> float:
    """Look up the frozen baseline for one hour."""
    local = pd.Timestamp(timestamp).tz_convert("Europe/Berlin")
    return float(table[str(local.month)][str(local.hour)])

# Which quantile columns bound each pre-registered interval.
INTERVAL_COLUMNS = {"80": (0, 4), "50": (1, 3)}
NOMINAL_COVERAGE = {"80": 0.80, "50": 0.50}


def _conformal_offsets(cal_quantiles: np.ndarray, cal_truth: np.ndarray) -> dict[str, float]:
    """Split-conformal corrections for each pre-registered interval.

    The conformity score is how far outside its own interval the truth fell:

        E_i = max(lo_i - y_i, y_i - hi_i)

    Negative when the truth was comfortably inside, so an over-wide interval is
    narrowed rather than only ever widened. Taking the
    ceil((n+1)(1-alpha))/n empirical quantile of E and expanding the interval by
    it gives marginal coverage of at least 1-alpha under exchangeability.

    Time series are not exchangeable, so this is an approximation rather than the
    finite-sample guarantee the theory offers. It is still far better founded
    than widening the intervals by a hand-picked constant until the number looks
    right, which would be fitting to the very thing being tested.
    """
    offsets: dict[str, float] = {}
    n = len(cal_truth)
    for name, (lo_col, hi_col) in INTERVAL_COLUMNS.items():
        lo = cal_quantiles[:, lo_col]
        hi = cal_quantiles[:, hi_col]
        scores = np.maximum(lo - cal_truth, cal_truth - hi)

        level = NOMINAL_COVERAGE[name]
        rank = min(int(np.ceil((n + 1) * level)), n)
        offsets[name] = float(np.sort(scores)[rank - 1])
    return offsets

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
    conformal_offsets: dict[str, float] = field(default_factory=dict)
    calibration_rows: int = 0
    climatology: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    # ---- fit ------------------------------------------------------------

    def fit(self, features: pd.DataFrame, prices: pd.Series) -> "DayAheadModel":
        matrix, target, anchor = self._align(features, prices)

        residual = target - anchor
        self.price_model = XGBRegressor(**PRICE_PARAMS).fit(matrix, residual)

        # Quantile model plus conformal calibration. Raw XGBoost quantile
        # regression fits the conditional quantiles of the *training* sample and
        # is reliably overconfident out of sample: the first walk-forward check
        # returned 0.67 coverage at a nominal 0.80 and 0.37 at a nominal 0.50.
        # Split-conformal (Romano et al., "Conformalized Quantile Regression")
        # fixes this by measuring how far outside its own interval the model
        # actually lands on held-out data, then widening by that amount.
        split = int(len(matrix) * (1 - CALIBRATION_FRACTION))
        if split < 24 * 30 or len(matrix) - split < 24 * 30:
            # Too little data for a stable holdout; fit on all rows and leave
            # the intervals uncalibrated.
            self.quantile_model = XGBRegressor(**QUANTILE_PARAMS).fit(matrix, residual)
            self.conformal_offsets = {}
        else:
            # Chronological split. A random one would let the calibration set
            # share days with the fit set through the lag features.
            fit_x, fit_y = matrix.iloc[:split], residual[:split]
            cal_x = matrix.iloc[split:]
            cal_truth = target[split:]
            cal_anchor = anchor[split:]

            self.quantile_model = XGBRegressor(**QUANTILE_PARAMS).fit(fit_x, fit_y)
            cal_q = np.sort(
                self.quantile_model.predict(cal_x) + cal_anchor[:, None], axis=1
            )
            self.conformal_offsets = _conformal_offsets(cal_q, cal_truth)
            self.calibration_rows = int(len(cal_x))

        # Freeze the call B baselines from training data only, for both the
        # primary threshold and the pre-declared fallback.
        target_series = pd.Series(target, index=matrix.index)
        self.climatology = {
            name: build_climatology(target_series, threshold)
            for name, threshold in CALL_B_THRESHOLDS.items()
        }

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

        # Apply the conformal corrections measured on held-out data.
        for name, (lo_col, hi_col) in INTERVAL_COLUMNS.items():
            offset = self.conformal_offsets.get(name)
            if offset is None:
                continue
            quantiles[:, lo_col] -= offset
            quantiles[:, hi_col] += offset
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
            "conformal_offsets": self.conformal_offsets,
            "calibration_rows": self.calibration_rows,
            "call_b_thresholds": CALL_B_THRESHOLDS,
            "climatology": self.climatology,
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
    def load(cls, directory: Path, verify: bool = True) -> "DayAheadModel":
        """Load a frozen model, verifying it matches its manifest.

        Without this check the manifest is decoration: a swapped or corrupted
        .ubj would produce predictions under a version label that no longer
        describes it, and the sealed hash would attest to the wrong thing.
        """
        directory = Path(directory)
        manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))

        if verify:
            verify_frozen_model(directory, manifest)

        model = cls(version=manifest["model_version"])
        model.trained_at_utc = manifest.get("trained_at_utc")
        model.training_rows = manifest.get("training_rows", 0)
        span = manifest.get("training_span_utc")
        model.training_span = tuple(span) if span else None
        model.negative_base_rate = manifest.get("negative_base_rate")
        model.conformal_offsets = manifest.get("conformal_offsets", {})
        model.calibration_rows = manifest.get("calibration_rows", 0)
        model.climatology = manifest.get("climatology", {})
        model.feature_columns = manifest["feature_columns"]

        model.price_model = XGBRegressor()
        model.price_model.load_model(str(directory / "price.ubj"))
        model.quantile_model = XGBRegressor()
        model.quantile_model.load_model(str(directory / "quantile.ubj"))

        if manifest.get("negative_model_fitted"):
            model.negative_model = XGBClassifier()
            model.negative_model.load_model(str(directory / "negative.ubj"))

        return model


def verify_frozen_model(directory: Path, manifest: dict) -> dict:
    """Check a frozen model directory against its manifest.

    Two separate checks with deliberately different severities:

    * **Artefact hashes** must match exactly. A frozen .ubj has no legitimate
      reason to change, so any mismatch is fatal.
    * **Source hash** must match the code that produced the manifest. A
      mismatch means prediction-determining code changed under a version label
      that no longer describes it. Per PREREGISTRATION.md section 4 that
      requires a new model version, not a quiet edit - so this is fatal too,
      with a message that says which route to take.
    """
    directory = Path(directory)
    expected = manifest.get("artefact_sha256") or {}
    if not expected:
        raise ModelIntegrityError(
            f"{directory} has no artefact_sha256 in its manifest; refusing to load "
            "a model whose contents cannot be verified"
        )

    mismatches: list[str] = []
    for name, want in expected.items():
        path = directory / f"{name}.ubj"
        if not path.exists():
            mismatches.append(f"{name}.ubj is missing")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            mismatches.append(f"{name}.ubj sha256 {got[:16]}... != manifest {want[:16]}...")

    if mismatches:
        raise ModelIntegrityError(
            f"frozen model at {directory} does not match its manifest:\n  "
            + "\n  ".join(mismatches)
        )

    want_source = manifest.get("source_sha256")
    got_source = source_hash()
    if want_source and got_source != want_source:
        raise ModelIntegrityError(
            f"prediction-determining source has changed since {manifest['model_version']} "
            f"was frozen (now {got_source[:16]}..., manifest {want_source[:16]}...).\n"
            "PREREGISTRATION.md section 4 requires a code change to take effect as a "
            "NEW model version rather than silently altering an existing one.\n"
            "Before the window opens (no sealed predictions yet) refresh the manifest "
            "with: python -m de_power_live.refreeze --version <v>\n"
            "Once predictions exist, train a new version instead."
        )

    return {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artefacts_verified": sorted(expected),
        "source_sha256": got_source,
    }


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
