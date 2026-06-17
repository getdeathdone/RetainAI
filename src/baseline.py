"""Classical ML baseline for RetainAI churn prediction.

The module expects already preprocessed train/test matrices from
`ChurnDataPreprocessor` and trains a RandomForestClassifier baseline. This
baseline gives us interpretable feature importances and a strong reference
point before adding a PyTorch sequence model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ChurnBaselineModel:
    """Train, evaluate and persist a RandomForest churn baseline."""

    n_estimators: int = 300
    max_depth: int | None = 12
    min_samples_leaf: int = 5
    random_state: int = 42
    n_jobs: int = -1
    artifact_path: Path = Path("artifacts/baseline_model.joblib")
    model: RandomForestClassifier = field(init=False)
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def train(
        self,
        X_train: NDArray[np.float32] | NDArray[np.float64],
        y_train: pd.Series | NDArray[np.int_],
        feature_names: list[str] | None = None,
    ) -> "ChurnBaselineModel":
        """Fit the RandomForest baseline on preprocessed features."""

        X_train_validated = self._validate_feature_matrix(X_train, matrix_name="X_train")
        y_train_validated = self._validate_target(y_train, target_name="y_train")

        if len(X_train_validated) != len(y_train_validated):
            raise ValueError(
                "X_train and y_train must contain the same number of rows. "
                f"Got {len(X_train_validated)} and {len(y_train_validated)}."
            )

        if feature_names is not None:
            if len(feature_names) != X_train_validated.shape[1]:
                raise ValueError(
                    "feature_names length must match number of columns in X_train. "
                    f"Got {len(feature_names)} names for {X_train_validated.shape[1]} columns."
                )
            self.feature_names = feature_names.copy()
        elif not self.feature_names:
            self.feature_names = [f"feature_{idx}" for idx in range(X_train_validated.shape[1])]

        LOGGER.info(
            "Training RandomForest baseline: rows=%s, features=%s, positive_rate=%.4f.",
            X_train_validated.shape[0],
            X_train_validated.shape[1],
            float(np.mean(y_train_validated)),
        )

        self.model.fit(X_train_validated, y_train_validated)
        LOGGER.info("RandomForest baseline training finished.")
        return self

    def evaluate(
        self,
        X_test: NDArray[np.float32] | NDArray[np.float64],
        y_test: pd.Series | NDArray[np.int_],
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate model quality with churn-relevant classification metrics."""

        self._validate_is_fitted()
        X_test_validated = self._validate_feature_matrix(X_test, matrix_name="X_test")
        y_test_validated = self._validate_target(y_test, target_name="y_test")

        if len(X_test_validated) != len(y_test_validated):
            raise ValueError(
                "X_test and y_test must contain the same number of rows. "
                f"Got {len(X_test_validated)} and {len(y_test_validated)}."
            )

        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        positive_probabilities = self.model.predict_proba(X_test_validated)[:, 1]
        predictions = (positive_probabilities >= threshold).astype(np.int64)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test_validated, positive_probabilities)),
            "f1": float(f1_score(y_test_validated, predictions, zero_division=0)),
            "precision": float(precision_score(y_test_validated, predictions, zero_division=0)),
            "recall": float(recall_score(y_test_validated, predictions, zero_division=0)),
        }

        LOGGER.info("Baseline evaluation metrics: %s", metrics)
        LOGGER.info(
            "Classification report:\n%s",
            classification_report(y_test_validated, predictions, zero_division=0),
        )
        return metrics

    def find_best_threshold(
        self,
        X_validation: NDArray[np.float32] | NDArray[np.float64],
        y_validation: pd.Series | NDArray[np.int_],
    ) -> tuple[float, float]:
        """Find the probability threshold that maximizes F1 on validation data."""

        self._validate_is_fitted()
        X_validation = self._validate_feature_matrix(X_validation, matrix_name="X_validation")
        y_validation_array = self._validate_target(y_validation, target_name="y_validation")
        probabilities = self.model.predict_proba(X_validation)[:, 1]

        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.arange(0.20, 0.81, 0.01):
            predictions = (probabilities >= threshold).astype(np.int64)
            current_f1 = float(f1_score(y_validation_array, predictions, zero_division=0))
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(threshold)

        LOGGER.info("Best F1 threshold: %.2f with F1=%.5f.", best_threshold, best_f1)
        return best_threshold, best_f1

    def get_feature_importances(self, top_n: int = 10) -> pd.DataFrame:
        """Return top-N most important features for explainability and LLM input."""

        self._validate_is_fitted()

        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        importances = self.model.feature_importances_
        if not self.feature_names:
            self.feature_names = [f"feature_{idx}" for idx in range(len(importances))]

        if len(self.feature_names) != len(importances):
            raise RuntimeError(
                "Feature name count does not match trained model feature count. "
                f"Got {len(self.feature_names)} names and {len(importances)} importances."
            )

        importance_frame = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        LOGGER.info("Top-%s feature importances:\n%s", top_n, importance_frame)
        return importance_frame

    def save_model(self, artifact_path: str | Path | None = None) -> Path:
        """Persist the trained baseline model and metadata to disk."""

        self._validate_is_fitted()

        output_path = Path(artifact_path) if artifact_path is not None else self.artifact_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "model": self.model,
            "feature_names": self.feature_names,
            "model_type": "RandomForestClassifier",
            "class_weight": "balanced",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }
        joblib.dump(payload, output_path)
        LOGGER.info("Saved baseline model artifact to %s.", output_path)
        return output_path

    def save_metrics(
        self,
        metrics: dict[str, float],
        artifact_path: str | Path = "artifacts/metrics.json",
        top_n_features: int = 10,
        best_threshold: float | None = None,
    ) -> Path:
        """Persist baseline metrics and feature importances for portfolio reporting."""

        output_path = Path(artifact_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_payload: dict[str, Any] = {}
        if output_path.exists():
            existing_payload = json.loads(output_path.read_text(encoding="utf-8"))

        top_features = self.get_feature_importances(top_n=top_n_features)
        existing_payload["baseline"] = {
            "model_type": "RandomForestClassifier",
            "metrics": metrics,
            "best_threshold": best_threshold,
            "top_features": top_features.to_dict(orient="records"),
            "saved_at": datetime.now(UTC).isoformat(),
        }

        output_path.write_text(
            json.dumps(existing_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("Saved baseline metrics to %s.", output_path)
        return output_path

    @classmethod
    def load_model(cls, artifact_path: str | Path) -> "ChurnBaselineModel":
        """Load a trained baseline model artifact."""

        input_path = Path(artifact_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Baseline model artifact not found: {input_path}")

        payload = joblib.load(input_path)
        baseline = cls(
            n_estimators=payload.get("n_estimators", 300),
            max_depth=payload.get("max_depth", 12),
            min_samples_leaf=payload.get("min_samples_leaf", 5),
            random_state=payload.get("random_state", 42),
            artifact_path=input_path,
        )
        baseline.model = payload["model"]
        baseline.feature_names = payload.get("feature_names", [])
        LOGGER.info("Loaded baseline model artifact from %s.", input_path)
        return baseline

    def predict_proba(
        self,
        X: NDArray[np.float32] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return churn probability for preprocessed feature rows."""

        self._validate_is_fitted()
        X_validated = self._validate_feature_matrix(X, matrix_name="X")
        return self.model.predict_proba(X_validated)[:, 1]

    @staticmethod
    def _validate_feature_matrix(
        X: NDArray[np.float32] | NDArray[np.float64],
        matrix_name: str,
    ) -> NDArray[np.float32] | NDArray[np.float64]:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{matrix_name} must be a numpy ndarray.")

        if X.ndim != 2:
            raise ValueError(f"{matrix_name} must be a 2D matrix. Got shape {X.shape}.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{matrix_name} must not be empty. Got shape {X.shape}.")

        if not np.isfinite(X).all():
            raise ValueError(f"{matrix_name} contains NaN or infinite values.")

        return X

    @staticmethod
    def _validate_target(
        y: pd.Series | NDArray[np.int_],
        target_name: str,
    ) -> NDArray[np.int64]:
        y_array = np.asarray(y, dtype=np.int64)

        if y_array.ndim != 1:
            raise ValueError(f"{target_name} must be a 1D target array.")

        if y_array.size == 0:
            raise ValueError(f"{target_name} must not be empty.")

        unique_values = set(np.unique(y_array).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"{target_name} must be binary with values 0/1. Got {unique_values}.")

        if len(unique_values) < 2:
            raise ValueError(f"{target_name} must contain both classes.")

        return y_array

    def _validate_is_fitted(self) -> None:
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("Baseline model must be trained before this operation.")


def _build_fallback_dataset(
    n_train: int = 800,
    n_test: int = 200,
    n_features: int = 32,
    random_state: int = 42,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.int64], list[str]]:
    """Create synthetic preprocessed matrices when the preprocessor is unavailable."""

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_train + n_test, n_features)).astype(np.float32)
    signal = 1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.6 * X[:, 2] + rng.normal(scale=0.8, size=X.shape[0])
    churn_probability = 1 / (1 + np.exp(-signal))
    y = rng.binomial(1, churn_probability).astype(np.int64)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    feature_names = [f"fallback_feature_{idx}" for idx in range(n_features)]

    return X_train, X_test, y_train, y_test, feature_names


def _build_demo_dataset() -> tuple[NDArray[np.float32], NDArray[np.float32], Any, Any, list[str]]:
    """Try the real preprocessing path first, then fall back to synthetic matrices."""

    try:
        from preprocessing import ChurnDataPreprocessor, build_fake_feature_mart

        fake_df = build_fake_feature_mart(n_rows=1000, random_state=42)
        preprocessor = ChurnDataPreprocessor()
        prepared = preprocessor.prepare_data_with_metadata(fake_df)
        preprocessor.save_preprocessor("artifacts/preprocessor.joblib")

        LOGGER.info("Demo dataset prepared through ChurnDataPreprocessor.")
        return (
            prepared.X_train,
            prepared.X_test,
            prepared.y_train,
            prepared.y_test,
            prepared.feature_names,
        )
    except ImportError as exc:
        LOGGER.warning(
            "Could not import ChurnDataPreprocessor. "
            "Using synthetic preprocessed matrices instead. Reason: %s",
            exc,
        )
        return _build_fallback_dataset()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    X_train_demo, X_test_demo, y_train_demo, y_test_demo, demo_feature_names = _build_demo_dataset()

    baseline_model = ChurnBaselineModel()
    baseline_model.train(
        X_train=X_train_demo,
        y_train=y_train_demo,
        feature_names=demo_feature_names,
    )

    best_threshold, _ = baseline_model.find_best_threshold(X_test_demo, y_test_demo)
    demo_metrics = baseline_model.evaluate(X_test=X_test_demo, y_test=y_test_demo, threshold=best_threshold)
    demo_importances = baseline_model.get_feature_importances(top_n=10)
    saved_artifact = baseline_model.save_model("artifacts/baseline_model.joblib")
    metrics_artifact = baseline_model.save_metrics(
        demo_metrics,
        "artifacts/metrics.json",
        best_threshold=best_threshold,
    )

    LOGGER.info("Smoke test metrics: %s", demo_metrics)
    LOGGER.info("Smoke test top features:\n%s", demo_importances)
    LOGGER.info("Smoke test model artifact: %s", saved_artifact)
    LOGGER.info("Smoke test metrics artifact: %s", metrics_artifact)
