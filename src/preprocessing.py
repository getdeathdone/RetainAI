"""Data preprocessing pipeline for RetainAI churn and LTV modeling.

This module converts rows from the PostgreSQL materialized view
`retainai.ml_customer_features` into model-ready matrices for classical ML
and PyTorch experiments.

Design notes:
    * Hard row filtering is kept outside the sklearn Pipeline because standard
      sklearn transformers return only X, not (X, y). Dropping rows inside a
      Pipeline would silently desynchronize features and labels.
    * The serializable artifact contains the fitted ColumnTransformer-based
      Pipeline used at inference time. Row filtering is a training-time quality
      gate and should not drop live users in the FastAPI layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger(__name__)


DEFAULT_NUMERIC_FEATURES: list[str] = [
    "account_age_days",
    "recency_days",
    "payment_recency_days",
    "paid_tx_count",
    "all_tx_count",
    "failed_tx_count",
    "refunded_tx_count",
    "product_category_count",
    "payment_method_count",
    "monetary_value",
    "avg_order_value",
    "max_order_value",
    "min_order_value",
    "avg_days_between_paid_tx",
    "ltv_observed",
    "last_rolling_3tx_avg_amount",
    "paid_tx_count_30d",
    "revenue_30d",
    "session_count",
    "total_page_views",
    "avg_page_views",
    "total_actions",
    "avg_actions",
    "avg_session_minutes",
    "support_tickets_count",
    "avg_days_between_sessions",
    "recent_3_sessions_avg_actions",
    "recent_3_sessions_avg_page_views",
    "last_rolling_5session_actions",
    "last_rolling_5session_page_views",
    "sessions_30d",
    "actions_30d",
    "page_views_30d",
    "support_tickets_30d",
    "avg_monthly_events_last_3m",
    "avg_monthly_events_before_3m",
    "monthly_activity_slope",
    "activity_drop_ratio",
    "revenue_per_account_day",
    "actions_per_session",
]

DEFAULT_CATEGORICAL_FEATURES: list[str] = [
    "country",
    "acquisition_channel",
    "plan_type",
    "gender",
    "ltv_segment",
]

DEFAULT_TARGET_COLUMN = "churn_label_45d"


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with both older and newer sklearn.

    scikit-learn renamed `sparse` to `sparse_output` in 1.2. This helper keeps
    the portfolio project easier to run across common local environments.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )


class HardOutlierFilter(BaseEstimator, TransformerMixin):
    """Remove extreme training rows by quantile caps for selected features.

    The transformer follows the sklearn estimator API, but exposes
    `filter_dataframe` for the production training workflow so X and y can be
    filtered together without index mismatches.

    Example:
        If `ltv_observed` has a 99th percentile of 5000 and
        `max_multiplier=1.5`, rows above 7500 are treated as hard outliers.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        upper_quantile: float = 0.99,
        max_multiplier: float = 1.5,
    ) -> None:
        self.columns = columns or ["ltv_observed", "paid_tx_count", "session_count"]
        self.upper_quantile = upper_quantile
        self.max_multiplier = max_multiplier

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HardOutlierFilter":
        """Estimate hard upper bounds from the training frame."""

        self._validate_input(X)
        self.upper_bounds_: dict[str, float] = {}

        for column in self.columns:
            if column not in X.columns:
                LOGGER.warning("Outlier column '%s' is missing; skipping it.", column)
                continue

            values = pd.to_numeric(X[column], errors="coerce").dropna()
            if values.empty:
                LOGGER.warning("Outlier column '%s' has no numeric values.", column)
                continue

            quantile_value = float(values.quantile(self.upper_quantile))
            self.upper_bounds_[column] = quantile_value * self.max_multiplier

        if not self.upper_bounds_:
            LOGGER.warning("No outlier bounds were learned; filtering will be skipped.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X unchanged for sklearn API compatibility.

        Row removal is intentionally performed through `filter_dataframe`,
        because sklearn's `transform` cannot return an aligned target vector.
        """

        self._validate_input(X)
        return X.copy()

    def filter_dataframe(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Drop rows exceeding learned hard upper bounds and align y."""

        self._validate_input(X)
        self._validate_is_fitted()

        mask = pd.Series(True, index=X.index)
        for column, upper_bound in self.upper_bounds_.items():
            values = pd.to_numeric(X[column], errors="coerce")
            mask &= values.isna() | (values <= upper_bound)

        X_filtered = X.loc[mask].copy()
        y_filtered = y.loc[mask].copy() if y is not None else None

        removed_rows = int((~mask).sum())
        LOGGER.info(
            "Hard outlier filtering removed %s rows from %s.",
            removed_rows,
            len(X),
        )

        return X_filtered, y_filtered

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HardOutlierFilter expects a pandas DataFrame.")

    def _validate_is_fitted(self) -> None:
        if not hasattr(self, "upper_bounds_"):
            raise RuntimeError("HardOutlierFilter must be fitted before filtering.")


@dataclass(slots=True)
class PreparedData:
    """Container with transformed train/test data and fitted metadata."""

    X_train: NDArray[np.float32]
    X_test: NDArray[np.float32]
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


@dataclass(slots=True)
class ChurnDataPreprocessor:
    """Prepare RetainAI feature-mart rows for churn models."""

    numeric_features: list[str] = field(default_factory=lambda: DEFAULT_NUMERIC_FEATURES.copy())
    categorical_features: list[str] = field(default_factory=lambda: DEFAULT_CATEGORICAL_FEATURES.copy())
    target_column: str = DEFAULT_TARGET_COLUMN
    artifact_path: Path = Path("artifacts/preprocessor.joblib")
    test_size: float = 0.2
    random_state: int = 42
    apply_outlier_filter: bool = True
    pipeline: Pipeline | None = field(default=None, init=False)
    outlier_filter: HardOutlierFilter | None = field(default=None, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        LOGGER.debug("Initialized ChurnDataPreprocessor.")

    def prepare_data(self, df: pd.DataFrame) -> tuple[NDArray[np.float32], NDArray[np.float32], pd.Series, pd.Series]:
        """Validate, split, fit preprocessing pipeline and transform features.

        Args:
            df: DataFrame with columns from `ml_customer_features`.

        Returns:
            Tuple of X_train, X_test, y_train, y_test. X arrays are already
            imputed, scaled and one-hot encoded.

        Raises:
            ValueError: If required columns are missing or the target is invalid.
            RuntimeError: If preprocessing fails unexpectedly.
        """

        try:
            LOGGER.info("Starting preprocessing for dataset with shape %s.", df.shape)
            self._validate_dataframe(df)

            model_frame = self._select_model_columns(df)
            X = model_frame.drop(columns=[self.target_column])
            y = model_frame[self.target_column].astype(np.int64)

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
            LOGGER.info(
                "Split data: train=%s rows, test=%s rows.",
                len(X_train_raw),
                len(X_test_raw),
            )

            if self.apply_outlier_filter:
                self.outlier_filter = HardOutlierFilter()
                self.outlier_filter.fit(X_train_raw)
                X_train_raw, filtered_y = self.outlier_filter.filter_dataframe(X_train_raw, y_train)
                if filtered_y is None:
                    raise RuntimeError("Outlier filtering returned no target vector.")
                y_train = filtered_y
                LOGGER.info("Train size after outlier filtering: %s rows.", len(X_train_raw))

            self.pipeline = self._build_pipeline()
            X_train = self.pipeline.fit_transform(X_train_raw, y_train)
            X_test = self.pipeline.transform(X_test_raw)
            self.feature_names_ = self._get_feature_names()

            LOGGER.info(
                "Preprocessing finished: X_train=%s, X_test=%s, features=%s.",
                X_train.shape,
                X_test.shape,
                len(self.feature_names_),
            )

            return (
                np.asarray(X_train, dtype=np.float32),
                np.asarray(X_test, dtype=np.float32),
                y_train.reset_index(drop=True),
                y_test.reset_index(drop=True),
            )
        except Exception as exc:
            LOGGER.exception("Failed to prepare churn dataset.")
            raise RuntimeError("Failed to prepare churn dataset.") from exc

    def prepare_data_with_metadata(self, df: pd.DataFrame) -> PreparedData:
        """Prepare data and return feature names together with matrices."""

        X_train, X_test, y_train, y_test = self.prepare_data(df)
        return PreparedData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.feature_names_.copy(),
        )

    def transform_new_data(self, df: pd.DataFrame) -> NDArray[np.float32]:
        """Transform new users with the fitted preprocessing pipeline.

        This method is intended for the FastAPI inference layer. It does not
        apply hard outlier row deletion because live users should still receive
        predictions even when they look unusual.
        """

        if self.pipeline is None:
            raise RuntimeError("Preprocessor pipeline is not fitted.")

        inference_frame = self._select_feature_columns(df)
        transformed = self.pipeline.transform(inference_frame)
        return np.asarray(transformed, dtype=np.float32)

    def save_preprocessor(self, artifact_path: str | Path | None = None) -> Path:
        """Persist the fitted sklearn Pipeline and metadata to disk."""

        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted preprocessor.")

        output_path = Path(artifact_path) if artifact_path is not None else self.artifact_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "pipeline": self.pipeline,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_column": self.target_column,
            "feature_names": self.feature_names_,
            "outlier_filter": self.outlier_filter,
        }
        joblib.dump(payload, output_path)
        LOGGER.info("Saved fitted preprocessor artifact to %s.", output_path)
        return output_path

    @classmethod
    def load_preprocessor(cls, artifact_path: str | Path) -> "ChurnDataPreprocessor":
        """Load a fitted preprocessor artifact produced by `save_preprocessor`."""

        input_path = Path(artifact_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found: {input_path}")

        payload = joblib.load(input_path)
        preprocessor = cls(
            numeric_features=payload["numeric_features"],
            categorical_features=payload["categorical_features"],
            target_column=payload["target_column"],
            artifact_path=input_path,
        )
        preprocessor.pipeline = payload["pipeline"]
        preprocessor.feature_names_ = payload.get("feature_names", [])
        preprocessor.outlier_filter = payload.get("outlier_filter")
        LOGGER.info("Loaded fitted preprocessor artifact from %s.", input_path)
        return preprocessor

    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn preprocessing Pipeline."""

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _build_one_hot_encoder()),
            ]
        )

        column_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.numeric_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )

        return Pipeline(steps=[("column_transformer", column_transformer)])

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate required columns and target quality."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        required_columns = set(self.numeric_features + self.categorical_features + [self.target_column])
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        target_values = set(pd.Series(df[self.target_column]).dropna().unique().tolist())
        if not target_values.issubset({0, 1, False, True}):
            raise ValueError(
                f"Target '{self.target_column}' must contain only binary values. "
                f"Got: {sorted(target_values)}"
            )

        if df[self.target_column].isna().any():
            raise ValueError(f"Target '{self.target_column}' contains missing values.")

        if df[self.target_column].nunique() < 2:
            raise ValueError("Target must contain both classes for stratified train/test split.")

    def _select_model_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only feature columns and target, avoiding leakage columns."""

        selected_columns = self.numeric_features + self.categorical_features + [self.target_column]
        return df.loc[:, selected_columns].copy()

    def _select_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only features required by the fitted Pipeline."""

        required_columns = set(self.numeric_features + self.categorical_features)
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(f"Missing required inference columns: {missing_columns}")

        return df.loc[:, self.numeric_features + self.categorical_features].copy()

    def _get_feature_names(self) -> list[str]:
        """Read output feature names from the fitted ColumnTransformer."""

        if self.pipeline is None:
            return []

        column_transformer = self.pipeline.named_steps["column_transformer"]
        try:
            return column_transformer.get_feature_names_out().tolist()
        except Exception:
            LOGGER.warning("Could not extract feature names from ColumnTransformer.")
            return []


def build_fake_feature_mart(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Create a fake `ml_customer_features`-like DataFrame for local testing."""

    rng = np.random.default_rng(random_state)

    account_age_days = rng.integers(30, 760, size=n_rows)
    recency_days = rng.integers(0, 120, size=n_rows)
    paid_tx_count = rng.poisson(lam=4.0, size=n_rows)
    session_count = rng.poisson(lam=8.0, size=n_rows)
    monetary_value = np.round(rng.gamma(shape=2.0, scale=120.0, size=n_rows), 2)
    paid_tx_count_30d = rng.binomial(n=np.maximum(paid_tx_count, 1), p=0.22)
    sessions_30d = rng.binomial(n=np.maximum(session_count, 1), p=0.28)
    avg_monthly_events_last_3m = rng.uniform(0, 20, size=n_rows)
    avg_monthly_events_before_3m = rng.uniform(0.1, 20, size=n_rows)

    # Inject a few extreme but realistic portfolio-test outliers.
    outlier_count = max(1, n_rows // 100)
    outlier_indices = rng.choice(n_rows, size=outlier_count, replace=False)
    monetary_value[outlier_indices] *= 25
    paid_tx_count[outlier_indices] += 80

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_rows + 1),
            "signup_ts": pd.Timestamp("2024-01-01"),
            "country": rng.choice(["US", "PL", "DE", "GB", "FR", "OTHER"], size=n_rows),
            "acquisition_channel": rng.choice(
                ["organic", "paid_search", "social", "referral", "partner"],
                size=n_rows,
            ),
            "plan_type": rng.choice(["free", "basic", "pro", "enterprise"], size=n_rows),
            "marketing_opt_in": rng.choice([True, False], size=n_rows),
            "age": rng.integers(18, 75, size=n_rows),
            "gender": rng.choice(["female", "male", "other"], size=n_rows),
            "account_age_days": account_age_days,
            "recency_days": recency_days,
            "payment_recency_days": np.clip(recency_days + rng.integers(-10, 30, size=n_rows), 0, None),
            "paid_tx_count": paid_tx_count,
            "all_tx_count": paid_tx_count + rng.poisson(lam=1.0, size=n_rows),
            "failed_tx_count": rng.poisson(lam=0.4, size=n_rows),
            "refunded_tx_count": rng.poisson(lam=0.2, size=n_rows),
            "product_category_count": rng.integers(1, 6, size=n_rows),
            "payment_method_count": rng.integers(1, 5, size=n_rows),
            "monetary_value": monetary_value,
            "avg_order_value": monetary_value / np.maximum(paid_tx_count, 1),
            "max_order_value": monetary_value / np.maximum(paid_tx_count, 1) * rng.uniform(1.1, 2.5, size=n_rows),
            "min_order_value": monetary_value / np.maximum(paid_tx_count, 1) * rng.uniform(0.2, 0.9, size=n_rows),
            "avg_days_between_paid_tx": rng.uniform(7, 65, size=n_rows),
            "ltv_observed": monetary_value,
            "last_rolling_3tx_avg_amount": monetary_value / np.maximum(paid_tx_count, 1),
            "paid_tx_count_30d": paid_tx_count_30d,
            "revenue_30d": paid_tx_count_30d * (monetary_value / np.maximum(paid_tx_count, 1)),
            "session_count": session_count,
            "total_page_views": session_count * rng.integers(2, 12, size=n_rows),
            "avg_page_views": rng.uniform(2, 12, size=n_rows),
            "total_actions": session_count * rng.integers(1, 10, size=n_rows),
            "avg_actions": rng.uniform(1, 10, size=n_rows),
            "avg_session_minutes": rng.uniform(3, 45, size=n_rows),
            "support_tickets_count": rng.poisson(lam=0.5, size=n_rows),
            "avg_days_between_sessions": rng.uniform(1, 35, size=n_rows),
            "recent_3_sessions_avg_actions": rng.uniform(0, 12, size=n_rows),
            "recent_3_sessions_avg_page_views": rng.uniform(1, 14, size=n_rows),
            "last_rolling_5session_actions": rng.uniform(0, 12, size=n_rows),
            "last_rolling_5session_page_views": rng.uniform(1, 14, size=n_rows),
            "sessions_30d": sessions_30d,
            "actions_30d": sessions_30d * rng.integers(0, 8, size=n_rows),
            "page_views_30d": sessions_30d * rng.integers(1, 10, size=n_rows),
            "support_tickets_30d": rng.poisson(lam=0.2, size=n_rows),
            "avg_monthly_events_last_3m": avg_monthly_events_last_3m,
            "avg_monthly_events_before_3m": avg_monthly_events_before_3m,
            "monthly_activity_slope": rng.normal(0, 0.00000001, size=n_rows),
            "activity_drop_ratio": avg_monthly_events_last_3m / avg_monthly_events_before_3m,
            "revenue_per_account_day": monetary_value / np.maximum(account_age_days, 1),
            "actions_per_session": (session_count * rng.integers(1, 10, size=n_rows)) / np.maximum(session_count, 1),
            "ltv_segment": pd.cut(
                monetary_value,
                bins=[-np.inf, 250, 1000, np.inf],
                labels=["low", "medium", "high"],
            ).astype(str),
            "extracted_at": pd.Timestamp.now(tz="UTC"),
        }
    )

    churn_logit = (
        -3.0
        + 0.045 * df["recency_days"]
        + 0.012 * df["payment_recency_days"]
        - 0.07 * df["paid_tx_count"]
        - 0.04 * df["session_count"]
        - 0.18 * df["sessions_30d"]
        - 0.28 * df["paid_tx_count_30d"]
        + 0.75 * (df["activity_drop_ratio"] < 0.60).astype(float)
        + 0.45 * (df["sessions_30d"] == 0).astype(float)
        + 0.35 * (df["paid_tx_count_30d"] == 0).astype(float)
        + 0.45 * (df["plan_type"] == "free").astype(float)
        + 0.25 * (df["support_tickets_count"] > 1).astype(float)
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    df["churn_label_45d"] = rng.binomial(1, churn_probability)

    # Add a couple of missing values to exercise imputers in the test run.
    missing_indices = rng.choice(n_rows, size=max(1, n_rows // 50), replace=False)
    df.loc[missing_indices, "avg_order_value"] = np.nan
    df.loc[missing_indices, "acquisition_channel"] = np.nan

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    fake_df = build_fake_feature_mart(n_rows=1000, random_state=42)
    preprocessor = ChurnDataPreprocessor()

    prepared = preprocessor.prepare_data_with_metadata(fake_df)
    artifact = preprocessor.save_preprocessor("artifacts/preprocessor.joblib")

    LOGGER.info("Smoke test artifact: %s", artifact)
    LOGGER.info("Smoke test train matrix shape: %s", prepared.X_train.shape)
    LOGGER.info("Smoke test test matrix shape: %s", prepared.X_test.shape)
    LOGGER.info("First 10 output features: %s", prepared.feature_names[:10])
