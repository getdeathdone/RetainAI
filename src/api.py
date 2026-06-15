"""FastAPI inference service for RetainAI churn prediction and LLM advice.

The service loads three artifacts at startup:
    * artifacts/preprocessor.joblib
    * artifacts/baseline_model.joblib
    * artifacts/dl_model.pth

It exposes a health endpoint and a prediction endpoint that combines classical
ML, PyTorch inference and a local Ollama-powered retention recommendation.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import asyncpg
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as ApiPath, Request, status
from pydantic import BaseModel, Field

try:
    from .baseline import ChurnBaselineModel
    from .dl_model import ChurnDeepLearningModel
    from .preprocessing import ChurnDataPreprocessor
except ImportError:
    from baseline import ChurnBaselineModel
    from dl_model import ChurnDeepLearningModel
    from preprocessing import ChurnDataPreprocessor


LOGGER = logging.getLogger(__name__)

ARTIFACT_DIR = Path(os.getenv("RETAINAI_ARTIFACT_DIR", "artifacts"))
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.joblib"
BASELINE_MODEL_PATH = ARTIFACT_DIR / "baseline_model.joblib"
DL_MODEL_PATH = ARTIFACT_DIR / "dl_model.pth"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "20"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "retainai")
POSTGRES_USER = os.getenv("POSTGRES_USER", "retainai")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "retainai_password")

APP_STATE: dict[str, Any] = {}


class CustomerData(BaseModel):
    """Input schema matching the feature columns expected by the preprocessor."""

    user_id: int | None = Field(default=None, ge=1)
    country: str = Field(default="US", min_length=2, max_length=32)
    acquisition_channel: Literal["organic", "paid_search", "social", "referral", "partner"] = "organic"
    plan_type: Literal["free", "basic", "pro", "enterprise"] = "basic"
    marketing_opt_in: bool = True
    age: int | None = Field(default=None, ge=16, le=90)
    gender: Literal["female", "male", "other"] = "other"

    account_age_days: int = Field(ge=0)
    recency_days: int = Field(ge=0)
    payment_recency_days: int = Field(ge=0)
    paid_tx_count: int = Field(ge=0)
    all_tx_count: int = Field(ge=0)
    failed_tx_count: int = Field(ge=0)
    refunded_tx_count: int = Field(ge=0)
    product_category_count: int = Field(ge=0)
    payment_method_count: int = Field(ge=0)
    monetary_value: float = Field(ge=0)
    avg_order_value: float = Field(ge=0)
    max_order_value: float = Field(ge=0)
    min_order_value: float = Field(ge=0)
    avg_days_between_paid_tx: float = Field(ge=0)
    ltv_observed: float = Field(ge=0)
    last_rolling_3tx_avg_amount: float = Field(ge=0)
    session_count: int = Field(ge=0)
    total_page_views: int = Field(ge=0)
    avg_page_views: float = Field(ge=0)
    total_actions: int = Field(ge=0)
    avg_actions: float = Field(ge=0)
    avg_session_minutes: float = Field(ge=0)
    support_tickets_count: int = Field(ge=0)
    avg_days_between_sessions: float = Field(ge=0)
    recent_3_sessions_avg_actions: float = Field(ge=0)
    recent_3_sessions_avg_page_views: float = Field(ge=0)
    last_rolling_5session_actions: float = Field(ge=0)
    last_rolling_5session_page_views: float = Field(ge=0)
    avg_monthly_events_last_3m: float = Field(ge=0)
    avg_monthly_events_before_3m: float = Field(ge=0)
    monthly_activity_slope: float = 0.0
    ltv_segment: Literal["low", "medium", "high"] = "low"

    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "user_id": 101,
                "country": "PL",
                "acquisition_channel": "paid_search",
                "plan_type": "pro",
                "marketing_opt_in": True,
                "age": 34,
                "gender": "female",
                "account_age_days": 240,
                "recency_days": 38,
                "payment_recency_days": 52,
                "paid_tx_count": 5,
                "all_tx_count": 6,
                "failed_tx_count": 1,
                "refunded_tx_count": 0,
                "product_category_count": 3,
                "payment_method_count": 2,
                "monetary_value": 720.5,
                "avg_order_value": 144.1,
                "max_order_value": 220.0,
                "min_order_value": 79.0,
                "avg_days_between_paid_tx": 31.5,
                "ltv_observed": 720.5,
                "last_rolling_3tx_avg_amount": 158.2,
                "session_count": 16,
                "total_page_views": 124,
                "avg_page_views": 7.75,
                "total_actions": 82,
                "avg_actions": 5.12,
                "avg_session_minutes": 18.4,
                "support_tickets_count": 2,
                "avg_days_between_sessions": 12.1,
                "recent_3_sessions_avg_actions": 2.4,
                "recent_3_sessions_avg_page_views": 4.2,
                "last_rolling_5session_actions": 3.1,
                "last_rolling_5session_page_views": 5.8,
                "avg_monthly_events_last_3m": 2.7,
                "avg_monthly_events_before_3m": 6.8,
                "monthly_activity_slope": -0.00000002,
                "ltv_segment": "medium",
            }
        }


class PredictionResponse(BaseModel):
    """Response schema with model probabilities and generated manager advice."""

    user_id: int | None
    baseline_churn_probability: float
    dl_churn_probability: float
    ensemble_churn_probability: float
    risk_segment: Literal["low", "medium", "high"]
    retention_advice: str
    llm_model: str
    llm_available: bool


class HealthResponse(BaseModel):
    """Operational health response."""

    status: Literal["ok", "degraded"]
    artifacts_loaded: dict[str, bool]
    database_connected: bool
    ollama_url: str
    ollama_model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts once on startup and release references on shutdown."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    LOGGER.info("Starting RetainAI API and loading artifacts.")
    try:
        APP_STATE["preprocessor"] = ChurnDataPreprocessor.load_preprocessor(PREPROCESSOR_PATH)
        APP_STATE["baseline_model"] = ChurnBaselineModel.load_model(BASELINE_MODEL_PATH)
        APP_STATE["dl_model"] = ChurnDeepLearningModel.load_model(DL_MODEL_PATH)
        APP_STATE["db_pool"] = await asyncpg.create_pool(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            min_size=1,
            max_size=5,
            command_timeout=30,
        )
        app.state.preprocessor = APP_STATE["preprocessor"]
        app.state.baseline_model = APP_STATE["baseline_model"]
        app.state.dl_model = APP_STATE["dl_model"]
        app.state.db_pool = APP_STATE["db_pool"]
        LOGGER.info("All RetainAI artifacts loaded successfully.")
    except FileNotFoundError as exc:
        LOGGER.exception("Required model artifact is missing.")
        raise RuntimeError(f"Required model artifact is missing: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Failed to load RetainAI artifacts.")
        raise RuntimeError("Failed to load RetainAI artifacts.") from exc

    try:
        yield
    finally:
        LOGGER.info("Shutting down RetainAI API and clearing model state.")
        APP_STATE.clear()
        app.state.preprocessor = None
        app.state.baseline_model = None
        app.state.dl_model = None
        db_pool = getattr(app.state, "db_pool", None)
        if db_pool is not None:
            await db_pool.close()
        app.state.db_pool = None


app = FastAPI(
    title="RetainAI Churn Intelligence API",
    description="Churn prediction service with classical ML, PyTorch and Ollama recommendations.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and loaded artifact status."""

    artifacts_loaded = {
        "preprocessor": "preprocessor" in APP_STATE,
        "baseline_model": "baseline_model" in APP_STATE,
        "dl_model": "dl_model" in APP_STATE,
    }
    database_connected = APP_STATE.get("db_pool") is not None
    service_ok = all(artifacts_loaded.values()) and database_connected

    return HealthResponse(
        status="ok" if service_ok else "degraded",
        artifacts_loaded=artifacts_loaded,
        database_connected=database_connected,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
    )


@app.post("/predict/churn", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData, request: Request) -> PredictionResponse:
    """Predict churn with both ML models and generate an LLM retention action."""

    customer_dict = _model_to_dict(customer)
    return await _predict_from_customer_dict(customer_dict, request)


@app.get("/customers/{user_id}/predict", response_model=PredictionResponse)
async def predict_customer_from_database(
    request: Request,
    user_id: int = ApiPath(ge=1),
) -> PredictionResponse:
    """Fetch customer features from Postgres and return churn prediction."""

    db_pool = getattr(request.app.state, "db_pool", None) or APP_STATE.get("db_pool")
    if db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool is not available.",
        )

    try:
        async with db_pool.acquire() as connection:
            record = await connection.fetchrow(
                "SELECT * FROM retainai.ml_customer_features WHERE user_id = $1",
                user_id,
            )
    except Exception as exc:
        LOGGER.exception("Failed to fetch customer %s from database.", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch customer features from database.",
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {user_id} was not found in ml_customer_features.",
        )

    customer_dict = _record_to_customer_dict(record)
    return await _predict_from_customer_dict(customer_dict, request)


async def _predict_from_customer_dict(customer_dict: dict[str, Any], request: Request) -> PredictionResponse:
    """Shared prediction flow for manual payloads and DB-backed customers."""

    preprocessor = getattr(request.app.state, "preprocessor", None) or APP_STATE.get("preprocessor")
    baseline_model = getattr(request.app.state, "baseline_model", None) or APP_STATE.get("baseline_model")
    dl_model = getattr(request.app.state, "dl_model", None) or APP_STATE.get("dl_model")

    if preprocessor is None or baseline_model is None or dl_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded.",
        )

    try:
        customer_frame = pd.DataFrame([customer_dict])
        features = preprocessor.transform_new_data(customer_frame)

        baseline_probability = float(baseline_model.predict_proba(features)[0])
        dl_probability = float(dl_model.predict_proba(features)[0])
        ensemble_probability = float(np.mean([baseline_probability, dl_probability]))

        advice, llm_available = await generate_retention_advice(
            customer=customer_dict,
            churn_prob=ensemble_probability,
        )

        return PredictionResponse(
            user_id=customer_dict.get("user_id"),
            baseline_churn_probability=round(baseline_probability, 6),
            dl_churn_probability=round(dl_probability, 6),
            ensemble_churn_probability=round(ensemble_probability, 6),
            risk_segment=_risk_segment(ensemble_probability),
            retention_advice=advice,
            llm_model=OLLAMA_MODEL,
            llm_available=llm_available,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        LOGGER.exception("Invalid prediction request.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check server logs for details.",
        ) from exc


async def generate_retention_advice(customer: dict[str, Any], churn_prob: float) -> tuple[str, bool]:
    """Generate retention advice using local Ollama, with a deterministic fallback."""

    prompt = _build_retention_prompt(customer, churn_prob)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 280,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            advice = str(data.get("response", "")).strip()

            if not advice:
                raise ValueError("Ollama returned an empty response.")

            return advice, True
    except Exception as exc:
        LOGGER.warning("Ollama advice generation failed; using fallback. Reason: %s", exc)
        return _fallback_retention_advice(customer, churn_prob), False


def _build_retention_prompt(customer: dict[str, Any], churn_prob: float) -> str:
    """Build a compact business prompt for Ollama."""

    return f"""
You are a senior customer retention strategist for a subscription business.
Your task is to give a concise, actionable recommendation to an account manager.

Rules:
- Write in clear business language.
- Mention the churn risk level.
- Give exactly 3 concrete retention actions.
- Do not invent private data.
- Keep the answer under 140 words.

Customer snapshot:
- Churn probability: {churn_prob:.1%}
- Plan: {customer.get("plan_type")}
- LTV observed: {customer.get("ltv_observed")}
- Monetary value: {customer.get("monetary_value")}
- Account age days: {customer.get("account_age_days")}
- Recency days: {customer.get("recency_days")}
- Payment recency days: {customer.get("payment_recency_days")}
- Paid transactions: {customer.get("paid_tx_count")}
- Sessions: {customer.get("session_count")}
- Recent avg actions: {customer.get("recent_3_sessions_avg_actions")}
- Support tickets: {customer.get("support_tickets_count")}
- Marketing opt-in: {customer.get("marketing_opt_in")}

Return only the recommendation.
""".strip()


def _fallback_retention_advice(customer: dict[str, Any], churn_prob: float) -> str:
    """Return a stable recommendation when Ollama is unavailable."""

    risk = _risk_segment(churn_prob)
    plan_type = customer.get("plan_type", "unknown")
    recency_days = customer.get("recency_days", "unknown")
    support_tickets = customer.get("support_tickets_count", 0)

    if risk == "high":
        return (
            f"High churn risk for a {plan_type} customer. "
            f"Prioritize a direct manager outreach, address {support_tickets} recent support issues, "
            f"and offer a targeted retention incentive tied to the customer's last active use case."
        )

    if risk == "medium":
        return (
            f"Medium churn risk. The customer has {recency_days} days since recent activity. "
            "Send a personalized reactivation message, recommend the next best feature, "
            "and monitor engagement over the next 7 days."
        )

    return (
        "Low churn risk. Keep the customer engaged with educational content, "
        "light-touch product tips, and periodic check-ins focused on expansion potential."
    )


def _risk_segment(churn_prob: float) -> Literal["low", "medium", "high"]:
    """Map churn probability to a manager-friendly risk segment."""

    if churn_prob >= 0.7:
        return "high"
    if churn_prob >= 0.4:
        return "medium"
    return "low"


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Support both Pydantic v1 and v2 serialization APIs."""

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _record_to_customer_dict(record: asyncpg.Record) -> dict[str, Any]:
    """Convert a feature-mart DB record into CustomerData-compatible dict."""

    raw_record = dict(record)
    allowed_fields = _customer_schema_fields()
    customer_dict: dict[str, Any] = {}

    for key in allowed_fields:
        value = raw_record.get(key)
        if isinstance(value, Decimal):
            value = float(value)
        customer_dict[key] = value

    return _model_to_dict(CustomerData(**customer_dict))


def _customer_schema_fields() -> set[str]:
    """Return CustomerData field names for both Pydantic v1 and v2."""

    if hasattr(CustomerData, "model_fields"):
        return set(CustomerData.model_fields.keys())
    return set(CustomerData.__fields__.keys())
