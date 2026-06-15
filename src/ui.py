"""Streamlit demo UI for RetainAI.

Run locally:
    streamlit run src/ui.py

The UI calls the FastAPI service, so keep `docker compose up --build` running
or start the API locally with uvicorn first.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st


API_URL = os.getenv("RETAINAI_API_URL", "http://localhost:8000").rstrip("/")


def fetch_prediction(user_id: int) -> dict[str, Any]:
    """Fetch DB-backed churn prediction from FastAPI."""

    with httpx.Client(timeout=30) as client:
        response = client.get(f"{API_URL}/customers/{user_id}/predict")
        response.raise_for_status()
        return response.json()


def fetch_health() -> dict[str, Any] | None:
    """Fetch API health; return None when the service is unavailable."""

    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{API_URL}/health")
            response.raise_for_status()
            return response.json()
    except Exception:
        return None


def risk_color(risk_segment: str) -> str:
    """Map risk segment to a simple display color."""

    return {
        "low": "#1f9d55",
        "medium": "#b7791f",
        "high": "#c53030",
    }.get(risk_segment, "#4a5568")


def main() -> None:
    st.set_page_config(page_title="RetainAI Churn Demo", layout="wide")

    st.title("RetainAI Churn Intelligence")
    st.caption("FastAPI + PostgreSQL + RandomForest + PyTorch + Ollama")

    health = fetch_health()
    if health is None:
        st.error(f"API is not reachable at {API_URL}. Start the backend first.")
        st.stop()

    status = health.get("status", "degraded")
    st.sidebar.header("Service")
    st.sidebar.write(f"API URL: `{API_URL}`")
    st.sidebar.write(f"Status: `{status}`")
    st.sidebar.write(f"Ollama model: `{health.get('ollama_model')}`")

    user_id = st.number_input(
        "Customer ID",
        min_value=1,
        max_value=15000,
        value=101,
        step=1,
        help="Synthetic users are generated with IDs from 1 to 15000.",
    )

    if st.button("Predict churn", type="primary"):
        try:
            result = fetch_prediction(int(user_id))
        except httpx.HTTPStatusError as exc:
            st.error(f"API returned {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except Exception as exc:
            st.error(f"Prediction request failed: {exc}")
            st.stop()

        risk_segment = result["risk_segment"]
        color = risk_color(risk_segment)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Baseline churn", f"{result['baseline_churn_probability']:.1%}")
        metric_cols[1].metric("DL churn", f"{result['dl_churn_probability']:.1%}")
        metric_cols[2].metric("Ensemble churn", f"{result['ensemble_churn_probability']:.1%}")
        metric_cols[3].markdown(
            f"""
            <div style="padding: 0.75rem; border-left: 6px solid {color}; background: #f7fafc;">
              <div style="font-size: 0.85rem; color: #4a5568;">Risk segment</div>
              <div style="font-size: 1.6rem; font-weight: 700; color: {color};">{risk_segment.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Retention recommendation")
        st.write(result["retention_advice"])

        st.caption(
            f"LLM available: {result['llm_available']} | "
            f"Model: {result['llm_model']} | "
            f"User ID: {result['user_id']}"
        )

        with st.expander("Raw API response"):
            st.json(result)


if __name__ == "__main__":
    main()
