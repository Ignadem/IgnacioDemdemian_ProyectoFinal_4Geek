from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "fraud_risk_model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"

LOW_THRESHOLD = 0.10
HIGH_THRESHOLD = 0.30


def load_model_and_metadata() -> tuple[object, dict]:
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    model = joblib.load(MODEL_PATH)
    return model, metadata


def render_feature_inputs(feature_columns: list[str], feature_stats: dict) -> pd.DataFrame:
    st.subheader("Company-year financial inputs")
    st.caption(
        "Enter one company record using the 12 approved features. "
        "This app returns a fraud-risk score for review prioritization."
    )

    inputs: dict[str, float] = {}

    cols = st.columns(3)
    for idx, feature in enumerate(feature_columns):
        col = cols[idx % len(cols)]
        with col:
            stats = feature_stats.get(feature, {})
            default = float(stats.get("median", 0.0))
            min_value = float(stats.get("min", -1e6))
            max_value = float(stats.get("max", 1e6))
            step = 1.0 if feature == "Financial_Year" else 0.01

            inputs[feature] = st.number_input(
                label=feature,
                min_value=min_value,
                max_value=max_value,
                value=default,
                step=step,
                format="%.6g",
            )

    return pd.DataFrame([inputs], columns=feature_columns)


def risk_bucket(score: float) -> tuple[str, str]:
    if score >= HIGH_THRESHOLD:
        return "High", "⚠️"
    if score >= LOW_THRESHOLD:
        return "Medium", "🟠"
    return "Low", "🟢"


def main() -> None:
    st.set_page_config(page_title="Fraud Risk Prioritization", page_icon="🧾", layout="centered")
    st.title("Fraud-Risk Prioritization (Demo)")

    st.markdown(
        """
        This app does **not** prove fraud.
        It estimates how closely a company-year record resembles known fraud-related cases
        and returns a review-priority score.
        """
    )

    try:
        model, metadata = load_model_and_metadata()
    except Exception as exc:  # pragma: no cover - runtime UI error handling
        st.error(f"Unable to load model artifacts: {exc}")
        return

    feature_columns = metadata.get("feature_columns") or []
    if not feature_columns:
        st.error("Model metadata does not contain feature_columns.")
        return

    feature_stats = metadata.get("feature_statistics", {})
    model_name = metadata.get("model_name", "Saved model")
    threshold = float(metadata.get("selected_model_threshold", 0.5))

    st.markdown(f"**Model:** `{model_name}`")
    st.markdown(f"**Review threshold used for training validation:** `{threshold:.4f}`")

    input_df = render_feature_inputs(feature_columns, feature_stats)

    if st.button("Run fraud-risk scoring", type="primary"):
        probabilities = model.predict_proba(input_df)[0]
        score = float(probabilities[1])
        risk_label, emoji = risk_bucket(score)

        col_left, col_right = st.columns(2)
        with col_left:
            st.metric("Fraud-risk probability", f"{score:.4%}")
        with col_right:
            st.metric("Review priority", f"{emoji} {risk_label}")

        st.progress(score)
        st.info(
            "Interpretation: "
            "`High` and `Medium` values suggest this row should be reviewed first by an analyst. "
            "This is not a legal judgment or audit proof of fraud."
        )

        st.divider()
        st.subheader("Scoring context")
        st.write(f"- Chosen low-risk threshold: `{LOW_THRESHOLD:.2f}`")
        st.write(f"- Chosen medium-high threshold: `{HIGH_THRESHOLD:.2f}`")
        st.write(f"- Disclaimer: {metadata.get('review_priority_disclaimer')}")

        with st.expander("What this score is and is not"):
            st.write(
                "- **What it is:** a model-generated risk score based on the 12-feature financial vector."
            )
            st.write("- **What it is not:** a verdict that a company committed fraud.")
            st.write(
                "- **Why this project has this format:** the training labels are based on "
                "`AAER_ID` presence and are incomplete for every real-world case."
            )


if __name__ == "__main__":
    main()
