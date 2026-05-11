from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "fraud_risk_model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
REPORTS_DIR = ROOT_DIR / "reports"
EDA_FIGURE_DIR = REPORTS_DIR / "eda" / "figures"
EDA_REPORT_PATH = REPORTS_DIR / "eda" / "focused_eda.md"

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


def _render_metric_cards(metadata: dict) -> None:
    dataset_rows = metadata.get("dataset_rows", 0)
    positive_ratio = float(metadata.get("dataset_positive_ratio", 0.0))
    model_name = metadata.get("model_name", "Saved model")
    threshold = float(metadata.get("selected_model_threshold", 0.5))

    metric_help = {
        "average_precision": "Area under the precision-recall curve; higher is better for finding rare frauds first.",
        "roc_auc": "ROC AUC: higher means better separation between fraud and non-fraud records.",
        "recall": "Recall: share of actual fraud cases the model catches; higher means fewer misses.",
        "precision": "Precision: share of flagged cases that are truly fraud-like; higher means fewer false alerts.",
        "f1": "F1 score: balance between precision and recall.",
        "threshold": "Decision threshold: probability cutoff used for review-priority label boundaries.",
    }

    def as_percent(value: float) -> str:
        return f"{float(value) * 100:.2f}%"

    def metric_block(model_label: str, metrics: dict) -> None:
        st.markdown(f"#### {model_label}")
        model_metrics = [
            ("average_precision", "Avg Precision"),
            ("roc_auc", "ROC AUC"),
            ("recall", "Recall"),
            ("precision", "Precision"),
            ("f1", "F1"),
            ("threshold", "Threshold"),
        ]

        for start in range(0, len(model_metrics), 3):
            cols = st.columns(3)
            row_metrics = model_metrics[start:start + 3]
            for idx, (key, label) in enumerate(row_metrics):
                with cols[idx]:
                    value = metrics.get(key, 0)
                    st.metric(
                        label=label,
                        value=as_percent(value),
                        help=metric_help.get(key),
                    )

        if "params" in metrics:
            st.caption(f"Model params: {metrics['params']}")
        st.divider()

    st.subheader("Model Evidence")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Model", model_name)
    with col_b:
        st.metric("Records used", f"{dataset_rows:,}")
    with col_c:
        st.metric("Known fraud-labeled ratio", f"{positive_ratio:.4%}")
    with col_d:
        st.metric(
            "Chosen threshold",
            as_percent(threshold),
            help=metric_help["threshold"],
        )

    st.markdown("### Validation and held-out test metrics")
    val = metadata.get("validation_metrics", {})
    test_metrics = metadata.get("test_metrics", {})

    if "logistic_regression" in val:
        metric_block("Logistic Regression (baseline)", val["logistic_regression"])
    if "random_forest_tuned" in val:
        metric_block("Random Forest (tuned)", val["random_forest_tuned"])

    if test_metrics:
        st.markdown("#### Held-out test (selected model)")
        cols = st.columns(3)
        with cols[0]:
            st.metric(
                "Avg Precision",
                as_percent(test_metrics.get("average_precision", 0)),
                help=metric_help["average_precision"],
            )
        with cols[1]:
            st.metric(
                "ROC AUC",
                as_percent(test_metrics.get("roc_auc", 0)),
                help=metric_help["roc_auc"],
            )
        with cols[2]:
            st.metric(
                "Recall",
                as_percent(test_metrics.get("recall", 0)),
                help=metric_help["recall"],
            )
        cols = st.columns(3)
        with cols[0]:
            st.metric(
                "Precision",
                as_percent(test_metrics.get("precision", 0)),
                help=metric_help["precision"],
            )
        with cols[1]:
            st.metric(
                "F1",
                as_percent(test_metrics.get("f1", 0)),
                help=metric_help["f1"],
            )
        with cols[2]:
            st.metric(
                "Threshold",
                as_percent(test_metrics.get("threshold", 0)),
                help=metric_help["threshold"],
            )
        st.caption(
            f"Test threshold used from validation: {as_percent(test_metrics.get('threshold', 0))}"
        )


def _render_project_overview() -> None:
    st.subheader("Project Overview")
    st.markdown(
        """
        This project is a **fraud-risk prioritization demo** for a machine learning course.

        - It uses only the 12 official financial features.
        - Labels come from `AAER_ID` presence (fraud-enforcement linkage).
        - The app ranks one input record by review-priority risk.
        - It does **not** prove fraud.
        """
    )
    st.markdown("### Labels")
    st.write(
        "- `target_fraud = 1`: `AAER_ID` exists in the source row."
        "\n- `target_fraud = 0`: `AAER_ID` is missing in this dataset."
    )


def _render_eda_gallery() -> None:
    st.subheader("EDA Visuals")
    if not EDA_FIGURE_DIR.exists():
        st.info("EDA figure folder not found yet.")
        return

    figures = [
        ("target_balance.png", "Target Balance"),
        ("fraud_rate_by_year.png", "Fraud Rate by Financial Year"),
        ("key_variable_distributions.png", "Key Variable Distributions"),
        ("fraud_vs_nonfraud_averages.png", "Fraud vs Non-Fraud Averages"),
        ("feature_correlation_heatmap.png", "Feature Correlation Heatmap"),
    ]
    for filename, title in figures:
        path = EDA_FIGURE_DIR / filename
        if not path.exists():
            continue
        with st.expander(title, expanded=True):
            st.image(str(path), use_container_width=True)

    if EDA_REPORT_PATH.exists():
        st.markdown("### EDA summary report")
        with st.expander("Open `focused_eda.md` summary"):
            st.code(EDA_REPORT_PATH.read_text(encoding="utf-8"), language="markdown")



def main() -> None:
    st.set_page_config(page_title="Fraud-Risk Prioritization", page_icon="🧾", layout="wide")
    st.title("Fraud-Risk Prioritization (Demo)")
    st.caption("Review-priority scoring for one company-year record. Not a legal or audit verdict.")

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

    tab_overview, tab_model, tab_eda, tab_predict = st.tabs(
        ["Project Overview", "Model Evidence", "EDA & Metrics", "Predict"]
    )

    with tab_predict:
        st.markdown(
            """
            This app does **not** prove fraud.
            It estimates how closely a company-year record resembles known fraud-related cases
            and returns a review-priority score.
            """
        )
        st.markdown(f"**Model:** `{model_name}`")
        st.markdown(f"**Review threshold used for training validation:** `{threshold:.2%}`")

        input_df = render_feature_inputs(feature_columns, feature_stats)
        if st.button("Run fraud-risk scoring", type="primary"):
            probabilities = model.predict_proba(input_df)[0]
            score = float(probabilities[1])
            risk_label, emoji = risk_bucket(score)

            left, right = st.columns(2)
            with left:
                st.metric("Fraud-risk probability", f"{score:.4%}")
            with right:
                st.metric("Review priority", f"{emoji} {risk_label}")

            st.progress(score)
            st.info(
                "Interpretation: "
                "`High` and `Medium` values suggest this row should be reviewed first by an analyst. "
                "This is not a legal judgment or audit proof of fraud."
            )

            st.divider()
            st.subheader("Scoring context")
            st.write(f"- Chosen low-risk threshold: `{LOW_THRESHOLD:.2%}`")
            st.write(f"- Chosen medium-high threshold: `{HIGH_THRESHOLD:.2%}`")
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

        with st.expander("Quick start values (defaults)"):
            st.write("Input defaults come from training data medians.")
            st.json(metadata.get("feature_statistics", {}))

    with tab_overview:
        _render_project_overview()

    with tab_model:
        _render_metric_cards(metadata)

    with tab_eda:
        _render_eda_gallery()


if __name__ == "__main__":
    main()
