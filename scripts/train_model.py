from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Cleaned_data_1995_2018.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_risk_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
RESULTS_PATH = BASE_DIR / "reports" / "modeling" / "model_results.md"

FEATURE_COLUMNS = [
    "Financial_Year",
    "sale",
    "ni",
    "at",
    "lt",
    "che",
    "rect",
    "invt",
    "cogs",
    "txt",
    "xint",
    "prcc_f",
]
TARGET_COLUMN = "target_fraud"
RANDOM_STATE = 42

RANDOM_FOREST_GRID = {
    "n_estimators": [200],
    "max_depth": [None, 12],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt"],
    "class_weight": ["balanced_subsample"],
}


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + ["AAER_ID"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")

    model_df = df[FEATURE_COLUMNS].copy()
    model_df["Financial_Year"] = (
        model_df["Financial_Year"].astype(str).str.replace("FY", "", regex=False).astype(int)
    )
    model_df[TARGET_COLUMN] = df["AAER_ID"].notna().astype(int)
    return model_df


def make_splits(df: pd.DataFrame):
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )


def make_random_forest_preprocessor() -> Pipeline:
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


def build_logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest(params: dict[str, object]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_random_forest_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **params,
                ),
            ),
        ]
    )


def choose_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if len(thresholds) == 0:
        return 0.5, 0

    f1_scores = np.zeros_like(precision[:-1], dtype=float)
    denominator = precision[:-1] + recall[:-1]
    valid = denominator > 0
    f1_scores[valid] = 2 * precision[:-1][valid] * recall[:-1][valid] / denominator[valid]

    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_predictions(y_true: pd.Series, y_scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, y_scores)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)],
    }


def evaluate_model(model: Pipeline, x_val: pd.DataFrame, y_val: pd.Series) -> tuple[float, dict[str, float]]:
    y_scores = model.predict_proba(x_val)[:, 1]
    threshold, _ = choose_best_threshold(y_val.to_numpy(), y_scores)
    metrics = evaluate_predictions(y_val, y_scores, threshold)
    return threshold, metrics


def feature_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    for col in FEATURE_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "min": float(series.min(skipna=True)),
            "max": float(series.max(skipna=True)),
            "median": float(series.median(skipna=True)),
            "mean": float(series.mean(skipna=True)),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
        }
    return stats


def tune_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict[str, object], dict[str, float]]:
    best_model = None
    best_metrics: dict[str, float] | None = None
    best_params: dict[str, object] | None = None

    for params in ParameterGrid(RANDOM_FOREST_GRID):
        candidate = build_random_forest(params)
        candidate.fit(x_train, y_train)
        threshold, metrics = evaluate_model(candidate, x_val, y_val)
        metrics.update({"model_name": "Random Forest (tuned)"})
        metrics["threshold"] = threshold
        metrics["params"] = params

        if best_metrics is None or metrics["average_precision"] > best_metrics["average_precision"]:
            best_model = candidate
            best_metrics = metrics
            best_params = params

    if best_model is None or best_metrics is None or best_params is None:
        raise RuntimeError("Random Forest tuning failed to produce any candidate.")

    return best_model, best_params, best_metrics


def format_confusion_matrix(metrics: dict[str, float]) -> str:
    tn, fp, fn, tp = [int(v) for v in metrics["confusion_matrix"]]
    return (
        "| true\\pred | 0 | 1 |\n"
        "| --- | ---: | ---: |\n"
        f"| 0 | {tn} | {fp} |\n"
        f"| 1 | {fn} | {tp} |"
    )


def format_metric_rows(metrics: dict[str, float]) -> str:
    ordered = ["average_precision", "roc_auc", "recall", "precision", "f1", "threshold"]
    lines = [
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in ordered:
        value = metrics[key]
        lines.append(f"| {key.replace('_', ' ').title()} | {value:.6f} |")
    return "\n".join(lines)


def write_report(
    row_count: int,
    positive_rate: float,
    logistic_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    selected_name: str,
    test_metrics: dict[str, float],
) -> None:
    feature_csv = ", ".join(FEATURE_COLUMNS)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    selected_row = "Logistic Regression"
    if selected_name == "Random Forest (tuned)":
        selected_row = "Random Forest (tuned)"

    lines = [
        "# Modeling Results: Phase 3",
        "",
        "## Dataset Summary",
        f"- Rows: {row_count}",
        f"- Features used: {len(FEATURE_COLUMNS)}",
        f"- Positive ratio: {positive_rate:.6f} ({positive_rate*100:.4f}%)",
        "",
        "## Why PR AUC / Average Precision is the primary metric",
        "Only about 0.6536% of records in this dataset have a known fraud-related label.",
        "With this imbalance, accuracy is misleading because predicting `0` for almost every row still looks good numerically but misses the risky cases.",
        "Priority for this course demo is *fraud-risk ranking*: use Average Precision so high scores are more likely to include known fraud-labeled cases.",
        "",
        "## Metric definitions",
        "- **Average Precision / PR AUC**: precision-recall ranking quality for rare positive labels.",
        "- **ROC AUC**: score separability between known fraud-labeled and non-labeled rows.",
        "- **Recall**: of known fraud-labeled rows, how many are flagged as suspicious.",
        "- **Precision**: of flagged suspicious rows, how many are actually fraud-labeled.",
        "- **F1**: combined recall/precision score at selected threshold.",
        "- **Confusion matrix**: counts of true negatives, false positives, false negatives, and true positives.",
        "",
        "## Official 12 feature set",
        feature_csv,
        "",
        "## Model training setup",
        "- Validation split: 70% train / 15% validation / 15% test (stratified).",
        "- Random state: 42.",
        "",
        "## Validation comparison",
        "",
        "### Logistic Regression (baseline)",
        format_metric_rows(logistic_metrics),
        "",
        "#### Confusion Matrix (validation split)",
        format_confusion_matrix(logistic_metrics),
        "",
        "### Random Forest (tuned)",
        format_metric_rows(rf_metrics),
        "",
        f"- Best params: `{json.dumps(rf_metrics['params'])}`",
        "",
        "#### Confusion Matrix (validation split)",
        format_confusion_matrix(rf_metrics),
        "",
        "## Final model selection",
        f"- Selected for test evaluation: **{selected_row}**",
        "",
        "## Held-out test metrics",
        format_metric_rows(test_metrics),
        "",
        "### Confusion Matrix (held-out test)",
        format_confusion_matrix(test_metrics),
        "",
        "",
        "## Deployment artifacts",
        f"- Saved model: `{MODEL_PATH}`",
        f"- Saved metadata: `{METADATA_PATH}`",
        "- The model output is **review priority**, not a legal or audit proof of fraud.",
        "- `target_fraud = 1` means an `AAER_ID` is present in the source dataset.",
        "- `target_fraud = 0` means no known `AAER_ID` label is present in this dataset.",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def train_and_report() -> None:
    model_df = load_dataset(DATA_PATH)

    print("Dataset summary")
    total_rows = len(model_df)
    positive_rate = float(model_df[TARGET_COLUMN].mean())
    print(f"Rows: {total_rows}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Fraud-labeled ratio: {positive_rate:.6f}")

    x_train, x_val, x_test, y_train, y_val, y_test = make_splits(model_df)
    print(f"Train/val/test: {len(x_train)}/{len(x_val)}/{len(x_test)}")

    # Baseline
    logistic_model = build_logistic_regression()
    logistic_model.fit(x_train, y_train)
    lr_threshold, lr_metrics = evaluate_model(logistic_model, x_val, y_val)
    lr_metrics.update({"model_name": "Logistic Regression"})
    lr_metrics["threshold"] = lr_threshold

    # Tuned Random Forest
    tuned_rf, rf_params, rf_metrics = tune_random_forest(
        x_train, y_train, x_val, y_val
    )
    rf_metrics["model_name"] = "Random Forest (tuned)"
    rf_metrics["params"] = rf_params

    validation_df = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "avg_precision": lr_metrics["average_precision"],
                "f1": lr_metrics["f1"],
            },
            {
                "model": "Random Forest (tuned)",
                "avg_precision": rf_metrics["average_precision"],
                "f1": rf_metrics["f1"],
            },
        ]
    )
    print("\nValidation metrics (higher AP is better):")
    print(validation_df.sort_values("avg_precision", ascending=False).to_string(index=False))

    if rf_metrics["average_precision"] > lr_metrics["average_precision"]:
        final_model = tuned_rf
        selected_name = "Random Forest (tuned)"
        selected_metrics = rf_metrics
    else:
        final_model = logistic_model
        selected_name = "Logistic Regression"
        selected_metrics = lr_metrics

    print(f"\nSelected model on validation: {selected_name}")

    # Save selected model with preprocessing included.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    # Evaluate selected model on test split with threshold chosen on validation.
    test_scores = final_model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_scores, selected_metrics["threshold"])

    # Save metadata for the app/validation context.
    metadata = {
        "model_name": selected_name,
        "model_path": str(MODEL_PATH),
        "feature_columns": FEATURE_COLUMNS,
        "target_definition": (
            "1 if AAER_ID exists, 0 if AAER_ID is missing in the source dataset"
        ),
        "dataset_rows": int(total_rows),
        "dataset_positive_ratio": positive_rate,
        "random_state": RANDOM_STATE,
        "feature_statistics": feature_stats(model_df[FEATURE_COLUMNS]),
        "validation_metrics": {
            "logistic_regression": {
                "average_precision": lr_metrics["average_precision"],
                "roc_auc": lr_metrics["roc_auc"],
                "recall": lr_metrics["recall"],
                "precision": lr_metrics["precision"],
                "f1": lr_metrics["f1"],
                "threshold": lr_metrics["threshold"],
                "confusion_matrix": lr_metrics["confusion_matrix"],
            },
            "random_forest_tuned": {
                "params": rf_metrics["params"],
                "average_precision": rf_metrics["average_precision"],
                "roc_auc": rf_metrics["roc_auc"],
                "recall": rf_metrics["recall"],
                "precision": rf_metrics["precision"],
                "f1": rf_metrics["f1"],
                "threshold": rf_metrics["threshold"],
                "confusion_matrix": rf_metrics["confusion_matrix"],
            },
        },
        "selected_model_threshold": float(selected_metrics["threshold"]),
        "test_metrics": {
            "average_precision": test_metrics["average_precision"],
            "roc_auc": test_metrics["roc_auc"],
            "recall": test_metrics["recall"],
            "precision": test_metrics["precision"],
            "f1": test_metrics["f1"],
            "threshold": test_metrics["threshold"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "review_priority_disclaimer": (
            "Fraud-risk scores indicate review priority only and are not legal or audit conclusions."
        ),
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Smoke-check that the joblib object can be used for scoring.
    loaded_model = joblib.load(MODEL_PATH)
    sample_row = pd.DataFrame([model_df[FEATURE_COLUMNS].iloc[0].to_dict()])
    sample_probability = float(loaded_model.predict_proba(sample_row)[0, 1])
    sample_prob_label = "high" if sample_probability >= 0.30 else "low/medium"

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Smoke check predict_proba sample: {sample_probability:.6f} ({sample_prob_label})")

    write_report(
        row_count=total_rows,
        positive_rate=positive_rate,
        logistic_metrics=lr_metrics,
        rf_metrics=rf_metrics,
        selected_name=selected_name,
        test_metrics=test_metrics,
    )

    print(f"Report written to: {RESULTS_PATH}")
    print(f"Selected final metrics (test): {test_metrics}")


if __name__ == "__main__":
    train_and_report()
