from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Cleaned_data_1995_2018.csv"
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


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS + ["AAER_ID"])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df)
    model_df = df[FEATURE_COLUMNS].copy()
    model_df["Financial_Year"] = (
        model_df["Financial_Year"]
        .astype(str)
        .str.replace("FY", "", regex=False)
        .astype(int)
    )
    model_df[TARGET_COLUMN] = df["AAER_ID"].notna().astype(int).to_numpy()
    return model_df


def split_dataset(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models():
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def optimize_random_forest(X_train, y_train, X_val, y_val):
    param_grid = {
        "n_estimators": [200],
        "max_depth": [None, 12],
        "min_samples_split": [2],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt"],
        "class_weight": ["balanced_subsample"],
    }

    best_model = None
    best_params = None
    best_metrics = None

    for params in ParameterGrid(param_grid):
        candidate_model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        **params,
                    ),
                ),
            ]
        )
        candidate_metrics, trained_candidate = evaluate_model(
            "random_forest_tuned",
            candidate_model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        if (
            best_metrics is None
            or candidate_metrics["average_precision"] > best_metrics["average_precision"]
        ):
            best_model = trained_candidate
            best_params = params
            best_metrics = candidate_metrics

    return best_model, best_params, best_metrics


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_val)[:, 1]
    threshold = select_best_threshold(y_val, y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": threshold,
        "average_precision": average_precision_score(y_val, y_scores),
        "roc_auc": roc_auc_score(y_val, y_scores),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
    }
    return metrics, model


def print_results(metrics_df: pd.DataFrame):
    sorted_metrics = metrics_df.sort_values(
        ["average_precision", "f1", "recall"], ascending=False
    )
    print("\nValidation metrics")
    print(sorted_metrics.round(4).to_string(index=False))


def select_best_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    best_threshold = 0.5
    best_f1 = -1.0

    for idx, threshold in enumerate(thresholds):
        p = precision[idx]
        r = recall[idx]
        if p + r == 0:
            continue
        current_f1 = 2 * p * r / (p + r)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return float(best_threshold)


def evaluate_best_model(best_name, best_model, threshold, X_test, y_test):
    y_scores = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    print(f"\nBest model on validation: {best_name}")
    print(f"Selected threshold: {threshold:.4f}")
    print(f"Test average_precision: {average_precision_score(y_test, y_scores):.4f}")
    print(f"Test roc_auc: {roc_auc_score(y_test, y_scores):.4f}")
    print("\nTest classification report")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


def main():
    df = load_dataset(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    print("Dataset summary")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Positive class ratio: {df[TARGET_COLUMN].mean():.4%}")

    trained_models = {}
    metrics = []

    logistic_model = build_models()["logistic_regression"]
    logistic_metrics, trained_logistic = evaluate_model(
        "logistic_regression", logistic_model, X_train, y_train, X_val, y_val
    )
    metrics.append(logistic_metrics)
    trained_models["logistic_regression"] = trained_logistic

    tuned_rf, best_rf_params, best_rf_metrics = optimize_random_forest(
        X_train, y_train, X_val, y_val
    )
    print("\nRandom Forest tuning")
    print(f"Best validation average_precision: {best_rf_metrics['average_precision']:.4f}")
    print(f"Best params: {best_rf_params}")
    metrics.append(best_rf_metrics)
    trained_models["random_forest_tuned"] = tuned_rf

    metrics_df = pd.DataFrame(metrics)
    print_results(metrics_df)

    best_row = metrics_df.sort_values(
        ["average_precision", "f1", "recall"], ascending=False
    ).iloc[0]
    best_name = best_row["model"]
    evaluate_best_model(
        best_name,
        trained_models[best_name],
        best_row["threshold"],
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()
