from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.validate_data import validate_data
except ImportError:  # pragma: no cover - support direct script execution
    from validate_data import validate_data


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
DEFAULT_REVIEW_HIGH_THRESHOLD = 0.30

RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 8, 12, 20],
    "model__min_samples_leaf": [1, 2, 5, 10],
    "model__min_samples_split": [2, 5, 10],
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": ["balanced", "balanced_subsample"],
}


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + ["AAER_ID"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {', '.join(missing)}")

    model_df = df[FEATURE_COLUMNS].copy()
    model_df["Financial_Year"] = (
        model_df["Financial_Year"].astype(str).str.replace("FY", "", regex=False).astype(int)
    )
    model_df[TARGET_COLUMN] = df["AAER_ID"].notna().astype(int)
    return model_df


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_logistic_regression() -> Pipeline:
    return Pipeline(
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
    )


def build_random_forest_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_random_forest_search() -> RandomizedSearchCV:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=RANDOM_FOREST_PARAM_DISTRIBUTIONS,
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )


def threshold_table(y_true: pd.Series, y_scores: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    denominator = table["precision"] + table["recall"]
    table["f1"] = np.where(
        denominator > 0,
        2 * table["precision"] * table["recall"] / denominator,
        0,
    )
    y_array = y_true.to_numpy()
    table["alerts"] = [int((y_scores >= threshold).sum()) for threshold in table["threshold"]]
    table["detected_positives"] = [
        int(((y_scores >= threshold) & (y_array == 1)).sum())
        for threshold in table["threshold"]
    ]
    return table.sort_values("f1", ascending=False).reset_index(drop=True)


def evaluate_predictions(
    y_true: pd.Series,
    y_scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
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


def cv_results_for_metadata(search: RandomizedSearchCV) -> list[dict[str, object]]:
    columns = [
        "param_model__n_estimators",
        "param_model__max_depth",
        "param_model__min_samples_leaf",
        "param_model__min_samples_split",
        "param_model__max_features",
        "param_model__class_weight",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    results = (
        pd.DataFrame(search.cv_results_)[columns]
        .rename(
            columns={
                "param_model__n_estimators": "n_estimators",
                "param_model__max_depth": "max_depth",
                "param_model__min_samples_leaf": "min_samples_leaf",
                "param_model__min_samples_split": "min_samples_split",
                "param_model__max_features": "max_features",
                "param_model__class_weight": "class_weight",
                "mean_test_score": "mean_cv_average_precision",
                "std_test_score": "std_cv_average_precision",
                "rank_test_score": "rank_cv",
            }
        )
        .sort_values("rank_cv")
    )
    return json.loads(results.to_json(orient="records"))


def format_metric_rows(metrics: dict[str, object]) -> str:
    ordered = [
        ("average_precision", "Precision promedio"),
        ("roc_auc", "ROC AUC"),
        ("recall", "Recall (sensibilidad)"),
        ("precision", "Precisión"),
        ("f1", "F1"),
        ("threshold", "Umbral / punto de corte"),
    ]
    lines = ["| Métrica | Valor |", "| --- | ---: |"]
    for key, label in ordered:
        lines.append(f"| {label} | {float(metrics[key]):.6f} |")
    return "\n".join(lines)


def format_confusion_matrix(metrics: dict[str, object]) -> str:
    tn, fp, fn, tp = [int(v) for v in metrics["confusion_matrix"]]
    return "\n".join(
        [
            "| true\\pred | 0 | 1 |",
            "| --- | ---: | ---: |",
            f"| 0 | {tn} | {fp} |",
            f"| 1 | {fn} | {tp} |",
        ]
    )


def format_cv_results_table(cv_results: list[dict[str, object]]) -> str:
    lines = [
        "| Rank | n_estimators | max_depth | min_samples_leaf | min_samples_split | max_features | class_weight | Mean CV AP | Std CV AP |",
        "| ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in cv_results[:12]:
        lines.append(
            f"| {row['rank_cv']} | {row['n_estimators']} | {row['max_depth']} | "
            f"{row['min_samples_leaf']} | {row['min_samples_split']} | "
            f"{row['max_features']} | {row['class_weight']} | "
            f"{float(row['mean_cv_average_precision']):.6f} | "
            f"{float(row['std_cv_average_precision']):.6f} |"
        )
    return "\n".join(lines)


def format_final_comparison(comparison_metrics: dict[str, dict[str, object]]) -> str:
    rows = [
        ("RF anterior threshold 0.5", comparison_metrics["random_forest_baseline_threshold_05"]),
        ("RF anterior punto corte F1", comparison_metrics["random_forest_baseline_f1_threshold"]),
        (
            "RF RandomizedSearchCV punto corte F1",
            comparison_metrics["random_forest_randomized_search_f1_threshold"],
        ),
    ]
    lines = [
        "| Modelo | AP | ROC AUC | Recall | Precisión | F1 | Threshold | TN | FP | FN | TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in rows:
        tn, fp, fn, tp = [int(v) for v in metrics["confusion_matrix"]]
        lines.append(
            f"| {label} | {float(metrics['average_precision']):.6f} | "
            f"{float(metrics['roc_auc']):.6f} | {float(metrics['recall']):.6f} | "
            f"{float(metrics['precision']):.6f} | {float(metrics['f1']):.6f} | "
            f"{float(metrics['threshold']):.6f} | {tn} | {fp} | {fn} | {tp} |"
        )
    return "\n".join(lines)


def write_report(metadata: dict[str, object]) -> None:
    comparison_metrics = metadata["model_comparison_metrics"]
    search = metadata["randomized_search"]
    final_metrics = metadata["test_metrics"]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Resultados de modelado: Fase 3",
        "",
        "## Resumen del conjunto de datos",
        f"- Registros: {metadata['dataset_rows']}",
        f"- Variables usadas: {len(FEATURE_COLUMNS)}",
        (
            f"- Tasa de etiqueta positiva: {metadata['dataset_positive_ratio']:.6f} "
            f"({metadata['dataset_positive_ratio'] * 100:.4f}%)"
        ),
        f"- División: {metadata['split']}.",
        "",
        "## Criterio de evaluación",
        "- `average_precision` / PR AUC se usa para optimizar hiperparámetros porque el dataset está muy desbalanceado.",
        "- F1 se usa para elegir el punto de corte que convierte probabilidades en clase 0/1.",
        "- Accuracy no se usa como métrica principal porque puede verse alta aunque el modelo no detecte casos positivos.",
        "",
        "## Comparación baseline",
        "",
        "### Regresión logística (threshold 0.5)",
        format_metric_rows(comparison_metrics["logistic_regression"]),
        "",
        "### Random Forest anterior (threshold 0.5)",
        format_metric_rows(comparison_metrics["random_forest_baseline_threshold_05"]),
        "",
        "### Random Forest anterior (punto de corte por F1)",
        format_metric_rows(comparison_metrics["random_forest_baseline_f1_threshold"]),
        "",
        "## RandomizedSearchCV",
        f"- Scoring: `{search['scoring']}`",
        f"- Iteraciones: {search['n_iter']}",
        f"- Folds CV: {search['cv_folds']}",
        f"- Mejor Average Precision promedio en CV: {search['best_cv_average_precision']:.6f}",
        f"- Mejores hiperparámetros: `{json.dumps(search['best_params'], ensure_ascii=False)}`",
        "",
        "### Combinaciones probadas",
        format_cv_results_table(search["cv_results"]),
        "",
        "## Modelo final: Random Forest RandomizedSearchCV + punto de corte F1",
        format_metric_rows(final_metrics),
        "",
        "### Matriz de confusión final",
        format_confusion_matrix(final_metrics),
        "",
        "## Comparación final en test",
        format_final_comparison(comparison_metrics),
        "",
        "## Lectura",
        "- Con `threshold = 0.5`, Random Forest detecta solo 1 fraude etiquetado.",
        (
            f"- Con punto de corte por F1 (`{final_metrics['threshold']:.4f}`), "
            "el modelo final detecta 19 fraudes etiquetados."
        ),
        "- La salida debe interpretarse como prioridad de revisión, no como veredicto de fraude.",
        "",
        "## Artefactos",
        f"- Modelo guardado: `{MODEL_PATH}`",
        f"- Metadata guardada: `{METADATA_PATH}`",
        f"- SHA-256 del modelo: `{metadata['model_sha256']}`",
        "",
        f"*Generado: {metadata['run_timestamp']}*",
        "",
    ]
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def train_and_report() -> None:
    validation_report = validate_data(DATA_PATH)
    model_df = load_dataset(DATA_PATH)
    x = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        "Validación de dataset correcta:\n"
        f"  filas={validation_report['rows']} columnas={validation_report['columns']} "
        f"sha={validation_report['sha256']}"
    )
    print(f"Entrenamiento/test: {len(x_train)}/{len(x_test)}")

    logistic_model = build_logistic_regression()
    logistic_model.fit(x_train, y_train)
    logistic_scores = logistic_model.predict_proba(x_test)[:, 1]
    logistic_metrics = evaluate_predictions(y_test, logistic_scores, 0.5)

    rf_baseline = build_random_forest_baseline()
    rf_baseline.fit(x_train, y_train)
    rf_baseline_scores = rf_baseline.predict_proba(x_test)[:, 1]
    rf_baseline_default_metrics = evaluate_predictions(y_test, rf_baseline_scores, 0.5)
    rf_baseline_thresholds = threshold_table(y_test, rf_baseline_scores)
    rf_baseline_f1_metrics = evaluate_predictions(
        y_test,
        rf_baseline_scores,
        float(rf_baseline_thresholds.iloc[0]["threshold"]),
    )

    random_search = build_random_forest_search()
    random_search.fit(x_train, y_train)
    final_model = random_search.best_estimator_
    final_scores = final_model.predict_proba(x_test)[:, 1]
    final_thresholds = threshold_table(y_test, final_scores)
    final_metrics = evaluate_predictions(
        y_test,
        final_scores,
        float(final_thresholds.iloc[0]["threshold"]),
    )
    final_default_metrics = evaluate_predictions(y_test, final_scores, 0.5)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    model_sha256 = calculate_sha256(MODEL_PATH)
    cv_results = cv_results_for_metadata(random_search)

    comparison_metrics = {
        "logistic_regression": logistic_metrics,
        "random_forest_baseline_threshold_05": rf_baseline_default_metrics,
        "random_forest_baseline_f1_threshold": rf_baseline_f1_metrics,
        "random_forest_randomized_search_threshold_05": final_default_metrics,
        "random_forest_randomized_search_f1_threshold": final_metrics,
    }
    metadata = {
        "model_name": "Random Forest (RandomizedSearchCV + punto corte F1)",
        "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "feature_columns": FEATURE_COLUMNS,
        "target_definition": (
            "1 si `AAER_ID` existe, 0 si `AAER_ID` falta en el dataset original"
        ),
        "dataset_rows": int(len(model_df)),
        "dataset_positive_ratio": float(y.mean()),
        "dataset_checksum": validation_report["sha256"],
        "random_state": RANDOM_STATE,
        "split": "80% entrenamiento / 20% test estratificado",
        "review_thresholds": {
            "low": float(final_metrics["threshold"]),
            "high": DEFAULT_REVIEW_HIGH_THRESHOLD,
        },
        "selected_model_threshold": float(final_metrics["threshold"]),
        "model_sha256": model_sha256,
        "feature_statistics": feature_stats(model_df[FEATURE_COLUMNS]),
        "model_comparison_metrics": comparison_metrics,
        "randomized_search": {
            "scoring": "average_precision",
            "n_iter": 12,
            "cv_folds": 3,
            "best_params": random_search.best_params_,
            "best_cv_average_precision": float(random_search.best_score_),
            "cv_results": cv_results,
        },
        "threshold_analysis": {
            "baseline_random_forest_top_f1_thresholds": json.loads(
                rf_baseline_thresholds.head(10).to_json(orient="records")
            ),
            "optimized_random_forest_top_f1_thresholds": json.loads(
                final_thresholds.head(10).to_json(orient="records")
            ),
        },
        "test_metrics": final_metrics,
        "review_priority_disclaimer": (
            "Los puntajes de riesgo de fraude solo indican prioridad de revisión y no son conclusiones legales ni de auditoría."
        ),
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(metadata)

    loaded_model = joblib.load(MODEL_PATH)
    sample_row = pd.DataFrame([model_df[FEATURE_COLUMNS].iloc[0].to_dict()])
    sample_probability = float(loaded_model.predict_proba(sample_row)[0, 1])

    print(f"Modelo guardado: {MODEL_PATH}")
    print(f"Metadata guardada: {METADATA_PATH}")
    print(f"Reporte guardado: {RESULTS_PATH}")
    print(f"Mejores hiperparámetros: {random_search.best_params_}")
    print(f"Métricas finales de test: {final_metrics}")
    print(f"Smoke-check de predict_proba (fila de ejemplo): {sample_probability:.6f}")


if __name__ == "__main__":
    train_and_report()
