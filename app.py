from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "Cleaned_data_1995_2018.csv"
MODEL_PATH = ROOT_DIR / "models" / "fraud_risk_model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
REPORTS_DIR = ROOT_DIR / "reports"
EDA_FIGURE_DIR = REPORTS_DIR / "eda" / "figures"
EDA_REPORT_PATH = REPORTS_DIR / "eda" / "focused_eda.md"
SQL_REPORT_PATH = REPORTS_DIR / "sql" / "phase1_sql_results.md"

LOW_THRESHOLD = 0.10
HIGH_THRESHOLD = 0.30
DEFAULT_LOW_THRESHOLD = LOW_THRESHOLD
DEFAULT_HIGH_THRESHOLD = HIGH_THRESHOLD
LOGGER = logging.getLogger(__name__)

FEATURE_DETAILS = {
    "Financial_Year": {
        "label": "Ejercicio fiscal",
        "description": "Año del registro financiero empresa-año.",
        "group": "Contexto",
    },
    "sale": {
        "label": "Ventas",
        "description": "Ingresos totales reportados por la empresa.",
        "group": "Tamaño y resultados",
    },
    "ni": {
        "label": "Utilidad neta",
        "description": "Resultado final después de gastos e impuestos.",
        "group": "Tamaño y resultados",
    },
    "at": {
        "label": "Activos totales",
        "description": "Valor total de lo que la empresa posee.",
        "group": "Tamaño y resultados",
    },
    "lt": {
        "label": "Pasivos totales",
        "description": "Deudas y obligaciones de la empresa.",
        "group": "Deuda y liquidez",
    },
    "che": {
        "label": "Efectivo e inversiones cortas",
        "description": "Recursos líquidos disponibles.",
        "group": "Deuda y liquidez",
    },
    "xint": {
        "label": "Gastos por intereses",
        "description": "Costos asociados a deuda financiera.",
        "group": "Deuda y liquidez",
    },
    "rect": {
        "label": "Cuentas por cobrar",
        "description": "Dinero pendiente de cobro a clientes.",
        "group": "Operación",
    },
    "invt": {
        "label": "Inventario",
        "description": "Bienes disponibles para venta o producción.",
        "group": "Operación",
    },
    "cogs": {
        "label": "Costo de ventas",
        "description": "Costo de bienes o servicios vendidos.",
        "group": "Operación",
    },
    "txt": {
        "label": "Impuestos",
        "description": "Impuestos totales pagados o a pagar.",
        "group": "Operación",
    },
    "prcc_f": {
        "label": "Precio de mercado",
        "description": "Valor de mercado incluido en el dataset.",
        "group": "Mercado",
    },
}

FEATURE_GROUP_ORDER = [
    "Contexto",
    "Tamaño y resultados",
    "Deuda y liquidez",
    "Operación",
    "Mercado",
]


def _as_percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _compact_number(value: float) -> str:
    number = float(value)
    abs_number = abs(number)
    if abs_number >= 1_000_000_000_000:
        return f"{number / 1_000_000_000_000:.2f}T"
    if abs_number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    if abs_number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs_number >= 1_000:
        return f"{number / 1_000:.2f}K"
    return f"{number:,.2f}"


def _html(value: object) -> str:
    return html.escape(str(value), quote=True)


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _feature_label(feature: str) -> str:
    return FEATURE_DETAILS.get(feature, {}).get("label", feature)


def _feature_description(feature: str) -> str:
    return FEATURE_DETAILS.get(feature, {}).get("description", "")


def load_model_and_metadata() -> tuple[object, dict]:
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    expected_hash = metadata.get("model_sha256")
    if expected_hash:
        _verify_model_checksum(expected_hash)

    model = joblib.load(MODEL_PATH)
    return model, metadata


@st.cache_data(show_spinner=False)
def _load_threshold_curve_data(
    data_path: str,
    model_path: str,
    model_sha256: str,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    _ = model_sha256  # invalida la cache si cambia el artefacto del modelo
    data_file = Path(data_path)
    model_file = Path(model_path)
    if not data_file.exists() or not model_file.exists() or not feature_columns:
        return pd.DataFrame()

    raw_df = pd.read_csv(data_file)
    required_columns = set(feature_columns) | {"AAER_ID"}
    if missing_columns := sorted(required_columns - set(raw_df.columns)):
        LOGGER.warning("No se puede construir curva de threshold. Faltan columnas: %s", missing_columns)
        return pd.DataFrame()

    x = raw_df[list(feature_columns)].copy()
    if "Financial_Year" in x.columns:
        x["Financial_Year"] = (
            x["Financial_Year"].astype(str).str.replace("FY", "", regex=False).astype(int)
        )
    y = raw_df["AAER_ID"].notna().astype(int)
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(model_file)
    scores = model.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    curve_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    denominator = curve_df["precision"] + curve_df["recall"]
    curve_df["f1"] = np.where(
        denominator > 0,
        2 * curve_df["precision"] * curve_df["recall"] / denominator,
        0,
    )
    return curve_df.sort_values("threshold").reset_index(drop=True)


def _verify_model_checksum(expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "El checksum del archivo del modelo no coincide. Entrena de nuevo y actualiza los artefactos."
        )


def _inject_layout_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #070a10;
            --bg-card: #10151f;
            --bg-card-2: #151b27;
            --bg-card-3: #1a2230;
            --line-soft: #2a3444;
            --line-strong: #3d4b61;
            --text-main: #f3f6fb;
            --text-soft: #aebad0;
            --text-muted: #7f8ca4;
            --accent: #66d9c6;
            --accent-2: #f0b35a;
            --accent-3: #8fb4ff;
            --danger: #f87171;
            --warning: #f0b35a;
            --success: #66d98b;
            --radius: 8px;
        }

        .stApp {
            background: var(--bg-main);
            color: var(--text-main);
            margin-top: 0 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        }

        header,
        .stApp > header,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stDeployButton {
            display: none !important;
        }

        [data-testid="stHeader"] {
            height: 0 !important;
            min-height: 0 !important;
        }

        [data-testid="stAppViewContainer"] {
            padding-top: 0 !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0rem !important;
        }

        .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1320px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding: 0.15rem 1.15rem 1.6rem !important;
        }

        .app-hero {
            border-radius: var(--radius);
            background:
                linear-gradient(135deg, rgba(102, 217, 198, 0.08) 0%, rgba(143, 180, 255, 0.06) 100%),
                #111722;
            padding: 0.68rem 0.78rem;
            color: #ffffff;
            margin: 0 0 0.45rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
            border: 1px solid rgba(143, 180, 255, 0.18);
            position: relative;
            overflow: hidden;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.5rem;
            align-items: center;
        }

        .app-hero h1 {
            margin: 0 0 0.28rem !important;
            padding: 0 !important;
            font-size: 1.14rem;
            color: #ffffff;
            line-height: 1.2;
            font-weight: 700;
        }

        .app-hero p {
            margin: 0 !important;
            padding: 0 !important;
            max-width: 1000px;
            color: #ecf3ff;
            line-height: 1.28;
            font-size: 0.78rem;
        }

        .hero-main {
            display: grid;
            grid-template-columns: 30px minmax(0, 1fr);
            align-items: start;
            gap: 0.48rem;
        }

        .hero-top-icon {
            width: 30px;
            height: 30px;
            border-radius: var(--radius);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 800;
            background: rgba(102, 217, 198, 0.13);
            border: 1px solid rgba(102, 217, 198, 0.35);
            color: white;
            flex: 0 0 auto;
        }

        .hero-copy {
            min-width: 0;
        }

        .hero-panel {
            display: none;
        }

        .hero-panel-title {
            color: #dce7fb;
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.38rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .hero-flow {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 0.35rem;
        }

        .hero-flow-step {
            border-radius: 7px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #eef3ff;
            font-size: 0.7rem;
            padding: 0.45rem 0.5rem;
            text-align: center;
        }

        .hero-meta {
            margin-top: 0.42rem;
            display: flex;
            gap: 0.32rem;
            flex-wrap: wrap;
        }

        .chip {
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            background: rgba(255, 255, 255, 0.075);
            color: #eff5ff;
            padding: 0.1rem 0.44rem;
            font-size: 0.68rem;
        }

        .section-header {
            color: var(--text-main);
            font-weight: 750;
            font-size: 1.22rem !important;
            margin: 0.95rem 0 0.48rem !important;
            padding: 0 !important;
            line-height: 1.18 !important;
            letter-spacing: 0 !important;
        }

        .panel-card {
            border-radius: var(--radius);
            border: 1px solid var(--line-soft);
            background: var(--bg-card);
            padding: 0.78rem 0.85rem;
            box-shadow: 0 8px 20px rgba(2, 8, 23, 0.36);
        }

        .panel-title {
            margin: 0 0 0.44rem;
            color: var(--text-main);
            font-weight: 700;
            font-size: 0.98rem;
            line-height: 1.2;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.72rem;
            margin-bottom: 0.58rem;
        }

        .kpi-card {
            border-radius: var(--radius);
            border: 1px solid var(--line-soft);
            background: linear-gradient(180deg, var(--bg-card-2), #101827);
            padding: 0.68rem 0.72rem;
            text-align: left;
            min-height: 82px;
            box-shadow: 0 8px 18px rgba(2, 8, 23, 0.28);
        }

        .kpi-label {
            color: var(--text-soft);
            font-size: 0.73rem;
            letter-spacing: 0.01em;
            margin-bottom: 0.12rem;
        }

        .kpi-value {
            color: var(--text-main);
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 0.18rem;
            word-break: break-word;
        }

        .kpi-sub {
            color: #94a3c4;
            font-size: 0.76rem;
            line-height: 1.2;
        }

        .overview-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 0.72rem;
            margin-top: 0.64rem;
        }

        .overview-card ul {
            margin: 0;
            padding-left: 1.05rem;
        }

        .overview-card li {
            color: #dbe5f8;
            margin: 0.28rem 0;
            line-height: 1.32;
            font-size: 0.9rem;
        }

        .status-card {
            display: grid;
            grid-template-columns: 1.25fr 1fr 1fr;
            gap: 0.65rem;
            align-items: stretch;
            margin-top: 0.72rem;
        }

        .status-item {
            border: 1px solid #2b3856;
            border-radius: var(--radius);
            background: #0e1626;
            padding: 0.62rem 0.7rem;
        }

        .status-label {
            color: var(--text-soft);
            font-size: 0.72rem;
            margin-bottom: 0.16rem;
        }

        .status-value {
            color: var(--text-main);
            font-size: 0.94rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .hint-box {
            border: 1px dashed #2e426f;
            border-radius: 12px;
            background: #111a2f;
            padding: 0.64rem 0.75rem;
            color: #c9d6ed;
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }

        .subtle-note {
            border: 1px solid #2e3f61;
            border-radius: var(--radius);
            background: #111a2f;
            color: #c9d7ef;
            font-size: 0.84rem;
            padding: 0.72rem 0.8rem;
            margin-top: 0.6rem;
        }

        .executive-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 0.58rem;
            margin-bottom: 0.85rem;
            align-items: stretch;
        }

        .lead-card {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: linear-gradient(180deg, #121824 0%, #0f141d 100%);
            padding: 0.82rem 0.88rem;
        }

        .lead-kicker {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }

        .lead-title {
            color: var(--text-main);
            font-size: 1.02rem;
            line-height: 1.18;
            font-weight: 800;
            margin-bottom: 0.32rem;
        }

        .lead-copy {
            color: #d7dfef;
            font-size: 0.82rem;
            line-height: 1.38;
            max-width: 860px;
        }

        .decision-list {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.58rem;
        }

        .decision-item {
            border: 1px solid var(--line-soft);
            border-left: 3px solid var(--accent);
            border-radius: var(--radius);
            background: #0f151f;
            padding: 0.56rem 0.62rem;
        }

        .decision-item:nth-child(2) {
            border-left-color: var(--accent-2);
        }

        .decision-item:nth-child(3) {
            border-left-color: var(--accent-3);
        }

        .decision-label {
            color: var(--text-soft);
            font-size: 0.72rem;
            margin-bottom: 0.15rem;
        }

        .decision-value {
            color: var(--text-main);
            font-size: 0.78rem;
            font-weight: 750;
            line-height: 1.25;
        }

        .story-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin-bottom: 0.9rem;
        }

        .story-step {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: #0f151f;
            padding: 0.72rem;
            min-height: 104px;
        }

        .story-step.featured {
            grid-column: 1 / -1;
            border-left: 3px solid var(--accent);
            background: linear-gradient(180deg, #111824, #0f151f);
            padding: 0.9rem 0.95rem;
            min-height: auto;
        }

        .story-number {
            color: var(--accent-2);
            font-size: 0.74rem;
            font-weight: 800;
            margin-bottom: 0.24rem;
        }

        .story-title {
            color: var(--text-main);
            font-weight: 780;
            font-size: 0.9rem;
            margin-bottom: 0.22rem;
            line-height: 1.2;
        }

        .story-step.featured .story-title {
            font-size: 1.04rem;
        }

        .story-copy {
            color: var(--text-soft);
            font-size: 0.78rem;
            line-height: 1.32;
        }

        .story-step.featured .story-copy {
            max-width: 1120px;
            font-size: 0.86rem;
            line-height: 1.42;
        }

        .chart-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .chart-note {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: #0f151f;
            padding: 0.72rem;
            margin-top: 0.45rem;
            color: #dbe5f5;
            font-size: 0.84rem;
            line-height: 1.36;
        }

        .eda-summary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.72rem;
            margin: 0.78rem 0 0.86rem;
        }

        .eda-summary-card {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: linear-gradient(180deg, #111824, #0f141d);
            padding: 0.78rem 0.82rem;
            min-height: 118px;
        }

        .eda-summary-label {
            color: var(--accent);
            font-size: 0.7rem;
            font-weight: 820;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 0.34rem;
        }

        .eda-summary-value {
            color: var(--text-main);
            font-size: 1.18rem;
            font-weight: 820;
            line-height: 1.12;
            margin-bottom: 0.28rem;
        }

        .eda-summary-copy {
            color: var(--text-soft);
            font-size: 0.78rem;
            line-height: 1.34;
        }

        .eda-spotlight {
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(300px, 0.92fr);
            gap: 0.78rem;
            align-items: start;
            margin-bottom: 0.9rem;
        }

        .eda-narrative-card,
        .eda-chart-card {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: #0f151f;
            padding: 0.82rem;
            box-shadow: 0 8px 18px rgba(2, 8, 23, 0.22);
        }

        .eda-narrative-card {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.8rem;
        }

        .eda-kicker {
            color: var(--accent-2);
            font-size: 0.7rem;
            font-weight: 820;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }

        .eda-title {
            color: var(--text-main);
            font-size: 1.05rem;
            font-weight: 820;
            line-height: 1.18;
            margin-bottom: 0.34rem;
        }

        .eda-copy {
            color: var(--text-soft);
            font-size: 0.82rem;
            line-height: 1.38;
            margin: 0;
        }

        .eda-step-list {
            display: grid;
            gap: 0.46rem;
        }

        .eda-step {
            display: grid;
            grid-template-columns: 28px minmax(0, 1fr);
            gap: 0.46rem;
            align-items: start;
            color: #dbe5f6;
            font-size: 0.78rem;
            line-height: 1.32;
        }

        .eda-step span {
            width: 24px;
            height: 24px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: rgba(102, 217, 198, 0.12);
            border: 1px solid rgba(102, 217, 198, 0.28);
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 820;
        }

        .eda-chart-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.78rem;
            margin-bottom: 0.9rem;
        }

        .eda-chart-card.wide {
            grid-column: 1 / -1;
        }

        .eda-chart-head {
            display: flex;
            justify-content: space-between;
            gap: 0.72rem;
            align-items: flex-start;
            margin-bottom: 0.62rem;
        }

        .eda-chart-title {
            color: var(--text-main);
            font-size: 0.98rem;
            font-weight: 820;
            line-height: 1.2;
            margin-bottom: 0.18rem;
        }

        .eda-chart-note {
            color: var(--text-soft);
            font-size: 0.78rem;
            line-height: 1.34;
        }

        .eda-chart-tag {
            flex: 0 0 auto;
            border: 1px solid rgba(143, 180, 255, 0.22);
            border-radius: 999px;
            color: #d8e4ff;
            background: rgba(143, 180, 255, 0.08);
            padding: 0.15rem 0.44rem;
            font-size: 0.66rem;
            font-weight: 760;
        }

        .eda-chart-image {
            display: block;
            width: 100%;
            max-height: 390px;
            object-fit: contain;
            background: #ffffff;
            border: 1px solid rgba(143, 180, 255, 0.16);
            border-radius: 7px;
        }

        .eda-chart-card.wide .eda-chart-image {
            max-height: 560px;
        }

        .eda-spotlight .eda-chart-image {
            max-height: 390px;
        }

        .eda-conclusion {
            border: 1px solid rgba(102, 217, 198, 0.26);
            border-radius: var(--radius);
            background: linear-gradient(135deg, rgba(102, 217, 198, 0.1), rgba(143, 180, 255, 0.07));
            padding: 0.8rem 0.9rem;
            color: #e8f7f4;
            margin: 0.2rem 0 0.82rem;
            font-size: 0.84rem;
            line-height: 1.38;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .insight-card {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: linear-gradient(180deg, #111824, #0f141d);
            padding: 0.88rem;
        }

        .insight-card.wide {
            grid-column: 1 / -1;
        }

        .insight-title {
            color: var(--text-main);
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .insight-copy {
            color: var(--text-soft);
            font-size: 0.82rem;
            line-height: 1.35;
            margin-bottom: 0.72rem;
        }

        .bar-row {
            margin: 0.56rem 0;
        }

        .bar-label {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            color: #dfe8f8;
            font-size: 0.78rem;
            margin-bottom: 0.24rem;
        }

        .bar-track {
            height: 12px;
            border-radius: 999px;
            background: #080d15;
            border: 1px solid #263244;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), #8fb4ff);
        }

        .bar-fill.warning {
            background: linear-gradient(90deg, var(--accent-2), #f87171);
        }

        .bar-fill.muted {
            background: linear-gradient(90deg, #53627a, #718098);
        }

        .bar-fill.winner {
            background: linear-gradient(90deg, var(--accent), #8fb4ff);
        }

        .insight-foot {
            color: #d6dfef;
            font-size: 0.78rem;
            margin-top: 0.58rem;
            border-top: 1px solid #263244;
            padding-top: 0.55rem;
        }

        .cta-card {
            border: 1px solid rgba(102, 217, 198, 0.28);
            border-radius: var(--radius);
            background: linear-gradient(135deg, rgba(102, 217, 198, 0.1), rgba(143, 180, 255, 0.07));
            padding: 0.85rem 0.95rem;
            color: #e8f7f4;
            margin-top: 0.72rem;
        }

        .cta-title {
            font-weight: 800;
            margin-bottom: 0.2rem;
            color: #ffffff;
        }

        .feature-code {
            color: var(--text-muted);
            font-size: 0.72rem;
            margin-top: 0.04rem;
        }

        .feature-description {
            color: #95a3ba;
            font-size: 0.72rem;
            line-height: 1.25;
            margin: 0.08rem 0 0.28rem;
        }

        .feature-glossary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.58rem;
        }

        .feature-glossary-item {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: #0f151f;
            padding: 0.62rem 0.68rem;
        }

        .model-explain-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.6rem 0 0.75rem;
        }

        .metric-explain {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: #0f151f;
            padding: 0.68rem;
        }

        .metric-explain strong {
            color: var(--text-main);
        }

        .metric-explain p {
            color: var(--text-soft);
            margin: 0.2rem 0 0;
            font-size: 0.8rem;
            line-height: 1.32;
        }

        .feature-head {
            margin-bottom: 0.12rem;
            color: var(--text-main);
            font-weight: 600;
            font-size: 0.78rem;
            letter-spacing: 0.005em;
        }

        .feature-field {
            display: flex;
            flex-direction: column;
            gap: 0.12rem;
            min-height: 68px;
            margin-bottom: 0.2rem;
        }

        .feature-field .feature-head,
        .feature-field .feature-code,
        .feature-field .feature-description {
            margin: 0;
            overflow-wrap: anywhere;
        }

        .feature-field .feature-description {
            display: block;
            line-height: 1.28;
            margin: 0.08rem 0 0 !important;
            padding-bottom: 0.35rem;
        }

        .form-heading {
            color: var(--text-main);
            font-size: 1.02rem;
            font-weight: 780;
            line-height: 1.2;
            margin: 0 0 0.32rem;
        }

        .form-subheading {
            color: var(--text-soft);
            font-size: 0.76rem;
            line-height: 1.35;
            margin: -0.08rem 0 0.58rem;
        }

        .form-section-title {
            color: var(--accent);
            font-size: 0.76rem !important;
            font-weight: 780;
            line-height: 1.1;
            margin: 0.72rem 0 0.42rem !important;
            padding-top: 0.34rem;
            border-top: 1px solid rgba(143, 180, 255, 0.12);
            letter-spacing: 0.035em;
            text-transform: uppercase;
        }

        .form-section-title.first {
            margin-top: 0.45rem !important;
            padding-top: 0;
            border-top: 0;
        }

        .feature-caption {
            color: #93a1bc;
            font-size: 0.7rem;
            margin-top: 0.2rem;
            margin-bottom: 0.55rem;
        }

        .input-caption {
            color: #8291ae;
            font-size: 0.65rem;
            margin: 0.14rem 0 0 !important;
            line-height: 1.25;
            display: inline-flex;
            width: fit-content;
            max-width: 100%;
            overflow-wrap: anywhere;
            border: 1px solid rgba(143, 180, 255, 0.13);
            border-radius: 999px;
            background: rgba(7, 10, 16, 0.45);
            padding: 0.16rem 0.42rem;
        }

        .section-subtitle {
            margin: 0 0 0.35rem;
            color: var(--text-soft);
            font-size: 0.88rem;
        }

        .field-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.2rem 0.5rem;
        }

        .demo-hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(290px, 360px);
            gap: 0.78rem;
            align-items: start;
            margin: 0 0 0.85rem;
        }

        .demo-intro-card,
        .demo-readout-card,
        .demo-side-card,
        .demo-placeholder-card {
            border: 1px solid rgba(143, 180, 255, 0.17);
            border-radius: var(--radius);
            background: linear-gradient(180deg, rgba(18, 25, 38, 0.98), rgba(12, 18, 29, 0.98));
            box-shadow: 0 12px 24px rgba(2, 8, 23, 0.18);
        }

        .demo-intro-card {
            padding: 1rem 1.05rem;
        }

        .demo-readout-card,
        .demo-side-card,
        .demo-placeholder-card {
            padding: 0.82rem;
        }

        .demo-kicker {
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.045em;
            text-transform: uppercase;
            margin: 0 0 0.45rem;
        }

        .demo-title {
            color: var(--text-main);
            font-size: 1.52rem;
            font-weight: 820;
            line-height: 1.08;
            margin: 0 0 0.52rem;
            letter-spacing: 0;
        }

        .demo-copy {
            color: #c9d4e8;
            font-size: 0.86rem;
            line-height: 1.45;
            max-width: 800px;
            margin: 0 0 0.75rem;
        }

        .demo-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
        }

        .demo-pill {
            border: 1px solid rgba(255, 255, 255, 0.13);
            background: rgba(255, 255, 255, 0.055);
            border-radius: 999px;
            color: #e9effb;
            font-size: 0.7rem;
            font-weight: 650;
            padding: 0.22rem 0.52rem;
            white-space: nowrap;
        }

        .demo-flow-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            margin-top: 0.9rem;
        }

        .demo-flow-chip {
            border: 1px solid rgba(102, 217, 198, 0.16);
            border-radius: var(--radius);
            background: rgba(102, 217, 198, 0.055);
            color: #dce9f7;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.42rem 0.48rem;
            text-align: center;
        }

        .demo-command-center {
            display: grid;
            grid-template-columns: minmax(0, 0.95fr) minmax(340px, 1.05fr);
            gap: 0.85rem;
            align-items: stretch;
            margin: 0.15rem 0 0.85rem;
        }

        .demo-signal-panel,
        .demo-brief-panel,
        .demo-editor-panel,
        .demo-control-panel {
            border: 1px solid rgba(143, 180, 255, 0.18);
            border-radius: 10px;
            background: linear-gradient(180deg, rgba(17, 24, 38, 0.98), rgba(10, 16, 27, 0.98));
            box-shadow: 0 18px 36px rgba(2, 8, 23, 0.22);
        }

        .demo-signal-panel {
            padding: 1rem;
            display: grid;
            grid-template-columns: 160px minmax(0, 1fr);
            gap: 1rem;
            align-items: center;
            min-height: 215px;
            position: relative;
            overflow: hidden;
        }

        .demo-signal-panel.low {
            border-color: rgba(102, 217, 139, 0.45);
            background: linear-gradient(145deg, rgba(12, 47, 36, 0.96), rgba(10, 19, 30, 0.98));
        }

        .demo-signal-panel.medium {
            border-color: rgba(240, 179, 90, 0.5);
            background: linear-gradient(145deg, rgba(55, 38, 12, 0.95), rgba(10, 19, 30, 0.98));
        }

        .demo-signal-panel.high {
            border-color: rgba(248, 113, 113, 0.55);
            background: linear-gradient(145deg, rgba(61, 20, 24, 0.95), rgba(10, 19, 30, 0.98));
        }

        .demo-score-ring {
            --score: 0%;
            --score-color: var(--accent-3);
            width: 142px;
            height: 142px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background:
                radial-gradient(circle at center, #0d1422 0 57%, transparent 58%),
                conic-gradient(var(--score-color) var(--score), rgba(143, 180, 255, 0.15) 0);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: inset 0 0 28px rgba(0, 0, 0, 0.28);
        }

        .demo-score-ring.low {
            --score-color: var(--success);
        }

        .demo-score-ring.medium {
            --score-color: var(--warning);
        }

        .demo-score-ring.high {
            --score-color: var(--danger);
        }

        .demo-ring-inner {
            text-align: center;
            width: 96px;
        }

        .demo-ring-value {
            color: var(--text-main);
            font-size: 1.25rem;
            font-weight: 850;
            line-height: 1;
        }

        .demo-ring-label {
            color: var(--text-muted);
            font-size: 0.64rem;
            font-weight: 750;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-top: 0.28rem;
        }

        .demo-signal-kicker,
        .demo-panel-kicker {
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 0 0 0.38rem;
        }

        .demo-signal-title {
            color: var(--text-main);
            font-size: 1.42rem;
            font-weight: 860;
            line-height: 1.08;
            margin: 0 0 0.46rem;
        }

        .demo-signal-copy {
            color: #cbd6ea;
            font-size: 0.82rem;
            line-height: 1.42;
            margin: 0;
        }

        .demo-thresholds {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem;
            margin-top: 0.75rem;
        }

        .demo-threshold {
            border: 1px solid rgba(143, 180, 255, 0.13);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.045);
            padding: 0.46rem 0.52rem;
        }

        .demo-threshold span {
            display: block;
            color: var(--text-muted);
            font-size: 0.64rem;
            margin-bottom: 0.1rem;
        }

        .demo-threshold strong {
            color: var(--text-main);
            font-size: 0.86rem;
        }

        .demo-brief-panel {
            padding: 0.92rem;
        }

        .demo-brief-title {
            color: var(--text-main);
            font-size: 1.05rem;
            font-weight: 820;
            line-height: 1.16;
            margin: 0 0 0.35rem;
        }

        .demo-brief-copy {
            color: var(--text-soft);
            font-size: 0.8rem;
            line-height: 1.42;
            margin: 0 0 0.7rem;
        }

        .demo-process {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.48rem;
        }

        .demo-process-step {
            border: 1px solid rgba(143, 180, 255, 0.13);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
            padding: 0.6rem;
            min-height: 95px;
        }

        .demo-process-number {
            width: 1.45rem;
            height: 1.45rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: rgba(102, 217, 198, 0.13);
            border: 1px solid rgba(102, 217, 198, 0.32);
            color: #cbfff4;
            font-size: 0.7rem;
            font-weight: 850;
            margin-bottom: 0.4rem;
        }

        .demo-process-title {
            color: var(--text-main);
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.18rem;
            line-height: 1.2;
        }

        .demo-process-copy {
            color: var(--text-soft);
            font-size: 0.68rem;
            line-height: 1.32;
        }

        .demo-lab-layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(285px, 0.32fr);
            gap: 0.85rem;
            align-items: start;
        }

        .demo-editor-panel {
            padding: 0.9rem;
        }

        .demo-editor-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 0.7rem;
            align-items: start;
            margin-bottom: 0.65rem;
        }

        .demo-editor-title {
            color: var(--text-main);
            font-size: 1.02rem;
            font-weight: 840;
            line-height: 1.18;
            margin: 0 0 0.18rem;
        }

        .demo-editor-copy {
            color: var(--text-soft);
            font-size: 0.74rem;
            line-height: 1.38;
            margin: 0;
        }

        .demo-editor-badge {
            border: 1px solid rgba(102, 217, 198, 0.24);
            border-radius: 999px;
            background: rgba(102, 217, 198, 0.08);
            color: #cdfef5;
            font-size: 0.68rem;
            font-weight: 800;
            padding: 0.22rem 0.55rem;
            white-space: nowrap;
        }

        .demo-control-panel {
            padding: 0.78rem;
            position: sticky;
            top: 0.75rem;
        }

        .demo-control-title {
            color: var(--text-main);
            font-size: 0.92rem;
            font-weight: 820;
            line-height: 1.2;
            margin: 0 0 0.35rem;
        }

        .demo-control-copy {
            color: var(--text-soft);
            font-size: 0.72rem;
            line-height: 1.38;
            margin: 0 0 0.7rem;
        }

        .demo-control-list {
            display: grid;
            gap: 0.46rem;
        }

        .demo-control-item {
            border: 1px solid rgba(143, 180, 255, 0.13);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
            padding: 0.5rem 0.55rem;
        }

        .demo-control-label {
            color: var(--text-muted);
            font-size: 0.64rem;
            margin-bottom: 0.08rem;
        }

        .demo-control-value {
            color: var(--text-main);
            font-size: 0.76rem;
            font-weight: 790;
            line-height: 1.26;
        }

        .demo-submit-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(260px, 0.36fr);
            gap: 0.75rem;
            align-items: center;
            margin-top: 0.7rem;
            padding-top: 0.7rem;
            border-top: 1px solid rgba(143, 180, 255, 0.14);
        }

        .demo-submit-note {
            color: var(--text-muted);
            font-size: 0.72rem;
            line-height: 1.38;
        }

        .case-submit-panel {
            border: 1px solid rgba(143, 180, 255, 0.16);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.035);
            padding: 0.68rem 0.72rem;
            margin: 0.08rem 0 0.56rem;
        }

        .case-submit-note {
            color: #aebad0;
            font-size: 0.74rem;
            line-height: 1.38;
            margin: 0;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid rgba(143, 180, 255, 0.16) !important;
            background: #0c1320 !important;
        }

        .demo-stage-header {
            border: 1px solid rgba(143, 180, 255, 0.18);
            border-radius: 10px;
            background:
                linear-gradient(135deg, rgba(102, 217, 198, 0.08), rgba(143, 180, 255, 0.05)),
                #0f1726;
            padding: 0.9rem 1rem;
            margin: 0.08rem 0 0.8rem;
        }

        .demo-stage-kicker {
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 0 0 0.28rem;
        }

        .demo-stage-title {
            color: var(--text-main);
            font-size: 1.2rem;
            line-height: 1.12;
            font-weight: 850;
            margin: 0 0 0.28rem;
        }

        .demo-stage-copy {
            color: #c7d2e8;
            font-size: 0.82rem;
            line-height: 1.38;
            max-width: 980px;
            margin: 0;
        }

        .case-panel-head {
            border: 1px solid rgba(143, 180, 255, 0.18);
            border-radius: 10px;
            background: linear-gradient(180deg, rgba(18, 25, 38, 0.98), rgba(11, 17, 28, 0.98));
            padding: 0.9rem;
            margin-bottom: 0.72rem;
            box-shadow: 0 16px 30px rgba(2, 8, 23, 0.18);
        }

        .case-panel-title-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: flex-start;
            margin-bottom: 0.48rem;
        }

        .case-panel-title {
            color: var(--text-main);
            font-size: 1.05rem;
            line-height: 1.16;
            font-weight: 840;
            margin: 0;
        }

        .case-panel-copy {
            color: var(--text-soft);
            font-size: 0.76rem;
            line-height: 1.38;
            max-width: 760px;
            margin: 0;
        }

        .case-badge {
            border: 1px solid rgba(102, 217, 198, 0.28);
            border-radius: 999px;
            background: rgba(102, 217, 198, 0.08);
            color: #cafff3;
            font-size: 0.68rem;
            font-weight: 820;
            padding: 0.22rem 0.58rem;
            white-space: nowrap;
        }

        .case-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.5rem;
            margin-top: 0.75rem;
        }

        .case-strip-item {
            border: 1px solid rgba(143, 180, 255, 0.13);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
            padding: 0.5rem 0.55rem;
        }

        .case-strip-label {
            color: var(--text-muted);
            font-size: 0.64rem;
            margin-bottom: 0.08rem;
        }

        .case-strip-value {
            color: var(--text-main);
            font-size: 0.76rem;
            font-weight: 780;
            line-height: 1.25;
        }

        .case-section-title {
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 860;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 0.7rem 0 0.36rem;
            padding-top: 0.48rem;
            border-top: 1px solid rgba(143, 180, 255, 0.14);
        }

        .case-section-title.first {
            margin-top: 0.15rem;
            padding-top: 0;
            border-top: 0;
        }

        .case-field-card {
            min-height: 84px;
        }

        .case-field-top {
            display: flex;
            justify-content: space-between;
            gap: 0.45rem;
            align-items: flex-start;
            margin-bottom: 0.32rem;
        }

        .case-field-label {
            color: var(--text-main);
            font-size: 0.82rem;
            font-weight: 790;
            line-height: 1.2;
        }

        .case-field-code {
            color: var(--text-muted);
            border: 1px solid rgba(143, 180, 255, 0.12);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.04);
            font-size: 0.62rem;
            padding: 0.08rem 0.36rem;
            white-space: nowrap;
        }

        .case-field-desc {
            color: #98a7bf;
            font-size: 0.7rem;
            line-height: 1.3;
            min-height: 1.82rem;
            margin-bottom: 0.38rem;
        }

        .case-field-range {
            color: var(--text-muted);
            font-size: 0.64rem;
            line-height: 1.2;
            margin-top: 0.18rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.case-field-card) {
            background: linear-gradient(180deg, rgba(16, 23, 36, 0.98), rgba(10, 16, 27, 0.98)) !important;
            border: 1px solid rgba(143, 180, 255, 0.18) !important;
            border-radius: 10px !important;
            box-shadow: 0 10px 20px rgba(2, 8, 23, 0.18) !important;
            margin-bottom: 0.58rem !important;
            overflow: hidden !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.case-field-card) > div {
            padding: 0.68rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.case-field-card) div[data-testid="stNumberInput"] {
            margin: 0 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.case-field-card) div[data-testid="stNumberInput"] button {
            display: none !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.case-field-card) div[data-baseweb="input"] {
            min-height: 38px !important;
            border-radius: 8px !important;
            background: #0d1524 !important;
            border: 1px solid rgba(143, 180, 255, 0.16) !important;
        }

        .result-panel {
            border: 1px solid rgba(143, 180, 255, 0.2);
            border-radius: 10px;
            background: linear-gradient(180deg, rgba(17, 24, 38, 0.98), rgba(8, 14, 24, 0.98));
            padding: 0.92rem;
            box-shadow: 0 18px 34px rgba(2, 8, 23, 0.25);
        }

        .result-panel.low {
            border-color: rgba(102, 217, 139, 0.48);
            background: linear-gradient(160deg, rgba(11, 52, 39, 0.98), rgba(8, 14, 24, 0.98));
        }

        .result-panel.medium {
            border-color: rgba(240, 179, 90, 0.55);
            background: linear-gradient(160deg, rgba(61, 43, 13, 0.98), rgba(8, 14, 24, 0.98));
        }

        .result-panel.high {
            border-color: rgba(248, 113, 113, 0.6);
            background: linear-gradient(160deg, rgba(63, 20, 25, 0.98), rgba(8, 14, 24, 0.98));
        }

        .result-kicker {
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.52rem;
        }

        .result-meter {
            --score: 0%;
            --meter-color: var(--accent-3);
            width: min(180px, 100%);
            aspect-ratio: 1;
            border-radius: 50%;
            display: grid;
            place-items: center;
            margin: 0 auto 0.78rem;
            background:
                radial-gradient(circle at center, #0b1220 0 58%, transparent 59%),
                conic-gradient(var(--meter-color) var(--score), rgba(143, 180, 255, 0.15) 0);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: inset 0 0 28px rgba(0, 0, 0, 0.32);
        }

        .result-meter.low {
            --meter-color: var(--success);
        }

        .result-meter.medium {
            --meter-color: var(--warning);
        }

        .result-meter.high {
            --meter-color: var(--danger);
        }

        .result-score {
            color: var(--text-main);
            font-size: 1.5rem;
            font-weight: 880;
            line-height: 1;
            text-align: center;
        }

        .result-score-label {
            color: var(--text-muted);
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-align: center;
            text-transform: uppercase;
            margin-top: 0.28rem;
        }

        .result-title {
            color: var(--text-main);
            font-size: 1.25rem;
            font-weight: 860;
            line-height: 1.12;
            margin: 0 0 0.36rem;
            text-align: center;
        }

        .result-copy {
            color: #c8d3e8;
            font-size: 0.78rem;
            line-height: 1.4;
            text-align: center;
            margin: 0 0 0.76rem;
        }

        .result-rule-grid {
            display: grid;
            gap: 0.48rem;
        }

        .result-rule {
            border: 1px solid rgba(143, 180, 255, 0.13);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
            padding: 0.52rem 0.58rem;
        }

        .result-rule span {
            display: block;
            color: var(--text-muted);
            font-size: 0.64rem;
            margin-bottom: 0.08rem;
        }

        .result-rule strong {
            color: var(--text-main);
            font-size: 0.82rem;
        }

        .result-note {
            border: 1px solid rgba(102, 217, 198, 0.16);
            border-radius: 8px;
            background: rgba(102, 217, 198, 0.055);
            color: var(--text-soft);
            font-size: 0.72rem;
            line-height: 1.38;
            margin-top: 0.62rem;
            padding: 0.56rem 0.62rem;
        }

        .result-summary {
            display: grid;
            gap: 0.72rem;
        }

        .result-headline {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 0.8rem;
            align-items: start;
            border-bottom: 1px solid rgba(143, 180, 255, 0.14);
            padding-bottom: 0.72rem;
        }

        .result-status-label {
            color: var(--text-muted);
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }

        .result-status-title {
            color: var(--text-main);
            font-size: 1.22rem;
            font-weight: 850;
            line-height: 1.12;
            margin: 0;
        }

        .result-score-pill {
            border: 1px solid rgba(143, 180, 255, 0.18);
            border-radius: 10px;
            background: rgba(5, 10, 18, 0.45);
            min-width: 108px;
            padding: 0.56rem 0.64rem;
            text-align: right;
        }

        .result-score-pill strong {
            color: var(--text-main);
            display: block;
            font-size: 1.18rem;
            line-height: 1;
        }

        .result-score-pill span {
            color: var(--text-muted);
            display: block;
            font-size: 0.62rem;
            font-weight: 780;
            letter-spacing: 0.04em;
            margin-top: 0.18rem;
            text-transform: uppercase;
        }

        .result-scale-card {
            border: 1px solid rgba(143, 180, 255, 0.14);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.04);
            padding: 0.68rem;
        }

        .result-scale-title {
            color: var(--text-main);
            font-size: 0.8rem;
            font-weight: 820;
            margin-bottom: 0.48rem;
        }

        .result-scale {
            position: relative;
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(102, 217, 139, 0.32), rgba(240, 179, 90, 0.34), rgba(248, 113, 113, 0.38));
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .result-scale-fill {
            display: block;
            height: 100%;
            min-width: 4px;
            border-radius: 999px;
            background: #f3f6fb;
            box-shadow: 0 0 12px rgba(243, 246, 251, 0.38);
        }

        .result-scale-markers {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.2rem;
            margin-top: 0.4rem;
            color: var(--text-muted);
            font-size: 0.62rem;
        }

        .result-scale-markers span:nth-child(2) {
            text-align: center;
        }

        .result-scale-markers span:nth-child(3) {
            text-align: right;
        }

        .result-interpretation {
            color: #cdd8ec;
            font-size: 0.78rem;
            line-height: 1.42;
            margin: 0;
        }

        .result-meta-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.46rem;
        }

        div[data-testid="stColumn"]:has(.result-panel) {
            position: sticky;
            top: 0.75rem;
            align-self: flex-start;
        }

        .demo-readout-title,
        .demo-side-title {
            color: #e8eefb;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.035em;
            text-transform: uppercase;
            margin: 0 0 0.62rem;
        }

        .demo-readout-list,
        .demo-step-list,
        .demo-spec-list {
            display: grid;
            gap: 0.48rem;
        }

        .demo-readout-item,
        .demo-step,
        .demo-spec-row {
            border: 1px solid rgba(143, 180, 255, 0.12);
            border-radius: var(--radius);
            background: rgba(255, 255, 255, 0.035);
        }

        .demo-readout-item {
            padding: 0.5rem 0.58rem;
        }

        .demo-readout-label,
        .demo-spec-label {
            color: var(--text-muted);
            font-size: 0.68rem;
            margin-bottom: 0.12rem;
        }

        .demo-readout-value,
        .demo-spec-value {
            color: var(--text-main);
            font-size: 0.86rem;
            font-weight: 760;
            line-height: 1.24;
        }

        .demo-workspace {
            margin-top: 0.15rem;
        }

        .demo-side-stack {
            display: grid;
            gap: 0.72rem;
        }

        div[data-testid="column"]:has(.demo-side-card),
        div[data-testid="stColumn"]:has(.demo-side-card) {
            position: sticky;
            top: 0.75rem;
            align-self: flex-start;
        }

        .demo-step {
            display: grid;
            grid-template-columns: 1.55rem minmax(0, 1fr);
            gap: 0.55rem;
            align-items: start;
            padding: 0.58rem;
        }

        .demo-step-number {
            width: 1.55rem;
            height: 1.55rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(102, 217, 198, 0.14);
            border: 1px solid rgba(102, 217, 198, 0.3);
            color: #c7fff3;
            font-size: 0.72rem;
            font-weight: 800;
        }

        .demo-step-title {
            color: var(--text-main);
            font-size: 0.8rem;
            font-weight: 760;
            line-height: 1.25;
            margin: 0;
        }

        .demo-step-copy {
            color: var(--text-soft);
            font-size: 0.7rem;
            line-height: 1.35;
            margin: 0.1rem 0 0;
        }

        .demo-spec-row {
            display: grid;
            grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
            gap: 0.45rem;
            padding: 0.48rem 0.55rem;
            align-items: center;
        }

        .demo-spec-label,
        .demo-spec-value {
            margin: 0;
        }

        .demo-placeholder-card {
            border-style: dashed;
            border-color: rgba(143, 180, 255, 0.24);
            background: rgba(14, 21, 34, 0.9);
        }

        .demo-side-card,
        .demo-placeholder-card {
            margin-bottom: 0.68rem;
        }

        .demo-placeholder-title {
            color: var(--text-main);
            font-weight: 780;
            font-size: 0.9rem;
            margin: 0 0 0.18rem;
        }

        .demo-placeholder-copy,
        .demo-note {
            color: var(--text-soft);
            font-size: 0.72rem;
            line-height: 1.42;
            margin: 0;
        }

        .demo-note {
            border: 1px solid rgba(102, 217, 198, 0.16);
            border-radius: var(--radius);
            background: rgba(102, 217, 198, 0.055);
            padding: 0.55rem 0.62rem;
            margin-top: 0.5rem;
        }

        .demo-score-track {
            height: 7px;
            border-radius: 999px;
            border: 1px solid rgba(143, 180, 255, 0.16);
            background: rgba(8, 13, 22, 0.9);
            overflow: hidden;
            margin: 0.48rem 0 0.58rem;
        }

        .demo-score-fill {
            display: block;
            height: 100%;
            min-width: 4px;
            border-radius: 999px;
            background: var(--success);
        }

        .demo-score-fill.medium {
            background: var(--warning);
        }

        .demo-score-fill.high {
            background: var(--danger);
        }

        .demo-action-help {
            color: var(--text-muted);
            font-size: 0.7rem;
            line-height: 1.35;
            padding-top: 0.28rem;
        }

        .risk-card {
            border-radius: var(--radius);
            border: 1px solid #2f4268;
            padding: 0.82rem;
            margin-top: 0;
        }

        .risk-card.low {
            background: linear-gradient(130deg, #0f2a22, #142f27);
            border-color: #2bcf79;
        }

        .risk-card.medium {
            background: linear-gradient(130deg, #2f2712, #3e3115);
            border-color: #eab308;
        }

        .risk-card.high {
            background: linear-gradient(130deg, #34171b, #411f24);
            border-color: #f87171;
        }

        .risk-title {
            text-transform: uppercase;
            letter-spacing: 0.03em;
            font-size: 0.75rem;
            color: var(--text-soft);
            margin-bottom: 0.15rem;
        }

        .risk-main {
            font-size: 1.8rem;
            color: var(--text-main);
            font-weight: 700;
            margin-bottom: 0.3rem;
            line-height: 1.1;
        }

        .risk-sub {
            color: var(--text-soft);
            font-size: 0.84rem;
        }

        .mini-title {
            font-weight: 700;
            color: var(--text-main);
            margin: 0 0 0.3rem;
            font-size: 0.98rem;
        }

        .viz-image {
            border-radius: 10px;
            border: 1px solid var(--line-soft);
        }

        .sql-report-section {
            border: 1px solid var(--line-soft);
            border-radius: var(--radius);
            background: #0f151f;
            padding: 0.85rem;
            margin: 0.75rem 0;
        }

        .sql-report-title {
            color: var(--text-main);
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }

        .sql-report-label {
            color: var(--text-soft);
            font-weight: 760;
            font-size: 0.82rem;
            margin: 0.35rem 0 0.28rem;
        }

        .sql-code {
            margin: 0 0 0.7rem;
            padding: 0.7rem 0.78rem;
            overflow-x: auto;
            border-radius: 7px;
            border: 1px solid #283447;
            background: #080d14;
            color: #d9e7ff;
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .sql-table-wrap {
            display: flex;
            justify-content: center;
            overflow-x: auto;
            width: 100%;
            padding-bottom: 0.15rem;
        }

        .sql-result-table {
            border-collapse: collapse;
            width: auto;
            min-width: min(100%, 520px);
            margin: 0 auto;
            color: var(--text-main);
            font-size: 0.82rem;
        }

        .sql-result-table th,
        .sql-result-table td {
            border: 1px solid var(--line-soft);
            padding: 0.45rem 0.62rem;
            text-align: center;
            white-space: nowrap;
        }

        .sql-result-table th {
            background: #171d28;
            color: #eef3ff;
            font-weight: 800;
        }

        .sql-result-table td {
            background: #0c1119;
        }

        .stTabs [data-baseweb="tab-list"] {
            display: inline-flex;
            width: fit-content;
            max-width: 100%;
            gap: 0.18rem;
            margin: 0.2rem 0 0.82rem;
            padding: 0.22rem;
            border: 1px solid #222b38;
            border-radius: 999px;
            background: #0d121a;
            overflow-x: auto;
        }

        .stTabs [role="tabpanel"] {
            padding-top: 0 !important;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 0;
            border-radius: 999px;
            min-height: 30px;
            padding: 0 0.72rem;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--blue-strong);
        }

        .stButton > button {
            border-radius: 10px;
            height: 44px;
            font-weight: 700;
            border: none;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            box-shadow: 0 10px 16px rgba(30, 64, 175, 0.35);
        }

        .stNumberInput > div > div > input,
        .stSelectbox div[data-baseweb="select"] {
            background: #0d1526 !important;
            color: var(--text-main) !important;
            border-color: #2e426f !important;
        }

        div[data-testid="stNumberInput"] {
            margin: 0.04rem 0 0.08rem !important;
        }

        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            min-height: 38px !important;
            border-radius: 8px !important;
            background: #0f1624 !important;
            border: 1px solid rgba(143, 180, 255, 0.12) !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }

        div[data-testid="stNumberInput"] input {
            min-height: 38px !important;
            height: 38px !important;
            font-size: 0.84rem !important;
            line-height: 1.1 !important;
            padding: 0 0.62rem !important;
        }

        div[data-testid="stNumberInput"] button {
            min-width: 30px !important;
            width: 30px !important;
            height: 38px !important;
        }

        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
            gap: 0.68rem !important;
            align-items: stretch !important;
        }

        div[data-testid="stForm"] div[data-testid="column"],
        div[data-testid="stForm"] div[data-testid="stColumn"] {
            min-width: 0;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-field) {
            background: linear-gradient(180deg, rgba(17, 24, 37, 0.98), rgba(12, 18, 29, 0.98)) !important;
            border: 1px solid rgba(143, 180, 255, 0.18) !important;
            border-radius: var(--radius) !important;
            box-shadow: 0 8px 18px rgba(2, 8, 23, 0.18) !important;
            margin: 0 0 0.54rem !important;
            overflow: visible !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-field) div[data-testid="stVerticalBlock"] {
            gap: 0.28rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.feature-field) > div {
            padding: 0.62rem !important;
        }

        div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {
            gap: 0.34rem;
        }

        div[data-testid="stFormSubmitButton"] {
            display: flex;
            justify-content: stretch;
        }

        div[data-testid="stFormSubmitButton"] button {
            min-width: 0;
            width: 100%;
            max-width: none;
            height: 46px;
        }

        .form-submit-divider {
            border-top: 1px solid rgba(143, 180, 255, 0.16);
            margin: 0.72rem 0 0.62rem;
            height: 0;
        }

        .stButton > button:hover {
            filter: brightness(1.06);
        }

        .stTabs [data-baseweb="tab-list"] button[role="tab"] {
            font-weight: 700;
            color: #aebad0;
            font-size: 0.76rem;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] [role="tab"] {
            background: #172033;
            border-color: transparent;
            color: #ffffff;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #172033;
            box-shadow: inset 0 0 0 1px rgba(102, 217, 198, 0.24);
        }

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(143, 180, 255, 0.18);
            border-radius: 10px;
            background:
                linear-gradient(180deg, rgba(17, 24, 38, 0.96), rgba(11, 17, 28, 0.98));
            padding: 0.82rem 0.86rem 0.9rem;
            box-shadow: 0 12px 24px rgba(2, 8, 23, 0.18);
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 0.72rem;
        }

        section[data-testid="stSidebar"] {
            background: transparent;
        }

        .stDataFrame,
        .stDataFrame div[data-testid="stTable"] {
            color: var(--text-main);
        }

        .kpi-layout [data-testid="stVerticalBlock"] {
            gap: 0.45rem;
        }

        .layout-grid {
            display: grid;
            gap: 0.8rem;
        }

        .stAlert {
            background: #111a2f;
            border-color: #2e426f;
        }

        .stAlert [data-testid="stMarkdownContainer"] {
            color: #cddaf5;
        }

        .stMetricValue,
        .stMetricLabel,
        .stMarkdown,
        .stText {
            color: var(--text-main) !important;
        }

        .stMetricDelta {
            color: #c7d6f5;
        }

        @media (max-width: 900px) {
            .hero-grid,
            .kpi-grid,
            .overview-grid,
            .status-card,
            .executive-grid,
            .eda-summary-grid,
            .eda-spotlight,
            .eda-chart-grid,
            .story-grid,
            .chart-grid,
            .insight-grid,
            .demo-hero,
            .demo-command-center,
            .demo-lab-layout,
            .case-strip,
            .feature-glossary-grid,
            .model-explain-grid {
                grid-template-columns: 1fr;
            }

            [data-testid="stAppViewContainer"] .main .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
            }

            .lead-title {
                font-size: 1.18rem;
            }

            .hero-flow {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .demo-title {
                font-size: 1.28rem;
            }

            .demo-intro-card {
                min-height: auto;
            }

            .demo-flow-row {
                grid-template-columns: 1fr;
            }

            .demo-signal-panel {
                grid-template-columns: 1fr;
            }

            .demo-process {
                grid-template-columns: 1fr;
            }

            .demo-control-panel {
                position: static;
            }

            div[data-testid="stColumn"]:has(.result-panel) {
                position: static;
            }
        }

        @media (max-width: 700px) {
            div[data-testid="stForm"] {
                padding: 0.9rem 0.85rem 1rem;
            }

            div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
                display: block !important;
                flex-direction: column !important;
                gap: 0 !important;
            }

            div[data-testid="stForm"] div[data-testid="column"],
            div[data-testid="stForm"] div[data-testid="stColumn"] {
                display: block !important;
                width: 100% !important;
                flex: 1 1 100% !important;
                margin-bottom: 0.7rem;
            }

            .feature-head {
                margin-bottom: 0.18rem;
            }

            .feature-field {
                min-height: auto;
                margin-bottom: 0.55rem;
            }

            .feature-field .feature-description {
                line-height: 1.36;
                padding-bottom: 0.45rem;
            }

            .demo-spec-row {
                grid-template-columns: 1fr;
            }

            .demo-submit-row,
            .demo-editor-header,
            .case-panel-title-row,
            .demo-thresholds {
                grid-template-columns: 1fr;
            }

            .case-panel-title-row {
                display: grid;
            }

            .result-meter {
                width: 132px;
            }

            .demo-score-ring {
                width: 122px;
                height: 122px;
            }

            .demo-pill {
                white-space: normal;
            }

            div[data-testid="stFormSubmitButton"] button {
                max-width: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_top_bar() -> None:
    st.markdown(
        """
        <div class="app-hero">
          <div class="hero-grid">
            <div class="hero-main">
                <span class="hero-top-icon">4G</span>
                <div class="hero-copy">
                  <h1>Proyecto final 4Geeks</h1>
                  <div class="hero-meta">
                    <span class="chip">Clasificación</span>
                    <span class="chip">Random Forest</span>
                    <span class="chip">Estados financieros 1995 a 2018</span>
                  </div>
                </div>
            </div>
            <div class="hero-panel">
              <div class="hero-panel-title">Flujo del proyecto</div>
              <div class="hero-flow">
                <div class="hero-flow-step">Objetivo</div>
                <div class="hero-flow-step">Datos</div>
                <div class="hero-flow-step">SQL</div>
                <div class="hero-flow-step">EDA</div>
                <div class="hero-flow-step">Variables</div>
                <div class="hero-flow-step">Modelo</div>
                <div class="hero-flow-step">Aplicación</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stat_tile(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{_html(label)}</div>
          <div class="kpi-value">{_html(value)}</div>
          <div class="kpi-sub">{_html(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_ordered_labels(columns: list[str]) -> list[str]:
    preferred = [
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
    ordered = [column for column in preferred if column in columns]
    ordered.extend(column for column in columns if column not in ordered)
    return ordered


def render_feature_inputs(feature_columns: list[str], feature_stats: dict) -> pd.DataFrame:
    ordered_features = _feature_ordered_labels(feature_columns)

    inputs: dict[str, float] = {}
    for group_index, group in enumerate(FEATURE_GROUP_ORDER):
        group_features = [
            feature
            for feature in ordered_features
            if FEATURE_DETAILS.get(feature, {}).get("group", "Otros") == group
        ]
        if not group_features:
            continue

        title_class = "form-section-title first" if group_index == 0 else "form-section-title"
        st.markdown(f'<div class="{title_class}">{_html(group)}</div>', unsafe_allow_html=True)
        column_count = 1 if len(group_features) == 1 else 2

        for row_start in range(0, len(group_features), column_count):
            row_features = group_features[row_start : row_start + column_count]
            cols = st.columns(column_count, gap="medium")
            for col, feature in zip(cols, row_features):
                with col:
                    with st.container(border=True):
                        stats = feature_stats.get(feature, {})
                        default = float(stats.get("median", 0.0))
                        min_value = float(stats.get("min", -1e6))
                        max_value = float(stats.get("max", 1e6))
                        step = 1.0 if feature == "Financial_Year" else 0.01
                        fmt = "%.0f" if feature == "Financial_Year" else "%.6g"

                        st.markdown(
                            f"""
                            <div class="feature-field">
                              <div class="feature-head">{_html(_feature_label(feature))}</div>
                              <div class="feature-code">{_html(feature)}</div>
                              <div class="feature-description">{_html(_feature_description(feature))}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        inputs[feature] = st.number_input(
                            label=feature,
                            min_value=min_value,
                            max_value=max_value,
                            value=default,
                            step=step,
                            format=fmt,
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            f'<div class="input-caption">Rango: {_compact_number(min_value)} a {_compact_number(max_value)}</div>',
                            unsafe_allow_html=True,
                        )

            if len(row_features) < column_count:
                for col in cols[len(row_features) :]:
                    with col:
                        st.empty()

    # Garantizar que se devuelvan todas las columnas requeridas por el modelo.
    for feature in feature_columns:
        inputs.setdefault(feature, 0.0)

    return pd.DataFrame([inputs], columns=feature_columns)


def render_demo_input_editor(
    feature_columns: list[str],
    feature_stats: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_features = _feature_ordered_labels(feature_columns)
    rows: list[dict[str, object]] = []
    for feature in ordered_features:
        stats = feature_stats.get(feature, {})
        default = float(stats.get("median", 0.0))
        rows.append(
            {
                "Variable": _feature_label(feature),
                "Valor": default,
            }
        )

    editor_df = pd.DataFrame(rows)
    edited_df = st.data_editor(
        editor_df,
        hide_index=True,
        width="stretch",
        height=455,
        disabled=["Variable"],
        column_config={
            "Variable": st.column_config.TextColumn("Variable", width="medium"),
            "Valor": st.column_config.NumberColumn(
                "Valor",
                help="Valor financiero usado por el modelo para esta simulación.",
                format="%.6g",
                width="small",
            ),
        },
        key="demo_feature_editor",
    )

    values_by_code = dict(zip(ordered_features, edited_df["Valor"].tolist()))
    inputs = {}
    for feature in feature_columns:
        value = values_by_code.get(feature, 0.0)
        inputs[feature] = 0.0 if pd.isna(value) else float(value)

    return pd.DataFrame([inputs], columns=feature_columns), edited_df


def render_case_builder_inputs(feature_columns: list[str], feature_stats: dict) -> pd.DataFrame:
    ordered_features = _feature_ordered_labels(feature_columns)
    inputs: dict[str, float] = {}

    for group_index, group in enumerate(FEATURE_GROUP_ORDER):
        group_features = [
            feature
            for feature in ordered_features
            if FEATURE_DETAILS.get(feature, {}).get("group", "Otros") == group
        ]
        if not group_features:
            continue

        title_class = "case-section-title first" if group_index == 0 else "case-section-title"
        st.markdown(f'<div class="{title_class}">{_html(group)}</div>', unsafe_allow_html=True)
        column_count = 1 if len(group_features) == 1 else 2

        for row_start in range(0, len(group_features), column_count):
            row_features = group_features[row_start : row_start + column_count]
            cols = st.columns(column_count, gap="medium")
            for col, feature in zip(cols, row_features):
                with col:
                    stats = feature_stats.get(feature, {})
                    default = float(stats.get("median", 0.0))
                    min_value = float(stats.get("min", -1e6))
                    max_value = float(stats.get("max", 1e6))
                    step = 1.0 if feature == "Financial_Year" else 0.01
                    fmt = "%.0f" if feature == "Financial_Year" else "%.6g"

                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div class="case-field-card">
                              <div class="case-field-top">
                                <div class="case-field-label">{_html(_feature_label(feature))}</div>
                                <div class="case-field-code">{_html(feature)}</div>
                              </div>
                              <div class="case-field-desc">{_html(_feature_description(feature))}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        inputs[feature] = st.number_input(
                            label=feature,
                            min_value=min_value,
                            max_value=max_value,
                            value=default,
                            step=step,
                            format=fmt,
                            key=f"case_input_{feature}",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            f'<div class="case-field-range">Rango observado: {_compact_number(min_value)} a {_compact_number(max_value)}</div>',
                            unsafe_allow_html=True,
                        )

            if len(row_features) < column_count:
                for col in cols[len(row_features) :]:
                    with col:
                        st.empty()

    for feature in feature_columns:
        inputs.setdefault(feature, 0.0)

    return pd.DataFrame([inputs], columns=feature_columns)


def risk_bucket(
    score: float,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> tuple[str, str]:
    low = min(max(0.0, low_threshold), 1.0)
    high = min(max(0.0, high_threshold), 1.0)
    if low > high:
        low, high = high, low

    if score >= high:
        return "Alto", "🔴"
    if score >= low:
        return "Medio", "🟠"
    return "Bajo", "🟢"


def _risk_card_class(level: str) -> str:
    return {"Bajo": "low", "Medio": "medium", "Alto": "high"}.get(level, "")


def _render_threshold_curve(metadata: dict) -> None:
    feature_columns = tuple(metadata.get("feature_columns", []))
    curve_df = _load_threshold_curve_data(
        str(DATA_PATH),
        str(MODEL_PATH),
        str(metadata.get("model_sha256", "")),
        feature_columns,
    )
    if curve_df.empty:
        st.info("No se pudo construir el gráfico de punto de corte con los datos locales.")
        return

    best_row = curve_df.loc[curve_df["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])

    st.markdown('<div class="section-header">Punto de corte según F1</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=120)
    ax.plot(curve_df["threshold"], curve_df["precision"], label="Precision", color="#4c78a8", linewidth=1.8)
    ax.plot(curve_df["threshold"], curve_df["recall"], label="Recall", color="#f58518", linewidth=1.8)
    ax.plot(curve_df["threshold"], curve_df["f1"], label="F1", color="#54a24b", linewidth=2.2)
    ax.axvline(
        best_threshold,
        color="#ff4f4f",
        linestyle="--",
        linewidth=1.6,
        label=f"Mejor F1: {best_threshold:.3f}",
    )
    ax.set_title("Precision, recall y F1 segun punto de corte")
    ax.set_xlabel("Threshold / punto de corte")
    ax.set_ylabel("Metrica")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown(
        """
        <div class="hint-box">
          La línea roja marca el punto de corte donde F1 alcanza su mejor equilibrio.
          A la izquierda se capturan más casos, pero con más alertas; a la derecha se exige más puntaje,
          baja el recall y quedan menos registros para revisión.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_cards(metadata: dict) -> None:
    positive_ratio = float(metadata.get("dataset_positive_ratio", 0.0))
    comparison_metrics = metadata.get("model_comparison_metrics", {})
    legacy_validation_metrics = metadata.get("validation_metrics", {})
    test_metrics = metadata.get("test_metrics", {})
    random_search = metadata.get("randomized_search", {})

    if comparison_metrics:
        model_rows = [
            ("Regresión logística (base, 0.5)", comparison_metrics.get("logistic_regression", {})),
            ("Random Forest anterior (0.5)", comparison_metrics.get("random_forest_baseline_threshold_05", {})),
            ("Random Forest anterior (punto corte F1)", comparison_metrics.get("random_forest_baseline_f1_threshold", {})),
            ("RF RandomizedSearchCV (punto corte F1)", comparison_metrics.get("random_forest_randomized_search_f1_threshold", {})),
        ]
        model_rows = [(label, metrics) for label, metrics in model_rows if metrics]
        logistic_metrics = comparison_metrics.get("logistic_regression", {})
        forest_metrics = comparison_metrics.get("random_forest_randomized_search_f1_threshold", {})
    else:
        model_rows = []
        for model_key, model_label in (
            ("logistic_regression", "Regresión logística (base)"),
            ("random_forest_tuned", "Random Forest (ajustada)"),
        ):
            if model_key not in legacy_validation_metrics:
                continue
            model_metrics = legacy_validation_metrics[model_key]
            model_rows.append((model_label, model_metrics))
        logistic_metrics = legacy_validation_metrics.get("logistic_regression", {})
        forest_metrics = legacy_validation_metrics.get("random_forest_tuned", {})

    if logistic_metrics and forest_metrics:
        lr_ap = float(logistic_metrics.get("average_precision", 0.0))
        rf_ap = float(forest_metrics.get("average_precision", 0.0))
        lr_f1 = float(logistic_metrics.get("f1", 0.0))
        rf_f1 = float(forest_metrics.get("f1", 0.0))
        lr_ap_width = min(max(lr_ap * 100, 0), 100)
        rf_ap_width = min(max(rf_ap * 100, 0), 100)
        lr_f1_width = min(max(lr_f1 * 100, 0), 100)
        rf_f1_width = min(max(rf_f1 * 100, 0), 100)
        st.markdown(
            f"""
            <div class="section-header">Evidencia de selección del modelo</div>
            <section class="insight-grid">
              <article class="insight-card wide">
                <div class="insight-title">Comparación de modelos</div>
                <div class="insight-copy">Se probó una base de Regresión Logística y se eligió Random Forest por mejor desempeño en señales raras.</div>
                <div class="insight-copy"><strong>Average Precision</strong> significa precisión promedio en el ranking: mide qué tan bien el modelo pone los casos con AAER_ID entre los primeros lugares de riesgo. Es útil porque hay muy pocos positivos.</div>
                <div class="bar-row">
                  <div class="bar-label"><span>RL · precisión promedio ranking</span><strong>{lr_ap:.4f}</strong></div>
                  <div class="bar-track"><div class="bar-fill muted" style="width: {lr_ap_width:.2f}%;"></div></div>
                </div>
                <div class="bar-row">
                  <div class="bar-label"><span>RF · precisión promedio ranking</span><strong>{rf_ap:.4f}</strong></div>
                  <div class="bar-track"><div class="bar-fill warning" style="width: {rf_ap_width:.2f}%;"></div></div>
                </div>
                <div class="bar-row">
                  <div class="bar-label"><span>RL · F1 Score</span><strong>{lr_f1:.4f}</strong></div>
                  <div class="bar-track"><div class="bar-fill muted" style="width: {lr_f1_width:.2f}%;"></div></div>
                </div>
                <div class="bar-row">
                  <div class="bar-label"><span>RF · F1 Score</span><strong>{rf_f1:.4f}</strong></div>
                  <div class="bar-track"><div class="bar-fill winner" style="width: {rf_f1_width:.2f}%;"></div></div>
                </div>
                <div class="insight-copy">F1 resume el equilibrio entre precision y recall. Cuanto más alto, mejor balance entre detectar casos con AAER_ID y evitar falsas alertas.</div>
                <div class="insight-foot">La elección se basa en métricas para clases desbalanceadas, no en accuracy.</div>
              </article>
            </section>
            """,
            unsafe_allow_html=True,
        )

    if model_rows:
        title = "Comparación final de modelos y punto de corte" if comparison_metrics else "Comparación en validación"
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
        compare_rows: list[dict[str, str]] = []
        for model_label, model_metrics in model_rows:
            compare_rows.append(
                {
                    "Modelo": model_label,
                    "Precision": _as_percent(model_metrics.get("precision", 0)),
                    "Average Precision": _as_percent(model_metrics.get("average_precision", 0)),
                    "Recall": _as_percent(model_metrics.get("recall", 0)),
                    "ROC AUC": _as_percent(model_metrics.get("roc_auc", 0)),
                    "F1 Score": _as_percent(model_metrics.get("f1", 0)),
                    "Threshold": _as_percent(model_metrics.get("threshold", 0)),
                }
            )
        st.dataframe(pd.DataFrame(compare_rows), width="stretch", hide_index=True)
        _render_threshold_curve(metadata)

    if test_metrics:
        st.markdown(
            """
            <div class="section-header">Cómo leer estas métricas</div>
            <section class="model-explain-grid">
              <article class="metric-explain">
                <strong>Precision</strong>
                <p>Indica qué parte de los registros marcados para revisión tenía AAER_ID. Ayuda a entender cuántas alertas positivas son realmente casos etiquetados.</p>
              </article>
              <article class="metric-explain">
                <strong>Average Precision</strong>
                <p>Mide si los registros con AAER_ID aparecen arriba en el ranking de riesgo. Es más útil que accuracy cuando la clase positiva es muy rara.</p>
              </article>
              <article class="metric-explain">
                <strong>Recall</strong>
                <p>Indica qué parte de los registros con AAER_ID logra encontrar el modelo. Sirve para ver cuántos casos etiquetados quedan por debajo del punto de corte.</p>
              </article>
              <article class="metric-explain">
                <strong>ROC AUC</strong>
                <p>Mide la capacidad general de ordenar casos positivos por encima de negativos. Sirve para comparar separación entre modelos, no para decidir por sí solo el punto de corte.</p>
              </article>
              <article class="metric-explain">
                <strong>F1 Score</strong>
                <p>Resume el equilibrio entre precision y recall. Se usa cuando importa detectar casos etiquetados, pero también evitar demasiadas alertas falsas.</p>
              </article>
              <article class="metric-explain">
                <strong>Threshold</strong>
                <p>Es el puntaje mínimo para clasificar un registro como Predicho 1. En este proyecto representa entrada a revisión humana, no confirmación de fraude.</p>
              </article>
            </section>
            <div class="hint-box">Lectura final: este modelo sirve para priorizar revisión humana. No confirma fraude ni reemplaza una auditoría.</div>
            """,
            unsafe_allow_html=True,
        )

    if random_search:
        best_params = random_search.get("best_params", {})
        best_cv_score = random_search.get("best_cv_average_precision")
        if best_params:
            st.markdown('<div class="section-header">Hiperparametrización</div>', unsafe_allow_html=True)
            param_descriptions = {
                "n_estimators": "cantidad de árboles",
                "min_samples_split": "mínimo de registros para dividir un nodo",
                "min_samples_leaf": "mínimo de registros por hoja",
                "max_features": "variables evaluadas en cada división",
                "max_depth": "profundidad máxima de cada árbol",
                "class_weight": "peso para compensar el desbalance de clases",
            }
            param_rows = []
            for key, value in best_params.items():
                param_name = key.replace("model__", "")
                description = param_descriptions.get(param_name)
                label = f"{param_name} ({description})" if description else param_name
                display_value = "Sin límite" if value is None else str(value)
                param_rows.append({"Hiperparámetro": label, "Valor": display_value})
            params_df = pd.DataFrame(param_rows)
            if best_cv_score is not None:
                st.caption(f"RandomizedSearchCV optimizó Average Precision. Mejor promedio CV: {_as_percent(best_cv_score)}")
            st.dataframe(params_df, width="stretch", hide_index=True)

    if test_metrics:
        confusion_matrix = test_metrics.get("confusion_matrix")
        if confusion_matrix and len(confusion_matrix) == 4:
            tn, fp, fn, tp = confusion_matrix
            st.markdown('<div class="section-header">Matriz de confusión en test</div>', unsafe_allow_html=True)
            confusion_df = pd.DataFrame(
                [
                    {"Real": "Sin etiqueta conocida", "Predicho 0": tn, "Predicho 1": fp},
                    {"Real": "Con AAER_ID", "Predicho 0": fn, "Predicho 1": tp},
                ]
            )
            st.dataframe(confusion_df, width="stretch", hide_index=True)
            reviewed_count = tp + fp
            actual_labeled_count = tp + fn
            st.markdown(
                f"""
                <div class="hint-box">
                  <strong>Cómo leer esta matriz:</strong> las filas muestran la etiqueta real disponible y las columnas muestran la decisión del modelo.
                  <strong>Predicho 1</strong> significa que el registro queda priorizado para revisión humana.
                  Con el punto de corte actual, el modelo envía <strong>{reviewed_count}</strong> registros a revisión:
                  <strong>{tp}</strong> tenían AAER_ID y <strong>{fp}</strong> no tenían etiqueta conocida.
                  De los <strong>{actual_labeled_count}</strong> registros con AAER_ID en test, detecta <strong>{tp}</strong> y deja
                  <strong>{fn}</strong> por debajo del punto de corte. Por eso el modelo se interpreta como ranking de prioridad,
                  no como confirmación final de fraude.
                </div>
                """,
                unsafe_allow_html=True,
            )

def _render_project_overview(metadata: dict) -> None:
    rows = int(metadata.get("dataset_rows", 0))
    positive_ratio = float(metadata.get("dataset_positive_ratio", 0.0))
    feature_count = len(metadata.get("feature_columns", []))

    st.markdown(
        f"""
        <div class="cta-card">
          <div class="cta-title">Sinopsis</div>
          El modelo no detecta fraude confirmado. Calcula un puntaje de riesgo para apoyar revisión humana y priorizar casos que merecen análisis.
        </div>
        <section class="decision-list">
          <div class="decision-item">
            <div class="decision-label">Registros analizados</div>
            <div class="decision-value">{rows:,}</div>
          </div>
          <div class="decision-item">
            <div class="decision-label">Clase positiva</div>
            <div class="decision-value">{positive_ratio:.4%} con AAER_ID</div>
          </div>
          <div class="decision-item">
            <div class="decision-label">Variables del modelo</div>
            <div class="decision-value">{feature_count} indicadores financieros</div>
          </div>
        </section>
        <div class="section-header">Proceso del análisis</div>
        <section class="story-grid">
          <article class="story-step featured">
            <div class="story-number">01</div>
            <div class="story-title">Objetivo del trabajo</div>
            <div class="story-copy">Este trabajo desarrolla un modelo de clasificación para analizar registros financieros históricos y estimar riesgo de fraude o manipulación contable. A partir de estados financieros empresa-año, la aplicación genera un puntaje de riesgo para apoyar la revisión humana. El resultado no confirma fraude ni reemplaza auditoría profesional.</div>
          </article>
          <article class="story-step">
            <div class="story-number">02</div>
            <div class="story-title">Datos y variable objetivo</div>
            <div class="story-copy">Se usa un dataset de estados financieros entre 1995 y 2018. Cada fila representa una empresa en un año fiscal. La variable AAER_ID se usa para construir la etiqueta: clase 1 cuando existe un caso conocido y clase 0 cuando no existe esa etiqueta.</div>
          </article>
          <article class="story-step">
            <div class="story-number">03</div>
            <div class="story-title">Base de datos y SQL</div>
            <div class="story-copy">Los datos se cargan en SQLite para trabajar con consultas SQL. Esto permite validar cantidad de registros, distribución de clases, años disponibles y comparaciones financieras básicas.</div>
          </article>
          <article class="story-step">
            <div class="story-number">04</div>
            <div class="story-title">Análisis exploratorio</div>
            <div class="story-copy">Se revisan distribuciones, valores extremos, desbalance de clases y relaciones entre variables financieras. Este paso ayuda a entender el problema antes de entrenar el modelo.</div>
          </article>
          <article class="story-step">
            <div class="story-number">05</div>
            <div class="story-title">Preparación de variables</div>
            <div class="story-copy">Se seleccionan 12 variables financieras relevantes para mantener el modelo simple, explicable y conectado con la consigna. Estas variables representan ventas, activos, deuda, liquidez, inventario, costos, impuestos, intereses y precio de mercado.</div>
          </article>
          <article class="story-step">
            <div class="story-number">06</div>
            <div class="story-title">Modelo y evaluación</div>
            <div class="story-copy">Se comparó Regresión Logística contra Random Forest y se eligió Random Forest por su mejor capacidad para priorizar casos de riesgo. Como la clase positiva es muy rara, se evalúa con métricas adecuadas para datos desbalanceados, como Average Precision, Recall, Precision y F1 Score.</div>
          </article>
          <article class="story-step">
            <div class="story-number">07</div>
            <div class="story-title">Aplicación final</div>
            <div class="story-copy">La app permite ingresar un nuevo registro financiero y obtener un puntaje de riesgo. El resultado sirve como apoyo para priorizar revisión humana, no como conclusión legal ni auditoría profesional.</div>
          </article>
        </section>
        """,
        unsafe_allow_html=True,
    )

def _eda_chart_card_html(
    filename: str,
    title: str,
    note: str,
    tag: str,
    *,
    wide: bool = False,
) -> str:
    image_path = EDA_FIGURE_DIR / filename
    card_class = "eda-chart-card wide" if wide else "eda-chart-card"
    if image_path.exists():
        image_markup = (
            f'<img class="eda-chart-image" src="{_image_data_uri(image_path)}" '
            f'alt="{_html(title)}">'
        )
    else:
        image_markup = '<div class="hint-box">No está disponible esta figura.</div>'

    return f"""<article class="{card_class}">
  <div class="eda-chart-head">
    <div>
      <div class="eda-chart-title">{_html(title)}</div>
      <div class="eda-chart-note">{_html(note)}</div>
    </div>
    <div class="eda-chart-tag">{_html(tag)}</div>
  </div>
  {image_markup}
</article>"""


def _render_eda_chart_card(
    filename: str,
    title: str,
    note: str,
    tag: str,
    *,
    wide: bool = False,
) -> None:
    st.markdown(
        _eda_chart_card_html(filename, title, note, tag, wide=wide),
        unsafe_allow_html=True,
    )


def _render_eda_gallery() -> None:
    st.markdown(
        """
        <section class="lead-card">
          <div class="lead-kicker">Análisis exploratorio</div>
          <div class="lead-title">Entender los datos antes de entrenar el modelo</div>
          <div class="lead-copy">
            El EDA revisa la estructura del dataset antes de modelar: cantidad de registros, balance de la etiqueta,
            evolución temporal, valores extremos y señales financieras. En este proyecto es clave porque la etiqueta
            positiva es muy rara; por eso el objetivo no es probar fraude, sino priorizar registros para revisión humana.
          </div>
        </section>
        <section class="eda-summary-grid">
          <article class="eda-summary-card">
            <div class="eda-summary-label">Objetivo del EDA</div>
            <div class="eda-summary-value">Explorar</div>
            <div class="eda-summary-copy">Antes de entrenar, se revisa si los datos son consistentes, desbalanceados o dominados por valores extremos.</div>
          </article>
          <article class="eda-summary-card">
            <div class="eda-summary-label">Desbalance</div>
            <div class="eda-summary-value">0.6536%</div>
            <div class="eda-summary-copy">Solo 575 de 87,974 registros tienen AAER_ID. Accuracy puede ser engañosa si el modelo predice casi todo como clase 0.</div>
          </article>
          <article class="eda-summary-card">
            <div class="eda-summary-label">Uso posterior</div>
            <div class="eda-summary-value">Priorizar</div>
            <div class="eda-summary-copy">La EDA justifica evaluar el modelo como ranking de revisión, no como una herramienta que declara fraude confirmado.</div>
          </article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not EDA_FIGURE_DIR.exists():
        st.info("Aún no se generaron las figuras de EDA.")
        return

    spotlight_chart = _eda_chart_card_html(
        "target_balance.png",
        "Balance de etiqueta objetivo",
        "El gráfico muestra la diferencia real entre registros sin etiqueta conocida y registros con AAER_ID.",
        "Punto de partida",
    )
    st.markdown(
        f"""
        <section class="eda-spotlight">
          {spotlight_chart}
          <article class="eda-narrative-card">
            <div>
              <div class="eda-kicker">Lectura del desbalance</div>
              <div class="eda-title">La clase positiva es extremadamente rara.</div>
              <p class="eda-copy">
                Hay 87,399 registros sin etiqueta conocida y solo 575 registros con AAER_ID. Esto significa que la clase
                positiva representa apenas 0.6536% del dataset. Si un modelo predijera todo como clase 0, tendría una
                accuracy muy alta, pero no ayudaría a encontrar los casos relevantes.
              </p>
              <p class="eda-copy">
                Por eso el proyecto usa métricas para datos desbalanceados y lee el resultado como prioridad de revisión:
                el modelo ordena casos que merecen atención, no confirma fraude por sí solo.
              </p>
            </div>
            <div class="eda-step-list">
              <div class="eda-step"><span>1</span><div>Se crea <code>target_fraud</code> a partir de AAER_ID para poder entrenar modelos de clasificación.</div></div>
              <div class="eda-step"><span>2</span><div>Se confirma que la clase positiva es muy minoritaria, por eso accuracy no alcanza.</div></div>
              <div class="eda-step"><span>3</span><div>Se buscan señales financieras y temporales que ayuden a priorizar registros para revisión.</div></div>
            </div>
          </article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Evidencia en orden de lectura</div>', unsafe_allow_html=True)
    chart_cards = [
        {
            "filename": "fraud_rate_by_year.png",
            "title": "Tasa de etiqueta por año fiscal",
            "note": "El pico alrededor de 2000-2002 coincide con un contexto de fuerte escrutinio contable: caída puntocom, Enron, WorldCom, Tyco y la aprobación de Sarbanes-Oxley en 2002. Debe leerse como concentración de casos AAER conocidos, no como todo el fraude real de la economía.",
            "tag": "Evolución",
            "wide": True,
        },
        {
            "filename": "receivables_to_sales_boxplot.png",
            "title": "Cuentas por cobrar / ventas",
            "note": "La caja de los registros con AAER_ID aparece más arriba: tienden a tener más cuentas por cobrar respecto de sus ventas. Hay solapamiento entre grupos, así que no prueba fraude; funciona como señal de alerta para revisión.",
            "tag": "Señal",
        },
        {
            "filename": "receivables_to_sales_bars.png",
            "title": "Resumen de rect / sale",
            "note": "La media sube de 0.36 a 0.94 y la mediana de 0.14 a 0.18 en registros con AAER_ID. La media aumenta más porque algunos casos extremos tienen cuentas por cobrar muy altas.",
            "tag": "Comparación",
        },
        {
            "filename": "key_variable_distributions.png",
            "title": "Distribución de montos financieros",
            "note": "Los montos están muy concentrados y también tienen valores extremos. La escala está comprimida para que el gráfico sea legible, pero el eje muestra montos originales aproximados como 100, 1K, 10K o 1M.",
            "tag": "Distribución",
            "wide": True,
        },
        {
            "filename": "feature_correlation_heatmap.png",
            "title": "Correlación entre variables del modelo",
            "note": "El mapa identifica relaciones fuertes entre variables de tamaño financiero. Correlación no implica causalidad ni prueba de fraude.",
            "tag": "Relaciones",
            "wide": True,
        },
    ]
    chart_cards_html = "\n".join(
        _eda_chart_card_html(
            card["filename"],
            card["title"],
            card["note"],
            card["tag"],
            wide=bool(card.get("wide", False)),
        )
        for card in chart_cards
    )
    st.markdown(f'<section class="eda-chart-grid">{chart_cards_html}</section>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="eda-conclusion">
          <strong>Conclusión de la EDA:</strong> la etiqueta positiva es rara, las señales no se distribuyen igual por año
          y algunas variables financieras muestran diferencias útiles. Esto respalda una app de priorización de revisión,
          no una app que declare fraude confirmado.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if EDA_REPORT_PATH.exists():
        with st.expander("Reporte EDA completo"):
            st.code(EDA_REPORT_PATH.read_text(encoding="utf-8"), language="markdown")


def _render_data_page(metadata: dict) -> None:
    rows = int(metadata.get("dataset_rows", 0))
    positive_ratio = float(metadata.get("dataset_positive_ratio", 0.0))
    positive_count = int(round(rows * positive_ratio))
    negative_count = rows - positive_count
    negative_ratio = (negative_count / rows) if rows else 0.0
    negative_bar_width = negative_ratio * 100
    positive_bar_width = positive_ratio * 100

    st.markdown(
        f"""
        <section class="lead-card">
          <div class="lead-kicker">Origen y etiqueta</div>
          <div class="lead-title">Dataset financiero histórico con etiqueta AAER_ID</div>
          <div class="lead-copy">
            El proyecto usa el dataset de Kaggle New Fraud Financial Dataset. Cada fila representa un registro financiero de una empresa en un año fiscal entre 1995 y 2018.
          </div>
        </section>
        <section class="overview-grid">
          <article class="panel-card overview-card">
            <div class="panel-title">Qué significa AAER_ID</div>
            <ul>
              <li>AAER_ID es un identificador de caso asociado a una acción regulatoria contable o de auditoría.</li>
              <li>Si existe, la fila se marca como caso relacionado con fraude conocido.</li>
              <li>El modelo aprende de esa etiqueta histórica; no descubre una verdad legal nueva.</li>
            </ul>
          </article>
          <article class="panel-card overview-card">
            <div class="panel-title">Origen regulatorio de AAER</div>
            <ul>
              <li>AAER proviene de reportes de la SEC, el regulador financiero de Estados Unidos.</li>
              <li>La SEC publica casos cuando investiga o sanciona irregularidades en reportes financieros.</li>
              <li>Puede incluir manipulación contable, ingresos mal reportados, fallas de auditoría o declaraciones financieras engañosas.</li>
              <li>Por eso AAER_ID funciona como etiqueta histórica de caso conocido, no como sentencia automática del modelo.</li>
            </ul>
          </article>
          <article class="panel-card overview-card">
            <div class="panel-title">Creación de target_fraud</div>
            <ul>
              <li>Se agrega la columna <code>target_fraud</code> al dataset para convertir <code>AAER_ID</code> en una variable objetivo.</li>
              <li>Esta columna permite entrenar, evaluar y comparar modelos de clasificación.</li>
              <li><code>target_fraud = 1</code>: la fila tiene <code>AAER_ID</code>, es decir, está vinculada a un caso conocido.</li>
              <li><code>target_fraud = 0</code>: no hay etiqueta conocida en este dataset; no garantiza que la empresa sea limpia.</li>
            </ul>
          </article>
        </section>
        <section class="status-card">
          <div class="status-item">
            <div class="status-label">Registros</div>
            <div class="status-value">{rows:,}</div>
          </div>
          <div class="status-item">
            <div class="status-label">Casos con AAER_ID</div>
            <div class="status-value">{positive_count:,}</div>
          </div>
          <div class="status-item">
            <div class="status-label">Tasa positiva</div>
            <div class="status-value">{positive_ratio:.4%}</div>
          </div>
        </section>
        <div class="section-header">Distribución de la variable objetivo</div>
        <section class="insight-grid">
          <article class="insight-card wide">
            <div class="insight-title">Etiqueta extremadamente desbalanceada</div>
            <div class="insight-copy">La clase positiva es muy rara. Esto define cómo se analiza el proyecto y por qué accuracy puede ser engañosa.</div>
            <div class="bar-row">
              <div class="bar-label"><span>Clase 0 · sin etiqueta conocida</span><strong>{negative_count:,} · {negative_ratio:.4%}</strong></div>
              <div class="bar-track"><div class="bar-fill muted" style="width: {negative_bar_width:.4f}%;"></div></div>
            </div>
            <div class="bar-row">
              <div class="bar-label"><span>Clase 1 · con AAER_ID</span><strong>{positive_count:,} · {positive_ratio:.4%}</strong></div>
              <div class="bar-track"><div class="bar-fill warning" style="width: {positive_bar_width:.4f}%;"></div></div>
            </div>
            <div class="insight-foot">Tasa positiva real: {positive_ratio:.4%} del dataset.</div>
          </article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Validación local</div>', unsafe_allow_html=True)
    st.code("/Users/igna/entorno/bin/python scripts/validate_data.py", language="bash")
    st.markdown(
        '<div class="hint-box">La validación comprueba existencia del CSV, cantidad de filas, checksum esperado y columnas requeridas para modelado.</div>',
        unsafe_allow_html=True,
    )


def _top_feature_importances(model: object, feature_columns: list[str], limit: int = 2) -> list[tuple[str, float]]:
    named_steps = getattr(model, "named_steps", None)
    estimator = named_steps.get("model") if named_steps else model
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_columns):
        return []

    ranked = sorted(
        zip(feature_columns, [float(value) for value in importances]),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:limit]


def _importance_explanation(feature: str) -> str:
    explanations = {
        "Financial_Year": (
            "Captura que los casos con AAER_ID no se distribuyen igual en el tiempo; "
            "hay años con mayor concentración histórica de casos etiquetados."
        ),
        "rect": (
            "Representa cuentas por cobrar. Ayuda porque valores altos frente a ventas pueden sugerir "
            "ingresos registrados que todavía no se transformaron en cobros."
        ),
    }
    return explanations.get(
        feature,
        "Aporta señal al ranking de riesgo según la importancia interna del Random Forest.",
    )


def _render_variables_page(metadata: dict, model: object) -> None:
    feature_columns = metadata.get("feature_columns", [])
    top_features = _top_feature_importances(model, feature_columns)
    feature_cards = "\n".join(
        f"""
        <article class="feature-glossary-item">
          <div class="feature-head">{_html(_feature_label(feature))}</div>
          <div class="feature-code">{_html(feature)}</div>
          <div class="feature-description">{_html(_feature_description(feature))}</div>
        </article>
        """
        for feature in feature_columns
    )
    important_cards = "\n".join(
        f"""
        <article class="eda-summary-card">
          <div class="eda-summary-label">Variable clave #{index}</div>
          <div class="eda-summary-value">{_html(_feature_label(feature))}</div>
          <div class="eda-summary-copy">
            Importancia en Random Forest: {_as_percent(importance)}. {_html(_importance_explanation(feature))}
          </div>
        </article>
        """
        for index, (feature, importance) in enumerate(top_features, start=1)
    )

    st.markdown(
        f"""
        <section class="lead-card">
          <div class="lead-kicker">Preparación de variables</div>
          <div class="lead-title">12 variables financieras usadas por el modelo</div>
          <div class="lead-copy">
            El modelo usa un subconjunto claro de 12 variables en lugar de las 120 columnas originales. Se mantiene
            acotado para que el proyecto sea explicable: las variables cubren contexto temporal, tamaño de la empresa,
            resultados, deuda, liquidez, operación, costos, impuestos, intereses y precio de mercado.
          </div>
        </section>
        <div class="section-header">Variables que más ayudan a predecir</div>
        <section class="eda-summary-grid">{important_cards}</section>
        <div class="hint-box">Estas importancias salen del Random Forest entrenado. Ayudan a explicar qué variables usa más el modelo para ordenar riesgo, pero no implican causalidad ni prueba directa de fraude.</div>
        <div class="section-header">Glosario de variables</div>
        <section class="feature-glossary-grid">{feature_cards}</section>
        """,
        unsafe_allow_html=True,
    )


def _markdown_table_to_html(table_markdown: str) -> str:
    lines = [
        line.strip()
        for line in table_markdown.strip().splitlines()
        if line.strip().startswith("|")
    ]
    if len(lines) < 2:
        return '<div class="hint-box">No se encontró tabla de resultado.</div>'

    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        rows.append(cells[: len(headers)])

    header_html = "".join(f"<th>{_html(header)}</th>" for header in headers)
    rows_html = "\n".join(
        "<tr>" + "".join(f"<td>{_html(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""
    <div class="sql-table-wrap">
      <table class="sql-result-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """


def _extract_sql_result_table(section_body: str) -> str:
    if "**Resultado:**" not in section_body:
        return ""

    result_block = section_body.split("**Resultado:**", 1)[1]
    table_lines: list[str] = []
    for line in result_block.splitlines():
        if line.strip().startswith("|"):
            table_lines.append(line)
        elif table_lines and line.strip():
            break
    return "\n".join(table_lines)


def _render_sql_report(report_text: str) -> None:
    sections = report_text.split("\n## ")
    intro = sections[0].replace("# Resultados SQL de Fase 1", "").strip()
    if intro:
        st.markdown(intro)

    for raw_section in sections[1:]:
        title, _, body = raw_section.partition("\n")
        query = ""
        if "```sql" in body:
            query_block = body.split("```sql", 1)[1]
            query = query_block.split("```", 1)[0].strip()

        table_html = _markdown_table_to_html(_extract_sql_result_table(body))
        st.markdown(
            f"""
            <section class="sql-report-section">
              <div class="sql-report-title">{_html(title.strip())}</div>
              <div class="sql-report-label">Consulta</div>
              <pre class="sql-code"><code>{_html(query)}</code></pre>
              <div class="sql-report-label">Resultado</div>
              {table_html}
            </section>
            """,
            unsafe_allow_html=True,
        )


def _render_sql_page() -> None:
    st.markdown(
        """
        <section class="lead-card">
          <div class="lead-kicker">Base de datos y SQL</div>
          <div class="lead-title">Consultas para validar y entender el dataset</div>
          <div class="lead-copy">
            Los datos se cargan en SQLite para usar SQL directamente sobre el dataset. Abajo se muestran las consultas
            aplicadas y su resultado: conteos, distribución de clases, evolución por año y comparaciones financieras básicas.
          </div>
        </section>
        <section class="story-grid">
          <article class="story-step">
            <div class="story-number">01</div>
            <div class="story-title">Total de registros</div>
            <div class="story-copy">Valida que la tabla contenga 87,974 filas cargadas desde el CSV local.</div>
          </article>
          <article class="story-step">
            <div class="story-number">02</div>
            <div class="story-title">Clases 0 y 1</div>
            <div class="story-copy">Cuenta registros sin AAER_ID y registros con AAER_ID para mostrar el desbalance.</div>
          </article>
          <article class="story-step">
            <div class="story-number">03</div>
            <div class="story-title">Tasa por año</div>
            <div class="story-copy">Revisa cómo cambia la presencia de casos etiquetados entre 1995 y 2018.</div>
          </article>
          <article class="story-step">
            <div class="story-number">04</div>
            <div class="story-title">Promedios financieros</div>
            <div class="story-copy">Compara ventas, utilidad, activos, pasivos y efectivo entre clases.</div>
          </article>
          <article class="story-step">
            <div class="story-number">05</div>
            <div class="story-title">Cuentas por cobrar</div>
            <div class="story-copy">Calcula cuentas por cobrar sobre ventas como señal exploratoria financiera.</div>
          </article>
          <article class="story-step">
            <div class="story-number">06</div>
            <div class="story-title">Pasivos sobre activos</div>
            <div class="story-copy">Calcula una razón de deuda para comparar estructura financiera entre clases.</div>
          </article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if SQL_REPORT_PATH.exists():
        with st.expander("Reporte SQL completo"):
            _render_sql_report(SQL_REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info("Aún no se generó el reporte SQL.")


def main() -> None:
    st.set_page_config(
        page_title="Priorización de riesgo de fraude",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_layout_styles()
    _render_top_bar()

    try:
        model, metadata = load_model_and_metadata()
        review_thresholds = metadata.get("review_thresholds", {})
        low_threshold = float(review_thresholds.get("low", LOW_THRESHOLD))
        high_threshold = float(review_thresholds.get("high", HIGH_THRESHOLD))
    except Exception:
        LOGGER.exception("No se pudieron cargar los artefactos del modelo")
        st.error("No se pueden cargar los artefactos del modelo. Reentrena o verifica archivos.")
        return

    feature_columns = metadata.get("feature_columns") or []
    if not feature_columns:
        st.error("La metadata del modelo no contiene `feature_columns`.")
        return

    feature_stats = metadata.get("feature_statistics", {})
    model_name = metadata.get("model_name", "Modelo guardado")
    if model_name == "Saved model":
        model_name = "Modelo guardado"
    (
        tab_inicio,
        tab_datos,
        tab_sql,
        tab_eda,
        tab_variables,
        tab_modelo,
        tab_aplicacion,
    ) = st.tabs(
        ["Inicio", "Datos", "SQL", "EDA", "Variables", "Modelo", "Aplicación"]
    )

    with tab_inicio:
        _render_project_overview(metadata)

    with tab_aplicacion:
        st.markdown(
            """
            <section class="demo-stage-header">
              <div class="demo-stage-kicker">Aplicación final</div>
              <div class="demo-stage-title">Ingresar un registro financiero y leer el puntaje al costado.</div>
              <p class="demo-stage-copy">
                La carga queda a la izquierda y el resultado queda fijo a la derecha. El puntaje es una señal de revisión:
                no declara fraude confirmado ni reemplaza auditoría profesional.
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        submitted = False
        input_df = pd.DataFrame(columns=feature_columns)
        last_score = st.session_state.get("demo_result", {}).get("score")

        workspace_cols = st.columns([0.70, 0.30], gap="medium")
        with workspace_cols[0]:
            with st.form("risk_form", clear_on_submit=False):
                st.markdown(
                    f"""
                    <section class="case-panel-head">
                      <div class="case-panel-title-row">
                        <div>
                          <div class="case-panel-title">Panel de datos financieros</div>
                          <p class="case-panel-copy">
                            Ajustá los valores de una empresa-año. Los campos están agrupados por lectura financiera para que no parezca una tabla cruda.
                          </p>
                        </div>
                        <div class="case-badge">{len(feature_columns)} variables</div>
                      </div>
                      <div class="case-strip">
                        <div class="case-strip-item">
                          <div class="case-strip-label">Entrada</div>
                          <div class="case-strip-value">Registro empresa-año</div>
                        </div>
                        <div class="case-strip-item">
                          <div class="case-strip-label">Valores iniciales</div>
                          <div class="case-strip-value">Medianas de entrenamiento</div>
                        </div>
                        <div class="case-strip-item">
                          <div class="case-strip-label">Resultado esperado</div>
                          <div class="case-strip-value">Prioridad de revisión</div>
                        </div>
                      </div>
                    </section>
                    """,
                    unsafe_allow_html=True,
                )
                input_df = render_case_builder_inputs(feature_columns, feature_stats)
                st.markdown('<div class="form-submit-divider"></div>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="case-submit-panel">
                      <p class="case-submit-note">
                        El cálculo usa el modelo guardado localmente. No guarda este registro ni modifica el dataset.
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                submitted = st.form_submit_button(
                    "Calcular prioridad", type="primary", width="stretch"
                )

        if submitted:
            try:
                probabilities = model.predict_proba(input_df)[0]
                last_score = float(probabilities[1])
                st.session_state["demo_result"] = {"score": last_score}
            except Exception:
                last_score = None
                LOGGER.exception("Error al calcular riesgo")
                st.error("No se pudo calcular la puntuación. Revisa los datos de entrada.")

        if last_score is None:
            result_class = ""
            score_display = "--"
            result_title = "Resultado pendiente"
            result_copy = "Completá o revisá los valores del panel izquierdo y calculá la prioridad."
            score_percent = 0.0
        else:
            last_score = float(last_score)
            result_label, _ = risk_bucket(
                last_score,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
            result_class = _risk_card_class(result_label)
            score_display = f"{last_score:.2%}"
            result_title = f"Prioridad {result_label}"
            result_copy = (
                "Este puntaje sirve para decidir qué caso revisar primero. "
                "No es conclusión legal ni auditoría."
            )
            score_percent = min(max(last_score, 0.0), 1.0) * 100

        with workspace_cols[1]:
            st.markdown(
                f"""
                <aside class="result-panel {result_class}">
                  <div class="result-kicker">Resultado</div>
                  <div class="result-summary">
                    <div class="result-headline">
                      <div>
                        <div class="result-status-label">Prioridad estimada</div>
                        <div class="result-status-title">{_html(result_title)}</div>
                      </div>
                      <div class="result-score-pill">
                        <strong>{score_display}</strong>
                        <span>puntaje</span>
                      </div>
                    </div>
                    <div class="result-scale-card">
                      <div class="result-scale-title">Escala de revisión</div>
                      <div class="result-scale">
                        <span class="result-scale-fill" style="width: {score_percent:.2f}%"></span>
                      </div>
                      <div class="result-scale-markers">
                        <span>Bajo</span>
                        <span>Medio {_as_percent(low_threshold)}</span>
                        <span>Alto {_as_percent(high_threshold)}</span>
                      </div>
                    </div>
                    <p class="result-interpretation">{_html(result_copy)}</p>
                    <div class="result-meta-grid">
                      <div class="result-rule">
                        <span>Modelo</span>
                        <strong>{_html(model_name)}</strong>
                      </div>
                      <div class="result-rule">
                        <span>Lectura correcta</span>
                        <strong>Señal de priorización, no veredicto.</strong>
                      </div>
                    </div>
                  </div>
                </aside>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Referencias de ingreso por variable"):
            if feature_stats:
                reference_rows = []
                for feature in _feature_ordered_labels(feature_columns):
                    stats = feature_stats.get(feature, {})
                    reference_rows.append(
                        {
                            "variable": _feature_label(feature),
                            "codigo": feature,
                            "mediana": stats.get("median"),
                            "min": stats.get("min"),
                            "max": stats.get("max"),
                        }
                    )
                st.dataframe(pd.DataFrame(reference_rows), width="stretch", hide_index=True)
            else:
                st.info("No se encontró metadata de estadísticas.")

    with tab_modelo:
        _render_metric_cards(metadata)

    with tab_eda:
        _render_eda_gallery()

    with tab_variables:
        _render_variables_page(metadata, model)

    with tab_sql:
        _render_sql_page()

    with tab_datos:
        _render_data_page(metadata)


if __name__ == "__main__":
    main()
