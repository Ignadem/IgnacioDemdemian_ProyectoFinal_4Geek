from __future__ import annotations

import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DB_PATH = BASE_DIR / "data" / "fraud_financial.db"
REPORT_DIR = BASE_DIR / "reports" / "eda"
FIGURES_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "focused_eda.md"
TABLE_NAME = "financial_records"

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

KEY_FINANCIAL_COLUMNS = ["sale", "ni", "at", "lt", "che"]

KEY_FINANCIAL_LABELS = {
    "sale": "Ventas",
    "ni": "Utilidad neta",
    "at": "Activos totales",
    "lt": "Pasivos totales",
    "che": "Efectivo",
}

FEATURE_LABELS = {
    "Financial_Year": "Año fiscal",
    "sale": "Ventas",
    "ni": "Utilidad neta",
    "at": "Activos totales",
    "lt": "Pasivos totales",
    "che": "Efectivo",
    "rect": "Cuentas por cobrar",
    "invt": "Inventario",
    "cogs": "Costo de ventas",
    "txt": "Impuestos",
    "xint": "Gasto intereses",
    "prcc_f": "Precio mercado",
}


def dataframe_to_markdown(df: pd.DataFrame, column_rename: dict[str, str] | None = None) -> str:
    if df.empty:
        return "_No se devolvieron filas._"

    working_df = df.rename(columns=column_rename) if column_rename else df

    headers = [str(column) for column in working_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in working_df.iterrows():
        lines.append("| " + " | ".join(format_value(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    return str(value)


def signed_log10(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.sign(values) * np.log10(1 + np.abs(values))


def build_receivables_to_sales_frame(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df["rect"] / df["sale"].replace(0, np.nan)
    plot_df = pd.DataFrame(
        {
            "target_fraud": df["target_fraud"],
            "sale": df["sale"],
            "receivables_to_sales": ratio,
        }
    ).replace([np.inf, -np.inf], np.nan)

    plot_df = plot_df[
        (plot_df["sale"] > 0)
        & plot_df["receivables_to_sales"].notna()
        & (plot_df["receivables_to_sales"] >= 0)
    ].copy()
    plot_df["etiqueta"] = plot_df["target_fraud"].map(
        {0: "Sin etiqueta conocida", 1: "Con AAER_ID"}
    )
    return plot_df


def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la base SQLite en {DB_PATH}. Ejecutá primero scripts/create_database.py."
        )

    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", connection)

    df["target_fraud"] = df["AAER_ID"].notna().astype(int)
    df["Financial_Year_Number"] = (
        df["Financial_Year"].astype(str).str.replace("FY", "", regex=False).astype(int)
    )

    for column in FEATURE_COLUMNS:
        if column == "Financial_Year":
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def save_target_balance_chart(df: pd.DataFrame) -> None:
    counts = df["target_fraud"].value_counts().sort_index()
    labels = ["Sin etiqueta de fraude conocida", "Con etiqueta de fraude relacionada"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values, color=["#4c78a8", "#d95f02"])
    ax.set_title("Balance de etiqueta objetivo")
    ax.set_ylabel("Registros")
    ax.set_yscale("log")
    ax.bar_label(bars, labels=[f"{value:,}" for value in counts.values], padding=3)
    ax.text(
        0.5,
        0.94,
        "Se usa escala logarítmica porque los registros con etiqueta de fraude son raros.",
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_balance.png", dpi=160)
    plt.close(fig)


def save_fraud_rate_by_year_chart(fraud_by_year: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        fraud_by_year["Financial_Year_Number"],
        fraud_by_year["fraud_rate_percent"],
        marker="o",
        color="#d95f02",
    )
    ax.set_title("Tasa de fraude por año fiscal")
    ax.set_xlabel("Año fiscal")
    ax.set_ylabel("Registros con etiqueta de fraude (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_rate_by_year.png", dpi=160)
    plt.close(fig)


def save_distribution_chart(df: pd.DataFrame) -> None:
    plot_df = df[KEY_FINANCIAL_COLUMNS].copy()
    for column in KEY_FINANCIAL_COLUMNS:
        plot_df[column] = signed_log10(plot_df[column])

    fig, axes = plt.subplots(1, len(KEY_FINANCIAL_COLUMNS), figsize=(18, 5.2), sharey=False)
    for ax, column in zip(axes, KEY_FINANCIAL_COLUMNS):
        sns.histplot(plot_df[column].dropna(), bins=40, ax=ax, color="#4c78a8")
        ax.set_title(KEY_FINANCIAL_LABELS[column], fontsize=12)
        ax.set_xlabel("escala logarítmica con signo", fontsize=10)
        ax.set_ylabel("Registros", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
    fig.suptitle("Distribución de variables financieras clave", fontsize=16, y=1.04)
    fig.tight_layout(pad=1.5, w_pad=2.2)
    fig.savefig(FIGURES_DIR / "key_variable_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_receivables_to_sales_boxplot(df: pd.DataFrame) -> None:
    plot_df = build_receivables_to_sales_frame(df)
    labels = ["Sin etiqueta conocida", "Con AAER_ID"]
    groups = [
        plot_df.loc[plot_df["target_fraud"] == 0, "receivables_to_sales"],
        plot_df.loc[plot_df["target_fraud"] == 1, "receivables_to_sales"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.45,
    )
    for patch, color in zip(box["boxes"], ["#4c78a8", "#d95f02"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(2)

    upper_limit = float(plot_df["receivables_to_sales"].quantile(0.95))
    ax.set_ylim(0, max(upper_limit * 1.15, 0.2))
    ax.set_title("Cuentas por cobrar / ventas por etiqueta")
    ax.set_xlabel("")
    ax.set_ylabel("rect / sale")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "receivables_to_sales_boxplot.png", dpi=160)
    plt.close(fig)


def save_receivables_to_sales_bar_chart(ratio_summary: pd.DataFrame) -> None:
    summary = ratio_summary.sort_values("target_fraud").copy()
    labels = summary["target_fraud"].map({0: "Sin etiqueta conocida", 1: "Con AAER_ID"})
    x = np.arange(len(summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    mean_bars = ax.bar(
        x - width / 2,
        summary["mean_receivables_to_sales_ratio"],
        width,
        label="Media",
        color="#4c78a8",
    )
    median_bars = ax.bar(
        x + width / 2,
        summary["median_receivables_to_sales_ratio"],
        width,
        label="Mediana",
        color="#d95f02",
    )
    ax.bar_label(mean_bars, labels=[f"{value:.2f}" for value in summary["mean_receivables_to_sales_ratio"]], padding=3)
    ax.bar_label(median_bars, labels=[f"{value:.2f}" for value in summary["median_receivables_to_sales_ratio"]], padding=3)
    ax.set_title("Resumen de cuentas por cobrar / ventas")
    ax.set_xlabel("")
    ax.set_ylabel("rect / sale")
    ax.set_xticks(x, labels)
    ax.legend(title="")
    ax.grid(axis="y", alpha=0.25)
    tallest_bar = float(
        summary[
            ["mean_receivables_to_sales_ratio", "median_receivables_to_sales_ratio"]
        ].to_numpy().max()
    )
    ax.set_ylim(0, max(tallest_bar * 1.18, 0.25))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "receivables_to_sales_bars.png", dpi=160)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    corr_df = pd.DataFrame()
    corr_df[FEATURE_LABELS["Financial_Year"]] = df["Financial_Year_Number"]
    for column in FEATURE_COLUMNS:
        if column == "Financial_Year":
            continue
        corr_df[FEATURE_LABELS[column]] = df[column]

    corr = corr_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        linewidths=0.4,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Mapa de correlación entre variables financieras", fontsize=15)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=180)
    plt.close(fig)
    return corr


def build_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    target_counts = (
        df["target_fraud"]
        .value_counts()
        .sort_index()
        .rename_axis("target_fraud")
        .reset_index(name="records")
    )
    target_counts["percent"] = (target_counts["records"] / len(df) * 100).round(4)

    fraud_by_year = (
        df.groupby(["Financial_Year_Number", "Financial_Year"])["target_fraud"]
        .agg(total_records="count", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
        .sort_values("Financial_Year_Number")
    )
    fraud_by_year["fraud_rate_percent"] = (fraud_by_year["fraud_rate"] * 100).round(4)
    fraud_by_year = fraud_by_year.drop(columns=["fraud_rate"])

    averages = (
        df.groupby("target_fraud")[KEY_FINANCIAL_COLUMNS]
        .mean()
        .round(2)
        .reset_index()
        .rename(
            columns={
                "sale": "avg_sales",
                "ni": "avg_net_income",
                "at": "avg_assets",
                "lt": "avg_liabilities",
                "che": "avg_cash",
            }
        )
    )
    averages.insert(1, "records", df.groupby("target_fraud").size().values)

    receivables_to_sales = build_receivables_to_sales_frame(df)
    ratio_summary = (
        receivables_to_sales
        .groupby("target_fraud")
        .agg(
            valid_ratio_records=("receivables_to_sales", "size"),
            mean_receivables_to_sales_ratio=("receivables_to_sales", "mean"),
            median_receivables_to_sales_ratio=("receivables_to_sales", "median"),
            p75_receivables_to_sales_ratio=("receivables_to_sales", lambda value: value.quantile(0.75)),
        )
        .round(4)
        .reset_index()
    )

    return {
        "target_counts": target_counts,
        "fraud_by_year": fraud_by_year,
        "averages": averages,
        "ratio_summary": ratio_summary,
    }


def _safe_lookup(df: pd.DataFrame, column: str, target_value: int, default_value: float) -> float:
    matching = df.loc[df["target_fraud"] == target_value, column]
    if matching.empty:
        return float(default_value)
    return float(matching.iloc[0])


def write_report(df: pd.DataFrame, summaries: dict[str, pd.DataFrame], corr: pd.DataFrame) -> None:
    target_counts = summaries["target_counts"]
    fraud_by_year = summaries["fraud_by_year"]
    ratio_summary = summaries["ratio_summary"]

    fraud_rows = int(
        _safe_lookup(
            target_counts,
            "records",
            1,
            0,
        )
    )
    non_fraud_rows = int(
        _safe_lookup(
            target_counts,
            "records",
            0,
            0,
        )
    )
    fraud_percent = float(
        _safe_lookup(
            target_counts,
            "percent",
            1,
            0.0,
        )
    )
    min_year = int(df["Financial_Year_Number"].min())
    max_year = int(df["Financial_Year_Number"].max())
    if fraud_by_year.empty:
        highest_year_label = "N/A"
        highest_year_rate = 0.0
    else:
        highest_year = fraud_by_year.sort_values("fraud_rate_percent", ascending=False).iloc[0]
        highest_year_label = str(highest_year["Financial_Year"])
        highest_year_rate = float(highest_year["fraud_rate_percent"])

    fraud_ratio = _safe_lookup(
        ratio_summary,
        "mean_receivables_to_sales_ratio",
        1,
        0.0,
    )
    non_fraud_ratio = _safe_lookup(
        ratio_summary,
        "mean_receivables_to_sales_ratio",
        0,
        0.0,
    )
    fraud_ratio_median = _safe_lookup(
        ratio_summary,
        "median_receivables_to_sales_ratio",
        1,
        0.0,
    )
    non_fraud_ratio_median = _safe_lookup(
        ratio_summary,
        "median_receivables_to_sales_ratio",
        0,
        0.0,
    )

    original_columns = len(df.columns) - 2  # target_fraud and Financial_Year_Number are helpers.

    lines = [
        "# EDA enfocado: detección de riesgo de fraude financiero",
        "",
        "Este reporte resume el análisis exploratorio enfocado para el proyecto de priorización de riesgo de fraude (versión 1).",
        "",
        "Mensaje clave: `target_fraud = 1` indica un caso vinculado a fraude conocido por `AAER_ID`.",
        "`target_fraud = 0` significa que este dataset no trae etiqueta de fraude conocida; no demuestra que el registro sea limpio.",
        "",
        "## 1. Resumen del dataset",
        "",
        f"- Registros: `{len(df):,}`",
        f"- Columnas originales: `{original_columns:,}`",
        "- Columnas auxiliares generadas para este reporte: `target_fraud`, `Financial_Year_Number`",
        f"- Años fiscales cubiertos: `{min_year}` a `{max_year}`",
        f"- Variables oficiales de modelado (v1): `{len(FEATURE_COLUMNS)}`",
        "",
        "Cada fila representa un registro empresa-año de estados financieros. Se usan 12 variables para que sea fácil de explicar en presentación de curso.",
        "",
        "Variables usadas:",
        "",
        ", ".join(f"`{column}`" for column in FEATURE_COLUMNS),
        "",
        "## 2. Balance de etiqueta objetivo",
        "",
        "![Balance de etiqueta objetivo](figures/target_balance.png)",
        "",
        dataframe_to_markdown(
            target_counts,
            {"target_fraud": "etiqueta_fraude", "records": "registros", "percent": "porcentaje"},
        ),
        "",
        f"Solo hay `{fraud_rows:,}` registros con etiqueta de fraude y `{non_fraud_rows:,}` sin etiqueta conocida.",
        f"Eso representa `{fraud_percent:.4f}%` del dataset. El problema está fuertemente desbalanceado y por eso no se debe usar accuracy como métrica principal.",
        "",
        "## 3. Tasa de fraude por año",
        "",
        "![Tasa de fraude por año](figures/fraud_rate_by_year.png)",
        "",
        dataframe_to_markdown(
            fraud_by_year[
                ["Financial_Year", "total_records", "fraud_cases", "fraud_rate_percent"]
            ],
            {
                "Financial_Year": "ejercicio_fiscal",
                "total_records": "total_registros",
                "fraud_cases": "casos_fraude",
                "fraud_rate_percent": "tasa_fraude_porcentaje",
            },
        ),
        "",
        f"La mayor tasa de etiqueta de fraude aparece en `{highest_year_label}` con `{highest_year_rate:.4f}%`.",
        "Esto muestra que los casos conocidos de fraude no están distribuidos de forma uniforme en el tiempo.",
        "El pico alrededor de 2000-2002 también coincide con un período de fuerte escrutinio contable en Estados Unidos: caída de la burbuja puntocom, casos como Enron, WorldCom y Tyco, y aprobación de Sarbanes-Oxley en 2002.",
        "Por eso conviene interpretarlo como concentración de registros con AAER_ID conocidos en ese contexto, no como una medición de todo el fraude real de la economía.",
        "",
        "## 4. Distribución de variables financieras clave",
        "",
        "![Distribución de variables financieras clave](figures/key_variable_distributions.png)",
        "",
        "Las variables `sale`, `ni`, `at`, `lt` y `che` tienen rangos amplios y outliers visibles.",
        "Se aplica transformación `signo(log10(1 + abs(valor)))` para graficarlas sin eliminar registros extremos.",
        "",
        "En términos simples, algunos registros muy grandes pueden dominar una escala lineal, por eso se usa escalado seguro para presentación.",
        "",
        "## 5. Cuentas por cobrar sobre ventas",
        "",
        "![Boxplot de cuentas por cobrar sobre ventas](figures/receivables_to_sales_boxplot.png)",
        "",
        "![Barras de cuentas por cobrar sobre ventas](figures/receivables_to_sales_bars.png)",
        "",
        "Resumen del ratio `rect / sale`:",
        "",
        dataframe_to_markdown(
            ratio_summary,
            {
                "target_fraud": "etiqueta_fraude",
                "valid_ratio_records": "registros_validos",
                "mean_receivables_to_sales_ratio": "media_rect_sale",
                "median_receivables_to_sales_ratio": "mediana_rect_sale",
                "p75_receivables_to_sales_ratio": "percentil_75_rect_sale",
            },
        ),
        "",
        f"Hallazgo útil: la media de `rect / sale` es `{fraud_ratio:.4f}` en casos etiquetados y `{non_fraud_ratio:.4f}` en no etiquetados.",
        f"La mediana también es mayor en casos etiquetados: `{fraud_ratio_median:.4f}` frente a `{non_fraud_ratio_median:.4f}`.",
        "No prueba fraude, pero sí da una señal de alerta operativa: ventas registradas que todavía no se transformaron en cobros pueden merecer revisión.",
        "",
        "## 6. Revisión de relación entre variables",
        "",
        "![Mapa de calor de correlación](figures/feature_correlation_heatmap.png)",
        "",
        "El mapa de calor muestra relaciones entre las 12 variables usadas. Puede indicar redundancia o relaciones fuertes, pero no significa causalidad ni prueba de fraude.",
        "",
        "Matriz de correlación:",
        "",
        dataframe_to_markdown(corr.round(3).reset_index().rename(columns={"index": "variable"})),
        "",
        "## Conclusión de la fase",
        "",
        "El dataset alcanza tamaño suficiente para modelar, pero la etiqueta positiva es muy rara.",
        "La EDA respalda enmarcar el proyecto como priorización de revisión, no como prueba de fraude.",
        "La diferencia en cuentas por cobrar/ventas aporta una señal financiera clara para la presentación final.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def generate_report() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    summaries = build_summaries(df)

    save_target_balance_chart(df)
    save_fraud_rate_by_year_chart(summaries["fraud_by_year"])
    save_distribution_chart(df)
    save_receivables_to_sales_boxplot(df)
    save_receivables_to_sales_bar_chart(summaries["ratio_summary"])
    corr = save_correlation_heatmap(df)

    write_report(df, summaries, corr)


def main() -> None:
    generate_report()
    print(f"Reporte EDA generado: {REPORT_PATH}")
    print(f"Figuras generadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
