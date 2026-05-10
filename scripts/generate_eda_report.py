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


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows returned._"

    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
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


def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_PATH}. Run scripts/create_database.py first."
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
    labels = ["No known fraud label", "Known fraud-related"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values, color=["#4c78a8", "#d95f02"])
    ax.set_title("Target Balance")
    ax.set_ylabel("Records")
    ax.set_yscale("log")
    ax.bar_label(bars, labels=[f"{value:,}" for value in counts.values], padding=3)
    ax.text(
        0.5,
        0.94,
        "Log scale used because fraud-labeled records are rare.",
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
    ax.set_title("Fraud Rate by Financial Year")
    ax.set_xlabel("Financial year")
    ax.set_ylabel("Fraud-labeled records (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_rate_by_year.png", dpi=160)
    plt.close(fig)


def save_distribution_chart(df: pd.DataFrame) -> None:
    plot_df = df[KEY_FINANCIAL_COLUMNS].copy()
    for column in KEY_FINANCIAL_COLUMNS:
        plot_df[column] = signed_log10(plot_df[column])

    fig, axes = plt.subplots(1, len(KEY_FINANCIAL_COLUMNS), figsize=(14, 4), sharey=False)
    for ax, column in zip(axes, KEY_FINANCIAL_COLUMNS):
        sns.histplot(plot_df[column].dropna(), bins=40, ax=ax, color="#4c78a8")
        ax.set_title(column)
        ax.set_xlabel("signed log10(1 + abs(value))")
        ax.set_ylabel("Records")
    fig.suptitle("Distribution of Key Financial Variables", y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "key_variable_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_fraud_vs_nonfraud_chart(summary: pd.DataFrame) -> None:
    plot_columns = ["avg_sales", "avg_net_income", "avg_assets", "avg_liabilities", "avg_cash"]
    plot_df = summary[["target_fraud"] + plot_columns].copy()
    plot_df["target_fraud"] = plot_df["target_fraud"].map(
        {0: "No known fraud label", 1: "Known fraud-related"}
    )
    melted = plot_df.melt(
        id_vars="target_fraud", var_name="metric", value_name="average_value"
    )
    melted["signed_log_average"] = signed_log10(melted["average_value"])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=melted,
        x="metric",
        y="signed_log_average",
        hue="target_fraud",
        ax=ax,
        palette=["#4c78a8", "#d95f02"],
    )
    ax.set_title("Fraud vs Non-Fraud Average Financial Values")
    ax.set_xlabel("")
    ax.set_ylabel("signed log10(1 + abs(average value))")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_vs_nonfraud_averages.png", dpi=160)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    corr_df = pd.DataFrame()
    corr_df["Financial_Year"] = df["Financial_Year_Number"]
    for column in FEATURE_COLUMNS:
        if column == "Financial_Year":
            continue
        corr_df[column] = df[column]

    corr = corr_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        linewidths=0.4,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap for the 12 Selected Features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=160)
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

    ratio_summary = (
        df.assign(
            receivables_to_sales=df["rect"] / df["sale"].replace(0, np.nan),
            liabilities_to_assets=df["lt"] / df["at"].replace(0, np.nan),
        )
        .groupby("target_fraud")
        .agg(
            records=("target_fraud", "size"),
            avg_receivables_to_sales_ratio=("receivables_to_sales", "mean"),
            avg_liabilities_to_assets_ratio=("liabilities_to_assets", "mean"),
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


def write_report(df: pd.DataFrame, summaries: dict[str, pd.DataFrame], corr: pd.DataFrame) -> None:
    target_counts = summaries["target_counts"]
    fraud_by_year = summaries["fraud_by_year"]
    averages = summaries["averages"]
    ratio_summary = summaries["ratio_summary"]

    fraud_rows = int(target_counts.loc[target_counts["target_fraud"] == 1, "records"].iloc[0])
    non_fraud_rows = int(target_counts.loc[target_counts["target_fraud"] == 0, "records"].iloc[0])
    fraud_percent = float(
        target_counts.loc[target_counts["target_fraud"] == 1, "percent"].iloc[0]
    )
    min_year = int(df["Financial_Year_Number"].min())
    max_year = int(df["Financial_Year_Number"].max())
    highest_year = fraud_by_year.sort_values("fraud_rate_percent", ascending=False).iloc[0]
    fraud_ratio = ratio_summary.loc[
        ratio_summary["target_fraud"] == 1, "avg_receivables_to_sales_ratio"
    ].iloc[0]
    non_fraud_ratio = ratio_summary.loc[
        ratio_summary["target_fraud"] == 0, "avg_receivables_to_sales_ratio"
    ].iloc[0]

    original_columns = len(df.columns) - 2  # target_fraud and Financial_Year_Number are helpers.

    lines = [
        "# Focused EDA: Financial Fraud Risk Detection",
        "",
        "This report summarizes the focused exploratory analysis for the first-version fraud-risk prioritization project.",
        "",
        "Important wording: `target_fraud = 1` means a record is linked to a known fraud-related case through `AAER_ID`. `target_fraud = 0` means no known fraud label is recorded in this dataset; it is not proof that the record was honest.",
        "",
        "## 1. Dataset Overview",
        "",
        f"- Records: `{len(df):,}`",
        f"- Original columns: `{original_columns:,}`",
        "- Generated helper columns for this report: `target_fraud`, `Financial_Year_Number`",
        f"- Financial years covered: `{min_year}` to `{max_year}`",
        f"- Official modeling feature count for v1: `{len(FEATURE_COLUMNS)}`",
        "",
        "Each row represents a company-year financial record. The project uses the 12 selected financial features because they are easier to explain in a course presentation and connect directly to financial statements.",
        "",
        "Selected features:",
        "",
        ", ".join(f"`{column}`" for column in FEATURE_COLUMNS),
        "",
        "## 2. Target Balance",
        "",
        "![Target balance](figures/target_balance.png)",
        "",
        dataframe_to_markdown(target_counts),
        "",
        f"Only `{fraud_rows:,}` records are fraud-labeled, compared with `{non_fraud_rows:,}` records with no known fraud label. That is `{fraud_percent:.4f}%` of the dataset, so this is a highly imbalanced classification problem. This is why later modeling should not rely on accuracy alone.",
        "",
        "## 3. Fraud Rate by Year",
        "",
        "![Fraud rate by year](figures/fraud_rate_by_year.png)",
        "",
        dataframe_to_markdown(
            fraud_by_year[
                ["Financial_Year", "total_records", "fraud_cases", "fraud_rate_percent"]
            ]
        ),
        "",
        f"The highest fraud-label rate appears in `{highest_year['Financial_Year']}` at `{highest_year['fraud_rate_percent']:.4f}%`. The year-level pattern helps show that fraud-labeled records are not evenly distributed over time.",
        "",
        "## 4. Distribution of Key Financial Variables",
        "",
        "![Distribution of key financial variables](figures/key_variable_distributions.png)",
        "",
        "The variables `sale`, `ni`, `at`, `lt`, and `che` have very wide ranges and visible outliers. The chart uses a signed log transform, `sign(value) * log10(1 + abs(value))`, so very large positive and negative values can fit into a readable presentation chart without deleting records.",
        "",
        "In simple terms, this section shows that financial statement data is not evenly shaped. A few very large companies or unusual records can dominate raw-scale charts, so presentation-safe scaling is necessary.",
        "",
        "## 5. Fraud vs Non-Fraud Comparison",
        "",
        "![Fraud vs non-fraud averages](figures/fraud_vs_nonfraud_averages.png)",
        "",
        dataframe_to_markdown(averages),
        "",
        "Additional ratio comparison:",
        "",
        dataframe_to_markdown(ratio_summary),
        "",
        f"The most useful presentation finding is the receivables-to-sales ratio: fraud-labeled records average `{fraud_ratio:.4f}`, while records with no known fraud label average `{non_fraud_ratio:.4f}`. This does not prove fraud, but it supports the business idea that unusually high receivables relative to sales can be a warning signal around revenue quality.",
        "",
        "## 6. Feature Relationship Check",
        "",
        "![Feature correlation heatmap](figures/feature_correlation_heatmap.png)",
        "",
        "The heatmap checks relationships among the 12 selected features. Correlation can reveal redundancy or strong relationships between inputs, but it is not causation and it is not proof of fraud.",
        "",
        "Correlation matrix:",
        "",
        dataframe_to_markdown(corr.round(3).reset_index().rename(columns={"index": "feature"})),
        "",
        "## Phase 2 Takeaway",
        "",
        "The data is large enough for modeling, but the positive fraud label is extremely rare. The EDA supports the project direction: this should be framed as ranking records for review, not proving fraud. The receivables-to-sales difference gives a simple financial signal that can be explained clearly in the final presentation.",
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
    save_fraud_vs_nonfraud_chart(summaries["averages"])
    corr = save_correlation_heatmap(df)

    write_report(df, summaries, corr)


def main() -> None:
    generate_report()
    print(f"Wrote EDA report: {REPORT_PATH}")
    print(f"Wrote figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
