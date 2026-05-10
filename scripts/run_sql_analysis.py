from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "fraud_financial.db"
REPORT_PATH = BASE_DIR / "reports" / "sql" / "phase1_sql_results.md"

QUERIES = [
    (
        "Total Records",
        """
        SELECT
          COUNT(*) AS total_records
        FROM financial_records;
        """,
    ),
    (
        "Fraud vs Non-Fraud Count",
        """
        SELECT
          CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS target_fraud,
          COUNT(*) AS records
        FROM financial_records
        GROUP BY target_fraud
        ORDER BY target_fraud;
        """,
    ),
    (
        "Fraud Rate by Financial Year",
        """
        SELECT
          Financial_Year,
          COUNT(*) AS total_records,
          SUM(CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END) AS fraud_cases,
          ROUND(AVG(CASE WHEN AAER_ID IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 4) AS fraud_rate_percent
        FROM financial_records
        GROUP BY Financial_Year
        ORDER BY Financial_Year;
        """,
    ),
    (
        "Average Key Financial Values by Fraud Label",
        """
        SELECT
          CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS target_fraud,
          COUNT(*) AS records,
          ROUND(AVG(sale), 2) AS avg_sales,
          ROUND(AVG(ni), 2) AS avg_net_income,
          ROUND(AVG(at), 2) AS avg_assets,
          ROUND(AVG(lt), 2) AS avg_liabilities,
          ROUND(AVG(che), 2) AS avg_cash
        FROM financial_records
        GROUP BY target_fraud
        ORDER BY target_fraud;
        """,
    ),
    (
        "Receivables-to-Sales Ratio by Fraud Label",
        """
        SELECT
          CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS target_fraud,
          COUNT(*) AS records,
          ROUND(AVG(rect), 2) AS avg_receivables,
          ROUND(AVG(sale), 2) AS avg_sales,
          ROUND(AVG(rect / NULLIF(sale, 0)), 4) AS avg_receivables_to_sales_ratio
        FROM financial_records
        GROUP BY target_fraud
        ORDER BY target_fraud;
        """,
    ),
    (
        "Liabilities-to-Assets Ratio by Fraud Label",
        """
        SELECT
          CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS target_fraud,
          COUNT(*) AS records,
          ROUND(AVG(lt), 2) AS avg_liabilities,
          ROUND(AVG(at), 2) AS avg_assets,
          ROUND(AVG(lt / NULLIF(at, 0)), 4) AS avg_liabilities_to_assets_ratio
        FROM financial_records
        GROUP BY target_fraud
        ORDER BY target_fraud;
        """,
    ),
]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows returned._"

    headers = [str(column) for column in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(value) for value in row.tolist()])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_queries() -> list[tuple[str, pd.DataFrame]]:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_PATH}. Run scripts/create_database.py first."
        )

    results = []
    with sqlite3.connect(DB_PATH) as connection:
        for title, query in QUERIES:
            results.append((title, pd.read_sql_query(query, connection)))
    return results


def write_report(results: list[tuple[str, pd.DataFrame]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 SQL Results",
        "",
        "Generated from `data/fraud_financial.db` using `scripts/run_sql_analysis.py`.",
        "",
    ]

    for title, df in results:
        lines.extend([f"## {title}", "", dataframe_to_markdown(df), ""])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = run_queries()
    write_report(results)
    print(f"Wrote SQL analysis report: {REPORT_PATH}")
    for title, df in results:
        print(f"- {title}: {len(df)} row(s)")


if __name__ == "__main__":
    main()
