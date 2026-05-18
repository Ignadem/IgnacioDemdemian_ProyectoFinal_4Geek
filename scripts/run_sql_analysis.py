from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "fraud_financial.db"
REPORT_PATH = BASE_DIR / "reports" / "sql" / "phase1_sql_results.md"

QUERIES = [
    (
        "Total de registros",
        """
        SELECT
          COUNT(*) AS total_registros
        FROM financial_records;
        """,
    ),
    (
        "Conteo de fraude vs no fraude",
        """
        SELECT
          CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS etiqueta_fraude,
          COUNT(*) AS registros
        FROM financial_records
        GROUP BY etiqueta_fraude
        ORDER BY etiqueta_fraude;
        """,
    ),
    (
        "Tasa de fraude por año fiscal",
        """
        SELECT
          Financial_Year AS ejercicio_fiscal,
          COUNT(*) AS total_registros,
          SUM(CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END) AS casos_fraude,
          ROUND(AVG(CASE WHEN AAER_ID IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 4) AS tasa_fraude_porcentaje
        FROM financial_records
        GROUP BY Financial_Year
        ORDER BY Financial_Year;
        """,
    ),
    (
        "Valores financieros clave promedio por etiqueta de fraude",
        """
        SELECT
          CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS etiqueta_fraude,
          COUNT(*) AS registros,
          ROUND(AVG(sale), 2) AS promedio_ventas,
          ROUND(AVG(ni), 2) AS promedio_ingreso_neto,
          ROUND(AVG(at), 2) AS promedio_activos,
          ROUND(AVG(lt), 2) AS promedio_pasivos,
          ROUND(AVG(che), 2) AS promedio_efectivo
        FROM financial_records
        GROUP BY etiqueta_fraude
        ORDER BY etiqueta_fraude;
        """,
    ),
    (
        "Ratio cuentas por cobrar / ventas por etiqueta de fraude",
        """
        SELECT
          CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS etiqueta_fraude,
          COUNT(*) AS registros,
          ROUND(AVG(rect), 2) AS promedio_cuentas_por_cobrar,
          ROUND(AVG(sale), 2) AS promedio_ventas,
          ROUND(AVG(rect / NULLIF(sale, 0)), 4) AS razon_cuentas_por_cobrar_ventas
        FROM financial_records
        GROUP BY etiqueta_fraude
        ORDER BY etiqueta_fraude;
        """,
    ),
    (
        "Ratio pasivos / activos por etiqueta de fraude",
        """
        SELECT
          CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS etiqueta_fraude,
          COUNT(*) AS registros,
          ROUND(AVG(lt), 2) AS promedio_pasivos,
          ROUND(AVG(at), 2) AS promedio_activos,
          ROUND(AVG(lt / NULLIF(at, 0)), 4) AS razon_pasivos_activos
        FROM financial_records
        GROUP BY etiqueta_fraude
        ORDER BY etiqueta_fraude;
        """,
    ),
]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No se devolvieron filas._"

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


def clean_query(query: str) -> str:
    return textwrap.dedent(query).strip()


def run_queries() -> list[tuple[str, str, pd.DataFrame]]:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la base SQLite: {DB_PATH}. Ejecutá primero scripts/create_database.py."
        )

    results = []
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA temp_store = MEMORY;")
        for title, query in QUERIES:
            results.append((title, clean_query(query), pd.read_sql_query(query, connection)))
    return results


def write_report(results: list[tuple[str, str, pd.DataFrame]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resultados SQL de Fase 1",
        "",
        "Generado desde `data/fraud_financial.db` usando `scripts/run_sql_analysis.py`.",
        "",
    ]

    for title, query, df in results:
        lines.extend(
            [
                f"## {title}",
                "",
                "**Consulta:**",
                "",
                "```sql",
                query,
                "```",
                "",
                "**Resultado:**",
                "",
                dataframe_to_markdown(df),
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = run_queries()
    write_report(results)
    print(f"Reporte SQL generado: {REPORT_PATH}")
    for title, _, df in results:
        print(f"- {title}: {len(df)} fila(s)")


if __name__ == "__main__":
    main()
