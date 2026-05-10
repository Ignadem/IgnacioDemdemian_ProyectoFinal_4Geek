from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from validate_data import DEFAULT_DATA_PATH, REQUIRED_COLUMNS, validate_data


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = BASE_DIR / "data" / "fraud_financial.db"
TABLE_NAME = "financial_records"


def create_database(
    csv_path: Path = DEFAULT_DATA_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, object]:
    validation = validate_data(csv_path)

    df = pd.read_csv(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        df.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_financial_year '
            f'ON {TABLE_NAME} ("Financial_Year")'
        )
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_aaer_id '
            f'ON {TABLE_NAME} ("AAER_ID")'
        )
        row_count = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({TABLE_NAME})")]

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(columns))
    if missing_columns:
        raise ValueError(f"Database table is missing required columns: {missing_columns}")

    if row_count != validation["rows"]:
        raise ValueError(
            f"Database row count mismatch. Expected {validation['rows']}, got {row_count}."
        )

    return {
        "database": db_path,
        "table": TABLE_NAME,
        "rows": row_count,
        "columns": len(columns),
    }


def main() -> None:
    result = create_database()
    print("SQLite database created")
    print(f"Database: {result['database']}")
    print(f"Table: {result['table']}")
    print(f"Rows loaded: {result['rows']}")
    print(f"Columns loaded: {result['columns']}")


if __name__ == "__main__":
    main()
