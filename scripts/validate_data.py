from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "Cleaned_data_1995_2018.csv"
EXPECTED_ROWS = 87_974
EXPECTED_SHA256 = "c279c7d662a286b6b18e2b9d01595b4683f4cc8e427749272626de04ea5d748c"
REQUIRED_COLUMNS = [
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
    "AAER_ID",
]


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data(path: Path = DEFAULT_DATA_PATH) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    checksum = calculate_sha256(path)
    if checksum != EXPECTED_SHA256:
        raise ValueError(
            "Unexpected dataset checksum. "
            f"Expected {EXPECTED_SHA256}, got {checksum}."
        )

    df = pd.read_csv(path)
    row_count = len(df)
    if row_count != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count. Expected {EXPECTED_ROWS}, got {row_count}.")

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    return {
        "path": path,
        "rows": row_count,
        "columns": len(df.columns),
        "sha256": checksum,
        "required_columns": REQUIRED_COLUMNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local fraud dataset CSV.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV path to validate. Defaults to {DEFAULT_DATA_PATH}.",
    )
    args = parser.parse_args()

    result = validate_data(args.path)
    print("Dataset validation passed")
    print(f"Path: {result['path']}")
    print(f"Rows: {result['rows']}")
    print(f"Columns: {result['columns']}")
    print(f"SHA-256: {result['sha256']}")
    print(f"Required columns: {', '.join(result['required_columns'])}")


if __name__ == "__main__":
    main()
