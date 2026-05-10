# Data Source

## Official Dataset

This project uses the Kaggle dataset **New Fraud Financial Dataset**.

- Kaggle URL: https://www.kaggle.com/datasets/abdulmalekalsalemi/new-fraud-financial-dataset
- Kaggle dataset slug: `abdulmalekalsalemi/new-fraud-financial-dataset`
- Local project file: `data/Cleaned_data_1995_2018.csv`

The dataset contains historical company financial records from fiscal years 1995 through 2018. Each row represents a company-year financial record.

## Reproducible Acquisition

The official acquisition method for this project is the Kaggle API/CLI:

```bash
kaggle datasets download -d abdulmalekalsalemi/new-fraud-financial-dataset
```

Then unzip the downloaded archive and place `Cleaned_data_1995_2018.csv` at:

```text
data/Cleaned_data_1995_2018.csv
```

This requires Kaggle credentials configured locally. If the CSV already exists locally, the project validation script can verify that it matches the expected dataset copy.

## Target Label

The original dataset includes the column `AAER_ID`.

`AAER_ID` means **Accounting and Auditing Enforcement Release ID**. In simple terms, it is an identifier for a known enforcement case related to accounting or auditing problems. When a row has an `AAER_ID`, the dataset is marking that company-year record as connected to a known fraud-related case.

For this project, we use `AAER_ID` as the source for the fraud label. The model is not trying to prove fraud by itself; it is learning from records that already have this enforcement-case identifier.

For this project, the modeling target is created as:

```text
target_fraud = 1 if AAER_ID exists
target_fraud = 0 if AAER_ID is missing
```

Interpretation:

- `target_fraud = 1`: the record is linked to a known fraud-related enforcement case.
- `target_fraud = 0`: no known fraud label is recorded in this dataset.

Important limitation: `target_fraud = 0` does not prove that a company-year record was honest. It only means the dataset does not contain a known fraud label for that row.

## Local Validation

Validate the local CSV with:

```bash
/Users/igna/entorno/bin/python scripts/validate_data.py
```

The validation checks:

- file exists;
- expected row count;
- expected SHA-256 checksum;
- required columns for Phase 1 and modeling.
