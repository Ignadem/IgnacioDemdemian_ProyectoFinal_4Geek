# Fraud Financial Risk Detection (Machine Learning Course Project)

This project demonstrates a full demo workflow for a **fraud-risk prioritization** system.

It is built to be understandable for non-technical review:

- get a public Kaggle financial dataset,
- store it in SQLite,
- run SQL + focused EDA,
- train and evaluate models with imbalance-aware metrics,
- save a model artifact,
- run a Streamlit demo that scores one company-year record.

The app and model output is a **review-priority signal**, not a legal or audit verdict.

## Project Goal

Predict whether a company-year record is similar to other rows that were linked to known accounting/auditing enforcement cases.

**Important:** `target_fraud = 1` means `AAER_ID` exists in the source dataset.  
`target_fraud = 0` means no known `AAER_ID` label is present in this dataset.

## Data source

- Official Kaggle dataset: `abdulmalekalsalemi/new-fraud-financial-dataset`
- Kaggle URL: https://www.kaggle.com/datasets/abdulmalekalsalemi/new-fraud-financial-dataset
- Local file: `data/Cleaned_data_1995_2018.csv`

See [`docs/data_source.md`](docs/data_source.md) for source and acquisition notes.

## Setup

Recommended environment:

```bash
cd /Users/igna/git/ProjectoFinal_4Geeks
/Users/igna/entorno/bin/python -m pip install -r requirements.txt
```

## Run the project

1. **Validate dataset**

```bash
/Users/igna/entorno/bin/python scripts/validate_data.py
```

2. **Build SQLite and SQL report**

```bash
/Users/igna/entorno/bin/python scripts/create_database.py
/Users/igna/entorno/bin/python scripts/run_sql_analysis.py
```

3. **Run EDA**

```bash
/Users/igna/entorno/bin/python scripts/generate_eda_report.py
```

4. **Train model + save artifacts**

```bash
/Users/igna/entorno/bin/python scripts/train_model.py
```

This creates:

- `models/fraud_risk_model.joblib`
- `models/model_metadata.json`
- `reports/modeling/model_results.md`

5. **Run Streamlit demo**

```bash
/Users/igna/entorno/bin/python -m streamlit run app.py
```

## Modeling summary

Feature set (12):

- `Financial_Year`
- `sale`
- `ni`
- `at`
- `lt`
- `che`
- `rect`
- `invt`
- `cogs`
- `txt`
- `xint`
- `prcc_f`

Baseline model:

- Logistic Regression (imbalance-aware)

Final model:

- Tuned Random Forest, selected by Average Precision (PR AUC)

## Evaluation metrics

- Primary: **Average Precision / PR AUC**
- Supporting: Recall, Precision, F1, ROC AUC, Confusion Matrix

`Accuracy is not used as a primary metric` because positive labels are very rare (~0.6536%).

## App behavior

- Inputs: 12 numeric features
- Output:
  - fraud-risk probability (`0` to `1`)
  - category:
    - Low: score < 0.10
    - Medium: 0.10 to < 0.30
    - High: >= 0.30
- **Output is review priority only.**

## Limitations

- The target uses `AAER_ID` from the dataset, so unlabeled rows are not proof of no fraud.
- One-model, one-horizon benchmark (no time-based drift handling).
- Not a production system: no authentication, audit trail, threshold governance, or batch upload yet.
- Metrics prioritize ranking for investigation, not legal certainty.

## Docs

- [`docs/data_source.md`](docs/data_source.md)
- [`docs/feature_glossary.md`](docs/feature_glossary.md)
- [`docs/presentation_outline.md`](docs/presentation_outline.md)
- [`reports/sql/phase1_sql_results.md`](reports/sql/phase1_sql_results.md)
- [`reports/eda/focused_eda.md`](reports/eda/focused_eda.md)
- [`reports/modeling/model_results.md`](reports/modeling/model_results.md)
