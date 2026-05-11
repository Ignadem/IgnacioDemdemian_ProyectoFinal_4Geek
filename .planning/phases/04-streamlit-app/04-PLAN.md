---
phase: 04-streamlit-app
plan: 01
type: execute
wave: 1
files_modified:
  - app.py
requirements:
  - APP-01
  - APP-02
  - APP-03
  - APP-04
  - APP-05
---

## Objective

Create a Streamlit app that loads the saved model and metadata to score one manually entered company-year record without retraining.

## Tasks

1. Build `app.py` UI with numeric inputs for all 12 features.
2. Load `models/fraud_risk_model.joblib` and `models/model_metadata.json`.
3. Compute fraud-risk score with `predict_proba` and map to Low/Medium/High categories:
   - Low < 0.10
   - Medium >= 0.10 and < 0.30
   - High >= 0.30
4. Show warning/disclaimer that output is review priority, not proof of fraud.

## Verification

- `python -m py_compile app.py`
- Check imports for missing files:
  - `/Models and metadata paths exist`
- Visual smoke test by launching `streamlit run app.py` and confirming a score appears for a filled record.
