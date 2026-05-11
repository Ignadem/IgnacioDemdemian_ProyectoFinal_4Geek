# Phase 4: Streamlit App Context

## Phase Boundary

Build a review-priority demo app that uses the persisted phase-3 model artifacts and does not retrain.

## Requirements

- APP-01: Manual inputs for the 12 approved features.
- APP-02: Load `models/fraud_risk_model.joblib`.
- APP-03: Show fraud-risk score as probability.
- APP-04: Show Low / Medium / High category with 10% and 30% thresholds.
- APP-05: Keep disclaimers around "review priority, not confirmed fraud."

## Inputs

- `models/fraud_risk_model.joblib`
- `models/model_metadata.json`

## Outputs

- `app.py`
