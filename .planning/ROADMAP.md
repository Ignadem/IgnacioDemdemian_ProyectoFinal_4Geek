# Roadmap: Financial Fraud Risk Detection

**Created:** 2026-05-10
**Mode:** lightweight, interactive, one phase at a time

## Phase 1: Data Source, Ingestion, SQLite, And SQL Analysis

**Goal:** Establish a reproducible data source story and satisfy the database/SQL requirements.

**Covers:** DATA-01, DATA-02, DATA-03, SQL-01, SQL-02, SQL-03, SQL-04, SQL-05, SQL-06, SQL-07

**Deliverables:**
- Kaggle dataset source documented.
- Ingestion script or documented CLI workflow.
- SQLite database generated from the CSV.
- Six SQL analyses implemented and captured for notebook/docs.

**Success Criteria:**
- A clean command can create or refresh the SQLite database from the CSV.
- SQL outputs confirm row counts, fraud imbalance, year patterns, average financial values, receivables-to-sales ratio, and liabilities-to-assets ratio.

## Phase 2: Focused EDA

**Goal:** Refine the exploratory analysis into a clear, presentation-ready story.

**Covers:** EDA-01, EDA-02, EDA-03, EDA-04, EDA-05, EDA-06

**Deliverables:**
- Notebook or report sections for the six agreed EDA areas.
- Charts/tables that are easy to explain.
- Plain-language interpretation of each finding.

**Success Criteria:**
- EDA clearly explains what the data is, how rare fraud labels are, and why the selected features are relevant.

## Phase 3: Modeling And Model Persistence

**Goal:** Train, evaluate, select, and save the final model.

**Covers:** MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06, MODEL-07

**Deliverables:**
- Logistic Regression baseline.
- Tuned Random Forest final model.
- Metrics table and interpretation.
- Saved `models/fraud_risk_model.joblib`.
- Saved `models/model_metadata.json`.

**Success Criteria:**
- Final model can be loaded without retraining and used to predict fraud-risk probability from the 12 features.

## Phase 4: Streamlit App

**Goal:** Build a simple demo app that uses the saved model.

**Covers:** APP-01, APP-02, APP-03, APP-04, APP-05

**Deliverables:**
- Streamlit app with manual input controls for the 12 features.
- Fraud-risk score output.
- Low/Medium/High category output.
- Review-priority disclaimer.

**Success Criteria:**
- App runs locally and returns a prediction without reading the full dataset or retraining the model.

## Phase 5: Documentation And Presentation Support

**Goal:** Package the project so it is understandable to evaluators and presentation-ready.

**Covers:** DOC-01, DOC-02, DOC-03, DOC-04

**Deliverables:**
- README.
- Simple feature glossary.
- Limitations section.
- Presentation outline or talking points.

**Success Criteria:**
- A reviewer can understand the problem, data, SQL, EDA, model, metrics, app, and limitations from the docs.

## Backlog

- v2: Curated expanded feature set beyond the 12 current features.
- v2: Feature importance or SHAP-style explanation.
- v2: Batch CSV upload in Streamlit.

## Phase Status

- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete
- Phase 5: complete

---
*Last updated: 2026-05-10 after end-to-end implementation*
