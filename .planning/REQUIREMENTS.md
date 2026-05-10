# Requirements: Financial Fraud Risk Detection

**Defined:** 2026-05-10
**Core Value:** Help a non-technical evaluator understand how financial statement data can be used to rank company-year records by fraud risk for human review.

## v1 Requirements

### Data Source And Ingestion

- [ ] **DATA-01**: Document Kaggle as the official dataset source with URL and dataset slug.
- [ ] **DATA-02**: Provide a reproducible Kaggle API/CLI acquisition path for the CSV.
- [ ] **DATA-03**: Validate the ingested CSV by row count, required columns, and checksum or equivalent integrity check.

### Database And SQL

- [ ] **SQL-01**: Create a SQLite database containing the financial records.
- [ ] **SQL-02**: Include SQL query for total record count.
- [ ] **SQL-03**: Include SQL query for fraud vs non-fraud count.
- [ ] **SQL-04**: Include SQL query for fraud rate by financial year.
- [ ] **SQL-05**: Include SQL query for average key financial values by fraud label.
- [ ] **SQL-06**: Include SQL query for receivables-to-sales ratio by fraud label.
- [ ] **SQL-07**: Include SQL query for liabilities-to-assets ratio by fraud label.

### Exploratory Data Analysis

- [ ] **EDA-01**: Present dataset overview, including rows, columns, years covered, and basic structure.
- [ ] **EDA-02**: Explain target balance and why the problem is highly imbalanced.
- [ ] **EDA-03**: Analyze fraud rate by year.
- [ ] **EDA-04**: Show distributions of key financial variables.
- [ ] **EDA-05**: Compare fraud vs non-fraud financial patterns.
- [ ] **EDA-06**: Include a feature relationship or correlation check for the 12 selected features.

### Modeling

- [ ] **MODEL-01**: Use the official 12-feature set for the first final version.
- [ ] **MODEL-02**: Train Logistic Regression as baseline comparison.
- [ ] **MODEL-03**: Train and tune Random Forest as final selected model.
- [ ] **MODEL-04**: Evaluate models using Average Precision, recall, precision, F1, ROC AUC, and confusion matrix.
- [ ] **MODEL-05**: Explain why accuracy is misleading for this imbalanced fraud dataset.
- [ ] **MODEL-06**: Save the final model with joblib.
- [ ] **MODEL-07**: Save model metadata including features, thresholds, and metrics.

### Streamlit App

- [ ] **APP-01**: Build a Streamlit app with manual inputs for the 12 model features.
- [ ] **APP-02**: Load the saved joblib model instead of retraining inside the app.
- [ ] **APP-03**: Display fraud-risk score as a probability or percentage.
- [ ] **APP-04**: Display Low, Medium, or High review-priority category using 10% and 30% thresholds.
- [ ] **APP-05**: Present model output as review priority, not confirmed fraud.

### Documentation And Presentation

- [ ] **DOC-01**: Create a README explaining project goal, data source, setup, SQL, EDA, model, metrics, and app usage.
- [ ] **DOC-02**: Explain the 12 features in simple human language.
- [ ] **DOC-03**: Prepare presentation-ready narrative aligned with the evaluation criteria.
- [ ] **DOC-04**: Include clear limitations, especially target-label uncertainty and rare-class imbalance.

## v2 Requirements

### Expanded Modeling

- **V2-MODEL-01**: Evaluate a curated expanded feature set beyond the 12 current features.
- **V2-MODEL-02**: Add feature importance analysis or SHAP-style explanations.
- **V2-MODEL-03**: Support CSV batch upload in the app.

## Out of Scope

| Feature | Reason |
|---------|--------|
| All 120 columns in v1 | Too broad for the first final course presentation and harder to explain defensibly |
| Confirmed fraud verdicts | Model output is not legal or audit proof |
| User accounts/authentication | Not required for a presentation demo |
| Production deployment hardening | The course rubric asks for a functional web app, not an enterprise platform |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| SQL-01 | Phase 1 | Pending |
| SQL-02 | Phase 1 | Pending |
| SQL-03 | Phase 1 | Pending |
| SQL-04 | Phase 1 | Pending |
| SQL-05 | Phase 1 | Pending |
| SQL-06 | Phase 1 | Pending |
| SQL-07 | Phase 1 | Pending |
| EDA-01 | Phase 2 | Pending |
| EDA-02 | Phase 2 | Pending |
| EDA-03 | Phase 2 | Pending |
| EDA-04 | Phase 2 | Pending |
| EDA-05 | Phase 2 | Pending |
| EDA-06 | Phase 2 | Pending |
| MODEL-01 | Phase 3 | Pending |
| MODEL-02 | Phase 3 | Pending |
| MODEL-03 | Phase 3 | Pending |
| MODEL-04 | Phase 3 | Pending |
| MODEL-05 | Phase 3 | Pending |
| MODEL-06 | Phase 3 | Pending |
| MODEL-07 | Phase 3 | Pending |
| APP-01 | Phase 4 | Pending |
| APP-02 | Phase 4 | Pending |
| APP-03 | Phase 4 | Pending |
| APP-04 | Phase 4 | Pending |
| APP-05 | Phase 4 | Pending |
| DOC-01 | Phase 5 | Pending |
| DOC-02 | Phase 5 | Pending |
| DOC-03 | Phase 5 | Pending |
| DOC-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 32
- Unmapped: 0

---
*Requirements defined: 2026-05-10*
*Last updated: 2026-05-10 after GSD initialization*
