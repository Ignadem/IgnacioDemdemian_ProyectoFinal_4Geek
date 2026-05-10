# Financial Fraud Risk Detection

## What This Is

This is a data science final project for a machine learning course. It builds an end-to-end fraud-risk prioritization workflow using a Kaggle financial fraud dataset, SQLite analysis, focused EDA, a tuned Random Forest model, and a Streamlit app.

The model estimates whether a company-year financial record resembles known fraud-related cases. It is intended to help prioritize audit or compliance review, not to prove that a company committed fraud.

## Core Value

Help a non-technical evaluator understand how financial statement data can be used to rank company-year records by fraud risk for human review.

## Requirements

### Validated

- ✓ Existing baseline notebook and script load the Kaggle CSV and train fraud-risk models — existing
- ✓ Existing model script compares Logistic Regression and tuned Random Forest with imbalance-aware metrics — existing

### Active

- [ ] Document the Kaggle dataset source and acquisition method.
- [ ] Load the dataset into SQLite and run SQL analyses that support the business story.
- [ ] Refine the EDA into six focused sections for presentation.
- [ ] Train and save the final tuned Random Forest model with joblib.
- [ ] Build a Streamlit app that accepts 12 manual financial inputs and returns a fraud-risk score/category.
- [ ] Produce project documentation and presentation-ready explanation.

### Out of Scope

- Using all 120 dataset columns in the first final version — harder to explain and unnecessary for the course presentation baseline.
- Claiming confirmed fraud from model output — predictions are risk-priority signals, not legal or audit conclusions.
- Building authentication, user accounts, or persistent app users — not needed for the course demo.
- Building a production-grade fraud detection platform — this project is scoped as an educational ML product demo.

## Context

- Evaluation criteria are stored in `doc/Criterios de Evaluacion.pdf`.
- Dataset source: Kaggle, `abdulmalekalsalemi/new-fraud-financial-dataset`.
- Local CSV: `data/Cleaned_data_1995_2018.csv`.
- The project CSV and `/Users/igna/Downloads/Cleaned_data_1995_2018.csv` were verified as identical by SHA-256.
- Dataset has 87,974 data rows and a very rare positive fraud label, around 0.65%.
- Fraud label logic: `target_fraud = 1` when `AAER_ID` exists; `target_fraud = 0` when `AAER_ID` is missing.
- Important interpretation: class `0` means no known fraud label in the dataset, not absolute proof of honest reporting.

## Constraints

- **Course scope**: Keep the project clear enough to present and explain in a machine learning course.
- **Data source**: Use Kaggle API/CLI as the official acquisition path.
- **Database**: Use SQLite for structured storage and SQL-based analysis.
- **Model features**: Use the current 12 understandable financial features for v1.
- **App framework**: Use Streamlit for a simple manual-input demo.
- **Model persistence**: Use joblib for the saved scikit-learn model.
- **Prediction language**: Present outputs as risk/review priority, not confirmed fraud.
- **Workflow**: Use lightweight GSD, one phase at a time, with explicit user checkpoints.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Frame product as fraud-risk prioritization | The model cannot prove fraud; it ranks similarity to known fraud cases | — Pending |
| Use Kaggle dataset source `abdulmalekalsalemi/new-fraud-financial-dataset` | Provides a documentable public source for the CSV | — Pending |
| Use Kaggle API/CLI as acquisition method | More reproducible than a manual file-only story | — Pending |
| Use SQLite for data storage and SQL analysis | Satisfies database requirement without server complexity | — Pending |
| Keep 12 current model features | Easier to explain and defend in a course presentation | — Pending |
| Use tuned Random Forest as final model | Current validation metrics beat Logistic Regression | — Pending |
| Keep Logistic Regression as baseline comparison | Provides a simple model comparison story | — Pending |
| Use Average Precision as primary metric | Better than accuracy for rare fraud labels and ranking use case | — Pending |
| Report recall, precision, F1, ROC AUC, confusion matrix | Explains tradeoffs between false alerts and missed fraud | — Pending |
| Save final model with joblib | Common scikit-learn deployment practice and simpler app runtime | — Pending |
| Build manual-input Streamlit app | Simple to demo and aligned with 12-feature model | — Pending |
| Use Low/Medium/High risk thresholds of 10% and 30% | Practical first-version review-priority categories | — Pending |
| Use six SQL analyses | Covers database requirement with useful analytical insight | — Pending |
| Use six focused EDA sections | Complete enough for rubric without overbuilding | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? Move to Out of Scope with reason.
2. Requirements validated? Move to Validated with phase reference.
3. New requirements emerged? Add to Active.
4. Decisions to log? Add to Key Decisions.
5. "What This Is" still accurate? Update if drifted.

**After each milestone**:
1. Full review of all sections.
2. Core Value check: still the right priority?
3. Audit Out of Scope: reasons still valid?
4. Update Context with current state.

---
*Last updated: 2026-05-10 after GSD initialization*
