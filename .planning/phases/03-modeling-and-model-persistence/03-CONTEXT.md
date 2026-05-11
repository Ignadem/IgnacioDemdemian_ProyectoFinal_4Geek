# Phase 3: Modeling And Model Persistence - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning
**Source:** Existing GSD project context, Phase 1/2 outputs, and user-approved modeling decisions

<domain>
## Phase Boundary

This phase trains, evaluates, selects, and saves the first final model for the course project.

The output should be a reproducible modeling pipeline that uses the official 12 features, compares a Logistic Regression baseline against a tuned Random Forest, reports metrics that make sense for rare fraud labels, and saves a joblib model that the later Streamlit app can load without retraining.

This phase should not build the Streamlit app. It should prepare the model artifacts that Phase 4 will consume.
</domain>

<decisions>
## Implementation Decisions

### Project Promise
- The model predicts fraud risk or suspicion level, not confirmed fraud.
- Use wording like "review priority" or "fraud-risk probability".
- Do not claim the model legally or audit-confirms fraud.

### Data And Target
- Use `data/Cleaned_data_1995_2018.csv` as the modeling input.
- Derive `target_fraud` from `AAER_ID`.
- `target_fraud = 1` means the record has a known fraud-related enforcement-case label.
- `target_fraud = 0` means no known fraud label is recorded in the dataset; it is not proof of honesty.

### Feature Scope
- Use only the official 12 v1 features:
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
- Do not expand to all 120 columns in v1.

### Model Scope
- Train Logistic Regression as a baseline.
- Train and tune Random Forest as the final selected model candidate.
- Use `class_weight` or equivalent imbalance-aware handling where appropriate.
- Use deterministic random state `42`.

### Metrics
- Primary metric: Average Precision / PR AUC.
- Supporting metrics: recall, precision, F1, ROC AUC, and confusion matrix.
- Include a plain-language explanation of why accuracy is misleading because only about `0.6536%` of rows are fraud-labeled.

### Model Persistence
- Save the final model with `joblib`.
- Expected final path: `models/fraud_risk_model.joblib`.
- Save metadata to `models/model_metadata.json`.
- Metadata must include feature list, threshold, selected model, parameters, metrics, and project wording/disclaimer.

### the agent's Discretion
- Refactor `baseline_fraud_model.py` or create `scripts/train_model.py`; prefer the cleaner maintainable path.
- Preserve useful existing modeling logic from `baseline_fraud_model.py`.
- Generate a Markdown report for evaluator-friendly metric explanation.
- Keep runtime reasonable for a course project.
</decisions>

<canonical_refs>
## Canonical References

### GSD Project Scope
- `.planning/PROJECT.md` - project framing and constraints.
- `.planning/REQUIREMENTS.md` - model requirement IDs.
- `.planning/ROADMAP.md` - Phase 3 goal and deliverables.

### Data And EDA Inputs
- `docs/data_source.md` - source and target label documentation.
- `reports/sql/phase1_sql_results.md` - SQL counts and ratios.
- `reports/eda/focused_eda.md` - EDA findings, imbalance explanation, and 12-feature context.

### Existing Modeling Work
- `baseline_fraud_model.py` - current training script with selected features, split logic, baseline model, Random Forest tuning, and metric functions.
- `data/Cleaned_data_1995_2018.csv` - local ignored modeling dataset.
</canonical_refs>

<specifics>
## Specific Ideas

Likely artifacts:
- `scripts/train_model.py`
- `reports/modeling/model_results.md`
- `models/fraud_risk_model.joblib`
- `models/model_metadata.json`

The saved joblib object should include all preprocessing needed for prediction, such as imputation and any scaling used by the selected model.
</specifics>

<deferred>
## Deferred Ideas

- Streamlit manual-input app is Phase 4.
- README and final presentation packaging are Phase 5.
- Expanded feature set and SHAP-style explanation remain v2 backlog.
</deferred>

---
*Phase: 03-modeling-and-model-persistence*
*Context gathered: 2026-05-10 via lightweight GSD planning*
