# Phase 2: Focused EDA - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning
**Source:** Existing GSD project context, Phase 1 outputs, and user-approved EDA scope

<domain>
## Phase Boundary

This phase turns the project exploration into a clear, presentation-ready EDA story. It should explain the dataset, the target imbalance, year-level fraud patterns, key financial variable distributions, fraud vs non-fraud differences, and relationships among the 12 selected features.

This phase should not retrain models, save joblib files, or build the Streamlit app. Those are later phases.
</domain>

<decisions>
## Implementation Decisions

### EDA Format
- Prefer a generated Markdown report over heavy notebook editing for this phase.
- Keep `analisis_inicial.ipynb` intact unless there is a clear need to touch it later.
- Report path: `reports/eda/focused_eda.md`.
- Figure output directory: `reports/eda/figures/`.

### Required EDA Sections
The final EDA must be limited to these six focused sections:
1. Dataset overview.
2. Target balance.
3. Fraud rate by year.
4. Distribution of key financial variables.
5. Fraud vs non-fraud comparison.
6. Feature relationship/correlation check.

### Data Inputs
- Use `data/fraud_financial.db` as the structured data source when practical.
- Use `reports/sql/phase1_sql_results.md` as supporting evidence from Phase 1.
- The target label is derived from `AAER_ID`.

### Feature Scope
- Use the official 12 features:
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

### Interpretation Style
- Explain findings in simple business language.
- Keep model-safe wording: fraud-labeled rows are known fraud-related cases; non-fraud rows mean no known fraud label.
- Highlight the rare-class imbalance and the receivables-to-sales signal found in Phase 1.

### the agent's Discretion
- Use pandas and matplotlib/seaborn if available.
- If charting libraries are missing, use tables and plain Markdown, but still satisfy all six EDA sections.
- Choose chart types that are easy to explain in a course presentation.
</decisions>

<canonical_refs>
## Canonical References

### GSD Project Scope
- `.planning/PROJECT.md` - project framing and constraints.
- `.planning/REQUIREMENTS.md` - EDA requirement IDs.
- `.planning/ROADMAP.md` - Phase 2 goal and deliverables.

### Phase 1 Outputs
- `docs/data_source.md` - source and target label documentation.
- `scripts/create_database.py` - database creation script.
- `data/fraud_financial.db` - generated local SQLite database.
- `reports/sql/phase1_sql_results.md` - SQL analysis results.

### Existing Analysis
- `analisis_inicial.ipynb` - current notebook with earlier exploration.
- `baseline_fraud_model.py` - official 12-feature model columns.
</canonical_refs>

<specifics>
## Specific Ideas

Likely artifacts:
- `scripts/generate_eda_report.py`
- `reports/eda/focused_eda.md`
- `reports/eda/figures/target_balance.png`
- `reports/eda/figures/fraud_rate_by_year.png`
- `reports/eda/figures/key_variable_distributions.png`
- `reports/eda/figures/fraud_vs_nonfraud_averages.png`
- `reports/eda/figures/feature_correlation_heatmap.png`

Use log scaling or clipping where raw financial variables have extreme outliers, but explain the choice.
</specifics>

<deferred>
## Deferred Ideas

- Model training and metric tables are Phase 3.
- Streamlit app is Phase 4.
- Final README/presentation packaging is Phase 5.
</deferred>

---
*Phase: 02-focused-eda*
*Context gathered: 2026-05-10 via lightweight GSD planning*
