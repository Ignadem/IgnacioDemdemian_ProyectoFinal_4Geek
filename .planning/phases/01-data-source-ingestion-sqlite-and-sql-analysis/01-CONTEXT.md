# Phase 1: Data Source, Ingestion, SQLite, And SQL Analysis - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning
**Source:** Existing GSD project context and user decisions

<domain>
## Phase Boundary

This phase establishes the data foundation for the course project. It must document the public Kaggle source, provide a reproducible acquisition path, validate the local CSV, create a SQLite database, and produce six SQL analyses that support the business/EDA story.

This phase does not modify model training, EDA charts, or the Streamlit app beyond creating reusable data/SQL artifacts those later phases can consume.
</domain>

<decisions>
## Implementation Decisions

### Dataset Source
- Use Kaggle dataset `abdulmalekalsalemi/new-fraud-financial-dataset`.
- Official URL: `https://www.kaggle.com/datasets/abdulmalekalsalemi/new-fraud-financial-dataset`.
- Official acquisition method: Kaggle API/CLI.
- Keep local CSV at `data/Cleaned_data_1995_2018.csv`.
- The CSV has already been verified against `/Users/igna/Downloads/Cleaned_data_1995_2018.csv` by SHA-256: `c279c7d662a286b6b18e2b9d01595b4683f4cc8e427749272626de04ea5d748c`.

### Database
- Use SQLite.
- Database path: `data/fraud_financial.db`.
- Main table name: `financial_records`.
- SQLite is for storage and SQL analysis. Later Streamlit predictions will use a saved joblib model, not SQLite.

### SQL Analyses
- Include exactly six first-version SQL analyses:
  1. Total records.
  2. Fraud vs non-fraud count.
  3. Fraud rate by financial year.
  4. Average key financial values by fraud label.
  5. Receivables-to-sales ratio by fraud label.
  6. Liabilities-to-assets ratio by fraud label.

### Target Label
- `target_fraud = 1` when `AAER_ID` exists.
- `target_fraud = 0` when `AAER_ID` is missing.
- Explain `0` as "no known fraud label in this dataset", not proof of honest reporting.

### the agent's Discretion
- Choose small script/module names that fit this repo.
- Use pandas and sqlite3/SQLAlchemy if already available; avoid adding heavy dependencies.
- Store generated query outputs as CSV or Markdown if useful for presentation.
</decisions>

<canonical_refs>
## Canonical References

Downstream implementation must read these before editing:

### GSD Project Scope
- `.planning/PROJECT.md` - project framing, constraints, and key decisions.
- `.planning/REQUIREMENTS.md` - Phase 1 requirement IDs and acceptance scope.
- `.planning/ROADMAP.md` - phase goal and deliverables.

### Existing Project Files
- `data/Cleaned_data_1995_2018.csv` - local data source.
- `baseline_fraud_model.py` - existing data-loading assumptions and feature/target names.
- `analisis_inicial.ipynb` - current notebook context and EDA/model baseline.
- `doc/Criterios de Evaluacion.pdf` - evaluation rubric.
</canonical_refs>

<specifics>
## Specific Ideas

Likely artifacts:
- `scripts/create_database.py`
- `sql/phase1_analysis.sql`
- `docs/data_source.md`
- generated `data/fraud_financial.db`
- optional generated SQL output tables in `reports/sql/`

The implementation should avoid committing or duplicating the 110 MB CSV. The CSV is already ignored in `.gitignore`.
</specifics>

<deferred>
## Deferred Ideas

- EDA charting is Phase 2.
- Model retraining and joblib saving are Phase 3.
- Streamlit app is Phase 4.
- README and presentation packaging are Phase 5.
</deferred>

---
*Phase: 01-data-source-ingestion-sqlite-and-sql-analysis*
*Context gathered: 2026-05-10 via lightweight GSD planning*
