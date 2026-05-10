# Plan 01-03 Summary: SQL Analysis

**Completed:** 2026-05-10
**Status:** Complete

## Changes

- Created `sql/phase1_analysis.sql` with six named SQL analyses:
  1. Total records.
  2. Fraud vs non-fraud count.
  3. Fraud rate by financial year.
  4. Average key financial values by fraud label.
  5. Receivables-to-sales ratio by fraud label.
  6. Liabilities-to-assets ratio by fraud label.
- Created `scripts/run_sql_analysis.py`.
- Generated `reports/sql/phase1_sql_results.md` from `data/fraud_financial.db`.

## Verification

- `rg -n "Total records|Receivables-to-sales|Liabilities-to-assets" sql/phase1_analysis.sql`
- `/Users/igna/entorno/bin/python scripts/run_sql_analysis.py`
- `rg -n "87974|Fraud Rate by Financial Year" reports/sql/phase1_sql_results.md`

## Key Outputs

- Total rows: `87974`
- Non-fraud labeled rows: `87399`
- Fraud labeled rows: `575`
- Highest fraud-rate years in the output occur around `FY2000` and `FY2001`.
- Average receivables-to-sales ratio is higher for fraud-labeled records (`0.9426`) than non-fraud records (`0.2931`).

## Requirements Covered

- SQL-02
- SQL-03
- SQL-04
- SQL-05
- SQL-06
- SQL-07
