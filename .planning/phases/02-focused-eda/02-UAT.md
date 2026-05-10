---
status: complete
phase: 02-focused-eda
source:
  - 02-01-SUMMARY.md
  - 02-02-SUMMARY.md
  - 02-EXECUTION-SUMMARY.md
started: 2026-05-10T14:27:00-03:00
updated: 2026-05-10T14:27:00-03:00
---

## Current Test

[testing complete]

## Tests

### 1. EDA Report Regenerates
expected: `/Users/igna/entorno/bin/python scripts/generate_eda_report.py` succeeds and writes `reports/eda/focused_eda.md`.
result: pass

### 2. Dataset Overview Is Present
expected: The report explains row count, original column count, helper columns, year range, selected features, and company-year record meaning.
result: pass

### 3. Target Balance Is Present
expected: The report shows `575` fraud-labeled records, `87,399` records with no known fraud label, the `0.6536%` fraud-label rate, and explains imbalance.
result: pass

### 4. Fraud Rate By Year Is Present
expected: The report includes year-level fraud rates and explains that fraud-labeled records are not evenly distributed over time.
result: pass

### 5. Key Variable Distributions Are Present
expected: The report includes `key_variable_distributions.png` and explains signed log scaling for wide-range financial variables.
result: pass

### 6. Fraud vs Non-Fraud Comparison Is Present
expected: The report includes `fraud_vs_nonfraud_averages.png`, financial average comparisons, and the receivables-to-sales finding without claiming it proves fraud.
result: pass

### 7. Feature Relationship Check Is Present
expected: The report includes `feature_correlation_heatmap.png` for the 12 selected features and explains that correlation is not causation or proof of fraud.
result: pass

### 8. All Figures Exist
expected: All five PNG figures exist and are non-empty.
result: pass

## Summary

total: 8
passed: 8
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

None.
