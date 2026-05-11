# Presentation Outline

## 1) Problem and framing

- "We are not proving fraud. We are ranking records by fraud-risk for review."
- Explain target label: `AAER_ID` presence in dataset.

## 2) Data and source

- Kaggle source and acquisition method.
- Scope: fiscal years 1995–2018, 87,974 rows.
- Imbalance context: only ~0.6536% labeled as fraud-related.

## 3) SQL story

- Total records and fraud balance.
- Fraud trend by year.
- Key comparisons:
  - average financial values by label
  - receivables-to-sales ratio
  - liabilities-to-assets ratio

## 4) EDA story

- Target imbalance.
- Distribution of key financial variables.
- Fraud vs non-fraud pattern comparison.
- Correlation heatmap on 12 features.

## 5) Modeling

- 12-feature baseline vs tuned Random Forest setup.
- Why PR AUC / Average Precision is primary due class imbalance.
- Validation comparison and held-out test metrics.

## 6) Demo

- Streamlit manual input.
- Risk score and Low/Medium/High category.
- Review-priority disclaimer.

## 7) Risks and limitations

- Unlabeled rows are not guaranteed to be non-fraud.
- Possible false positives/false negatives.
- Scope is educational/demo, not production audit.
