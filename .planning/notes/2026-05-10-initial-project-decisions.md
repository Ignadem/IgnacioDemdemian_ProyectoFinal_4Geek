---
date: "2026-05-10 00:00"
promoted: false
---

Initial project decisions:

- Project is a fraud-risk prioritization tool, not a definitive fraud detector.
- Dataset source is Kaggle: `abdulmalekalsalemi/new-fraud-financial-dataset`.
- Official acquisition method is Kaggle API/CLI.
- The local project CSV and Downloads CSV were verified as identical by SHA-256: `c279c7d662a286b6b18e2b9d01595b4683f4cc8e427749272626de04ea5d748c`.
- Use SQLite for storage and SQL-based analysis.
- SQL analysis will include six queries: total records, fraud/non-fraud count, fraud rate by year, average financial values by fraud label, receivables-to-sales ratio by fraud label, and liabilities-to-assets ratio by fraud label.
- Use the current 12 features: `Financial_Year`, `sale`, `ni`, `at`, `lt`, `che`, `rect`, `invt`, `cogs`, `txt`, `xint`, `prcc_f`.
- Target is `target_fraud`, created from `AAER_ID`.
- `target_fraud = 1` means known fraud-related record; `target_fraud = 0` means no known fraud label in the dataset.
- Final model will be tuned Random Forest; Logistic Regression remains the baseline comparison.
- Main metric is Average Precision; also report recall, precision, F1, ROC AUC, confusion matrix, and classification report.
- Save the final model with joblib in `models/fraud_risk_model.joblib`.
- Save model metadata in `models/model_metadata.json`.
- Build a manual-input Streamlit app using the 12 features.
- App outputs fraud-risk score, Low/Medium/High category, and review-priority recommendation.
- Risk categories: Low under 10%, Medium 10% to under 30%, High 30% and above.
- Final EDA has six sections: dataset overview, target balance, fraud rate by year, distributions of key variables, fraud vs non-fraud comparison, and feature relationship/correlation check.
