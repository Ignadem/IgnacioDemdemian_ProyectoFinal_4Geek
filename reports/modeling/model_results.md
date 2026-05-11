# Modeling Results: Phase 3

## Dataset Summary
- Rows: 87974
- Features used: 12
- Positive ratio: 0.006536 (0.6536%)

## Why PR AUC / Average Precision is the primary metric
Only about 0.6536% of records in this dataset have a known fraud-related label.
With this imbalance, accuracy is misleading because predicting `0` for almost every row still looks good numerically but misses the risky cases.
Priority for this course demo is *fraud-risk ranking*: use Average Precision so high scores are more likely to include known fraud-labeled cases.

## Metric definitions
- **Average Precision / PR AUC**: precision-recall ranking quality for rare positive labels.
- **ROC AUC**: score separability between known fraud-labeled and non-labeled rows.
- **Recall**: of known fraud-labeled rows, how many are flagged as suspicious.
- **Precision**: of flagged suspicious rows, how many are actually fraud-labeled.
- **F1**: combined recall/precision score at selected threshold.
- **Confusion matrix**: counts of true negatives, false positives, false negatives, and true positives.

## Official 12 feature set
Financial_Year, sale, ni, at, lt, che, rect, invt, cogs, txt, xint, prcc_f

## Model training setup
- Validation split: 70% train / 15% validation / 15% test (stratified).
- Random state: 42.

## Validation comparison

### Logistic Regression (baseline)
| metric | value |
| --- | --- |
| Average Precision | 0.009280 |
| Roc Auc | 0.643906 |
| Recall | 0.651163 |
| Precision | 0.011740 |
| F1 | 0.023064 |
| Threshold | 0.531959 |

#### Confusion Matrix (validation split)
| true\pred | 0 | 1 |
| --- | ---: | ---: |
| 0 | 8396 | 4714 |
| 1 | 30 | 56 |

### Random Forest (tuned)
| metric | value |
| --- | --- |
| Average Precision | 0.097975 |
| Roc Auc | 0.866022 |
| Recall | 0.186047 |
| Precision | 0.228571 |
| F1 | 0.205128 |
| Threshold | 0.186926 |

- Best params: `{"class_weight": "balanced_subsample", "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 5, "n_estimators": 200}`

#### Confusion Matrix (validation split)
| true\pred | 0 | 1 |
| --- | ---: | ---: |
| 0 | 13056 | 54 |
| 1 | 70 | 16 |

## Final model selection
- Selected for test evaluation: **Random Forest (tuned)**

## Held-out test metrics
| metric | value |
| --- | --- |
| Average Precision | 0.068777 |
| Roc Auc | 0.843148 |
| Recall | 0.103448 |
| Precision | 0.147541 |
| F1 | 0.121622 |
| Threshold | 0.186926 |

### Confusion Matrix (held-out test)
| true\pred | 0 | 1 |
| --- | ---: | ---: |
| 0 | 13058 | 52 |
| 1 | 78 | 9 |


## Deployment artifacts
- Saved model: `/Users/igna/git/ProjectoFinal_4Geeks/models/fraud_risk_model.joblib`
- Saved metadata: `/Users/igna/git/ProjectoFinal_4Geeks/models/model_metadata.json`
- The model output is **review priority**, not a legal or audit proof of fraud.
- `target_fraud = 1` means an `AAER_ID` is present in the source dataset.
- `target_fraud = 0` means no known `AAER_ID` label is present in this dataset.

*Generated: 2026-05-10 22:04:36*
