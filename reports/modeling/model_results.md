# Resultados de modelado: Fase 3

## Resumen del conjunto de datos
- Registros: 87974
- Variables usadas: 12
- Tasa de etiqueta positiva: 0.006536 (0.6536%)
- División: 80% entrenamiento / 20% test estratificado.

## Criterio de evaluación
- `average_precision` / PR AUC se usa para optimizar hiperparámetros porque el dataset está muy desbalanceado.
- F1 se usa para elegir el punto de corte que convierte probabilidades en clase 0/1.
- Accuracy no se usa como métrica principal porque puede verse alta aunque el modelo no detecte casos positivos.

## Comparación baseline

### Regresión logística (threshold 0.5)
| Métrica | Valor |
| --- | ---: |
| Precision promedio | 0.009729 |
| ROC AUC | 0.662691 |
| Recall (sensibilidad) | 0.721739 |
| Precisión | 0.011326 |
| F1 | 0.022303 |
| Umbral / punto de corte | 0.500000 |

### Random Forest anterior (threshold 0.5)
| Métrica | Valor |
| --- | ---: |
| Precision promedio | 0.104562 |
| ROC AUC | 0.867118 |
| Recall (sensibilidad) | 0.008696 |
| Precisión | 0.111111 |
| F1 | 0.016129 |
| Umbral / punto de corte | 0.500000 |

### Random Forest anterior (punto de corte por F1)
| Métrica | Valor |
| --- | ---: |
| Precision promedio | 0.104562 |
| ROC AUC | 0.867118 |
| Recall (sensibilidad) | 0.165217 |
| Precisión | 0.253333 |
| F1 | 0.200000 |
| Umbral / punto de corte | 0.206787 |

## RandomizedSearchCV
- Scoring: `average_precision`
- Iteraciones: 12
- Folds CV: 3
- Mejor Average Precision promedio en CV: 0.076405
- Mejores hiperparámetros: `{"model__n_estimators": 200, "model__min_samples_split": 5, "model__min_samples_leaf": 5, "model__max_features": "log2", "model__max_depth": null, "model__class_weight": "balanced_subsample"}`

### Combinaciones probadas
| Rank | n_estimators | max_depth | min_samples_leaf | min_samples_split | max_features | class_weight | Mean CV AP | Std CV AP |
| ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: |
| 1 | 200 | None | 5 | 5 | log2 | balanced_subsample | 0.076405 | 0.016881 |
| 2 | 300 | None | 10 | 2 | sqrt | balanced | 0.071979 | 0.019357 |
| 3 | 200 | 20 | 1 | 10 | log2 | balanced | 0.055451 | 0.003254 |
| 4 | 100 | 20 | 5 | 2 | sqrt | balanced | 0.052861 | 0.004222 |
| 5 | 300 | 12 | 1 | 2 | log2 | balanced_subsample | 0.040361 | 0.005164 |
| 6 | 100 | 12 | 10 | 5 | log2 | balanced_subsample | 0.039232 | 0.006091 |
| 7 | 100 | 12 | 2 | 10 | sqrt | balanced_subsample | 0.035841 | 0.003615 |
| 8 | 200 | 8 | 2 | 2 | log2 | balanced | 0.034740 | 0.004896 |
| 8 | 200 | 8 | 2 | 2 | sqrt | balanced | 0.034740 | 0.004896 |
| 10 | 300 | 8 | 5 | 2 | sqrt | balanced_subsample | 0.033874 | 0.003097 |
| 11 | 200 | 8 | 1 | 5 | log2 | balanced_subsample | 0.031839 | 0.002619 |
| 12 | 100 | 8 | 1 | 10 | log2 | balanced_subsample | 0.031556 | 0.004216 |

## Modelo final: Random Forest RandomizedSearchCV + punto de corte F1
| Métrica | Valor |
| --- | ---: |
| Precision promedio | 0.101710 |
| ROC AUC | 0.862764 |
| Recall (sensibilidad) | 0.165217 |
| Precisión | 0.263889 |
| F1 | 0.203209 |
| Umbral / punto de corte | 0.210032 |

### Matriz de confusión final
| true\pred | 0 | 1 |
| --- | ---: | ---: |
| 0 | 17427 | 53 |
| 1 | 96 | 19 |

## Comparación final en test
| Modelo | AP | ROC AUC | Recall | Precisión | F1 | Threshold | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF anterior threshold 0.5 | 0.104562 | 0.867118 | 0.008696 | 0.111111 | 0.016129 | 0.500000 | 17472 | 8 | 114 | 1 |
| RF anterior punto corte F1 | 0.104562 | 0.867118 | 0.165217 | 0.253333 | 0.200000 | 0.206787 | 17424 | 56 | 96 | 19 |
| RF RandomizedSearchCV punto corte F1 | 0.101710 | 0.862764 | 0.165217 | 0.263889 | 0.203209 | 0.210032 | 17427 | 53 | 96 | 19 |

## Lectura
- Con `threshold = 0.5`, Random Forest detecta solo 1 fraude etiquetado.
- Con punto de corte por F1 (`0.2100`), el modelo final detecta 19 fraudes etiquetados.
- La salida debe interpretarse como prioridad de revisión, no como veredicto de fraude.

## Artefactos
- Modelo guardado: `/Users/igna/git/ProjectoFinal_4Geeks/models/fraud_risk_model.joblib`
- Metadata guardada: `/Users/igna/git/ProjectoFinal_4Geeks/models/model_metadata.json`
- SHA-256 del modelo: `82bdc6f70de92f07dca20b9975d33decb23e10a12fff3e6ad61359a89c2a9316`

*Generado: 2026-05-16 18:35:05*
