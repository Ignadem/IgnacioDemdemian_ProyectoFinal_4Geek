# Detección de Riesgo de Fraude Financiero (Proyecto de Machine Learning)

Este proyecto muestra un flujo de trabajo completo para una demostración de **priorización de riesgo financiero**.

Está diseñado para que sea entendible en revisiones no técnicas:

- descargar un dataset financiero público desde Kaggle,
- guardarlo en SQLite,
- ejecutar SQL + EDA enfocada,
- entrenar y evaluar modelos con métricas para clases desbalanceadas,
- guardar el artefacto del modelo,
- ejecutar una demo en Streamlit que puntúa un registro empresa-año.

La salida de la app y del modelo es una **señal de prioridad de revisión**, no un veredicto legal o de auditoría.

## Objetivo del proyecto

Predecir si un registro empresa-año se parece a otros registros vinculados a casos conocidos de cumplimiento contable/auditivo.

**Importante:** `target_fraud = 1` significa que `AAER_ID` existe en el dataset original.  
`target_fraud = 0` significa que no hay una etiqueta `AAER_ID` conocida en este dataset.

## Fuente de datos

- Dataset oficial de Kaggle: `abdulmalekalsalemi/new-fraud-financial-dataset`
- Kaggle URL: https://www.kaggle.com/datasets/abdulmalekalsalemi/new-fraud-financial-dataset
- Local file: `data/Cleaned_data_1995_2018.csv`

Ver [`docs/data_source.md`](docs/data_source.md) para detalles de origen y descarga.

## Configuración

Entorno recomendado:

```bash
cd /Users/igna/git/ProjectoFinal_4Geeks
/Users/igna/entorno/bin/python -m pip install -r requirements.txt
```

## Ejecutar el proyecto

1. **Validar dataset**

```bash
/Users/igna/entorno/bin/python scripts/validate_data.py
```

2. **Crear SQLite y reporte SQL**

```bash
/Users/igna/entorno/bin/python scripts/create_database.py
/Users/igna/entorno/bin/python scripts/run_sql_analysis.py
```

3. **Ejecutar EDA**

```bash
/Users/igna/entorno/bin/python scripts/generate_eda_report.py
```

4. **Entrenar modelo + guardar artefactos**

```bash
/Users/igna/entorno/bin/python scripts/train_model.py
```

Esto crea:

- `models/fraud_risk_model.joblib`
- `models/model_metadata.json`
- `reports/modeling/model_results.md`

5. **Ejecutar demo de Streamlit**

```bash
/Users/igna/entorno/bin/python -m streamlit run app.py
```

## Resumen del modelado

Conjunto de variables (12):

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

Modelo base:

- Regresión logística (con tratamiento de desbalance)

Modelo final:

- Random Forest optimizado con `RandomizedSearchCV`, seleccionado por **Average Precision (PR AUC)** y usado con punto de corte por **F1**.

## Métricas de evaluación

- Primaria: **Average Precision / PR AUC**
- Complementarias: Sensibilidad (Recall), Precisión, F1, ROC AUC, Matriz de confusión
- Punto de corte final: aproximadamente `0.21`, elegido por F1.

No se usa `accuracy` como métrica principal porque las etiquetas positivas son muy raras (~0,6536%).

## Comportamiento de la app

- Entradas: 12 variables numéricas
- Salida:
  - probabilidad de riesgo de fraude (`0` a `1`)
  - categoría:
    - Bajo: puntuación < 0.21
    - Medio: 0.21 a < 0.30
    - Alto: >= 0.30
- **La salida es solo prioridad de revisión.**

## Limitaciones

- La etiqueta objetivo usa `AAER_ID` del dataset, por lo que una fila sin etiqueta no prueba ausencia de fraude.
- Modelo base de un solo enfoque y un solo horizonte temporal (sin ajuste por drift temporal).
- No es un sistema de producción: aún no hay autenticación, trazabilidad de auditoría, control de umbrales ni carga por lotes.
- Las métricas priorizan el ranking para la investigación, no la certeza legal.

## Documentación

- [`docs/data_source.md`](docs/data_source.md)
- [`docs/feature_glossary.md`](docs/feature_glossary.md)
- [`docs/presentation_outline.md`](docs/presentation_outline.md)
- [`reports/sql/phase1_sql_results.md`](reports/sql/phase1_sql_results.md)
- [`reports/eda/focused_eda.md`](reports/eda/focused_eda.md)
- [`reports/modeling/model_results.md`](reports/modeling/model_results.md)
