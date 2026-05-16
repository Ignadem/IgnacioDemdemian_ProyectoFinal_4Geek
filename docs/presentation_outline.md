# Guion de presentación

## 1) Problema y planteamiento

- "No estamos probando fraude. Estamos ordenando registros por riesgo de revisión."
- Explicar la etiqueta objetivo: presencia de `AAER_ID` en el dataset.

## 2) Datos y origen

- Origen en Kaggle y método de descarga.
- Alcance: años fiscales 1995–2018, 87.974 filas.
- Desbalance: solo ~0,6536% tiene etiqueta relacionada con fraude.

## 3) Historia con SQL

- Cantidad total de registros y balance de clases (fraude/no fraude).
- Tendencia de fraude por año.
- Comparaciones clave:
  - promedio de variables financieras por etiqueta
  - ratio cuentas por cobrar / ventas
  - ratio pasivos / activos

## 4) Historia de EDA

- Desbalance de la etiqueta objetivo.
- Distribución de variables financieras clave.
- Comparación de patrones fraude vs no fraude.
- Mapa de calor de correlación de las 12 variables.

## 5) Modelado

- Comparación inicial: Regresión Logística vs Random Forest.
- Por qué PR AUC / Average Precision es la métrica principal por el desbalance.
- Optimización de Random Forest con RandomizedSearchCV.
- Punto de corte elegido por F1: aproximadamente 0,21.
- Comparación final: threshold 0,5 vs punto de corte F1.

## 6) Demo

- Input manual en Streamlit.
- Puntuación de riesgo y categoría (Bajo/Medio/Alto).
- Descargo: salida de prioridad de revisión, no veredicto de fraude.

## 7) Riesgos y limitaciones

- Fila sin etiqueta conocida no garantiza que sea no fraude.
- Posibles falsos positivos y falsos negativos.
- Alcance educativo/demostración, no auditoría productiva.
