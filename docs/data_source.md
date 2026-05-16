# Origen de datos

## Dataset oficial

Este proyecto usa el dataset de Kaggle **New Fraud Financial Dataset**.

- Kaggle URL: https://www.kaggle.com/datasets/abdulmalekalsalemi/new-fraud-financial-dataset
- Kaggle dataset slug: `abdulmalekalsalemi/new-fraud-financial-dataset`
- Local project file: `data/Cleaned_data_1995_2018.csv`

El dataset contiene registros financieros históricos de empresas de los ejercicios fiscales 1995–2018.
Cada fila representa un registro financiero de una empresa en un año.

## Adquisición reproducible

El método oficial de adquisición para este proyecto es la API/CLI de Kaggle:

```bash
kaggle datasets download -d abdulmalekalsalemi/new-fraud-financial-dataset
```

Luego descomprimí el archivo descargado y colocá `Cleaned_data_1995_2018.csv` en:

```text
data/Cleaned_data_1995_2018.csv
```

Esto requiere credenciales de Kaggle configuradas localmente. Si el CSV ya existe localmente, el script de validación puede confirmar que coincide con la copia esperada del dataset.

## Etiqueta objetivo

El dataset original incluye la columna `AAER_ID`.

`AAER_ID` significa **Accounting and Auditing Enforcement Release ID**.
En términos simples, es el identificador de un caso de enforcement conocido relacionado con temas contables o de auditoría.
Cuando una fila tiene `AAER_ID`, el dataset está marcando ese registro empresa-año como vinculado a un caso relacionado con fraude.

En este proyecto usamos `AAER_ID` como fuente para la etiqueta de fraude.
El modelo no intenta probar fraude por sí solo; aprende de registros que ya vienen etiquetados con ese identificador de caso.

Para este proyecto, la variable objetivo se crea como:

```text
`target_fraud = 1` si `AAER_ID` existe
`target_fraud = 0` si `AAER_ID` falta
```

Interpretación:

- `target_fraud = 1`: el registro está vinculado a un caso de enforcement relacionado con fraude conocido.
- `target_fraud = 0`: no hay etiqueta de fraude conocida en este dataset.

Importante: `target_fraud = 0` no prueba que un registro empresa-año sea limpio. Solo significa que
este dataset no contiene una etiqueta de fraude conocida para esa fila.

## Validación local

Validá el CSV local con:

```bash
/Users/igna/entorno/bin/python scripts/validate_data.py
```

La validación comprueba:

- que el archivo exista;
- cantidad esperada de filas;
- checksum SHA-256 esperado;
- columnas requeridas para Fase 1 y modelado.
