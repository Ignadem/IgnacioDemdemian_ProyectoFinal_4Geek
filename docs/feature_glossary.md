# Glosario de variables (12 variables)

Esta es la lista oficial de las 12 variables que usa el modelo en la versión 1.

## Variables

- `Financial_Year`: ejercicio fiscal, convertido a valor numérico.
- `sale`: ventas totales (ingresos) del ejercicio.
- `ni`: utilidad neta (ganancia después de gastos e impuestos).
- `at`: activos totales.
- `lt`: pasivos totales (deudas y obligaciones de la empresa).
- `che`: efectivo y equivalentes / inversiones de corto plazo (recursos líquidos).
- `rect`: cuentas por cobrar (dinero pendiente de cobro de clientes).
- `invt`: inventario.
- `cogs`: costo de ventas (costo de bienes/servicios vendidos).
- `txt`: impuestos totales pagados o a pagar.
- `xint`: gastos por intereses.
- `prcc_f`: valor de mercado (precio por acción del dataset); se usa tal cual viene.

Notas de interpretación:

- Valores más altos de `sale`, `at` y `che` suelen corresponder a empresas más grandes.
- Un `rect` alto en relación con `sale` puede sugerir patrones de reconocimiento de ingresos más agresivos.
- `xint`, `lt` elevados y `ni` negativo pueden reflejar presión financiera y potenciales tensiones de reporte.
