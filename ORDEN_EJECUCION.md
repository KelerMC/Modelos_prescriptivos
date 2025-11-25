# 📋 ORDEN DE EJECUCIÓN DE NOTEBOOKS

## 🎯 Secuencia Obligatoria

### **1️⃣ 01_eda.ipynb** - Análisis Exploratorio y Preparación de Datos
**Propósito:** Base fundamental del proyecto
**Genera:** `metricas_venta_integradas.csv`

**Celdas principales:**
- ✅ Carga de 4 datasets (ventas, productos, inventario, costos)
- ✅ Homogenización de columnas
- ✅ Validación de tipos y limpieza
- ✅ Análisis estadístico descriptivo
- ✅ Análisis temporal (mensual y semanal)
- ✅ Integración de datasets
- ✅ Generación de métricas consolidadas por SKU-Tienda

**Output crítico:**
```
metricas_venta_integradas.csv
```

---

### **2️⃣ 02_modelo_sQ.ipynb** - Sistema de Revisión Continua (s, Q)
**Propósito:** Calcular política de inventario óptima
**Requiere:** `metricas_venta_integradas.csv`, `Costos_Logisticos.csv`
**Genera:** `resultados_modelo_sQ.csv`, `resumen_escenarios.csv`

**Celdas principales:**
- ✅ Carga de datos preparados
- ✅ Cálculo de Q* (lote óptimo) y s (punto de reorden)
- ✅ Evaluación de 5 escenarios logísticos
- ✅ Cálculo de CTA y CTE (costos totales)
- ✅ Visualización comparativa

**Output:**
```
resultados_modelo_sQ.csv
resumen_escenarios.csv
```

---

### **3️⃣ 05_forecasting_demanda.ipynb** - Pronóstico de Demanda (ARIMA)
**Propósito:** Generar pronósticos de demanda futura
**Requiere:** `metricas_venta_integradas.csv`
**Genera:** `ventas.csv` (sintéticas), `resultados_forecast.csv`

**Celdas principales:**
- ✅ Generación de series temporales sintéticas
- ✅ Modelo ARIMA por SKU-Tienda
- ✅ Pronóstico de 8 semanas futuras
- ✅ Validación de resultados

**Output:**
```
ventas.csv (series sintéticas)
resultados_forecast.csv
```

---

### **4️⃣ 03_opt_multi_periodo.ipynb** - Optimización Multi-Período
**Propósito:** Optimizar pedidos en múltiples períodos
**Requiere:** `resultados_modelo_sQ.csv`, `resultados_forecast.csv`
**Genera:** `resultados_multi_periodo.csv`

**Celdas principales:**
- ✅ Instalación de dependencias (PuLP)
- ✅ Definición de parámetros de optimización
- ✅ Creación del modelo de programación lineal
- ✅ Restricciones de balance de inventario
- ✅ Resolución con solver CBC

**Output:**
```
resultados_multi_periodo.csv
```

---

### **5️⃣ 04_transferencias_tiendas.ipynb** - Sistema de Transferencias
**Propósito:** Optimizar transferencias entre tiendas
**Requiere:** `metricas_venta_integradas.csv`, `diccionario_costos.csv`
**Genera:** `data/tiendas.csv`, `data/estado_tiendas.csv`, `resultados_transferencias.csv`

**Celdas principales:**
- ✅ Creación de tabla de tiendas
- ✅ Clasificación de estados (excedente/déficit/normal)
- ✅ Modelo de optimización de transferencias
- ✅ Cálculo de costos logísticos

**Output:**
```
data/tiendas.csv
data/estado_tiendas.csv
resultados_transferencias.csv
```

---

### **6️⃣ 06_consolidado_prescriptivo.ipynb** - Consolidación de Resultados
**Propósito:** Integrar todos los modelos prescriptivos
**Requiere:** Todos los archivos generados anteriormente
**Genera:** `consolidado_prescriptivo.parquet`

**Celdas principales:**
- ✅ Carga y normalización de todos los resultados
- ✅ Unificación de métricas (forecast, pedidos, stock, transferencias)
- ✅ Auditoría de integridad
- ✅ Análisis y visualización prescriptiva

**Output:**
```
consolidado_prescriptivo.parquet
```

---

### **7️⃣ dashboard_prescriptivo.py** - Dashboard Interactivo
**Propósito:** Visualización ejecutiva de resultados
**Requiere:** `consolidado_prescriptivo.parquet`

**Ejecución:**
```bash
streamlit run dashboard_prescriptivo.py
```

**Funcionalidades:**
- 📊 KPIs dinámicos
- 📈 Gráficos interactivos
- 🔍 Filtros por tienda y SKU
- ⚠️ Sistema de alertas inteligentes

---

## 📊 DEPENDENCIAS ENTRE ARCHIVOS

```
01_eda.ipynb
    ↓
    metricas_venta_integradas.csv
    ↓
    ├─→ 02_modelo_sQ.ipynb → resultados_modelo_sQ.csv
    │                     → resumen_escenarios.csv
    │
    └─→ 05_forecasting_demanda.ipynb → ventas.csv (sintéticas)
                                     → resultados_forecast.csv
         ↓
         ├─→ 03_opt_multi_periodo.ipynb → resultados_multi_periodo.csv
         │
         └─→ 04_transferencias_tiendas.ipynb → resultados_transferencias.csv
                                              → data/tiendas.csv
              ↓
              06_consolidado_prescriptivo.ipynb
                    ↓
              consolidado_prescriptivo.parquet
                    ↓
              dashboard_prescriptivo.py
```

---

## ⚠️ NOTAS IMPORTANTES

### **Orden CRÍTICO:**
1. **Siempre ejecutar `01_eda.ipynb` primero**
2. Luego `02_modelo_sQ.ipynb`
3. Después `05_forecasting_demanda.ipynb`
4. Seguir con `03_opt_multi_periodo.ipynb` y `04_transferencias_tiendas.ipynb` (en cualquier orden)
5. Finalmente `06_consolidado_prescriptivo.ipynb`
6. Lanzar dashboard con Streamlit

### **Archivos Clave a Verificar:**
- ✅ `metricas_venta_integradas.csv` (generado por 01)
- ✅ `resultados_forecast.csv` (generado por 05)
- ✅ `resultados_modelo_sQ.csv` (generado por 02)
- ✅ `resultados_multi_periodo.csv` (generado por 03)
- ✅ `resultados_transferencias.csv` (generado por 04)
- ✅ `consolidado_prescriptivo.parquet` (generado por 06)

### **Si hay errores:**
1. Verificar que cada notebook anterior se ejecutó completamente
2. Revisar que los archivos CSV/Parquet existan en la carpeta notebooks/
3. Comprobar que no haya valores nulos en columnas críticas

---

## 🎯 RESUMEN EJECUTIVO

| # | Notebook | Propósito | Output Principal | Tiempo Aprox. |
|---|----------|-----------|------------------|---------------|
| 1 | 01_eda.ipynb | Preparación de datos | metricas_venta_integradas.csv | 2-3 min |
| 2 | 02_modelo_sQ.ipynb | Modelo (s,Q) | resultados_modelo_sQ.csv | 1-2 min |
| 3 | 05_forecasting_demanda.ipynb | Pronóstico ARIMA | resultados_forecast.csv | 3-5 min |
| 4 | 03_opt_multi_periodo.ipynb | Optimización PL | resultados_multi_periodo.csv | 3-5 min |
| 5 | 04_transferencias_tiendas.ipynb | Transferencias | resultados_transferencias.csv | 1-2 min |
| 6 | 06_consolidado_prescriptivo.ipynb | Consolidación | consolidado_prescriptivo.parquet | 1 min |
| 7 | dashboard_prescriptivo.py | Visualización | Dashboard web | - |

**TIEMPO TOTAL ESTIMADO: 11-18 minutos**

---

## ✅ CHECKLIST DE EJECUCIÓN

- [ ] 1. Ejecutar 01_eda.ipynb completo
- [ ] 2. Verificar que existe metricas_venta_integradas.csv
- [ ] 3. Ejecutar 02_modelo_sQ.ipynb
- [ ] 4. Ejecutar 05_forecasting_demanda.ipynb
- [ ] 5. Verificar que existe resultados_forecast.csv
- [ ] 6. Ejecutar 03_opt_multi_periodo.ipynb
- [ ] 7. Ejecutar 04_transferencias_tiendas.ipynb
- [ ] 8. Ejecutar 06_consolidado_prescriptivo.ipynb
- [ ] 9. Verificar que existe consolidado_prescriptivo.parquet
- [ ] 10. Lanzar dashboard: `streamlit run dashboard_prescriptivo.py`

---

**Proyecto Completo y Optimizado ✅**
