# ===============================================================
# DASHBOARD PRESCRIPTIVO CON MODO OSCURO Y KPIs DINÁMICOS
# ===============================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuración general ---
st.set_page_config(page_title="Dashboard Prescriptivo", layout="wide")
st.title("Dashboard Prescriptivo Integrado")

st.markdown("""
**Sistema de soporte a decisiones — Retail Analytics**
> Análisis integrado de pronóstico de demanda, política (s,Q) y costos logísticos.
""")

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_parquet("consolidado_prescriptivo.parquet")
    
    # Cargar productos para obtener categorías
    productos = pd.read_csv("../data/productos.csv")
    productos.columns = productos.columns.str.strip().str.lower()
    
    # Crear nombres descriptivos: "SKU - Categoría"
    df = df.merge(productos[['sku', 'categoria']], on='sku', how='left')
    df['sku_nombre'] = df['sku'] + ' - ' + df['categoria']
    
    return df

df = load_data()

# --- Filtros ---
st.sidebar.header("Filtros de análisis")
tiendas = sorted(df["tienda"].unique())
skus = sorted(df["sku_nombre"].unique())

tienda_sel = st.sidebar.selectbox("Seleccionar tienda", ["Todas"] + tiendas)
sku_sel = st.sidebar.selectbox("Seleccionar SKU", ["Todos"] + skus)

df_filtrado = df.copy()
if tienda_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["tienda"] == tienda_sel]
if sku_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["sku_nombre"] == sku_sel]

# ===============================================================
# 1. MÉTRICAS CON COLORES DINÁMICOS
# ===============================================================
demanda_total = df_filtrado["prediccion_ARIMA"].sum()
pedido_total = df_filtrado["pedido"].sum()
stock_promedio = df_filtrado["stock"].mean()
costo_transfer = df_filtrado["costo_transferencia_total"].sum()

# Calcular eficiencias
eficiencia_pedido = (pedido_total / demanda_total) * 100 if demanda_total > 0 else 0
alerta_color = "green" if eficiencia_pedido >= 90 else ("orange" if eficiencia_pedido >= 70 else "red")

# --- Layout de métricas ---
st.markdown("### Indicadores clave de desempeño (KPIs)")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Demanda total", f"{demanda_total:,.0f}")
col2.metric("Pedido total", f"{pedido_total:,.0f}")
col3.metric("Stock promedio", f"{stock_promedio:,.2f}")
col4.metric("Costo transferencia (S/)", f"{costo_transfer:,.2f}")
col5.markdown(f"**Eficiencia pedido:** <span style='color:{alerta_color};font-size:24px'>{eficiencia_pedido:.1f}%</span>", unsafe_allow_html=True)

# ===============================================================
# 2. GRÁFICO DEMANDA VS PEDIDO
# ===============================================================
st.markdown("### Evolución de demanda pronosticada vs pedido (s,Q)")
if sku_sel != "Todos":
    serie = df_filtrado.groupby("semana")[["prediccion_ARIMA", "pedido"]].sum()
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(serie.index, serie["prediccion_ARIMA"], label="Demanda pronosticada", marker="o", linestyle="--")
    ax.plot(serie.index, serie["pedido"], label="Pedido óptimo (s,Q)", marker="x", linewidth=2)
    ax.set_xlabel("Semana")
    ax.set_ylabel("Unidades")
    ax.set_title(f"SKU {sku_sel} - {tienda_sel if tienda_sel!='Todas' else 'Todas las tiendas'}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Selecciona un SKU para visualizar la serie temporal.")

# ===============================================================
# 3. COSTO LOGÍSTICO POR TIENDA
# ===============================================================
st.markdown("### Costos logísticos por tienda")
costo_tienda = df.groupby("tienda")["costo_transferencia_total"].sum().sort_values(ascending=False)
st.bar_chart(costo_tienda)

# ===============================================================
# 4. TOP 10 SKUs POR DEMANDA
# ===============================================================
st.markdown("### Top 10 SKUs por demanda pronosticada")
top_skus = df.groupby("sku_nombre")["prediccion_ARIMA"].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_skus)

# ===============================================================
# 5. TABLA DETALLADA
# ===============================================================
st.markdown("### Datos detallados filtrados")
st.dataframe(
    df_filtrado[["sku_nombre", "tienda", "semana", "prediccion_ARIMA", "pedido", "stock", "costo_transferencia_total"]]
    .sort_values(by=["sku_nombre", "semana"])
    .reset_index(drop=True)
)
# ===============================================================
# 6. ALERTAS INTELIGENTES — RIESGO LOGÍSTICO
# ===============================================================
st.markdown("### Alertas inteligentes del sistema")

# Definir umbrales
umbral_stock_bajo = df["stock"].mean() * 0.6
umbral_costo_alto = df["costo_transferencia_total"].mean() * 1.5
umbral_eficiencia_alta = 130

alertas = []

# --- 1. Bajo stock con alta demanda ---
riesgo_stock = df_filtrado[
    (df_filtrado["stock"] < umbral_stock_bajo) &
    (df_filtrado["prediccion_ARIMA"] > df_filtrado["stock"])
]
if not riesgo_stock.empty:
    alertas.append(f"ALERTA: **{len(riesgo_stock)} combinaciones SKU-Tienda** presentan *bajo stock* y *alta demanda esperada*.")

# --- 2. Ineficiencia de pedidos ---
if eficiencia_pedido > umbral_eficiencia_alta:
    alertas.append("ALTA: La **eficiencia de pedidos supera el 130%**, se recomienda revisar la política de reabastecimiento (s,Q).")
elif eficiencia_pedido < 70:
    alertas.append("BAJA: La **eficiencia de pedidos es menor al 70%**, podrían existir roturas de stock o sobreventas.")

# --- 3. Costos logísticos desbalanceados ---
costo_por_tienda = df.groupby("tienda")["costo_transferencia_total"].sum()
tiendas_costosas = costo_por_tienda[costo_por_tienda > umbral_costo_alto]
if not tiendas_costosas.empty:
    alertas.append(f"COSTO: Las tiendas con *mayor costo logístico* son: {', '.join(tiendas_costosas.index.tolist())}.")

# --- Mostrar resultados ---
if alertas:
    for a in alertas:
        st.warning(a)
else:
    st.success("OK: No se detectaron alertas críticas en este escenario.")
