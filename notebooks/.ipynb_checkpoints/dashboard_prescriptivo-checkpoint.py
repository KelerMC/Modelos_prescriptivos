# ===============================================================
# 🧩 BLOQUE 3 — DASHBOARD PRESCRIPTIVO INTERACTIVO (STREAMLIT)
# ===============================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuración general ---
st.set_page_config(page_title="Dashboard Prescriptivo", layout="wide")
st.title("📊 Dashboard Prescriptivo Integrado")
st.markdown("""
Visualiza los resultados del modelo de planificación integrada:
- Pronóstico de demanda (ARIMA)
- Política de reabastecimiento (s, Q)
- Costos logísticos de transferencia
""")

# --- Cargar datos ---
@st.cache_data
def load_data():
    return pd.read_parquet("consolidado_prescriptivo.parquet")

df = load_data()
st.sidebar.header("Filtros")

# --- Filtros dinámicos ---
tiendas = sorted(df["tienda"].unique())
skus = sorted(df["sku"].unique())

tienda_sel = st.sidebar.selectbox("Seleccionar tienda", ["Todas"] + tiendas)
sku_sel = st.sidebar.selectbox("Seleccionar SKU", ["Todos"] + skus)

df_filtrado = df.copy()
if tienda_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["tienda"] == tienda_sel]
if sku_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["sku"] == sku_sel]

# --- Resumen general ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("🛒 Demanda total", f"{df_filtrado['prediccion_ARIMA'].sum():,.0f}")
col2.metric("📦 Pedido total", f"{df_filtrado['pedido'].sum():,.0f}")
col3.metric("🏬 Stock promedio", f"{df_filtrado['stock'].mean():.2f}")
col4.metric("🚚 Costo transferencia (S/)", f"{df_filtrado['costo_transferencia'].sum():,.2f}")

# --- Gráfico 1: Demanda vs Pedido ---
st.subheader("📈 Demanda pronosticada vs Pedido óptimo")
if sku_sel != "Todos":
    serie = df_filtrado.groupby("semana")[["prediccion_ARIMA", "pedido"]].sum()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(serie.index, serie["prediccion_ARIMA"], label="Demanda pronosticada", marker="o")
    ax.plot(serie.index, serie["pedido"], label="Pedido óptimo (s,Q)", marker="x")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Unidades")
    ax.set_title(f"SKU {sku_sel} - {tienda_sel}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Selecciona un SKU para visualizar la serie temporal.")

# --- Gráfico 2: Costo logístico por tienda ---
st.subheader("💰 Costos logísticos por tienda")
costo_tienda = (
    df.groupby("tienda")["costo_transferencia"]
    .sum()
    .sort_values(ascending=False)
)
st.bar_chart(costo_tienda)

# --- Gráfico 3: Top SKUs por demanda ---
st.subheader("🏆 Top 10 SKUs por demanda pronosticada")
top_skus = (
    df.groupby("sku")["prediccion_ARIMA"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(top_skus)
