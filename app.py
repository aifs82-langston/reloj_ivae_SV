import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

# 1. Configuración de la página
st.set_page_config(page_title="Monitor IVAE El Salvador", layout="centered")

# 2. Caché de datos
@st.cache_data
def cargar_datos():
    df = pd.read_excel('Tendencia_Ciclo_Sectores_IVAE_20022026.xlsx', parse_dates=['Fecha'], index_col='Fecha').sort_index()
    return df

df_base = cargar_datos()

sectores_nombres = {
    "IVAE_General": "IVAE GENERAL",
    "IVAE_Agropecuario": "SECTOR AGROPECUARIO",
    "IPI": "PRODUCCIÓN INDUSTRIAL (IPI)",
    "IVAE_Construccion": "SECTOR CONSTRUCCIÓN",
    "IVAE_Comercio_Servicios": "COMERCIO Y SERVICIOS",
    "IVAE_Info_Comunicaciones": "INFORMACIÓN Y COMUNICACIONES",
    "IVAE_Financiero": "ACTIVIDADES FINANCIERAS Y DE SEGUROS",
    "IVAE_Inmobiliario": "ACTIVIDADES INMOBILIARIAS",
    "IVAE_Servicios_Profesionales": "SERVICIOS PROFESIONALES Y TÉCNICOS",
    "IVAE_Servicios_Publicos": "ADMINISTRACIÓN PÚBLICA Y DEFENSA"
}

st.title("Reloj de la Tendencia-Ciclo")
st.markdown("Monitor macroeconómico comparativo de los últimos 3 años.")

# ==========================================
# 3. CONTROLES DINÁMICOS INTELIGENTES
# ==========================================
col1, col2 = st.columns(2)

with col1:
    sector_elegido_nombre = st.selectbox("Seleccione el Sector:", list(sectores_nombres.values()))

with col2:
    # Extraer todos los meses disponibles para que el usuario elija el "Mes de Cierre"
    # Lo mostramos de más reciente a más antiguo
    meses_disponibles = df_base.index.strftime('%Y-%m').unique().sort_values(ascending=False)
    mes_cierre_str = st.selectbox("Mes de Referencia (Cierre):", meses_disponibles)

# Lógica del sector
col_r = list(sectores_nombres.keys())[list(sectores_nombres.values()).index(sector_elegido_nombre)]
col_tc = f"{col_r}_TC"

# 4. PROCESAMIENTO MATEMÁ
