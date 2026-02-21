import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

# 1. Configuración de la página
st.set_page_config(page_title="Análisis de los gráficos de reloj del IVAE El Salvador", page_icon="🔄", layout="centered")

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

st.title("🟢📈 → 🟡⛰️ → 🔴📉 → 🔵🔄 Análisis de los gráficos de reloj de la tendencia-ciclo del IVAE de El Salvador")
st.image("FullLogo.png", width=300)
st.subheader("Comparativo de los últimos 3 años.")
st.markdown("Alfredo Ibrahim Flores Sarria ©2026")

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

# 4. PROCESAMIENTO MATEMÁTICO (Filtro HP)
df = df_base.copy()
df['log_serie'] = np.log(df[col_tc])
cycle, trend = hpfilter(df['log_serie'], lamb=129600)
df['Ciclo_norm'] = (cycle - cycle.mean()) / cycle.std()
df['Delta'] = df['Ciclo_norm'].diff()

# ==========================================
# 5. CÁLCULO DE LOS 3 AÑOS COMPARATIVOS
# ==========================================
# A partir del mes de cierre, calculamos 3 ventanas de 12 meses exactos hacia atrás
fecha_fin_p3 = pd.to_datetime(mes_cierre_str)
fecha_inicio_p3 = fecha_fin_p3 - pd.DateOffset(months=12)

fecha_fin_p2 = fecha_inicio_p3
fecha_inicio_p2 = fecha_fin_p2 - pd.DateOffset(months=12)

fecha_fin_p1 = fecha_inicio_p2
fecha_inicio_p1 = fecha_fin_p1 - pd.DateOffset(months=12)

periodos = [
    (fecha_inicio_p1.strftime('%Y-%m'), fecha_fin_p1.strftime('%Y-%m')), # Hace 3 años
    (fecha_inicio_p2.strftime('%Y-%m'), fecha_fin_p2.strftime('%Y-%m')), # Hace 2 años
    (fecha_inicio_p3.strftime('%Y-%m'), fecha_fin_p3.strftime('%Y-%m'))  # Último año
]

# Nombres dinámicos para las pestañas
titulos_tabs = [
    f"{fecha_inicio_p1.year}-{fecha_fin_p1.year}",
    f"{fecha_inicio_p2.year}-{fecha_fin_p2.year}",
    f"{fecha_inicio_p3.year}-{fecha_fin_p3.year}"
]

st.divider()
st.subheader(f"Análisis Sectorial: {sector_elegido_nombre}")

# Crear las 3 pestañas comparativas
tabs = st.tabs(titulos_tabs)

config = {
    'colors': ['#2A5CAA', '#2E8B57', '#D93F3F'],  # Azul, Verde, Rojo
    'styles': ['-', '--', '-.'],
    'markers': ['o', 's', 'D']
}

# ==========================================
# 6. RENDERIZADO VISUAL
# ==========================================
def dibujar_reloj_individual(start, end, color, idx):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    meses_esp = {'01':'Ene', '02':'Feb', '03':'Mar', '04':'Abr', '05':'May', '06':'Jun', 
                 '07':'Jul', '08':'Ago', '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dic'}
    
    df_periodo = df.loc[start:end].sort_index()
    mask = df_periodo['Delta'].notna()

    ax.set_xlabel('Variación del Ciclo (ΔC$_t$)', fontsize=12)
    ax.set_ylabel('Ciclo Normalizado (C$_t$)', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle=':')

    # Trayectoria mensual
    ax.plot(df_periodo[mask]['Delta'], df_periodo[mask]['Ciclo_norm'],
            color=color, linestyle=config['styles'][idx], linewidth=2.5, alpha=0.9)

    # Marcadores de inicio y fin
    for i, (fecha_str, marker) in enumerate(zip([start, end], config['markers'][:2])):
        try:
            # Buscar el punto exacto
            punto = df_periodo.loc[df_periodo.index.astype(str).str.startswith(fecha_str)].iloc[0]
            x_val, y_val = punto['Delta'], punto['Ciclo_norm']
            
            if pd.isna(x_val) or pd.isna(y_val): continue
                
            ax.scatter(x_val, y_val, color=color, s=150 + (i*30), edgecolor='black', marker=marker, zorder=5)

            mes_texto = meses_esp.get(fecha_str[5:7], '')
            offset_x = 25 if i == 0 else -25
            va_pos = 'bottom' if y_val > 0 else 'top'
            
            ax.annotate(f'{mes_texto} {fecha_str[:4]}\nΔ: {x_val:.2f}\nC: {y_val:.2f}',
                        (x_val, y_val), textcoords="offset points",
                        xytext=(offset_x, 15 if i == 0 else -15),
                        ha='right' if i == 1 else 'left', va=va_pos,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.95))
        except IndexError:
            pass

    # Cuadrantes
    for xy, color_cuad in [((0.5, 0.5), '#90EE90'), ((-0.5, 0.5), '#FFB6C1'), 
                           ((-0.5, -0.5), '#87CEEB'), ((0.5, -0.5), '#DDA0DD')]:
        ax.add_patch(plt.Rectangle(xy, 0.5, 0.5, color=color_cuad, alpha=0.1, transform=ax.transAxes))

    ax.axhline(0, color='#333333', linestyle=':', linewidth=1.5)
    ax.axvline(0, color='#333333', linestyle=':', linewidth=1.5)

    # Limites
    x_buf = 0.2 * (df_periodo['Delta'].max() - df_periodo['Delta'].min())
    y_buf = 0.2 * (df_periodo['Ciclo_norm'].max() - df_periodo['Ciclo_norm'].min())
    
    x_buf = x_buf if pd.notna(x_buf) and x_buf > 0 else 0.5
    y_buf = y_buf if pd.notna(y_buf) and y_buf > 0 else 0.5
    
    ax.set_xlim(df_periodo['Delta'].min() - x_buf, df_periodo['Delta'].max() + x_buf)
    ax.set_ylim(df_periodo['Ciclo_norm'].min() - y_buf, df_periodo['Ciclo_norm'].max() + y_buf)
    
    plt.tight_layout()
    return fig

# Generar gráficos en cada pestaña
for idx, (start, end) in enumerate(periodos):
    with tabs[idx]:
        figura = dibujar_reloj_individual(start, end, config['colors'][idx], idx)
        st.pyplot(figura, use_container_width=True)

# 7. Leyenda 
with st.expander("Ver Leyenda y Criterios de Interpretación", expanded=False):
    st.markdown("""
    **Trayectoria del Reloj:**
    * 🔵 **Círculo:** Inicio del período (Hace 12 meses).
    * 🟦 **Cuadrado:** Fin del período (Mes de cierre).
    
    **Interpretación de Cuadrantes:**
    * 🟩 **Superior Derecho:** Crecimiento por encima de la tendencia.
    * 🟥 **Superior Izquierdo:** Decrecimiento por encima de la tendencia (Desaceleración).
    * 🟦 **Inferior Izquierdo:** Decrecimiento por debajo de la tendencia (Recesión).
    * 🟪 **Inferior Derecho:** Crecimiento por debajo de la tendencia (Recuperación).
    """)
