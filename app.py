import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

# 1. Configuración de la página (Debe ser la primera instrucción)
st.set_page_config(page_title="Monitor IVAE El Salvador", layout="centered")

# 2. Caché de datos para que la app sea ultrarrápida al cambiar opciones
@st.cache_data
def cargar_datos():
    # En GitHub, este archivo deberá estar en la misma carpeta que app.py
    df = pd.read_excel('Tendencia_Ciclo_Sectores_IVAE_20022026.xlsx', parse_dates=['Fecha'], index_col='Fecha').sort_index()
    return df

df_base = cargar_datos()

# Diccionario de sectores
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

# 3. INTERFAZ DE USUARIO (Mobile-Friendly, sin barras laterales)
st.title("Reloj de la Tendencia-Ciclo")
st.markdown("Monitor macroeconómico de alta frecuencia para El Salvador.")

# Controles en el cuerpo principal (uno debajo del otro o en columnas si la pantalla es ancha)
col1, col2 = st.columns(2)
with col1:
    sector_elegido_nombre = st.selectbox("Seleccione el Sector:", list(sectores_nombres.values()))

# Encontrar el código original de R basado en la selección del usuario
col_r = list(sectores_nombres.keys())[list(sectores_nombres.values()).index(sector_elegido_nombre)]
col_tc = f"{col_r}_TC"

# 4. PROCESAMIENTO MATEMÁTICO (Solo para el sector seleccionado)
df = df_base.copy()
df['log_serie'] = np.log(df[col_tc])
cycle, trend = hpfilter(df['log_serie'], lamb=129600)
df['Ciclo_norm'] = (cycle - cycle.mean()) / cycle.std()
df['Delta'] = df['Ciclo_norm'].diff()

# 5. RENDERIZADO VISUAL CON PESTAÑAS (Ideal para celulares)
st.subheader(f"Análisis: {sector_elegido_nombre}")

periodos = [
    ('2022-11', '2023-11'),
    ('2023-11', '2024-11'),
    ('2024-11', '2025-11')
]

# Crear pestañas interactivas en lugar de paneles horizontales
tab1, tab2, tab3 = st.tabs(["2022-2023", "2023-2024", "2024-2025"])
tabs = [tab1, tab2, tab3]

config = {
    'colors': ['#2A5CAA', '#2E8B57', '#D93F3F'],
    'styles': ['-', '--', '-.'],
    'markers': ['o', 's', 'D']
}

def dibujar_reloj_individual(start, end, color, idx):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100) # Tamaño cuadrado perfecto para móvil
    
    meses_esp = {'01':'Ene', '02':'Feb', '03':'Mar', '04':'Abr', '05':'May', '06':'Jun', 
                 '07':'Jul', '08':'Ago', '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dic'}
    
    df_periodo = df.loc[start:end].sort_index()
    mask = df_periodo['Delta'].notna()

    ax.set_xlabel('Variación del Ciclo (ΔC$_t$)', fontsize=12)
    ax.set_ylabel('Ciclo Normalizado (C$_t$)', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle=':')

    ax.plot(df_periodo[mask]['Delta'], df_periodo[mask]['Ciclo_norm'],
            color=color, linestyle=config['styles'][idx], linewidth=2.5, alpha=0.9)

    for i, (fecha, marker) in enumerate(zip([start, end], config['markers'][:2])):
        punto = df_periodo.loc[fecha]
        x_val, y_val = punto['Delta'].iloc[0], punto['Ciclo_norm'].iloc[0]
        ax.scatter(x_val, y_val, color=color, s=150 + (i*30), edgecolor='black', zorder=5)

        mes_texto = meses_esp.get(fecha[5:7], '')
        offset_x = 25 if i == 0 else -25
        va_pos = 'bottom' if y_val > 0 else 'top'
        ax.annotate(f'{mes_texto} {fecha[:4]}\nΔ: {x_val:.2f}\nC: {y_val:.2f}',
                    (x_val, y_val), textcoords="offset points",
                    xytext=(offset_x, 15 if i == 0 else -15),
                    ha='right' if i == 1 else 'left', va=va_pos,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.95))

    # Cuadrantes
    for xy, color_cuad in [((0.5, 0.5), '#90EE90'), ((-0.5, 0.5), '#FFB6C1'), 
                           ((-0.5, -0.5), '#87CEEB'), ((0.5, -0.5), '#DDA0DD')]:
        ax.add_patch(plt.Rectangle(xy, 0.5, 0.5, color=color_cuad, alpha=0.1, transform=ax.transAxes))

    ax.axhline(0, color='#333333', linestyle=':', linewidth=1.5)
    ax.axvline(0, color='#333333', linestyle=':', linewidth=1.5)

    x_buf = 0.2 * (df_periodo['Delta'].max() - df_periodo['Delta'].min())
    y_buf = 0.2 * (df_periodo['Ciclo_norm'].max() - df_periodo['Ciclo_norm'].min())
    ax.set_xlim(df_periodo['Delta'].min() - x_buf, df_periodo['Delta'].max() + x_buf)
    ax.set_ylim(df_periodo['Ciclo_norm'].min() - y_buf, df_periodo['Ciclo_norm'].max() + y_buf)
    
    plt.tight_layout()
    return fig

# Generar y mostrar el gráfico en cada pestaña
for idx, (start, end) in enumerate(periodos):
    with tabs[idx]:
        figura = dibujar_reloj_individual(start, end, config['colors'][idx], idx)
        st.pyplot(figura, use_container_width=True)

# 6. Leyenda y Criterios al final (En un Expander para ahorrar espacio vertical)
with st.expander("Ver Leyenda y Criterios de Interpretación"):
    st.markdown("""
    **Trayectoria:**
    * 🟢 Inicio del período | ⬛ Fin del período
    
    **Cuadrantes:**
    * **Superior Derecho:** Δ > 0, C_t > 0 (Crecimiento por encima de la tendencia)
    * **Superior Izquierdo:** Δ < 0, C_t > 0 (Decrecimiento por encima de la tendencia)
    * **Inferior Izquierdo:** Δ < 0, C_t < 0 (Decrecimiento por debajo de la tendencia)
    * **Inferior Derecho:** Δ > 0, C_t < 0 (Crecimiento por debajo de la tendencia)
    """)
