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

# 3. INTERFAZ DE USUARIO PRINCIPAL
st.title("Reloj de la Tendencia-Ciclo")
st.markdown("Monitor macroeconómico de alta frecuencia para El Salvador.")

# Selección de Sector
sector_elegido_nombre = st.selectbox("Seleccione el Sector Analítico:", list(sectores_nombres.values()))

# Procesamiento del nombre de la columna
col_r = list(sectores_nombres.keys())[list(sectores_nombres.values()).index(sector_elegido_nombre)]
col_tc = f"{col_r}_TC"

# 4. PROCESAMIENTO MATEMÁTICO (HP Filter)
df = df_base.copy()
df['log_serie'] = np.log(df[col_tc])
cycle, trend = hpfilter(df['log_serie'], lamb=129600)
df['Ciclo_norm'] = (cycle - cycle.mean()) / cycle.std()
df['Delta'] = df['Ciclo_norm'].diff()

st.divider()

# ==========================================
# 5. LA MAGIA DINÁMICA: SLIDER DE TIEMPO
# ==========================================
st.subheader(f"Análisis: {sector_elegido_nombre}")

# Obtener fechas límite del dataframe
min_date = df.index.min().to_pydatetime()
max_date = df.index.max().to_pydatetime()

# Calcular un período por defecto (ej. los últimos 2 años)
try:
    default_start = max_date.replace(year=max_date.year - 2)
except ValueError:
    default_start = min_date

# El Slider interactivo
fechas = st.slider(
    "Deslice para definir el período exacto de análisis:",
    min_value=min_date,
    max_value=max_date,
    value=(default_start, max_date),
    format="MMM YYYY"  # Formato amigable: Ene 2023, Feb 2024, etc.
)

# Convertir la selección del usuario a texto para el filtro de Pandas
start_str = fechas[0].strftime('%Y-%m')
end_str = fechas[1].strftime('%Y-%m')

# ==========================================
# 6. RENDERIZADO VISUAL ÚNICO Y DINÁMICO
# ==========================================
def dibujar_reloj_dinamico(start, end):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    meses_esp = {'01':'Ene', '02':'Feb', '03':'Mar', '04':'Abr', '05':'May', '06':'Jun', 
                 '07':'Jul', '08':'Ago', '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dic'}
    
    # Filtrar el dataframe con las fechas exactas del slider
    df_periodo = df.loc[start:end].sort_index()
    
    if len(df_periodo) < 2:
        st.warning("Seleccione un período más amplio (al menos 2 meses) para ver la trayectoria.")
        return fig
        
    mask = df_periodo['Delta'].notna()
    color_grafico = '#2A5CAA' # Azul profesional por defecto

    ax.set_xlabel('Variación del Ciclo (ΔC$_t$)', fontsize=12)
    ax.set_ylabel('Ciclo Normalizado (C$_t$)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Línea de trayectoria
    ax.plot(df_periodo[mask]['Delta'], df_periodo[mask]['Ciclo_norm'],
            color=color_grafico, linestyle='-', linewidth=2.5, alpha=0.9)

    # Puntos de inicio y fin
    marcadores = ['o', 's']
    for i, (fecha, marker) in enumerate(zip([start, end], marcadores)):
        try:
            # Buscar la fecha exacta en el índice. Si el usuario escoge un mes que es NaN por el diff(), lo manejamos
            punto = df_periodo.loc[df_periodo.index.astype(str).str.startswith(fecha)].iloc[0]
            x_val, y_val = punto['Delta'], punto['Ciclo_norm']
            
            if pd.isna(x_val) or pd.isna(y_val):
                continue
                
            ax.scatter(x_val, y_val, color=color_grafico, s=150 + (i*50), edgecolor='black', marker=marker, zorder=5)

            mes_texto = meses_esp.get(fecha[5:7], '')
            offset_x = 25 if i == 0 else -25
            va_pos = 'bottom' if y_val > 0 else 'top'
            
            ax.annotate(f'{mes_texto} {fecha[:4]}\nΔ: {x_val:.2f}\nC: {y_val:.2f}',
                        (x_val, y_val), textcoords="offset points",
                        xytext=(offset_x, 15 if i == 0 else -15),
                        ha='right' if i == 1 else 'left', va=va_pos,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_grafico, alpha=0.95))
        except IndexError:
            pass # Si no encuentra el mes exacto (raro, pero previene caídas)

    # Cuadrantes de interpretación
    for xy, color_cuad in [((0.5, 0.5), '#90EE90'), ((-0.5, 0.5), '#FFB6C1'), 
                           ((-0.5, -0.5), '#87CEEB'), ((0.5, -0.5), '#DDA0DD')]:
        ax.add_patch(plt.Rectangle(xy, 0.5, 0.5, color=color_cuad, alpha=0.1, transform=ax.transAxes))

    ax.axhline(0, color='#333333', linestyle=':', linewidth=1.5)
    ax.axvline(0, color='#333333', linestyle=':', linewidth=1.5)

    # Ajuste dinámico de márgenes
    x_buf = 0.2 * (df_periodo['Delta'].max() - df_periodo['Delta'].min())
    y_buf = 0.2 * (df_periodo['Ciclo_norm'].max() - df_periodo['Ciclo_norm'].min())
    
    # Prevenir errores si el buffer es 0
    x_buf = x_buf if x_buf > 0 else 0.5
    y_buf = y_buf if y_buf > 0 else 0.5
    
    ax.set_xlim(df_periodo['Delta'].min() - x_buf, df_periodo['Delta'].max() + x_buf)
    ax.set_ylim(df_periodo['Ciclo_norm'].min() - y_buf, df_periodo['Ciclo_norm'].max() + y_buf)
    
    plt.tight_layout()
    return fig

# Mostrar el gráfico dinámico
figura_actual = dibujar_reloj_dinamico(start_str, end_str)
st.pyplot(figura_actual, use_container_width=True)

# 7. Leyenda colapsable
with st.expander("Ver Leyenda y Criterios de Interpretación", expanded=False):
    st.markdown("""
    **Trayectoria del Reloj:**
    * 🔵 **Círculo:** Inicio del período seleccionado.
    * 🟦 **Cuadrado:** Fin del período seleccionado.
    
    **Interpretación de Cuadrantes:**
    * 🟩 **Superior Derecho:** Crecimiento por encima de la tendencia.
    * 🟥 **Superior Izquierdo:** Decrecimiento por encima de la tendencia (Desaceleración).
    * 🟦 **Inferior Izquierdo:** Decrecimiento por debajo de la tendencia (Recesión).
    * 🟪 **Inferior Derecho:** Crecimiento por debajo de la tendencia (Recuperación).
    """)
