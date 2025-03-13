import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

# Configuración de estilo global
st.set_page_config(
    page_title="ACM - RE/MAX IRON",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    /* Fondo de la aplicación y color del texto general */
    .stApp {
        background-color: #0c2749;
        color: white;
    }

    /* Estilo de los labels (etiquetas) */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: white !important;
        font-size: 16px;
        font-weight: bold;
    }

    /* Estilo de las barras de entrada */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #485c77;
        color: white;
        border-radius: 5px;
        padding: 8px;
        border: 1px solid #004AAD;
    }

    /* Estilo del botón */
    .stButton>button {
        background-color: #004AAD;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00387D;
    }

    /* Estilo de los títulos y subtítulos */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: white;
        font-weight: bold;
    }

    /* Estilo de los textos generales */
    .stMarkdown p {
        color: white;
    }

    /* Estilo de las métricas */
    .stMetric {
        color: white;
        background-color: #1e3a5f;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stMetric label {
        font-size: 14px;
        font-weight: bold;
        color: #ffffff;
    }
    .stMetric div {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
    }

    /* Estilo de las tablas */
    .stDataFrame {
        background-color: #1e3a5f;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stDataFrame th {
        background-color: #004AAD;
        color: white;
        font-weight: bold;
    }
    .stDataFrame td {
        background-color: #485c77;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cargar los modelos de venta 
model_casas = joblib.load('random_forest_model.pkl')
model_departamentos = joblib.load('random_forest_model_du.pkl')
modelo_ventalocal = joblib.load('random_forest_model_vl.pkl')
# Cargar los modelos de cierre de venta
model_sigi_cu = joblib.load('modelo_cu.pkl')
model_sigi_du = joblib.load('modelo_du.pkl')
modelo_sigi_vl = joblib.load('modelo_vl.pkl')
# Cargar datasets de venta
data_casas = pd.read_csv('dataset.csv').drop(columns=['Municipio_num'], errors='ignore')
data_departamentos = pd.read_csv('dataset_du.csv').drop(columns=['Municipio_num'], errors='ignore')
data_ventalocal = pd.read_csv('dataset_vl.csv').drop(columns=['Municipio_num'], errors='ignore')
# Cargar datasets de cierre de venta
data_sigi_cu = pd.read_csv('data_cu.csv').drop(columns=['Municipio_num'], errors='ignore')
data_sigi_du = pd.read_csv('data_du.csv').drop(columns=['Municipio_num'], errors='ignore')
data_sigi_vl = pd.read_csv('data_vl.csv').drop(columns=['Municipio_num'], errors='ignore')

# Cargar los modelos de alquiler
model_alquiler_casas = joblib.load('random_forest_model_cau.pkl')
model_alquiler_departamentos = joblib.load('random_forest_model_dau.pkl') 
model_alquiler_local = joblib.load('random_forest_model_al.pkl') 
# Cargar los modelos de cierre de alquiler
model_alquiler_cierre_casas = joblib.load('modelo_ac.pkl') 
model_alquiler_cierre_departamentos = joblib.load('modelo_ad.pkl') 
model_alquiler_cierre_local = joblib.load('modelo_lc.pkl')
# Cargar datasets de alquiler
data_alquiler_casas = pd.read_csv('dataset_cau.csv').drop(columns=['Municipio_num'], errors='ignore')
data_alquiler_departamentos = pd.read_csv('dataset_dau.csv').drop(columns=['Municipio_num'], errors='ignore') 
data_alquiler_local = pd.read_csv('dataset_al.csv').drop(columns=['Municipio_num'], errors='ignore') 
# Cargar datasets de cierre de alquiler
data_alquiler_cierre_casas = pd.read_csv('data_ac.csv').drop(columns=['Municipio_num'], errors='ignore') 
data_alquiler_cierre_departamentos = pd.read_csv('data_ad.csv').drop(columns=['Municipio_num'], errors='ignore') 
data_alquiler_cierre_local = pd.read_csv('data_lc.csv').drop(columns=['Municipio_num'], errors='ignore') 


# Diccionario de zonas (distritos)
zonas = {
    "Ancón": 1, "Puente Piedra": 2, "Carabayllo": 3, "Comas": 4, "Los Olivos": 5, "Independencia": 6, "San Martín de Porres": 7,
    "San Juan de Lurigancho": 8, "El Agustino": 9, "Santa Anita": 10, "San Luis": 11, "Ate Vitarte": 12, "Cieneguilla": 13, "Chaclacayo": 14, "Chosica": 15,
    "San Isidro": 16, "Miraflores": 17, "Barranco": 18, "San Borja": 19, "Santiago de Surco": 20, "La Molina": 21,
    "Cercado de Lima": 22, "Breña": 23, "La Victoria": 24, "Rímac": 25,
    "Jesús María": 26, "Lince": 27, "Magdalena del Mar": 28, "Pueblo Libre": 29, "San Miguel": 30, "Surquillo": 31,
    "Villa María del Triunfo": 32, "Villa El Salvador": 33, "Chorrillos": 34, "Lurín": 35, "Pachacamac": 36, "Pucusana": 37,
    "San Juan de Miraflores": 38, "Punta Hermosa": 39, "Punta Negra": 40, "San Bartolo": 41, "Santa María del Mar": 42,
    "Callao": 43, "Barranca": 44, "Canta": 45, "Cañete": 46, "Huaral": 47, "Huarochirí": 48, "Huaura": 49, "Oyón": 50, "Yauyos": 51, "Cajatambo": 52
}


# Diccionario de municipios con la nueva categorización
municipios = {
    'Lima Norte': ['Ancón', 'Carabayllo', 'Comas', 'Independencia', 'Los Olivos', 'Puente Piedra', 'San Martín de Porres'],
    'Lima Este': ['Ate Vitarte', 'Chaclacayo', 'Chosica', 'Cieneguilla', 'El Agustino', 'San Juan de Lurigancho', 'San Juan de Miraflores', 'San Luis', 'Santa Anita'],
    'Lima Top': ['Barranco', 'San Borja', 'Santiago de Surco', 'Miraflores', 'San Isidro', 'La Molina'],
    'Lima Centro': ['Breña', 'Cercado de Lima', 'La Victoria', 'Rímac'],
    'Lima Moderna': ['Jesús María', 'Lince', 'Magdalena del Mar', 'Pueblo Libre', 'San Miguel', 'Surquillo'],
    'Lima Sur': ['Chorrillos', 'Lurín', 'Pachacamac', 'Pucusana', 'Punta Hermosa', 'Punta Negra', 'San Bartolo', 'Santa María del Mar', 'Villa El Salvador', 'Villa María del Triunfo'],
    'Lima Callao': ['Callao'],
    'Fuera de Lima': ['Barranca', 'Canta', 'Cañete', 'Huaral', 'Huarochirí', 'Huaura', 'Oyón', 'Yauyos', 'Cajatambo']
}

# Función para obtener municipio basado en zona
def obtener_municipio(zona):
    for municipio, distritos in municipios.items():
        if zona in distritos:
            return municipio
    return 'Municipio desconocido'

# Función para predecir precio y propiedades similares
def predecir_precio_y_similares(area_total, dormitorios, banos, estacionamiento, zona_num, data, model):
    entrada = pd.DataFrame({
        'Área Total log': [np.log1p(area_total)],
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Estacionamiento': [estacionamiento],
        'Zona_num': [zona_num],
    })

    # Predicción del precio en logaritmo
    prediccion_log = model.predict(entrada)
    precio_venta_pred = np.expm1(prediccion_log)[0]

    # Filtrar propiedades en la zona seleccionada
    propiedades_en_zona = data[data['Zona_num'] == zona_num].copy()

    # Si no hay propiedades en la zona, mostrar todas las propiedades en la zona
    if propiedades_en_zona.empty:
        st.warning(f"⚠️ No se encontraron propiedades similares en el distrito seleccionado. Mostrando todas las propiedades en la zona.")
        propiedades_en_zona = data.copy()  # Mostrar todas las propiedades disponibles

    # Calcular distancias para propiedades similares
    features = ['Área Total log', 'Zona_num']
    distancias = pairwise_distances(entrada[features], propiedades_en_zona[features])
    indices_similares = np.argsort(distancias[0])[:10]  # Tomar las 10 más cercanas
    propiedades_similares_mostradas = propiedades_en_zona.iloc[indices_similares].copy()

    # Revertir logaritmos
    propiedades_similares_mostradas['Área Total'] = np.expm1(propiedades_similares_mostradas['Área Total log'])
    propiedades_similares_mostradas['Precio Venta'] = np.expm1(propiedades_similares_mostradas['Precio Venta log'])
    propiedades_similares_mostradas = propiedades_similares_mostradas[['Área Total', 'Dormitorios', 'Baños', 'Estacionamiento', 'Precio Venta', 'Enlaces']]

    # Asignar la zona y el municipio
    zona = [nombre for nombre, num in zonas.items() if num == zona_num][0]
    municipio = obtener_municipio(zona)
    return precio_venta_pred, propiedades_similares_mostradas, zona, municipio


# Interfaz de usuario
col1, col2 = st.columns([1, 4])
with col1:
    st.image("V1.png", width=200)
with col2:
    st.title("Análisis Comparativo de Mercado (ACM)")
    st.markdown("""
        <p style='font-size: 16px; color: white;'>
            Ingresa los datos de la propiedad y selecciona el tipo de inmueble para obtener
            una estimación del precio de mercado. También podrás ver propiedades similares
            y comparar su relación con el valor estimado, lo que te ayudará a tomar decisiones más informadas.
        </p>
    """, unsafe_allow_html=True)

# Opción para seleccionar el tipo de propiedad
tipo_propiedad = st.selectbox("Selecciona el tipo de propiedad", ["Casa", "Departamento","Local Comercial"])
tipo_operacion = st.selectbox("Selecciona el tipo de operación", ["Venta", "Alquiler"])

# Datos del propietario
st.subheader("📌 Datos del Propietario")
nombre_propietario = st.text_input("Nombre del Propietario")
correo_propietario = st.text_input("Correo del Propietario")
telefono_propietario = st.text_input("Teléfono del Propietario")

# Datos del agente
st.subheader("📌 Datos del Agente")
nombre_agente = st.text_input("Nombre del Agente")
correo_agente = st.text_input("Correo del Agente")
telefono_agente = st.text_input("Teléfono del Agente")

# Almacenar los datos (aunque no se usen por ahora)
datos_adicionales = {
    "Propietario": {
        "Nombre": nombre_propietario,
        "Correo": correo_propietario,
        "Teléfono": telefono_propietario
    },
    "Agente": {
        "Nombre": nombre_agente,
        "Correo": correo_agente,
        "Teléfono": telefono_agente
    }
}

st.subheader("📌 Datos de Propiedad")
# Formulario de entrada
area_total = st.number_input("📏 Área Total (m²)", min_value=10.0, format="%.2f")
dormitorios = st.number_input("🛏 Número de Dormitorios", min_value=1)
banos = st.number_input("🚿 Número de Baños", min_value=0)
estacionamiento = st.number_input("🚗 Número de Estacionamientos", min_value=0)
zona_select = st.selectbox("📍 Selecciona el Distrito", list(zonas.keys()))
zona_num = zonas[zona_select]


# Botón para realizar la predicción
if st.button("Predecir Precio"):
    if tipo_operacion == "Venta":
        if tipo_propiedad == "Casa":
            modelo = model_casas
            data = data_casas
        elif tipo_propiedad == "Departamento":
            modelo = model_departamentos
            data = data_departamentos
        else:  # Local Comercial
            modelo = modelo_ventalocal
            data = data_ventalocal
    elif tipo_operacion == "Alquiler":
        if tipo_propiedad == "Casa":
            modelo = model_alquiler_casas
            data = data_alquiler_casas
        elif tipo_propiedad == "Departamento":
            modelo = model_alquiler_departamentos
            data = data_alquiler_departamentos
        else:  # Local Comercial
            modelo = model_alquiler_local
            data = data_alquiler_local
    else:
        st.error("Operación no válida. Selecciona 'Venta' o 'Alquiler'.")

    # Llamar a la función de predicción
    precio_estimado, propiedades_similares, zona, municipio = predecir_precio_y_similares(
        area_total, dormitorios, banos, estacionamiento, zona_num, data, modelo
    )

    propiedades_similares['Área Total'] = propiedades_similares['Área Total'].round(2)
    propiedades_similares['Estacionamiento'] = propiedades_similares['Estacionamiento'].astype(int)
    propiedades_similares['Dormitorios'] = propiedades_similares['Dormitorios'].astype(int)
    propiedades_similares['Baños'] = propiedades_similares['Baños'].astype(int)
 
    # Mostrar los resultados
    tipo_cambio = 3.71  # Tipo de cambio de soles a dólares

    # Convertir el precio estimado a dólares
    precio_estimado_dolares = precio_estimado / tipo_cambio
    # Calcular precio por metro cuadrado
    precio_m2 = precio_estimado / area_total
    # Convertir el precio por m2 a dólares
    precio_m2_dolares = precio_m2 / tipo_cambio

    # Mostrar resultados
    st.subheader(f"📊 Resultados para la propiedad en {zona}, {municipio}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precio Estimado", f"S/. {precio_estimado:,.2f}")
        st.metric("Precio Estimado por m²", f"S/. {precio_m2:,.2f}/m²")
    with col2:
        st.metric("💵 Precio Estimado en dólares", f"$ {precio_estimado_dolares:,.2f}")
        st.metric("💵 Precio Estimado por m² en dólares", f"$ {precio_m2_dolares:,.2f}/m²")
    st.markdown(f"<p style='font-size: 10px;'>Tipo de cambio utilizado: {tipo_cambio:,.2f} soles por dólar</p>", unsafe_allow_html=True)

    if not propiedades_similares.empty:
        # Calcular valores clave
        precio_min = propiedades_similares['Precio Venta'].min()
        precio_max = propiedades_similares['Precio Venta'].max()
        diferencia_min = precio_estimado - precio_min
        diferencia_max = precio_max - precio_estimado

        area_min = propiedades_similares['Área Total'].min()
        area_max = propiedades_similares['Área Total'].max()

        # Evitar división por cero
        if area_min == 0 or area_max == 0:
            precio_m2_area_min, precio_m2_area_max = 0, 0
        else:
            precio_m2_area_min = precio_min / area_min
            precio_m2_area_max = precio_max / area_max

        # Ajustar límites para incluir el estimado sin perder referencias originales
        limite_min = min(precio_min, precio_estimado)
        limite_max = max(precio_max, precio_estimado)

        limite_m2_min = min(precio_m2_area_min, precio_m2)
        limite_m2_max = max(precio_m2_area_max, precio_m2)

        # Crear columnas
        col1, col2 = st.columns([1, 1])
        # 🎨 **Colores personalizados**
        color_barra = "#004AAD"  # Azul Remax
        color_punto_base = "#E02020"  # Rojo Remax
        color_punto_estimado = "#18171C"  # Punto en negro para destacar
        
        size_puntos = 100  # Tamaño de los puntos

        # 📊 **Gráfico de Comparación de Precios**
        # 📊 **Gráfico de Comparación de Precios**
        with col1:
            st.subheader("📈 Comparación de Precios")
            fig, ax = plt.subplots(figsize=(8, 2))

            # Definir los límites del gráfico
            limite_inferior = min(precio_min, precio_estimado) - 5000  # Margen adicional
            limite_superior = max(precio_max, precio_estimado) + 5000  # Margen adicional

            # Línea base con degradado
            x_vals = np.linspace(limite_inferior, limite_superior, 100)
            ax.plot(x_vals, [1] * 100, color=color_barra, linewidth=3, alpha=0.8, label="Rango de precios")

            # Puntos de referencia
            ax.scatter([precio_min, precio_max], [1, 1], color=color_punto_base, s=size_puntos, edgecolors="white", linewidth=2, label="Precio mínimo/máximo")

            # Punto estimado (rojo si está fuera del rango)
            color_estimado = "red" if precio_estimado < precio_min or precio_estimado > precio_max else color_punto_estimado
            ax.scatter(precio_estimado, 1, color=color_estimado, s=size_puntos, edgecolors="white", linewidth=2, label="Precio estimado")

            # Textos mejor alineados
            ax.text(precio_min, 1.05, f"S/. {precio_min:,.0f}", ha='center', fontsize=10, fontweight='light', color=color_punto_base)
            ax.text(precio_estimado, 1.10, f"S/. {precio_estimado:,.0f}", ha='center', fontsize=12, fontweight='light', color=color_estimado)
            ax.text(precio_max, 1.05, f"S/. {precio_max:,.0f}", ha='center', fontsize=10, fontweight='light', color=color_punto_base)

            # Etiquetas
            ax.text(limite_inferior - 5000, 0.95, "Precio mínimo en la zona", ha='left', fontsize=10, fontweight='light', color='white')
            ax.text(limite_superior + 5000, 0.95, "Precio máximo en la zona", ha='right', fontsize=10, fontweight='light', color='white')
            ax.text(precio_estimado, 1.15, "Precio estimado", ha='center', fontsize=12, fontweight='light', color=color_estimado)

            # Líneas de referencia
            ax.vlines(precio_min, 0.95, 1.05, color=color_punto_base, linestyle="--", alpha=0.5)
            ax.vlines(precio_max, 0.95, 1.05, color=color_punto_base, linestyle="--", alpha=0.5)
            ax.vlines(precio_estimado, 0.95, 1.10, color=color_estimado, linestyle="--", alpha=0.5)

            # Leyenda
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10, frameon=False)

            # Estética
            ax.set_xlim(limite_inferior, limite_superior)  # Rango dinámico
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_frame_on(False)

            st.pyplot(fig)

        # 📊 **Gráfico de Comparación de Precios por m²**
        # 📊 **Gráfico de Comparación de Precios por m²**
        with col2:
            st.subheader("📈 Comparación de Precios por m²")
            fig, ax = plt.subplots(figsize=(8, 2))

            # Definir los límites del gráfico
            limite_inferior_m2 = min(precio_m2_area_min, precio_m2) - 100  # Margen adicional
            limite_superior_m2 = max(precio_m2_area_max, precio_m2) + 100  # Margen adicional

            # Línea base
            x_vals = np.linspace(limite_inferior_m2, limite_superior_m2, 100)
            ax.plot(x_vals, [1] * 100, color=color_barra, linewidth=3, alpha=0.8, label="Rango de precios por m²")

            # Puntos de referencia
            ax.scatter([precio_m2_area_min, precio_m2_area_max], [1, 1], color=color_punto_base, s=size_puntos, edgecolors="white", linewidth=2, label="Precio mínimo/máximo por m²")

            # Punto estimado
            color_estimado_m2 = "red" if precio_m2 < precio_m2_area_min or precio_m2 > precio_m2_area_max else color_punto_estimado
            ax.scatter(precio_m2, 1, color=color_estimado_m2, s=size_puntos, edgecolors="white", linewidth=2, label="Precio estimado por m²")

            # Textos más claros
            ax.text(precio_m2_area_min, 1.05, f"S/. {precio_m2_area_min:,.0f}", ha='center', fontsize=10, fontweight='light', color=color_punto_base)
            ax.text(precio_m2, 1.10, f"S/. {precio_m2:,.0f}", ha='center', fontsize=12, fontweight='light', color=color_estimado_m2)
            ax.text(precio_m2_area_max, 1.05, f"S/. {precio_m2_area_max:,.0f}", ha='center', fontsize=10, fontweight='light', color=color_punto_base)

            # Etiquetas
            ax.text(limite_inferior_m2 - 100, 0.95, "Precio mínimo por m² en la zona", ha='left', fontsize=10, fontweight='light', color='white')
            ax.text(limite_superior_m2 + 100, 0.95, "Precio máximo por m² en la zona", ha='right', fontsize=10, fontweight='light', color='white')
            ax.text(precio_m2, 1.15, "Precio estimado por m²", ha='center', fontsize=12, fontweight='light', color=color_estimado_m2)

            # Líneas de referencia
            ax.vlines(precio_m2_area_min, 0.95, 1.05, color=color_punto_base, linestyle="--", alpha=0.5)
            ax.vlines(precio_m2_area_max, 0.95, 1.05, color=color_punto_base, linestyle="--", alpha=0.5)
            ax.vlines(precio_m2, 0.95, 1.10, color=color_estimado_m2, linestyle="--", alpha=0.5)

            # Leyenda
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10, frameon=False)

            # Estética
            ax.set_xlim(limite_inferior_m2, limite_superior_m2)  # Rango dinámico
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_frame_on(False)

            st.pyplot(fig)
            
        # Tabla de propiedades similares
        st.subheader("🏘 Propiedades Similares")
        propiedades_similares = propiedades_similares.reset_index(drop=True)    
        st.write(propiedades_similares)
    else:
        st.warning("⚠️ No se encontraron propiedades similares en esta zona.")
        
    precio_venta = precio_estimado
    
    # Asegurarse de usar el modelo adecuado
    if tipo_operacion == "Venta":
        if tipo_propiedad == "Casa":
            modelo_sigi = model_sigi_cu
            data_sigi = data_sigi_cu
        elif tipo_propiedad == "Departamento":
            modelo_sigi = model_sigi_du
            data_sigi = data_sigi_du
        else:      # Local Comercial
            modelo_sigi = modelo_sigi_vl
            data_sigi = data_sigi_vl
    elif tipo_operacion == "Alquiler":
        if tipo_propiedad == "Casa":
            modelo_sigi = model_alquiler_cierre_casas
            data_sigi = data_alquiler_cierre_casas
        elif tipo_propiedad == "Departamento":
            modelo_sigi = model_alquiler_cierre_departamentos
            data_sigi = data_alquiler_cierre_departamentos
        else:      # Local Comercial
            modelo_sigi = model_alquiler_cierre_local
            data_sigi = data_alquiler_cierre_local
        

    # Crear el DataFrame para la predicción de precio de cierre
    entrada = pd.DataFrame({
        'Área Total log': [np.log1p(area_total)],
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Estacionamiento': [estacionamiento],
        'Zona_num': [zona_num],
        'Precio Venta log': [np.log1p(precio_venta)]  # Precio venta log
    })

    # Predicción del precio de cierre en logaritmo
    prediccion_log = modelo.predict(entrada)
    precio_cierre_pred = np.expm1(prediccion_log)[0]

    # Calcular precio por metro cuadrado de cierre
    precio_cierre_m2 = precio_cierre_pred / area_total

    # Convertir el precio de cierre por m2 a dólares
    precio_cierre_m2_dolares = precio_cierre_m2 / tipo_cambio

    # Filtrar propiedades de la misma zona
    propiedades_similares = data[data['Zona_num'] == zona_num].copy()

    # Si no hay propiedades en la zona, buscar en todo el dataset
    if propiedades_similares.empty:
        propiedades_similares = data.copy()
        st.warning("⚠️ No se encontraron propiedades similares en esta zona. Mostrando propiedades similares a nivel general.")

    # Calcular la distancia euclidiana entre la entrada y el dataset
    features = ['Área Total log', 'Precio Venta log']
    distancias = pairwise_distances(entrada[features], propiedades_similares[features])
    indices_similares = np.argsort(distancias[0])[:10]  # Tomar las 10 más cercanas

    # Seleccionar propiedades similares
    propiedades_similares_mostradas = propiedades_similares.iloc[indices_similares].copy()

    # Revertir logaritmo para mostrar los valores originales
    propiedades_similares_mostradas['Área Total'] = np.expm1(propiedades_similares_mostradas['Área Total log'])
    propiedades_similares_mostradas['Precio Cierre'] = np.expm1(propiedades_similares_mostradas['Precio Cierre log'])

    # Calcular Precio por Metro Cuadrado para cada propiedad
    propiedades_similares_mostradas['Precio Cierre por m²'] = propiedades_similares_mostradas['Precio Cierre'] / propiedades_similares_mostradas['Área Total']

    # Redondear valores
    propiedades_similares_mostradas['Precio Cierre por m²'] = propiedades_similares_mostradas['Precio Cierre por m²'].round(2)

    # Incluir la columna en la tabla
    propiedades_similares_mostradas = propiedades_similares_mostradas[['Área Total', 'Dormitorios', 'Baños', 'Estacionamiento', 'Precio Cierre', 'Precio Cierre por m²', 'Codigo']]

    # Asegurar que "Codigo" sea tipo str
    propiedades_similares_mostradas["Codigo"] = propiedades_similares_mostradas["Codigo"].astype(str)

    # Obtener la zona y municipio
    zonas_municipios = {num: (zona, obtener_municipio(zona)) for zona, num in zonas.items()}
    zona, municipio = zonas_municipios[zona_num]

    precio_estimado_cierre_dolares = precio_cierre_pred / tipo_cambio

    # Mostrar resultados
    st.subheader("📊 Resultados de Precio de Cierre")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precio Estimado de Cierre", f"S/. {precio_cierre_pred:,.2f}")
        st.metric("Precio Estimado de Cierre por m²", f"S/. {precio_cierre_m2:,.2f}/m²")
    with col2:
        st.metric("💵 Precio Estimado de Cierre en dólares", f"$ {precio_estimado_cierre_dolares:,.2f}")
        st.metric("💵 Precio Estimado de Cierre por m² en dólares", f"$ {precio_cierre_m2_dolares:,.2f}/m²")
    st.markdown(f"<p style='font-size: 10px;'>Tipo de cambio utilizado: {tipo_cambio:,.2f} soles por dólar</p>", unsafe_allow_html=True)

    # Mostrar propiedades similares
    if not propiedades_similares_mostradas.empty:
        st.write(propiedades_similares_mostradas)
    else:
        st.warning("⚠️ No se encontraron propiedades similares para el precio de cierre.")
    
    # Actualizar el valor del precio estimado para el siguiente ciclo
    st.session_state.precio_estimado = precio_venta