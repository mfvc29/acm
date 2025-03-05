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


# Cargar los modelos previamente guardados
model_casas = joblib.load('random_forest_model.pkl')
model_departamentos = joblib.load('random_forest_model_du.pkl')
model_sigi_cu = joblib.load('modelo_cu.pkl')
model_sigi_du = joblib.load('modelo_du.pkl')

# Cargar datasets
data_casas = pd.read_csv('dataset.csv').drop(columns=['Municipio_num'], errors='ignore')
data_departamentos = pd.read_csv('dataset_du.csv').drop(columns=['Municipio_num'], errors='ignore')
data_sigi_cu = pd.read_csv('data_cu.csv').drop(columns=['Municipio_num'], errors='ignore')
data_sigi_du = pd.read_csv('data_du.csv').drop(columns=['Municipio_num'], errors='ignore')

# Diccionario de zonas (distritos)
# Mapa de zonas con números actualizados
zonas = {
    "Ancón": 1, "Ate Vitarte": 2, "Barranco": 3, "Breña": 4, "Carabayllo": 5, "Cercado de Lima": 6,
    "Chaclacayo": 7, "Chorrillos": 8, "Chosica": 9, "Cieneguilla": 10, "Comas": 11, "El Agustino": 12,
    "Independencia": 13, "Jesús María": 14, "La Molina": 15, "La Victoria": 16, "Lince": 17,
    "Los Olivos": 18, "Lurín": 19, "Magdalena del Mar": 20, "Miraflores": 21, "Pachacamac": 22,
    "Pucusana": 23, "Pueblo Libre": 24, "Puente Piedra": 25, "Punta Hermosa": 26, "Punta Negra": 27,
    "Rímac": 28, "San Bartolo": 29, "San Borja": 30, "San Isidro": 31, "San Juan de Lurigancho": 32,
    "San Juan de Miraflores": 33, "San Luis": 34, "San Martín de Porres": 35, "San Miguel": 36,
    "Santa Anita": 37, "Santa María del Mar": 38, "Santiago de Surco": 39, "Surquillo": 40,
    "Villa El Salvador": 41, "Villa María del Triunfo": 42, "Callao": 43, "Bellavista": 44,
    "Carmen de la Legua Reynoso": 45, "La Perla": 46, "La Punta": 47, "Ventanilla": 48, "Mi Perú": 49,
    "Barranca": 50, "Canta": 51, "Cañete": 52, "Huaral": 53, "Huarochirí": 54, "Huaura": 55,
    "Oyón": 56, "Yauyos": 57, "Cajatambo": 58
}
 
# Diccionario de municipios con la nueva categorización
municipios = {
    'Lima Norte': ['Ancón', 'Carabayllo', 'Comas', 'Independencia', 'Los Olivos', 'Puente Piedra', 'San Martín de Porres'],
    'Lima Este': ['Ate Vitarte', 'Chaclacayo', 'Chosica', 'Cieneguilla', 'El Agustino', 'San Juan de Lurigancho', 'San Juan de Miraflores', 'San Luis', 'Santa Anita'],
    'Lima Top': ['Barranco', 'San Borja', 'Santiago de Surco', 'Miraflores', 'San Isidro', 'La Molina'],
    'Lima Centro': ['Breña', 'Cercado de Lima', 'La Victoria', 'Rímac'],
    'Lima Moderna': ['Jesús María', 'Lince', 'Magdalena del Mar', 'Pueblo Libre', 'San Miguel', 'Surquillo'],
    'Lima Sur': ['Chorrillos', 'Lurín', 'Pachacamac', 'Pucusana', 'Punta Hermosa', 'Punta Negra', 'San Bartolo', 'Santa María del Mar', 'Villa El Salvador', 'Villa María del Triunfo'],
    'Lima Callao': ['Callao', 'Bellavista', 'Carmen de la Legua Reynoso', 'La Perla', 'La Punta', 'Ventanilla', 'Mi Perú'],
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

    # Filtrar propiedades similares por zona
    propiedades_similares = data[data['Zona_num'] == zona_num].copy()
    if propiedades_similares.empty:
        return precio_venta_pred, pd.DataFrame(), None, None

    # Distancias
    features = ['Área Total log', 'Zona_num']
    distancias = pairwise_distances(entrada[features], propiedades_similares[features])
    indices_similares = np.argsort(distancias[0])[:10]
    propiedades_similares_mostradas = propiedades_similares.iloc[indices_similares].copy()

    # Revertir logaritmos
    propiedades_similares_mostradas['Área Total'] = np.expm1(propiedades_similares_mostradas['Área Total log'])
    propiedades_similares_mostradas['Precio Venta'] = np.expm1(propiedades_similares_mostradas['Precio Venta log'])
    propiedades_similares_mostradas = propiedades_similares_mostradas[['Área Total', 'Dormitorios', 'Baños', 'Estacionamiento', 'Precio Venta','Enlaces']]
    
    # Asignar la zona y el municipio
    zona = [nombre for nombre, num in zonas.items() if num == zona_num][0]
    municipio = obtener_municipio(zona)
    return precio_venta_pred, propiedades_similares_mostradas, zona, municipio

# Función para obtener municipio basado en zona
def obtener_municipio(zona):
    for municipio, distritos in municipios.items():
        if zona in distritos:
            return municipio
    return 'Municipio desconocido'

# Interfaz de usuario
st.set_page_config(page_title="ACM - RE/MAX IRON", page_icon="🏡", layout="wide")
# Mostrar el logo
#st.image("Fondo.jpeg", width=150)  # Ajusta el ancho según lo necesites

# Título
col1, col2 = st.columns([1, 4])
with col1:
    st.image("Fondo.jpeg", width=200)
with col2:
    st.title("Análisis Comparativo de Mercado (ACM)")

# Descripción
st.write(
    "Ingresa los datos de la propiedad y selecciona el tipo de inmueble para obtener "
    "una estimación del precio de mercado. También podrás ver propiedades similares "
    "y comparar su relación con el valor estimado, lo que te ayudará a tomar decisiones más informadas."
)

# Opción para seleccionar el tipo de propiedad
tipo_propiedad = st.selectbox("Selecciona el tipo de propiedad", ["Casa", "Departamento"])

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

st.subheader("📌 Datos de Propidad")
# Formulario de entrada
area_total = st.number_input("📏 Área Total (m²)", min_value=10.0, format="%.2f")
dormitorios = st.number_input("🛏 Número de Dormitorios", min_value=1)
banos = st.number_input("🚿 Número de Baños", min_value=0)
estacionamiento = st.number_input("🚗 Número de Estacionamientos", min_value=0)
zona_select = st.selectbox("📍 Selecciona el Distrito", list(zonas.keys()))
zona_num = zonas[zona_select]

# Botón para realizar la predicción
if st.button("Predecir Precio"):
    if tipo_propiedad == "Casa":
        modelo = model_casas
        data = data_casas
    else:
        modelo = model_departamentos
        data = data_departamentos
    
    precio_estimado, propiedades_similares, zona, municipio = predecir_precio_y_similares(
        area_total, dormitorios, banos, estacionamiento, zona_num, data, modelo)

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
    st.metric("Precio Estimado", f"{precio_estimado:,.2f} soles")
    st.metric("💵 Precio Estimado en dólares", f"{precio_estimado_dolares:,.2f} dólares*")
    st.markdown(f"<p style='font-size: 10px;'>Tipo de cambio utilizado: {tipo_cambio:,.2f} soles por dólar</p>", unsafe_allow_html=True)
    st.metric("Precio Estimado por m²", f"{precio_m2:,.2f} soles/m²")
    st.metric("💵 Precio Estimado por m² en dólares", f"{precio_m2_dolares:,.2f} dólares/m²")

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
        color_punto_estimado = "black"  # Punto en negro para destacar
        size_puntos = 100  # Tamaño de los puntos

        # 📊 **Gráfico de Comparación de Precios**
        with col1:
                st.subheader("📈 Comparación de Precios")
                fig, ax = plt.subplots(figsize=(5, 1))

                # Línea base con degradado
                x_vals = np.linspace(limite_min, limite_max, 100)
                ax.plot(x_vals, [1] * 100, color=color_barra, linewidth=3, alpha=0.8)

                # Puntos de referencia
                ax.scatter([precio_min, precio_max], [1, 1], color=color_punto_base, s=size_puntos, edgecolors="white", linewidth=2)

                # Punto estimado (rojo si está fuera del rango)
                color_estimado = "red" if precio_estimado < precio_min or precio_estimado > precio_max else color_punto_estimado
                ax.scatter(precio_estimado, 1, color=color_estimado, s=size_puntos, edgecolors="white", linewidth=2)

                # Textos mejor alineados
                ax.text(precio_min, 1.05, f"S/. {precio_min:,.0f}", ha='center', fontsize=9, fontweight='bold')
                ax.text(precio_estimado, 1.10, f"S/. {precio_estimado:,.0f}", ha='center', fontsize=10, fontweight='bold', color=color_estimado)
                ax.text(precio_max, 1.05, f"S/. {precio_max:,.0f}", ha='center', fontsize=9, fontweight='bold')

                # Estética
                ax.set_xlim(limite_min - 5000, limite_max + 5000)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_frame_on(False)

                st.pyplot(fig)

        # 📊 **Gráfico de Comparación de Precios por m²**
        with col2:
                st.subheader("📈 Comparación de Precios por m²")
                fig, ax = plt.subplots(figsize=(5, 1))

                # Línea base
                x_vals = np.linspace(limite_m2_min, limite_m2_max, 100)
                ax.plot(x_vals, [1] * 100, color=color_barra, linewidth=3, alpha=0.8)

                # Puntos de referencia
                ax.scatter([precio_m2_area_min, precio_m2_area_max], [1, 1], color=color_punto_base, s=size_puntos, edgecolors="white", linewidth=2)

                # Punto estimado
                color_estimado_m2 = "red" if precio_m2 < precio_m2_area_min or precio_m2 > precio_m2_area_max else color_punto_estimado
                ax.scatter(precio_m2, 1, color=color_estimado_m2, s=size_puntos, edgecolors="white", linewidth=2)

                # Textos más claros
                ax.text(precio_m2_area_min, 1.05, f"S/. {precio_m2_area_min:,.0f}", ha='center', fontsize=9, fontweight='bold')
                ax.text(precio_m2, 1.10, f"S/. {precio_m2:,.0f}", ha='center', fontsize=10, fontweight='bold', color=color_estimado_m2)
                ax.text(precio_m2_area_max, 1.05, f"S/. {precio_m2_area_max:,.0f}", ha='center', fontsize=9, fontweight='bold')

                # Estética
                ax.set_xlim(limite_m2_min - 100, limite_m2_max + 100)
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
    if tipo_propiedad == "Casa":
        modelo = model_sigi_cu
        data = data_sigi_cu
    else:
        modelo = model_sigi_du
        data = data_sigi_du

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

    #  "Codigo" tipo str
    propiedades_similares_mostradas["Codigo"] = propiedades_similares_mostradas["Codigo"].astype(str)
    
    # Obtener la zona y municipio
    zonas_municipios = {num: (zona, obtener_municipio(zona)) for zona, num in zonas.items()}
    zona, municipio = zonas_municipios[zona_num]
    
    precio_estimado_cierre_dolares = precio_cierre_pred / tipo_cambio

    # Mostrar resultados
    st.metric("Precio Estimado de Cierre", f"{precio_cierre_pred:,.2f} soles")
    st.metric("💵 Precio Estimado de Cierre en dólares", f"{precio_estimado_cierre_dolares:,.2f} dólares*")
    st.markdown(f"<p style='font-size: 10px;'>Tipo de cambio utilizado: {tipo_cambio:,.2f} soles por dólar</p>", unsafe_allow_html=True)

    # Mostrar propiedades similares
    if not propiedades_similares_mostradas.empty:
        st.write(propiedades_similares_mostradas)
    else:
        st.warning("⚠️ No se encontraron propiedades similares para el precio de cierre.")
    
    # Actualizar el valor del precio estimado para el siguiente ciclo
    st.session_state.precio_estimado = precio_venta
