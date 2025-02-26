 
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Cargar los modelos previamente guardados
model_casas = joblib.load('random_forest_model.pkl')
model_departamentos = joblib.load('random_forest_model_du.pkl')
model_cu = joblib.load('modelo_cu.pkl')  # Nuevo modelo agregado

# Cargar datasets
data_casas = pd.read_csv('dataset.csv').drop(columns=['Municipio_num'], errors='ignore')
data_departamentos = pd.read_csv('dataset_du.csv').drop(columns=['Municipio_num'], errors='ignore')
data_cu = pd.read_csv('data_cu.csv').drop(columns=['Municipio_num'], errors='ignore')  # Nuevo dataset agregado

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

    return precio_venta_pred, propiedades_similares_mostradas

# Nueva función para predecir precio de cierre
def predecir_precio_cierre(area_total, dormitorios, banos, estacionamiento, zona_num, precio_venta_pred, data):
    entrada = pd.DataFrame({
        'Área Total log': [np.log1p(area_total)],
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Estacionamiento': [estacionamiento],
        'Zona_num': [zona_num],
        'Precio Venta log': [np.log1p(precio_venta_pred)]
    })

    prediccion_log = model_cu.predict(entrada)
    precio_cierre_pred = np.expm1(prediccion_log)[0]

    return precio_cierre_pred

# Interfaz de usuario
st.title("🏡 Predicción de Precios de Propiedades en Lima")

# Selección del tipo de propiedad
tipo_propiedad = st.selectbox("Selecciona el tipo de propiedad", ["Casa", "Departamento"])

# Entrada del usuario
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
    
    precio_estimado, propiedades_similares = predecir_precio_y_similares(
        area_total, dormitorios, banos, estacionamiento, zona_num, data, modelo)
    
    # Predicción de precio de cierre solo para casas
    if tipo_propiedad == "Casa":
        precio_cierre_estimado = predecir_precio_cierre(area_total, dormitorios, banos, estacionamiento, zona_num, precio_estimado, data_cu)
        st.metric("📉 Precio de Cierre Estimado", f"{precio_cierre_estimado:,.2f} soles")
    
    st.metric("Precio Estimado", f"{precio_estimado:,.2f} soles")
