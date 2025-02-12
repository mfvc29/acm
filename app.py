import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Cargar los modelos previamente guardados
model_casas = joblib.load('random_forest_model.pkl')
model_departamentos = joblib.load('random_forest_model_du.pkl')

# Cargar datasets
data_casas = pd.read_csv('dataset.csv').drop(columns=['Municipio_num'], errors='ignore')
data_departamentos = pd.read_csv('dataset_du.csv').drop(columns=['Municipio_num'], errors='ignore')

# Diccionario de zonas (distritos)
zonas = {
    "Ancón": 1, "Ate Vitarte": 2, "Barranco": 3, "Breña": 4, "Carabayllo": 5, "Cercado de Lima": 6,
    "Chaclacayo": 7, "Chorrillos": 8, "Chosica": 9, "Cieneguilla": 10, "Comas": 11, "El Agustino": 12,
    "Independencia": 13, "Jesús María": 14, "La Molina": 15, "La Victoria": 16, "Lince": 17,
    "Los Olivos": 18, "Lurín": 19, "Magdalena del Mar": 20, "Miraflores": 21, "Pachacamac": 22,
    "Pucusana": 23, "Pueblo Libre": 24, "Puente Piedra": 25, "Punta Hermosa": 26, "Punta Negra": 27,
    "Rímac": 28, "San Bartolo": 29, "San Borja": 30, "San Isidro": 31, "San Juan de Lurigancho": 32,
    "San Juan de Miraflores": 33, "San Luis": 34, "San Martín de Porres": 35, "San Miguel": 36,
    "Santa Anita": 37, "Santa María del Mar": 38, "Santiago de Surco": 39, "Surquillo": 40,
    "Villa El Salvador": 41, "Villa María del Triunfo": 42, "Callao": 43
}

# Diccionario de municipios
municipios = {
    'Lima Top': ['Barranco', 'San Borja', 'Santiago de Surco', 'Miraflores', 'San Isidro', 'La Molina'],
    'Lima Moderna': ['Jesús María', 'Pueblo Libre', 'Lince', 'San Miguel', 'Magdalena del Mar', 'Surquillo'],
    'Lima Centro': ['Cercado de Lima', 'La Victoria', 'Breña', 'Rímac'],
    'Lima Norte': ['Carabayllo', 'Comas', 'San Martín de Porres', 'Independencia', 'Los Olivos', 'Ancón', 'Puente Piedra'],
    'Lima Sur': ['Chorrillos', 'Punta Hermosa', 'San Bartolo', 'Punta Negra', 'Villa El Salvador', 'Villa María del Triunfo', 'San Juan de Miraflores', 'Lurín', 'Pucusana', 'Pachacamac', 'Santa María del Mar'],
    'Lima Este': ['Ate Vitarte', 'Chaclacayo', 'Chosica', 'San Luis', 'El Agustino', 'Cieneguilla', 'Santa Anita', 'San Juan de Lurigancho'],
    'Lima Callao': ['Callao']
}

def obtener_municipio(zona):
    for municipio, distritos in municipios.items():
        if zona in distritos:
            return municipio
    return 'Municipio desconocido'

def predecir_precio_y_similares(area_total, dormitorios, banos, estacionamiento, zona_num, data, model):
    entrada = pd.DataFrame({
        'Área Total log': [np.log1p(area_total)],
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Estacionamiento': [estacionamiento],
        'Zona_num': [zona_num],
    })
    prediccion_log = model.predict(entrada)
    precio_venta_pred = np.expm1(prediccion_log)[0]
    propiedades_similares = data[data['Zona_num'] == zona_num].copy()
    if propiedades_similares.empty:
        return precio_venta_pred, pd.DataFrame(), None, None
    distancias = pairwise_distances(entrada[['Área Total log', 'Zona_num']], propiedades_similares[['Área Total log', 'Zona_num']])
    indices_similares = np.argsort(distancias[0])[:10]
    propiedades_similares_mostradas = propiedades_similares.iloc[indices_similares].copy()
    propiedades_similares_mostradas['Área Total'] = np.expm1(propiedades_similares_mostradas['Área Total log'])
    propiedades_similares_mostradas['Precio Venta'] = np.expm1(propiedades_similares_mostradas['Precio Venta log'])
    zona = next((k for k, v in zonas.items() if v == zona_num), None)
    municipio = obtener_municipio(zona) if zona else None
    return precio_venta_pred, propiedades_similares_mostradas, zona, municipio

st.title("🏡 Predicción de Precios de Propiedades en Lima")
tipo_propiedad = st.selectbox("Selecciona el tipo de propiedad", ["Casa", "Departamento"])
area_total = st.number_input("📏 Área Total (m²)", min_value=0.1, format="%.2f")
dormitorios = st.number_input("🛏 Número de Dormitorios", min_value=1)
banos = st.number_input("🚿 Número de Baños", min_value=0)
estacionamiento = st.number_input("🚗 Número de Estacionamientos", min_value=0)
zona_select = st.selectbox("📍 Selecciona el Distrito", list(zonas.keys()))
zona_num = zonas[zona_select]

if st.button("Predecir Precio"):
    modelo = model_casas if tipo_propiedad == "Casa" else model_departamentos
    data = data_casas if tipo_propiedad == "Casa" else data_departamentos
    precio_estimado, propiedades_similares, zona, municipio = predecir_precio_y_similares(area_total, dormitorios, banos, estacionamiento, zona_num, data, modelo)
    st.metric("Precio Estimado", f"{precio_estimado:,.2f} soles")
    if not propiedades_similares.empty:
        st.subheader("🏘 Propiedades Similares")
        st.write(propiedades_similares)
    else:
        st.warning("⚠️ No se encontraron propiedades similares en esta zona.")
