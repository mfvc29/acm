import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Cargar modelos
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
    "Carmen de la Legua Reynoso": 45, "La Perla": 46, "La Punta": 47, "Ventanilla": 48, "Mi Perú": 49
}

# Diccionario de municipios
municipios = {
    'Lima Norte': ['Ancón', 'Carabayllo', 'Comas', 'Independencia', 'Los Olivos', 'Puente Piedra', 'San Martín de Porres'],
    'Lima Este': ['Ate Vitarte', 'Chaclacayo', 'Chosica', 'Cieneguilla', 'El Agustino', 'San Juan de Lurigancho', 'San Juan de Miraflores', 'San Luis', 'Santa Anita'],
    'Lima Top': ['Barranco', 'San Borja', 'Santiago de Surco', 'Miraflores', 'San Isidro', 'La Molina'],
    'Lima Centro': ['Breña', 'Cercado de Lima', 'La Victoria', 'Rímac'],
    'Lima Moderna': ['Jesús María', 'Lince', 'Magdalena del Mar', 'Pueblo Libre', 'San Miguel', 'Surquillo'],
    'Lima Sur': ['Chorrillos', 'Lurín', 'Pachacamac', 'Pucusana', 'Punta Hermosa', 'Punta Negra', 'San Bartolo', 'Santa María del Mar', 'Villa El Salvador', 'Villa María del Triunfo'],
    'Lima Callao': ['Callao', 'Bellavista', 'Carmen de la Legua Reynoso', 'La Perla', 'La Punta', 'Ventanilla', 'Mi Perú']
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
    zona = [nombre for nombre, num in zonas.items() if num == zona_num][0]
    municipio = obtener_municipio(zona)
    
    modelo_sigi = model_sigi_cu if tipo_propiedad == "Casa" else model_sigi_du
    entrada['Precio Venta log'] = np.log1p(precio_venta_pred)
    prediccion_log_sigi = modelo_sigi.predict(entrada)
    precio_cierre_pred_sigi = np.expm1(prediccion_log_sigi)[0]
    
    return precio_venta_pred, zona, municipio, precio_cierre_pred_sigi

# Streamlit UI
st.title("🏡 Predicción de Precios de Propiedades en Lima")
tipo_propiedad = st.selectbox("Selecciona el tipo de propiedad", ["Casa", "Departamento"])
area_total = st.number_input("📏 Área Total (m²)", min_value=10.0, format="%.2f")
dormitorios = st.number_input("🛏 Dormitorios", min_value=1)
banos = st.number_input("🚿 Baños", min_value=0)
estacionamiento = st.number_input("🚗 Estacionamientos", min_value=0)
zona_num = st.number_input("📍 Zona (Código)", min_value=1)

if st.button("Predecir Precio"):
    modelo = model_casas if tipo_propiedad == "Casa" else model_departamentos
    data = data_casas if tipo_propiedad == "Casa" else data_departamentos
    precio_estimado, zona, municipio, precio_cierre_estimado_sigi = predecir_precio_y_similares(area_total, dormitorios, banos, estacionamiento, zona_num, data, modelo)
    st.metric("💰 Precio Estimado", f"{precio_estimado:,.2f} soles")
    st.write(f"📍 Zona: {zona}, Municipio: {municipio}")
    st.metric("📉 Precio de Cierre Estimado SIGI", f"{precio_cierre_estimado_sigi:,.2f} soles")
