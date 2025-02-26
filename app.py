import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Cargar los modelos previamente guardados
model_casas = joblib.load('random_forest_model.pkl')
model_departamentos = joblib.load('random_forest_model_du.pkl')
model_cierre_casas = joblib.load('modelo_cu.pkl')
#model_cierre_departamentos = joblib.load('modelo_cu_departamentos.pkl')

# Cargar datasets
data_casas = pd.read_csv('dataset.csv').drop(columns=['Municipio_num'], errors='ignore')
data_departamentos = pd.read_csv('dataset_du.csv').drop(columns=['Municipio_num'], errors='ignore')
data_cierre_casas = pd.read_csv('data_cu.csv').drop(columns=['Municipio_num'], errors='ignore')
#data_cierre_departamentos = pd.read_csv('data_cu_departamentos.csv').drop(columns=['Municipio_num'], errors='ignore')

# Diccionario de zonas (distritos)
zonas = {"Ancón": 1, "Ate Vitarte": 2, "Barranco": 3, "Breña": 4, "Carabayllo": 5, "Cercado de Lima": 6,
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
         "Oyón": 56, "Yauyos": 57, "Cajatambo": 58}

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

# Función para predecir precio de venta
def predecir_precio_venta(area_total, dormitorios, banos, estacionamiento, zona_num, data, model):
    entrada = pd.DataFrame({
        'Área Total log': [np.log1p(area_total)],
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Estacionamiento': [estacionamiento],
        'Zona_num': [zona_num],
    })
    prediccion_log = model.predict(entrada)
    precio_pred = np.expm1(prediccion_log)[0]
    return precio_pred

# Función para predecir precio de cierre con el precio de venta como nueva variable de entrada
def predecir_precio_cierre(area_total, dormitorios, banos, estacionamiento, zona_num, precio_venta, model_cierre):
    entrada = pd.DataFrame({
        'Área Total log': [np.log1p(area_total)],
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Estacionamiento': [estacionamiento],
        'Zona_num': [zona_num],
        'Precio Venta': [precio_venta],
    })
    prediccion_log = model_cierre.predict(entrada)
    precio_pred = np.expm1(prediccion_log)[0]
    return precio_pred

# Interfaz de usuario
st.set_page_config(page_title="Predicción de Precios de Propiedades", layout="wide")
st.title("🏡 Predicción de Precios de Propiedades en Lima")
st.sidebar.header("Parámetros de Entrada")
