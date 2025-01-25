import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances

# Cargar el modelo previamente guardado
model = joblib.load('random_forest_model.pkl')

# Función para predecir el precio y las propiedades similares
def predecir_precio_y_similares(area_total, dormitorios, banos, estacionamiento, zona_num, data):
    entrada = pd.DataFrame({
        'Área Total log': [np.log1p(area_total)],  # Aplicar log para el modelo
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Estacionamiento': [estacionamiento],
        'Zona_num': [zona_num],
    })

    # Predicción del precio en logaritmo
    prediccion_log = model.predict(entrada)

    # Convertir la predicción de logaritmo a la escala original
    precio_venta_pred = np.expm1(prediccion_log)[0]

    # Calcular distancias para encontrar propiedades similares
    propiedades_similares = data.copy()

    # Calcular la distancia euclidiana entre la entrada y el dataset
    features = ['Área Total log', 'Dormitorios', 'Baños', 'Estacionamiento', 'Zona_num']
    distancias = pairwise_distances(entrada[features], propiedades_similares[features])
    indices_similares = np.argsort(distancias[0])[:10]  # Tomar los 5 más cercanos

    # Seleccionar propiedades similares
    propiedades_similares_mostradas = propiedades_similares.iloc[indices_similares].copy()

    # Revertir logaritmo para mostrar los valores originales
    propiedades_similares_mostradas['Área Total'] = np.expm1(propiedades_similares_mostradas['Área Total log'])
    propiedades_similares_mostradas['Precio Venta'] = np.expm1(propiedades_similares_mostradas['Precio Venta log'])

    # Eliminar las columnas logarítmicas para claridad
    propiedades_similares_mostradas = propiedades_similares_mostradas[['Área Total', 'Dormitorios', 'Baños', 'Estacionamiento', 'Zona_num', 'Precio Venta']]

    # Zonas y municipios
    zonas_municipios = {
        0: 'Lima Top',
        1: 'Lima Moderna',
        2: 'Lima Centro',
        3: 'Lima Norte',
        4: 'Lima Sur',
        5: 'Lima Este',
        6: 'Lima Callao'
    }

    # Mapear la zona numérica a su nombre de municipio
    if zona_num <= 5:
        zona = zonas_municipios[0]
        municipio = ["Barranco", "San Borja", "Santiago de Surco", "Miraflores", "San Isidro", "La Molina"][zona_num]
    else:
        zona = 'Zona desconocida'
        municipio = 'Municipio desconocido'

    return precio_venta_pred, propiedades_similares_mostradas, zona, municipio

# Cargar el dataset (asegúrate de que el archivo "dataset.csv" esté en la misma carpeta)
data = pd.read_csv('dataset.csv').drop(columns=['Municipio_num'], errors='ignore')

# Interfaz de usuario con Streamlit
st.title("Predicción de Precio de Propiedades en Lima")
st.write("Introduce los datos de la propiedad para obtener una estimación de su precio y las propiedades similares.")

# Formularios para los datos de entrada
area_total = st.number_input("Área Total (m²)", min_value=1)
dormitorios = st.number_input("Número de Dormitorios", min_value=1)
banos = st.number_input("Número de Baños", min_value=1)
estacionamiento = st.number_input("Número de Estacionamientos", min_value=0)
zona_num = st.number_input("Número de Zona (0-35)", min_value=0, max_value=35)

if st.button("Predecir Precio"):
    precio_estimado, propiedades_similares, zona, municipio = predecir_precio_y_similares(area_total, dormitorios, banos, estacionamiento, zona_num, data)
    
    # Mostrar resultados
    st.subheader(f"El precio estimado de la propiedad es: {precio_estimado:.2f} dólares.")
    st.write(f"Zona: {zona} - Municipio: {municipio}")
    
    st.subheader("Propiedades Similares:")
    propiedades_similares = propiedades_similares.reset_index(drop=True)
    st.write(propiedades_similares)
