import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances

# Cargar el modelo previamente guardado
model = joblib.load('random_forest_model.pkl')

# Diccionario de zonas (distritos)
zonas = {
    'Barranco': 0, 'San Borja': 1, 'Santiago de Surco': 2, 'Miraflores': 3, 'San Isidro': 4, 'La Molina': 5,
    'Jesús María': 6, 'Pueblo Libre': 7, 'Lince': 8, 'San Miguel': 9, 'Magdalena del Mar': 10, 'Surquillo': 11,
    'Cercado de Lima': 12, 'La Victoria': 13, 'Breña': 14, 'Rímac': 15, 'Carabayllo': 16, 'Comas': 17,
    'San Martín de Porres': 18, 'Independencia': 19, 'Los Olivos': 20, 'Ancón': 21, 'Chorrillos': 22,
    'Punta Hermosa': 23, 'San Bartolo': 24, 'Punta Negra': 25, 'Cerro Azul': 26, 'Ate Vitarte': 27,
    'Chaclacayo': 28, 'Chosica': 29, 'San Luis': 30, 'El Agustino': 31, 'Cieneguilla': 32, 'La Perla': 33,
    'Callao': 34, 'Bellavista': 35
}

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
    precio_venta_pred = np.expm1(prediccion_log)[0]  # Convertir logaritmo a escala original

    # Calcular propiedades similares
    propiedades_similares = data.copy()
    features = ['Área Total log', 'Dormitorios', 'Baños', 'Estacionamiento', 'Zona_num']
    distancias = pairwise_distances(entrada[features], propiedades_similares[features])
    indices_similares = np.argsort(distancias[0])[:10]
    propiedades_similares_mostradas = propiedades_similares.iloc[indices_similares].copy()

    # Revertir logaritmos
    propiedades_similares_mostradas['Área Total'] = np.expm1(propiedades_similares_mostradas['Área Total log'])
    propiedades_similares_mostradas['Precio Venta'] = np.expm1(propiedades_similares_mostradas['Precio Venta log'])
    propiedades_similares_mostradas = propiedades_similares_mostradas[['Área Total', 'Dormitorios', 'Baños', 'Estacionamiento', 'Zona_num', 'Precio Venta']]

    # Asignar la zona y municipio según la selección
    zona = list(zonas.keys())[zona_num]  # Obtener el nombre de la zona por su índice
    municipio = zona  # Municipio es igual a la zona seleccionada

    return precio_venta_pred, propiedades_similares_mostradas, zona, municipio

# Cargar el dataset
data = pd.read_csv('dataset.csv').drop(columns=['Municipio_num'], errors='ignore')

# Interfaz de usuario con Streamlit
st.title("Predicción de Precio de casas en Lima")
st.write("Introduce los datos de la propiedad para obtener una estimación de su precio y las propiedades similares.")

# Formularios para los datos de entrada
area_total = st.number_input("Área Total (m²)", min_value=1)
dormitorios = st.number_input("Número de Dormitorios", min_value=1)
banos = st.number_input("Número de Baños", min_value=1)
estacionamiento = st.number_input("Número de Estacionamientos", min_value=0)

# Usar un dropdown para seleccionar el distrito
zona_select = st.selectbox("Selecciona el Distrito", list(zonas.keys()))

# Obtener el número de zona correspondiente
zona_num = zonas[zona_select]

if st.button("Predecir Precio"):
    precio_estimado, propiedades_similares, zona, municipio = predecir_precio_y_similares(area_total, dormitorios, banos, estacionamiento, zona_num, data)
    
    # Mostrar resultados
    st.subheader(f"El precio estimado de la propiedad es: {precio_estimado:.2f} soles.")
    st.write(f"Distrito: {zona} - Municipio: {municipio}")
    
    st.subheader("Propiedades Similares:")
    propiedades_similares = propiedades_similares.reset_index(drop=True)
    st.write(propiedades_similares)
