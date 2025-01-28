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
    'Barranco': 0, 'San Borja': 1, 'Santiago de Surco': 2, 'Miraflores': 3, 'San Isidro': 4, 'La Molina': 5,
    'Jesús María': 6, 'Pueblo Libre': 7, 'Lince': 8, 'San Miguel': 9, 'Magdalena del Mar': 10, 'Surquillo': 11,
    'Cercado de Lima': 12, 'La Victoria': 13, 'Breña': 14, 'Rímac': 15, 'Carabayllo': 16, 'Comas': 17,
    'San Martín de Porres': 18, 'Independencia': 19, 'Los Olivos': 20, 'Ancón': 21, 'Chorrillos': 22,
    'Punta Hermosa': 23, 'San Bartolo': 24, 'Punta Negra': 25, 'Cerro Azul': 26, 'Ate Vitarte': 27,
    'Chaclacayo': 28, 'Chosica': 29, 'San Luis': 30, 'El Agustino': 31, 'Cieneguilla': 32, 'La Perla': 33,
    'Callao': 34, 'Bellavista': 35
}

# Diccionario de municipios (por zona)
municipios = {
    'Lima Top': ['Barranco', 'San Borja', 'Santiago de Surco', 'Miraflores', 'San Isidro', 'La Molina'],
    'Lima Moderna': ['Jesús María', 'Pueblo Libre', 'Lince', 'San Miguel', 'Magdalena del Mar', 'Surquillo'],
    'Lima Centro': ['Cercado de Lima', 'La Victoria', 'Breña', 'Rímac'],
    'Lima Norte': ['Carabayllo', 'Comas', 'San Martín de Porres', 'Independencia', 'Los Olivos', 'Ancón'],
    'Lima Sur': ['Chorrillos', 'Punta Hermosa', 'San Bartolo', 'Punta Negra', 'Cerro Azul'],
    'Lima Este': ['Ate Vitarte', 'Chaclacayo', 'Chosica', 'San Luis', 'El Agustino', 'Cieneguilla'],
    'Lima Callao': ['La Perla', 'Callao', 'Bellavista']
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
    propiedades_similares_mostradas = propiedades_similares_mostradas[['Área Total', 'Dormitorios', 'Baños', 'Estacionamiento', 'Precio Venta']]

    # Asignar la zona y el municipio
    zona = list(zonas.keys())[zona_num]
    municipio = obtener_municipio(zona)
    return precio_venta_pred, propiedades_similares_mostradas, zona, municipio

# Función para obtener municipio basado en zona
def obtener_municipio(zona):
    for municipio, distritos in municipios.items():
        if zona in distritos:
            return municipio
    return 'Municipio desconocido'

# Interfaz de usuario
st.title("🏡 Predicción de Precios de Propiedades en Lima")
st.write("Selecciona el tipo de propiedad y proporciona los datos correspondientes para obtener una estimación del precio y ver las propiedades similares.")

# Opción para seleccionar el tipo de propiedad
tipo_propiedad = st.selectbox("Selecciona el tipo de propiedad", ["Casa", "Departamento"])

# Formulario de entrada
area_total = st.number_input("📏 Área Total (m²)", min_value=1)
dormitorios = st.number_input("🛏 Número de Dormitorios", min_value=1)
banos = st.number_input("🚿 Número de Baños", min_value=1)
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

    # Mostrar resultados
    st.subheader(f"📊 Resultados para la propiedad en {zona}, {municipio}")
    st.metric("Precio Estimado", f"{precio_estimado:,.2f} soles")

    if not propiedades_similares.empty:
        # Calcular valores clave
        precio_min = propiedades_similares['Precio Venta'].min()
        precio_max = propiedades_similares['Precio Venta'].max()
        diferencia_min = precio_estimado - precio_min
        diferencia_max = precio_max - precio_estimado

        # Indicadores adicionales
        st.metric("Precio Mínimo", f"{precio_min:,.2f} soles", f"Diferencia: {diferencia_min:,.2f}")
        st.metric("Precio Máximo", f"{precio_max:,.2f} soles", f"Diferencia: {diferencia_max:,.2f}")

        # Gráfico de barras
        st.subheader("📈 Comparación de Precios")
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['Precio Mínimo', 'Precio Estimado', 'Precio Máximo']
        valores = [precio_min, precio_estimado, precio_max]
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c']

        ax.barh(labels, valores, color=colores, alpha=0.8, edgecolor='black')
        for i, valor in enumerate(valores):
            ax.text(valor, i, f"{valor:,.2f} soles", va='center', ha='left', fontsize=10)
        ax.set_xlabel("Precio en soles")
        ax.set_title("Comparación de Precios")
        st.pyplot(fig)

        # Tabla de propiedades similares
        st.subheader("🏘 Propiedades Similares")
        propiedades_similares = propiedades_similares.reset_index(drop=True)    
        st.write(propiedades_similares)
    else:
        st.warning("⚠️ No se encontraron propiedades similares en esta zona.")
