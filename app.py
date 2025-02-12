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
    "Villa El Salvador": 41, "Villa María del Triunfo": 42, "Callao": 43,"Asia": 44, "Barranca": 45, "Canta": 46, "Cañete": 47, "Huacho": 48, "Huaral": 49,
    "Huarochiri": 50, "Huaura": 51, "Mala": 52, "Quilmana": 53, "Ricardo Palma": 54
}

# Diccionario de municipios
municipios = {
    'Lima Top': ['Barranco', 'San Borja', 'Santiago de Surco', 'Miraflores', 'San Isidro', 'La Molina'],
    'Lima Moderna': ['Jesús María', 'Pueblo Libre', 'Lince', 'San Miguel', 'Magdalena del Mar', 'Surquillo'],
    'Lima Centro': ['Cercado de Lima', 'La Victoria', 'Breña', 'Rímac'],
    'Lima Norte': ['Carabayllo', 'Comas', 'San Martín de Porres', 'Independencia', 'Los Olivos', 'Ancón', 'Puente Piedra'],
    'Lima Sur': ['Chorrillos', 'Punta Hermosa', 'San Bartolo', 'Punta Negra', 'Villa El Salvador', 'Villa María del Triunfo', 'San Juan de Miraflores', 'Lurín', 'Pucusana', 'Pachacamac', 'Santa María del Mar'],
    'Lima Este': ['Ate Vitarte', 'Chaclacayo', 'Chosica', 'San Luis', 'El Agustino', 'Cieneguilla', 'Santa Anita', 'San Juan de Lurigancho'],
    'Lima Callao': ['Callao'],
    'Lima Provincias': ['Asia', 'Barranca', 'Canta', 'Cañete', 'Huacho', 'Huaral', 'Huarochiri', 'Huaura', 'Mala', 'Quilmana', 'Ricardo Palma']
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
    propiedades_similares_mostradas = propiedades_similares_mostradas[['Área Total', 'Dormitorios', 'Baños', 'Estacionamiento', 'Precio Venta']]
    
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
st.title("🏡 Predicción de Precios de Propiedades en Lima")
st.write("Selecciona el tipo de propiedad y proporciona los datos correspondientes para obtener una estimación del precio y ver las propiedades similares.")

# Opción para seleccionar el tipo de propiedad
tipo_propiedad = st.selectbox("Selecciona el tipo de propiedad", ["Casa", "Departamento"])

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
    tipo_cambio = 3.80  # Tipo de cambio de soles a dólares

    # Convertir el precio estimado a dólares
    precio_estimado_dolares = precio_estimado / tipo_cambio
        
    # Mostrar resultados
    st.subheader(f"📊 Resultados para la propiedad en {zona}, {municipio}")
    st.metric("Precio Estimado", f"{precio_estimado:,.2f} soles")
    st.metric("💵 Precio Estimado en dólares", f"{precio_estimado_dolares:,.2f} dólares*")
    st.markdown(f"<p style='font-size: 10px;'>Tipo de cambio utilizado: {tipo_cambio:,.2f} soles por dólar</p>", unsafe_allow_html=True)


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
