import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Diccionario de zonas y municipios
zonas = {
    0: 'Lima Top',
    1: 'Lima Moderna',
    2: 'Lima Centro',
    3: 'Lima Norte',
    4: 'Lima Sur',
    5: 'Lima Este',
    6: 'Lima Callao'
}

municipios = {
    0: 'Barranco',
    1: 'Jesús María',
    2: 'Cercado de Lima',
    3: 'Carabayllo',
    4: 'Chorrillos',
    5: 'Ate Vitarte',
    6: 'La Perla',
    7: 'San Borja',
    8: 'Pueblo Libre',
    9: 'La Victoria',
    10: 'Comas',
    11: 'Punta Hermosa',
    12: 'Chaclacayo',
    13: 'Callao',
    14: 'Santiago de Surco',
    15: 'Lince',
    16: 'Breña',
    17: 'San Martín de Porres',
    18: 'San Bartolo',
    19: 'Chosica',
    20: 'Bellavista',
    21: 'Miraflores',
    22: 'San Miguel',
    23: 'Rímac',
    24: 'Independencia',
    25: 'Punta Negra',
    26: 'San Luis',
    27: 'San Isidro',
    28: 'Magdalena del Mar',
    29: 'Los Olivos',
    30: 'Cerro Azul',
    31: 'El Agustino',
    32: 'La Molina',
    33: 'Surquillo',
    34: 'Ancón',
    35: 'Cieneguilla'
}

# Cargar los datos generados
data = pd.read_csv("data_prueba.csv")

# Preprocesar la variable 'Zona' y 'Municipio' (codificación)
label_encoder = LabelEncoder()
data['Zona'] = label_encoder.fit_transform(data['Zona'])
data['Municipio'] = label_encoder.fit_transform(data['Municipio'])

# Definir las características (X) y el objetivo (y)
X = data[['Área Total', 'Zona', 'Dormitorios', 'Baños', 'Estacionamiento', 'Municipio']]
y = data['Precio Venta']

# Entrenamiento de RandomForest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X, y)

# Función para realizar la predicción
def realizar_prediccion(area_total, zona, dormitorios, banos, estacionamiento, municipio):
    input_data = pd.DataFrame({
        'Área Total': [area_total],
        'Zona': [zona],
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Estacionamiento': [estacionamiento],
        'Municipio': [municipio]
    })

    # Realizar la predicción
    precio_estimado = model_rf.predict(input_data)[0]
    
    # Filtrar propiedades similares
    propiedades_similares = data[ 
        (data['Zona'] == zona) & 
        (data['Municipio'] == municipio)
    ]
    
    # Calcular precio mínimo y máximo en propiedades similares
    precio_minimo = propiedades_similares['Precio Venta'].min()
    precio_maximo = propiedades_similares['Precio Venta'].max()

    # Contar propiedades dentro de los rangos
    num_propiedades_min_max = len(propiedades_similares[(propiedades_similares['Precio Venta'] >= precio_minimo) & 
                                                        (propiedades_similares['Precio Venta'] <= precio_maximo)])
    
    num_propiedades_estimadas = len(propiedades_similares[(propiedades_similares['Precio Venta'] >= precio_estimado - 10000) & 
                                                           (propiedades_similares['Precio Venta'] <= precio_estimado + 10000)])

    return precio_estimado, propiedades_similares, precio_minimo, precio_maximo, num_propiedades_min_max, num_propiedades_estimadas

# Título de la aplicación
st.title("Predicción de Precio de Propiedades")

# Mostrar diccionario de zonas y municipios
st.subheader("Consideraciones de Zonas y Municipios")
st.markdown("""
A continuación se presentan las zonas y municipios correspondientes a cada identificador (ID):
""")

st.markdown("### Zonas")
for key, value in zonas.items():
    st.markdown(f"**ID {key}:** {value}")

st.markdown("### Municipios")
for key, value in municipios.items():
    st.markdown(f"**ID {key}:** {value}")

# Descripción de la app
st.markdown(""" 
Esta aplicación te permite calcular el precio estimado de una propiedad basado en su área total, zona, número de dormitorios, baños, estacionamiento y municipio. 
Los resultados incluyen propiedades similares, el precio mínimo y máximo, y otros indicadores relacionados.
""")

# Crear formulario para ingreso de datos
st.subheader("Ingresa los detalles de la propiedad")

area_total = st.number_input("Área Total (m²)", min_value=0.0, step=1.0)
zona = st.number_input("Zona (ID de zona)", min_value=0, step=1)
dormitorios = st.number_input("Dormitorios", min_value=0, step=1)
banos = st.number_input("Baños", min_value=0, step=1)
estacionamiento = st.number_input("Estacionamiento", min_value=0, step=1)
municipio = st.number_input("Municipio (ID de municipio)", min_value=0, step=1)

# Botón para ejecutar la predicción
if st.button("Calcular Precio"):
    precio_estimado, propiedades_similares, precio_minimo, precio_maximo, num_propiedades_min_max, num_propiedades_estimadas = realizar_prediccion(
        area_total, zona, dormitorios, banos, estacionamiento, municipio)

    # Mostrar resultados
    st.subheader(f"Precio Estimado: {precio_estimado:.2f} USD")
    
    # Mostrar tabla de propiedades similares
    st.subheader(f"Propiedades Similares:")
    st.write(propiedades_similares[['Área Total', 'Zona', 'Dormitorios', 'Baños', 'Estacionamiento', 'Municipio', 'Precio Venta']])

    st.subheader(f"Precio Mínimo: {precio_minimo:.2f} USD")
    st.subheader(f"Precio Máximo: {precio_maximo:.2f} USD")
    st.subheader(f"Número de Propiedades con Precio Mínimo y Máximo: {num_propiedades_min_max}")
    st.subheader(f"Número de Propiedades con Precio Estimado y Máximo: {num_propiedades_estimadas}")
    
    # Consejos adicionales
    st.markdown("""
    **Consejos:**
    - Si el precio estimado está fuera del rango de propiedades similares, es posible que la propiedad tenga características únicas.
    - Las propiedades similares te ofrecen una mejor visión del mercado en la zona específica.
    """)
