import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Definir el diccionario de zonas y municipios
zonas_municipios = {
    'Lima Top': {
        0: 'Barranco', 1: 'San Borja', 2: 'Santiago de Surco', 3: 'Miraflores',
        4: 'San Isidro', 5: 'La Molina'
    },
    'Lima Moderna': {
        6: 'Jesús María', 7: 'Pueblo Libre', 8: 'Lince', 9: 'San Miguel',
        10: 'Magdalena del Mar', 11: 'Surquillo'
    },
    'Lima Centro': {
        12: 'Cercado de Lima', 13: 'La Victoria', 14: 'Breña', 15: 'Rímac'
    },
    'Lima Norte': {
        16: 'Carabayllo', 17: 'Comas', 18: 'San Martín de Porres', 19: 'Independencia',
        20: 'Los Olivos', 21: 'Ancón'
    },
    'Lima Sur': {
        22: 'Chorrillos', 23: 'Punta Hermosa', 24: 'San Bartolo', 25: 'Punta Negra',
        26: 'Cerro Azul'
    },
    'Lima Este': {
        27: 'Ate Vitarte', 28: 'Chaclacayo', 29: 'Chosica', 30: 'San Luis',
        31: 'El Agustino', 32: 'Cieneguilla'
    },
    'Lima Callao': {
        33: 'La Perla', 34: 'Callao', 35: 'Bellavista'
    }
}

# Cargar los datos generados
data = pd.read_csv("base_cu.csv")

# Preprocesar la variable 'Zona' y 'Municipio' (codificación)
label_encoder_zona = LabelEncoder()
label_encoder_municipio = LabelEncoder()

# Codificar las zonas y municipios
data['Zona'] = label_encoder_zona.fit_transform(data['Zona'])
data['Municipio'] = label_encoder_municipio.fit_transform(data['Municipio'])

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
st.title("🏡 **Predicción de Precio de Propiedades**")

# Agregar un texto de introducción más estilizado
st.markdown("""
Bienvenido a la herramienta de predicción de precios de propiedades. 
Aquí podrás calcular el valor estimado de una propiedad en Lima con base en sus características.
""")

# Mostrar diccionario de zonas y municipios con un formato más atractivo
st.subheader("🌍 **Zonas y Municipios en Lima**")

# Mostrar Zonas con sus respectivos números
st.markdown("### **Zonas de Lima (ID asignado)**")
for key, value in zonas_municipios.items():
    zona_nombres = [f"**ID {zona_id}: {zona_nombre}**" for zona_id, zona_nombre in value.items()]
    st.markdown(f"**{key}:** {', '.join(zona_nombres)}")

# Agregar espacio entre las secciones
st.markdown("---")

# Descripción de la app
st.markdown("""
Esta aplicación te permite calcular el precio estimado de una propiedad basado en su área total, zona, número de dormitorios, baños, estacionamiento y municipio. 
A continuación, ingresa los datos de la propiedad y obtendrás el precio estimado junto con información adicional sobre propiedades similares.
""")

# Crear formulario para ingreso de datos
st.subheader("📊 **Ingresa los detalles de la propiedad**")

col1, col2 = st.columns(2)

with col1:
    area_total = st.number_input("Área Total (m²)", min_value=0.0, step=1.0)
    dormitorios = st.number_input("Dormitorios", min_value=0, step=1)
    banos = st.number_input("Baños", min_value=0, step=1)

with col2:
    estacionamiento = st.number_input("Estacionamiento", min_value=0, step=1)
    zona = st.selectbox("Zona (ID de zona)", list(label_encoder_zona.classes_))
    municipio = st.selectbox("Municipio (ID de municipio)", list(label_encoder_municipio.classes_))

# Botón para ejecutar la predicción
if st.button("🔮 **Calcular Precio**"):
    precio_estimado, propiedades_similares, precio_minimo, precio_maximo, num_propiedades_min_max, num_propiedades_estimadas = realizar_prediccion(
        area_total, zona, dormitorios, banos, estacionamiento, municipio)

    # Mostrar los resultados de forma más organizada
    st.subheader(f"💰 **Precio Estimado: {precio_estimado:.2f} USD**")
    
    # Mostrar tabla de propiedades similares
    st.subheader("🏠 **Propiedades Similares**")
    st.write(propiedades_similares[['Área Total', 'Zona', 'Dormitorios', 'Baños', 'Estacionamiento', 'Municipio', 'Precio Venta']])

    st.subheader(f"📉 **Precio Mínimo: {precio_minimo:.2f} USD**")
    st.subheader(f"📈 **Precio Máximo: {precio_maximo:.2f} USD**")
    st.subheader(f"🔢 **Número de Propiedades con Precio Mínimo y Máximo: {num_propiedades_min_max}**")
    st.subheader(f"🔮 **Número de Propiedades con Precio Estimado: {num_propiedades_estimadas}**")

    # Consejos adicionales
    st.markdown("""
    **Consejos:**
    - Si el precio estimado está fuera del rango de propiedades similares, es posible que la propiedad tenga características únicas.
    - Las propiedades similares te ofrecen una mejor visión del mercado en la zona específica.
    """)

    # Agregar una línea divisoria
    st.markdown("---")
