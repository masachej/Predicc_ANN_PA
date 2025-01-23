import streamlit as st
import tensorflow as tf
import numpy as np
import joblib  # Para cargar el escalador

# Cargar el modelo
model = tf.keras.models.load_model('ANN_modelo_PPA.h5')

# Cargar el escalador (asegúrate de tener el archivo 'scaler.pkl' en el mismo directorio)
scaler = joblib.load('scaler.pkl')

# Función para realizar la predicción
def make_prediction(tcm, rendimiento, toneladas_jugo):
    # Escalar los datos de entrada usando el mismo escalador
    data = np.array([[tcm, rendimiento, toneladas_jugo]])
    data_scaled = scaler.transform(data)  # Escalar los datos de entrada
    prediction = model.predict(data_scaled)  # Hacer la predicción
    return prediction[0][0]  # Devolver la predicción (único valor en la predicción)

# Título principal centrado
st.markdown(
    """
    <div style="text-align: center;">
        <h1>MONTERREY AZUCARERA LOJANA</h1>
        <h2>PREDICCIÓN DE LA PRODUCCIÓN DE AZÚCAR</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mostrar logo centrado
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("")  # Espacio vacío
with col2:
    st.image("logom.png", use_container_width=True)  # Cambia "logo.png" por la ruta correcta
with col3:
    st.write("")  # Espacio vacío

# Texto explicativo sobre la utilidad del aplicativo
st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave: Toneladas Caña Molida (TCM), Rendimiento y Toneladas de
