import streamlit as st
import tensorflow as tf
import numpy as np
import joblib  # Para cargar el escalador
import os  # Para verificar la existencia del archivo

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

# Título principal
st.title("MONTERREY AZUCARERA LOJANA")

# Cargar el logo
logo_path = "logo.png"  # Cambia a la ruta correcta si es necesario
if os.path.exists(logo_path):
    # Usamos HTML para centrar el logo
    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{open(logo_path, "rb").read().encode("base64")}" width="300"></div>', 
        unsafe_allow_html=True
    )
else:
    st.warning("El logo no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")

# Título secundario
st.subheader("Predicción de la Producción de Azúcar")

# Texto explicativo sobre la utilidad del aplicativo
st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave: Toneladas Caña Molida (TCM), Rendimiento y Toneladas de Jugo.
La herramienta es útil para los profesionales en la industria azucarera, facilitando la toma de decisiones informadas basadas en datos.
""")

st.write("""
La predicción se realiza mediante un algoritmo de machine learning, utilizando un algoritmo de Red Neuronal Artificial (ANN) entrenada con datos históricos diarios de producción azucarera del Ingenio Azucarero Monterrey C.A.
""")

st.write("""
Ingrese los valores en los campos a continuación para obtener una estimación de la producción 
