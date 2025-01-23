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

# Mostrar logo centrado con `st.image`
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("")  # Espacio vacío
with col2:
    st.image("logom.png", use_column_width=True)  # Cambia "logo.png" por la ruta correcta
with col3:
    st.write("")  # Espacio vacío

# Mostrar títulos centrados
st.markdown(
    """
    <div style="text-align: center;">
        <h1>MONTERREY AZUCARERA LOJANA</h1>
        <h2>PREDICCIÓN DE LA PRODUCCIÓN DE AZÚCAR</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Texto explicativo sobre la utilidad del aplicativo
st.write("""
Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave: Toneladas Caña Molida (TCM), Rendimiento y Toneladas de Jugo.
La herramienta es útil para los profesionales en la industria azucarera, facilitando la toma de decisiones informadas basadas en datos.""")

st.write("""
La predicción se realiza mediante un algoritmo de machine learning, utilizando un algoritmo de Red Neuronal Artificial (ANN) entrenada con datos históricos diarios de producción azucarera del Ingenio Azucarero Monterrey C.A.
""")

st.write("""
Ingrese los valores en los campos a continuación para obtener una estimación de la producción de azúcar en sacos.
""")

# Entrada de datos
tcm = st.number_input("Ingrese el valor de Toneladas Caña Molida (ton)", min_value=0.0, value=0.0, step=0.01)
rendimiento = st.number_input("Ingrese el valor de Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)

# Botón para hacer la predicción
if st.button("Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        st.write(f"La predicción de producción es: {result:.2f} sacos.")  # Mostrar la predicción

