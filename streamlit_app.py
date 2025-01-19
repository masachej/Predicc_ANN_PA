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

# Título de la app
st.title("Predicción de Producción de Azúcar")

# Entrada de datos
tcm = st.number_input("Ingrese el valor de TCM")
rendimiento = st.number_input("Ingrese el valor de Rendimiento")
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo")

# Botón para hacer la predicción
if st.button("Realizar Predicción"):
    result = make_prediction(tcm, rendimiento, toneladas_jugo)
    st.write(f"La predicción de producción es: {result:.2f}")  # Mostrar la predicción
