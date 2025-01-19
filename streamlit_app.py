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
tcm = st.number_input("Ingrese el valor de TCM", min_value=0.0, value=0.0, step=0.01)
rendimiento = st.number_input("Ingrese el valor de Rendimiento", min_value=0.0, value=0.0, step=0.01)
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo", min_value=0.0, value=0.0, step=0.01)

# Botón para hacer la predicción
if st.button("Realizar Predicción"):
    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
        st.warning("Por favor, ingrese valores mayores a 0 en todos los campos.")
    else:
        result = make_prediction(tcm, rendimiento, toneladas_jugo)
        st.write(f"La predicción de producción es: {result:.2f}")  # Mostrar la predicción
