import streamlit as st
import tensorflow as tf
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model('ANN_modelo_PPA.h5')

# Función para realizar la predicción
def make_prediction(tcm, rendimiento, toneladas_jugo):
    data = np.array([[tcm, rendimiento, toneladas_jugo]])
    prediction = model.predict(data)
    return prediction[0][0]

# Título de la app
st.title("Predicción de Producción de Azúcar")

# Entrada de datos
tcm = st.number_input("Ingrese el valor de TCM")
rendimiento = st.number_input("Ingrese el valor de Rendimiento")
toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo")

# Botón para hacer la predicción
if st.button("Realizar Predicción"):
    result = make_prediction(tcm, rendimiento, toneladas_jugo)
    st.write(f"La predicción de producción es: {result}")