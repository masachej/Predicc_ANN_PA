# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cZ02JpAESuVADgfQMqMO-ppaiRUsu64L
"""

import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Título de la app
st.title("Predicción de Producción con Modelo de Red Neuronal")

# Subir archivo CSV
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Mostrar las primeras filas del archivo cargado
    st.write("Datos cargados:")
    st.write(data.head())

    # Asegúrate de que las columnas estén disponibles
    if 'Tcm' in data.columns and 'Rendimiento' in data.columns and 'Toneladas_jugo' in data.columns:
        # Cargar el modelo entrenado
        model = load_model("path/to/your/ANN_modelo_PPA.h5")

        # Preprocesar los datos antes de hacer predicción (normalización, etc.)
        # Asegúrate de ajustar esto dependiendo del preprocesamiento que usaste durante el entrenamiento
        input_data = data[['Tcm', 'Rendimiento', 'Toneladas_jugo']].values
        input_data_scaled = input_data  # Aquí aplicarías el mismo escalador que usaste durante el entrenamiento

        # Realizar la predicción
        predictions = model.predict(input_data_scaled)

        # Mostrar las predicciones
        st.write("Predicciones de Producción:")
        st.write(predictions)

    else:
        st.error("El archivo CSV no contiene las columnas necesarias ('Tcm', 'Rendimiento', 'Toneladas_jugo').")