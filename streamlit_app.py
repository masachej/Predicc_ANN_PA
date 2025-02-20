7	import streamlit as st
8	import tensorflow as tf
9	import numpy as np
10	import joblib  # Para cargar el escalador
11	import os  # Para verificar la existencia del archivo
12	import base64  # Para codificar la imagen en base64
13	
14	# Cargar el modelo
15	model = tf.keras.models.load_model('ANN_modelo_PPA.h5')
16	
17	# Cargar el escalador (asegúrate de tener el archivo 'scaler.pkl' en el mismo directorio)
18	scaler = joblib.load('scaler.pkl')
19	
20	# Función para realizar la predicción
21	def make_prediction(tcm, rendimiento, toneladas_jugo):
22	    # Escalar los datos de entrada usando el mismo escalador
23	    data = np.array([[tcm, rendimiento, toneladas_jugo]])
24	    data_scaled = scaler.transform(data)  # Escalar los datos de entrada
25	    prediction = model.predict(data_scaled)  # Hacer la predicción
26	    return prediction[0][0]  # Devolver la predicción (único valor en la predicción)
27	
28	# Cargar el logo
29	logo_path = "logom.png"  # Cambia a la ruta correcta si es necesario
30	if os.path.exists(logo_path):
31	    # Codificar la imagen en base64
32	    with open(logo_path, "rb") as image_file:
33	        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
34	    
35	    # Usar HTML para centrar el logo
36	    st.markdown(
37	        f'<div style="text-align: center;"><img src="data:image/png;base64,{encoded_image}" width="300"></div>', 
38	        unsafe_allow_html=True
39	    )
40	else:
41	    st.warning("El logo no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")
42	# Título principal
43	#st.subheader("MONTERREY AZUCARERA LOJANA")
44	st.title("MONTERREY AZUCARERA LOJANA")
45	# Título secundario
46	st.subheader("Predicción de la Producción de Azúcar")
47	
48	# Texto explicativo sobre la utilidad del aplicativo
49	st.write("""
50	Este aplicativo permite predecir la producción de azúcar a partir de tres variables clave: Toneladas Caña Molida (TCM), Rendimiento y Toneladas de Jugo.
51	La herramienta es útil para los profesionales e ingenieros azucareros de la empresa, facilitando la toma de decisiones informadas basadas en datos.
52	""")
53	
54	st.write("""
55	La predicción se realiza mediante un algoritmo de machine learning, utilizando un modelo de Red Neuronal Artificial (ANN) entrenada con datos históricos diarios de producción azucarera del Ingenio Azucarero Monterrey C.A.
56	""")
57	
58	st.write("""
59	Ingrese los valores en los campos a continuación para obtener una estimación de la producción de azúcar en sacos.
60	""")
61	
62	# Entrada de datos
63	tcm = st.number_input("Ingrese el valor de Toneladas Caña Molida (ton)", min_value=0.0, value=0.0, step=0.01)
64	rendimiento = st.number_input("Ingrese el valor de Rendimiento (kg/TCM)", min_value=0.0, value=0.0, step=0.01)
65	toneladas_jugo = st.number_input("Ingrese el valor de Toneladas de Jugo (ton)", min_value=0.0, value=0.0, step=0.01)
66	
67	# Botón para hacer la predicción
68	if st.button("Realizar Predicción"):
69	    if tcm == 0.0 or rendimiento == 0.0 or toneladas_jugo == 0.0:
70	        st.warning("Por favor, ingrese valores mayores a 0 en todos los campos.")
71	    else:
72	        result = make_prediction(tcm, rendimiento, toneladas_jugo)
73	        st.write(f"La predicción de producción es: {result:.2f} sacos.")  # Mostrar la predicción
