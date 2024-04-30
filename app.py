import streamlit as st
import pandas as pd
import pickle as pl
import numpy as np
st.title("Predicción del valor de un inmueble")
data = pd.read_csv("D:\descargas\hprice.csv")
with open("xgb_model.pickle", "rb") as f:
    xgb_rmse = pl.load(f)
# Widgets para la entrada de datos
numero_habitaciones = st.selectbox("Número de habitaciones:", data["numero_habitaciones"].unique())
metros_cuadrados = st.number_input("Metros cuadrados:", min_value=1)
anio_construccion = st.number_input("Año de construcción:", min_value=1900)
vars = np.array([[numero_habitaciones, metros_cuadrados, anio_construccion, ...]])
# Predecir el precio usando los modelos
xgb_prediction = xgb_rmse.predict(np.array([[numero_habitaciones, metros_cuadrados, anio_construccion, ...]]))[0]
st.write("XGBoost Prediction:", xgb_prediction)