import streamlit as st
import pandas as pd
import pickle as pl
import numpy as np
st.title("Predicción del valor de un inmueble")
data = pd.read_csv("hprice.csv")
with open("rmse.pickle", "rb") as f:
    xgb_rmse = pl.load(f)
# Widgets para la entrada de datos
lotsize = st.slider("Tamaño del lote (lotsize)", float(data["lotsize"].min()), float(data["lotsize"].max()), float(data["lotsize"].mean()))
asses = st.slider("Evaluación del vecindario (asses)", float(data["asses"].min()), float(data["asses"].max()), float(data["asses"].mean()))
colonial = st.radio("¿Es colonial?", ("Sí", "No"))
sqrft = st.slider("Pies cuadrados (sqrft)", float(data["sqrft"].min()), float(data["sqrft"].max()), float(data["sqrft"].mean()))

colonial_binary = 1 if colonial == "Sí" else 0
prediction = xgb_rmse.predict(np.array([[lotsize, asses, colonial_binary, sqrft]]))[0]
st.write(f"El valor predicho del inmueble es: ${prediction:.2f}")