import streamlit as st
import pandas as pd
import pickle as pl
import numpy as np
st.title("Predicción del valor de un inmueble")
data = pd.read_csv("hprice.csv")
with open("gs_xgb.pickle", "rb") as f:
    gs_xgb = pl.load(f)
# Widgets para la entrada de datos
lotsize = st.slider("Tamaño del lote (lotsize)", float(data["lotsize"].min()), float(data["lotsize"].max()), float(data["lotsize"].mean()))
assess = st.slider("Evaluación del vecindario (assess)", float(data["assess"].min()), float(data["assess"].max()), float(data["assess"].mean()))
colonial = st.radio("¿Es colonial?", ("Sí", "No"))
sqrft = st.slider("Pies cuadrados (sqrft)", float(data["sqrft"].min()), float(data["sqrft"].max()), float(data["sqrft"].mean()))
bdrms = st.slider("Numero de habitaciones (bdrms)", float(data["bdrms"].min()), float(data["bdrms"].max()), float(data["bdrms"].mean()))

print("lotsize:", lotsize)
print("assess:", assess)
print("colonial_binary:", colonial)
print("sqrft:", sqrft)
print("bdrms:", bdrms)

colonial_binary = 1 if colonial == "Sí" else 0
preds = gs_xgb.preds(np.array([[lotsize, assess, colonial_binary, sqrft,bdrms]]))[0]
st.write(f"El valor predicho del inmueble es: ${preds:.2f}")
