import streamlit as st
import pandas as pd
import pickle as pl
import numpy as np
st.title("Predicción del valor de un inmueble")
data = pd.read_csv("hprice.csv")
with open("Lr.pickle", "rb") as f:
    Lr = pl.load(f)
# Widgets para la entrada de datos
lotsize = st.slider("Tamaño del lote (lotsize)", float(data["lotsize"].min()), float(data["lotsize"].max()), float(data["lotsize"].mean()))
assess = st.slider("assess", float(data["assess"].min()), float(data["assess"].max()), float(data["assess"].mean()))
sqrft = st.slider("Pies cuadrados (sqrft)", int(data["sqrft"].min()), int(data["sqrft"].max()), int(data["sqrft"].mean()))
bdrms = st.slider("Numero de habitaciones (bdrms)", int(data["bdrms"].min()), int(data["bdrms"].max()), int(data["bdrms"].mean()))
colonial = st.radio("¿Es colonial?", ("Sí", "No"))

print("lotsize:", lotsize)
print("assess:", assess)
print("colonial_binary:", colonial)
print("sqrft:", sqrft)
print("bdrms:", bdrms)

colonial_binary = 1 if colonial == "Sí" else 0
valores=np.array([[lotsize,assess,colonial_binary,sqrft,bdrms]])
p=Lr.predict([[lotsize,assess,colonial_binary,sqrft,bdrms]])
st.write(st.write("El valor del inmueble es " + str(p)))
