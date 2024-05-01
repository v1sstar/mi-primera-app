import streamlit as st
import pandas as pd
import numpy as np
import pickle as pl
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
model= pd.read_csv("hprice.csv")

X = model[["lotsize", "assess","bdrms","colonial","sqrft"]]
y=model["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#XGBoost
param_grid = {"eta":[1],
              "max_depth":[3,4,5,6,7,8,9],
              "lambda":[2,3,4,5,6],
              "alpha":[1,2,3,4,5,6,7,8,9]}
gs_xgb = GridSearchCV(XGBRegressor(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=5)
gs_xgb.fit(X_train, y_train)
best_params = gs_xgb.best_params_
print("Mejores parámetros:", best_params)
preds = gs_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:", rmse)


with open("gs_xgb.pickle", "wb") as f:
    pl.dump(gs_xgb, f)