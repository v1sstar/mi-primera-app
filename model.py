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
model.info()
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

#regression tree 
param_gridtree = {'max_depth': [3,8],
              'min_samples_split': [4,8,16],
              'min_samples_leaf': [1,2,3]}
rf_grid= GridSearchCV(DecisionTreeRegressor(),param_gridtree,scoring="neg_mean_squared_error", cv=3)
rf_grid.fit(X_train, y_train)
rf_grid.best_params_
y_pred = rf_grid.predict(X_test)
rmse_tree= np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse_tree)

#linear regression
Lr=LinearRegression()
Lr.fit(X_train, y_train)
Lr_pred= Lr.predict(X_test)
mse = mean_squared_error(y_test, Lr_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

with open("Lr.pickle", "wb") as f:
    pl.dump(Lr, f)