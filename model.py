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
y=model["price"]
X=model.drop(columns="price")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#XGBoost
param_grid = {"eta":[1],
              "max_depth":[3,4,5,6,7,8,9],
              "lambda":[2,3,4,5,6],
              "alpha":[1,2,3,4,5,6,7,8,9]}
gs_xgb = GridSearchCV(XGBRegressor(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=5)
gs_xgb.fit(X_train, y_train)
preds = gs_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

#LinearRegression
Lr=LinearRegression()
Lr.fit(X,y)
Lr_pred= Lr.predict(X_test)
with open("xgb_model.pickle", "wb") as f:
    pl.dump(rmse, f)