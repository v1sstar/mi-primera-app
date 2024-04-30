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
model= pd.read_csv("D:\descargas\hprice.csv")
e=np.random.normal(0,1,size=100)
y=model["assess"]+model["bdrms"]+model["lotsize"]+model["sqrft"]+model["colonial"]+e
X=model.drop(columns="price")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#XGBoost
param_grid = {"eta":[1],
              "max_depth":[3,4,5,6,7,8,9],
              "lambda":[2,3,4,5,6],
              "alpha":[1,2,3,4,5,6,7,8,9]}
gs_xgb = GridSearchCV(XGBRegressor(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=5)
print(gs_xgb)
gs_xgb.fit(X_train, y_train)
preds = gs_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

#LinearRegression
Lr=LinearRegression()
Lr.fit(X,y)
Lr_pred= Lr.predict(X_test)

#Decisiontree
param_grid = {'max_depth':[6,7,8,],
              'min_samples_split': [2,4,8,12],
              'min_samples_leaf': [2,3,6,7,8]}
tree_grid = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, cv=5)
tree_grid.fit(X, y)
tree_grid.best_params_
y_pred = tree_grid.predict(model)

with open("xgb_model.pickle", "wb") as f:
    pl.dump(rmse, f)

with open("lr_model.pickle", "wb") as f:
    pl.dump(Lr_pred, f)

with open("tree_model.pickle", "wb") as f:
    pl.dump(y_pred, f)