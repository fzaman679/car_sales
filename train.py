#importing libraries 
import os, sys

# Data Analysis and visualation libraries
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Creating folders for variables 
current_folder_location = os.getcwd()
src = current_folder_location + '\\src'
data = src + '\\DATA'
raw_data = data+ '\\RAW\\'
processed_data = data+ '\\PROCESSED\\'

# Reading the CSV file
cars = pd.read_csv(raw_data + 'cars.csv')

cars_ml = cars.copy()

# Encode Categorical Columns
categ = ['manufacturer', 'model', 'transmission', 'fuelType']

le = LabelEncoder()
cars_ml[categ] = cars_ml[categ].apply(le.fit_transform)

# Prepearing the dataset for training 
X = cars_ml.drop(['price'], axis=1)
y = cars_ml['price']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#calling an instance of StandardScaler 
scaler = StandardScaler()

#transforming the X_train and X_test 

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Importing CatBoostRegressor model 
cat_boost_model = CatBoostRegressor()

# Training the model
cat_boost_model.fit(X_train_scaled, y_train)

#Predicting the results
cat_boost_model_preds_test = cat_boost_model.predict(X_test_scaled)

print('-----------------------------------------------')
print('The model performance for test set')
print('MAE test', mean_absolute_error(y_test, cat_boost_model_preds_test))
print('MSE test', mean_squared_error(y_test, cat_boost_model_preds_test))
print('RMSE test', np.sqrt(mean_squared_error(y_test, cat_boost_model_preds_test)))
print('R2 score', r2_score(y_test, cat_boost_model_preds_test))





