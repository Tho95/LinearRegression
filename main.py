#Thomas Hübscher/ 04.02.2021
# Example for Linear Regression / predicting insurance cost

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import dataInfo

###################################
#read examine and preprocess data

filePath = 'insurance.csv'
X = pd.read_csv(filePath)
#print(X.head())

y = X['charges'] #(target we want to predict)

X.drop('charges', axis = 1, inplace=True)
print(X.head())

dataInfo.general(X)
dataInfo.missing_value_per_column(X)
categorical, numerical = dataInfo.colType(X)

print("categorical columns: ",categorical)

print(X.sex.unique())              # nominal data --> one hot encode
print(X.smoker.unique())           # ordinal data --> Ordinalencode
print(X.region.unique())           # nominal data --> onehot encode

