#Thomas Hübscher/ 04.02.2021
# Example for Linear Regression / predicting insurance cost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


import dataInfo
import preprocess
import model
import plot


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
print(X.smoker.unique())           # ordinal data --> ordinal encode
print(X.region.unique())           # nominal data --> one hot encode



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

preprocessor = preprocess.encode()

trans=preprocessor.fit_transform(X_train) #####################################new dataframe
plot.regplot(X_train, y_train)
'''
###for plotting

print(trans)
df = pd.DataFrame(trans, #mit ohe codierten...
                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])    ##recreate dataframe
plot.pairplot(df,y)
#####
'''

#linear Regression
###################################################################
model = model.linearRegression()

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_valid)

print('\n \nMAE Linear Regression with all values :', mean_absolute_error(y_valid, predictions))
print('relative error: ',mean_absolute_error(y_valid, predictions)/ np.mean(y))
print('mean squared error: ',mean_squared_error(y_valid, predictions))
print('R2 score: ', r2_score(y_valid, predictions))
print('Slope: ', model.coef_)
print('\n \nIntercept: ', model.intercept_)
#R2 score is a statistical measure of how close the data is to the fitted line (coefficent of (multiple)
#determination)), it's the percentage of the response variable variation
# 0...100%, the higher the better the model fits the data

#Conclusion for the model: Linear Regression is okay for the dataset. the R2 value is high. the plots show that
#the variance for each plot is pretty high. Nevertheless we have a more or less good prediction for the values
#Even a xgboost model ( which was not optimized) was just a little bit better than our LinearRegessrion model.




###### xgboost reference
modelxgb = XGBRegressor(random_state=0, colsample_bytree=0.9, max_depth=20, n_estimators=1400, reg_alpha=1.5,
                         reg_lambda=1.1, subsample=0.7)

pipe1 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', modelxgb)])

pipe1.fit(X_train, y_train)

predictions = pipe1.predict(X_valid)

print('MAE XGB :', mean_absolute_error(y_valid, predictions))
print('relative error: ',mean_absolute_error(y_valid, predictions)/ np.mean(y))


############################################
#Linear Regression only with numerical values
X.drop(['smoker','sex','region'],axis=1,inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
print(X_train.head())
model1 = LinearRegression()
model1.fit(X_train,y_train)
predictions = model1.predict(X_valid)
print('MAE Linear Regression with numeric values :', mean_absolute_error(y_valid, predictions))
print('relative error: ',mean_absolute_error(y_valid, predictions)/ np.mean(y))




