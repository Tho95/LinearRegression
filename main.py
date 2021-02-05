#Thomas Hübscher/ 04.02.2021
# Example for Linear Regression / predicting insurance cost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

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
'''
###for plotting

print(trans)
df = pd.DataFrame(trans, #mit ohe codierten...
                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])    ##recreate dataframe
plot.pairplot(df,y)
#####
'''
print(trans[:,0])
np.savetxt('transformed.csv',trans,delimiter=",")
plot.regplot(X_train, y_train)
model = model.linearRegression()

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_valid)

print('MAE LinReg :', mean_absolute_error(y_valid, predictions))


#pruefe()



'''
###### xgboost reference
modelxgb = model.xgboost()

pipe1 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', modelxgb)])

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_valid)

print('MAE XGB :', mean_absolute_error(y_valid, predictions))

'''