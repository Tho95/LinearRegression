#different models to use on the dataset

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def linearRegression():
    '''Function returns a linear Regression model'''
    model = LinearRegression()
    print('bin in model')
    return model

def xgboost():
    '''Function returns a xgboost model'''
    model = XGBRegressor(random_state=0, colsample_bytree=0.9, max_depth=20, n_estimators=1400, reg_alpha=1.5,
                         reg_lambda=1.1, subsample=0.7)
    return model