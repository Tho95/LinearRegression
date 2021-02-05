# module for showing information about a dataframe

import pandas as pd

def general(X):
    '''Function shows the shape of the Dataframe X'''
    # see the shape of training data:
    print('Shape of training data(rows, columns): ', X.shape, '\n')
    return X.shape[0], X.shape[1]

def missing_value_per_column(X):
    '''Function returns and shows the name of columns with missing values and how many
     values are missing'''
    missing_cols = [col for col in X.columns if X[col].isnull().sum() > 0]
    missing = []
    namemissing = []

    for col in X.columns:
        if X[col].isnull().sum() > 0:
            missing.append(X[col].isnull().sum())
    for col in range(0, len(missing)):
        namemissing.append(missing_cols[col])
        print(missing_cols[col], 'missing values: ',missing[col])
    print('\n', len(missing),' columns have missing values.\n')
    return namemissing, missing

def colType(X):
    '''Function shows the number of numeric and object columns and returns them as dataframes'''
    object_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
    print('categorial columns: ', len(object_cols))
    numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    print('numerical columns: ', len(numeric_cols))
    return object_cols,numeric_cols



