#file for encoding the categorical columns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def encode():
    '''Function for encoding the categorical columns with either ordinal encoding or nominal encoding'''
    ordinal_cols = list(['smoker'])
    nominal_cols = list(['sex', 'region'])

    ordinal_encoder = OrdinalEncoder(categories=[['no', 'yes']])

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(remainder='passthrough',
                                     transformers=[('ordi', ordinal_encoder, ordinal_cols),
                                                   ('cate', one_hot_encoder, nominal_cols)])
    print('bin im preprocessor')
    return preprocessor


