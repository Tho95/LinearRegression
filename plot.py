# module for plotting the data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

def regplot(X,y):
    '''Function for plotting the variables of input X against the target y'''

    fig, axes = plt.subplots(2,3,figsize=(16,8))
    fig.suptitle('charges for insurance')

    sns.regplot(ax=axes[0,0], x='bmi', y=y, data=X);
    #axes[0].set_xlabel('bmi')
    sns.regplot(ax=axes[0,1], x = 'children', y=y, data = X);

    sns.regplot(ax=axes[1,0], x='age', y=y, data=X);

    sns.regplot(ax=axes[1,1], x='age', y=y, data=X);


    ###### encoding of ordinal data
    ordinal_encoder = OrdinalEncoder(categories=[['no', 'yes']])
    ordinal_encoder.fit(X[['smoker']])
    smoker_encoded = pd.DataFrame(ordinal_encoder.transform(X[['smoker']]))
######
    sns.regplot(ax=axes[0,2], x=smoker_encoded, y=y);
    axes[0,2].set_xlabel('smoker(0:No, 1:Yes]')

    ordinal_encoder1 = OrdinalEncoder(categories=[['female', 'male']])
    ordinal_encoder1.fit(X[['sex']])
    sex_encoded = pd.DataFrame(ordinal_encoder1.transform(X[['sex']]))

    sns.regplot(ax=axes[1, 2], x=sex_encoded, y=y);
    axes[1, 2].set_xlabel('sex(0:f, 1:m]')
    axes[1, 2].set_xlim(0,1)
    axes[1, 2].set_ylim(0, 60000)
    plt.show()