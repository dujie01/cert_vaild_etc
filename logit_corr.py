import os
import pandas as pd
import numpy as np

import math
import sys
import random

from statsmodels.regression.linear_model import OLS, GLS
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit, MNLogit

sample_dir = '/group/regular/Paper/all_logit_city_class'
sample_files = []
for r,l,f in os.walk(sample_dir):
    for ff in f:
        if ff.endswith('.parquet'):
            sample_files.append(os.path.join(r, ff))
df = pd.concat([pd.read_parquet(file) for file in sample_files], ignore_index=True)

df.groupby(['job_city_class']).count()

def get_city_cate(c):
    if c=='one' or c=='new_one' or c=='two' or c=='three' or c=='four':
        return c
    else:
        return 'other'
    
df['job_city_class'] = df['job_city_class'].apply(get_city_cate)

for col in data.columns:
    data[col] = data[col].replace(np.NaN, 0)

data1 = data[(data['job_city_class'] == 'new_one') | (data['job_city_class'] == 'one')]
data1 = data1.drop(['job_city_class'], axis = 1)

### logit1
Y1 = data1.iloc[:,:1]
X1 = data1.drop(['label_callback','three_class'], axis = 1)
for i in X1.columns:
    print(i,X1[i].unique())

Model1 = sm.Logit(Y1,X1).fit()

Model1.summary()


### logit2
data_new = data[(data.three_class != 1.0)]
data_new = data_new[(data_new['job_city_class'] == 'new_one') | (data_new['job_city_class'] == 'one')]
data_new = data_new.drop(['job_city_class'], axis = 1)
for col in data_new.columns:
    data_new[col] = data_new[col].astype(float)
Y1 = data_new.iloc[:,:1]
X1 = data_new.drop(['label_callback','three_class'], axis = 1)
for i in X1.columns:
    print(i,X1[i].unique())
Model1 = sm.Logit(Y1,X1).fit()
Model1.summary()

data_new = data[(data.three_class != 1.0)]
data_new = data_new[(data_new['job_city_class'] == 'other')]
data_new = data_new.drop(['job_city_class'], axis = 1)
for col in data_new.columns:
    data_new[col] = data_new[col].astype(float)
Y1 = data_new.iloc[:,:1]
X1 = data_new.drop(['label_callback','three_class'], axis = 1)
for i in X1.columns:
    print(i,X1[i].unique())
Model1 = sm.Logit(Y1,X1).fit()
Model1.summary()


