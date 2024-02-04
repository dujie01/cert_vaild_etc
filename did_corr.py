import os
import pandas as pd
import numpy as np

import math
import sys
import random

# import psmatching.match as psm
# import pytest
# from psmatching.utilities import *

from statsmodels.regression.linear_model import OLS, GLS
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit, MNLogit

import datetime
import time
import copy

data = pd.read_csv('/group/regular/Paper/DiD_before_after_cert/parallel_test_past_data2.csv')
# data = data.drop(['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','id','geek_id','boss_id','job_id','expect_id','first_add_time_x','kl_time_ts','birthday','degree'], axis=1)

def get_city_cate(c):
    if c=='one' or c=='new_one' or c=='two' or c=='three' or c=='four':
        return c
    else:
        return 'other'
    
data['job_city_class'] = data['job_city_class'].apply(get_city_cate)

data['degree_type'] = data['degree_type'].apply(str)
data['work_years'] = data['work_years'].apply(int)
data['gender'] = data['gender'].apply(int)

data['cert_general'] = data['cert_general'].apply(int)
data['intercept_'] = data['cert_general']*data['diff']

data['label_exp_salary'] = data['exp_salary'].apply(float)
data['label_salary'] = data['job_salary'].apply(float)
data['l1_code'] = data['l1_code'].apply(str)

data['wk'] = data['wk'].apply(str)
mydata = data[['label_callback','label_salary',\
               'age','gender','work_years','wk','degree_type',\
               'l1_code','job_city_class','Time_FE']]

y_data = mydata[['label_callback']]
x_data = mydata.iloc[:,2:]
X = pd.get_dummies(x_data)
print('Dummy Finished')

X = X.drop(['wk_control','degree_type_0','job_city_class_other','Time_FE_2023-01','l1_code_100000'], axis = 1)
X = sm.add_constant(X)
print('Constant Added')
Y = y_data

Logit_model = Logit(Y, X)
print('Model built success')

result = Logit_model.fit()
print('Model fit success')

result.summary()

data['mth'] = data['mth'].apply(str)
mydata = data[['label_callback','label_salary',\
               'age','gender','work_years','mth','degree_type',\
               'l1_code','job_city_class','Time_FE']]

y_data = mydata[['label_callback']]
x_data = mydata.iloc[:,2:]
X = pd.get_dummies(x_data)
print('Dummy Finished')

X = X.drop(['mth_control','degree_type_0','job_city_class_other','Time_FE_2023-01','l1_code_100000'], axis = 1)
X = sm.add_constant(X)
print('Constant Added')
Y = y_data

Logit_model = Logit(Y, X)
print('Model built success')

result = Logit_model.fit()
print('Model fit success')

result.summary()


cbdata = data[['label_callback','label_exp_salary','label_salary','intercept_',#'cert_general','diff',\
               'age','gender','work_years','degree_type',\
               'l1_code','job_city_class','Time_FE']]

Y_callback = cbdata[['label_callback']]
x_data = cbdata.iloc[:,3:]
X = pd.get_dummies(x_data)
X = X.drop(['degree_type_0','job_city_class_other','Time_FE_2023-01','l1_code_100000'], axis = 1)
X = sm.add_constant(X)

Logit_model = Logit(Y_callback, X)
print('Model built success')

result = Logit_model.fit()
print('Model fit success')

result.summary()


Y_salary = cbdata[['label_salary']]

Ols_model = sm.OLS(Y_salary, X)
print('Model built success')

model_reg = Ols_model.fit()
print('Model fit success')

model_reg.summary()

c_data = data[data['diff']==1]
c_data['degree_type'] = c_data['degree_type'].map({'0':'below_associate','1':'associate','2':'bachelor','3':'master'})
c_data['cert_general'] = c_data['cert_general'].apply(str)
c_data['intercept'] = c_data['degree_type']+'_'+c_data['cert_general']

cd_data = c_data[['label_callback','label_salary',#'cert_general','diff',\
                  'age','gender','work_years','intercept',\
                  'l1_code','job_city_class','Time_FE']]

Y_callback = cd_data[['label_callback']]
x_data = cd_data.iloc[:,2:]
X = pd.get_dummies(x_data)
X = X.drop(['intercept_below_associate_0','job_city_class_other','Time_FE_2023-01','l1_code_100000'], axis = 1)
X = sm.add_constant(X)

Logit_model = Logit(Y_callback, X)
print('Model built success')

result = Logit_model.fit()
print('Model fit success')

result.summary()

Y_salary = cd_data[['label_salary']]

Ols_model = sm.OLS(Y_salary, X)
print('Model built success')

model_reg = Ols_model.fit()
print('Model fit success')

model_reg.summary()

age_data = data[data['diff']==1]

import matplotlib.pyplot as plt
plt.hist(age_data['age'], bins=range(min(age_data['age']), max(age_data['age']) + 1))
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Column1')
plt.show()

age_data = age_data[(age_data['age'] >= 20) & (age_data['age'] <= 50)]
age_data

age_data['age_group'] = pd.cut(age_data['age'], bins=10)
age_data['age_group'] = age_data['age_group'].astype(str)
age_data['cert_general'] = age_data['cert_general'].apply(str)
age_data['intercept'] = age_data['age_group']+'_'+age_data['cert_general']
age_data['intercept']
def aaa(a):
    if '_1' in a:
        return a
    if '_0' in a:
        return '0'

age_data['intercept'] = age_data['intercept'].apply(aaa)

aged_data = age_data[['label_callback','cert_general','label_salary',#'cert_general','diff',\
                  'age','degree','gender','work_years','intercept',\
                  'l1_code','job_city_class','Time_FE']]

cd_data = aged_data.copy()
Y_callback = cd_data[['label_callback']]
x_data = cd_data.iloc[:,3:]
X = pd.get_dummies(x_data)


X = X.drop(['intercept_0','job_city_class_other','degree_polytechnic','Time_FE_2023-01','l1_code_100000'], axis = 1)
X = sm.add_constant(X)

Logit_model = Logit(Y_callback, X)
print('Model built success')

result = Logit_model.fit()
print('Model fit success')

result.summary()



