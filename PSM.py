import os
import pandas as pd
import numpy as np

import math
import sys
import random

import psmatching.match as psm
import pytest
from psmatching.utilities import *

from statsmodels.regression.linear_model import OLS, GLS
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit, MNLogit

sample_dir = '/group/regular/Paper/raw_data' + insert_code
sample_files = []
for r,l,f in os.walk(sample_dir):
    for ff in f:
        if ff.endswith('.parquet'):
            sample_files.append(os.path.join(r, ff))
df = pd.concat([pd.read_parquet(file) for file in sample_files], ignore_index=True)

data = df.copy()
data['cert_amount'] = data['cert_amount'].replace(np.NaN, '0')

a = data[['geek_id','age','gender','degree','work_years','fresh_graduate','cert_amount']]
b = pd.get_dummies(data.major_std)
b = b.drop(['people_resource'], axis = 1)
cc= pd.concat([a,b], axis = 1)

# c1 = cc[cc['cert_amount']=='1'].sample(n=1000)
# c2 = cc[cc['cert_amount']=='0'].sample(n=3000)#(frac =.50)
c = cc.copy()
# c = pd.concat([c1,c2], axis = 0).reset_index(drop = True)

major_dict = {}
for i, n in enumerate(data.major_std.unique()):
    major_dict[n] = 'c'+str(i)

c = c.rename(columns = major_dict)

md = str('cert_amount ~ ')+str(list(c.columns)).replace("'cert_amount', ",'').replace("'geek_id', ",'').replace("', '",' + ').replace("'",'').replace('[','').replace(']','')

k = "1"
model = md
m = psm.PSMatch(df, model, k)
print("\finish ...", end = " ")
glm_binom = sm.formula.glm(formula = model, data = c, family = sm.families.Binomial())
result = glm_binom.fit()
propensity_scores = result.fittedvalues
c["PROPENSITY"] = propensity_scores

groups = c.cert_amount
propensity = c.PROPENSITY
groups = groups == groups.unique()[1]
n = len(groups)

n1 = groups[groups==1].sum()
n2 = n-n1
g1, g2 = propensity[groups==0], propensity[groups==1]

if n1 > n2:
    n1, n2, g1, g2 = n2, n1, g2, g1

m_order = list(np.random.permutation(groups[groups==1].index))

matches = {}
k = int(k)
for m in m_order:
    dist = abs(g1[m]-g2)
    array = np.array(dist)

    if k < len(array):
        k_smallest = np.partition(array, k)[:k].tolist()
        caliper = None
        if caliper:
            caliper = float(caliper)
            keep_diffs = [i for i in k_smallest if i <= caliper]
            keep_ids = np.array(dist[dist.isin(keep_diffs)].index)
        else:
            keep_ids = np.array(dist[dist.isin(k_smallest)].index)
        if len(keep_ids) > k:
            matches[m] = list(np.random.choice(keep_ids, k, replace=False))
        elif len(keep_ids) < k:
            while len(matches[m]) <= k:
                matches[m].append("NA")
        else:
            matches[m] = keep_ids.tolist()
        replace = False
        if not replace:
            g2 = g2.drop(matches[m])
print("\nmatch_finish!")

