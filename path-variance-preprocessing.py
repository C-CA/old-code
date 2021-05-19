# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:04:02 2021

@author: tfahry
"""
import pandas as pd

ldf = pd.read_csv('Lateness.csv', index_col='Date')

locset = list(set(ldf['Geography']))

loclist = []
temploc = []

for loc in ldf['Geography']:
    if ';Terminate' in loc:
        temploc.append(loc)
        loclist.append(temploc)
        temploc = []
    else:
        temploc.append(loc)
        

def compare(a,b):
    global loclist
    
    for sublist in loclist:
        if a in sublist and b in sublist:
            return sublist.index(a) - sublist.index(b)
        
    else:
        return 0
    
from functools import cmp_to_key

locset.sort(key = cmp_to_key(compare))

print(locset)

#%%
import numpy as np
df = pd.read_csv(r'C:\Users\Tfarhy\OneDrive - Network Rail\2021.02.15_Path Variance demos\pivoted-lateness.csv', index_col=0)
df = df.mask(df.eq('#ERROR'))
df = df.dropna(axis=1,thresh=len(df)*.98).dropna()
columns = sorted(list(df.columns), key = cmp_to_key(compare))
df  = df[columns]

for col in columns:
    df[col]= df[col].str.replace(',','')
df = df.astype(float)

bdf = pd.read_csv(r'C:\Users\Tfarhy\OneDrive - Network Rail\2021.02.15_Path Variance demos\pivoted-busyness.csv', index_col=0)
for col in bdf.columns:
    bdf[col] = bdf[col].str.replace(',','')
bdf = bdf.astype(float)

odf = pd.read_csv(r'C:\Users\Tfarhy\OneDrive - Network Rail\2021.02.15_Path Variance demos\pivoted-offpeak.csv', index_col=0)
odf = odf.fillna(value=0)
odf = odf.astype(float)

X = df.merge(odf,how='left', left_index=True, right_index=True)
X = X.merge(bdf,how='left', left_index=True, right_index=True)

X = X[['OFF-PEAK','PEAK']]

y=df.values
X=X.values

#%%
# example of evaluating chained multioutput regression with an SVM model

import numpy as np
np.set_printoptions(suppress=True)

from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR

# define dataset
#X, y = make_regression(n_samples=1000, n_features=1,n_informative=1, n_targets=3, random_state=1, noise=0.5)

# define base model
model = LinearSVR()
# define the chained multioutput wrapper model
wrapper = RegressorChain(model)
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(wrapper, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))



wrapper.fit(X,y)

print(wrapper.predict(X[0:20]))
print(y[0:20])

