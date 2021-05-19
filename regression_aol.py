# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:35:14 2021

@author: TFahry
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

import tensorflow as tf

import keras
from keras import layers
from keras.layers.experimental import preprocessing

dataframe = pd.read_csv('C:\\Users\\Tfarhy\\OneDrive - Network Rail\\2021.01.21_Keras demos\\ttperf4.csv')
dataframe = dataframe.dropna()

for floatvar in ['ppm','canx','ontime','righttime','originontime']:
     dataframe[floatvar] = dataframe[floatvar].str.rstrip('%').astype('float') / 100.0
     
dataframe = dataframe.drop(columns = 'operator')

print(dataframe.head())

train_dataset = dataframe.sample(frac=0.8, random_state=0)
test_dataset = dataframe.drop(train_dataset.index)

target = 'ontime'

mydf = dataframe.copy()
mylabels = mydf.pop(target)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop(target)
test_labels = test_features.pop(target)

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

test_results = {}

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

#%%dnn  model
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

dnn_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

#%%fitting 
history = dnn_model.fit(
    train_features, train_labels, 
    epochs=100,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

plot_loss(history)

dnn_model.evaluate(test_features, test_labels, verbose=1)

#%%plotting prediction accuracy
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Target]')
  plt.legend()
  plt.grid(True)

def plot_planned(x, y):
  plt.scatter(train_features, train_labels, label='Data', s = 5)
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('planned')
  plt.ylabel('Target')
  plt.legend()

def p(i):
    global mydf
    global tdf
    global mylabels
    global dnn_model
    

    pred = dnn_model.predict(np.array(mydf[i:i+1]))[0,0]
    actual = mylabels[i]
    
    print(f'Predicted AOL : {pred} Actual AOL: {actual}')
    
def parserow(dfrow):
    a = np.array(dfrow)[1:]
    
    # for index in [1,2,3,4,7]:
    #     a[index] = float(a[index].rstrip('%'))/100
    
    return a[:-1]

#%% our tests
tdf = pd.read_csv('C:\\Users\\Tfarhy\\OneDrive - Network Rail\\2021.01.21_Keras demos\\forplot.csv')
tdf.dropna()

for floatvar in ['PPM%','Canx%','On Time% (WTT)','Right Time%','Origin On Time% (WTT)']:
     tdf[floatvar] = tdf[floatvar].str.rstrip('%').astype('float') / 100.0
     

#sns.pairplot(tdf[['Recorded Stops','Canx%','Origin On Time% (WTT)','On Time (OTM)','PPM']], diag_kind='kde' ,plot_kws={"s": 2}, height=3)



#%%
tdf = tdf.drop(columns = ['operator','PPM'])























