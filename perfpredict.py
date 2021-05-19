# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:29:37 2021

@author: tfahry
"""

from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
import pydotplus
import pydot as pyd
import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.optimizers import SGD

dataframe = pd.read_csv(
    'C:\\Users\\Tfarhy\\OneDrive - Network Rail\\2021.01.21_Keras demos\\ttperf3.csv')

dataframe = dataframe.dropna()
dataframe = dataframe.drop(columns='operator')

for floatvar in ['ppm', 'canx', 'ontime', 'righttime', 'originontime']:
    dataframe[floatvar] = dataframe[floatvar].str.rstrip(
        '%').astype('float') / 100.0

val_dataframe = dataframe.sample(frac=0.05, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(f'Using {len(train_dataframe)} samples for training and {len(val_dataframe)} for validation')


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


def encode_numerical_feature(feature, name, dataset):

    normalizer = Normalization()

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_string_categorical_feature(feature, name, dataset):

    index = StringLookup()

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    index.adapt(feature_ds)

    encoded_feature = index(feature)

    encoder = CategoryEncoding(output_mode="binary")

    feature_ds = feature_ds.map(index)

    encoder.adapt(feature_ds)

    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):

    encoder = CategoryEncoding(output_mode="binary")

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    encoder.adapt(feature_ds)

    encoded_feature = encoder(feature)
    return encoded_feature


#operator = keras.Input(shape=(1,), name="operator", dtype="string")


planned = keras.Input(shape=(1,), name="planned", )
# full = keras.Input(shape=(1,), name="full")
# part = keras.Input(shape=(1,), name="part")


ppm = keras.Input(shape=(1,), name="ppm",)
# canx = keras.Input(shape=(1,), name="canx")
# ontime = keras.Input(shape=(1,), name="ontime")
# righttime = keras.Input(shape=(1,), name="righttime")
# originontime = keras.Input(shape=(1,), name="originontime")
# aol = keras.Input(shape=(1,), name="aol")
# recstops = keras.Input(shape=(1,), name="recstops")
# ontimeotm = keras.Input(shape=(1,), name="ontimeotm")

all_inputs = [
    planned,
    # full,
    # part,
    ppm,
    # canx,
    # ontime,
    # righttime,
    # originontime,
    # aol,
    # recstops,
    # ontimeotm
]

#operator_encoded = encode_string_categorical_feature(operator, "operator", train_ds)

planned_encoded = encode_numerical_feature(planned, "planned", train_ds)
# full_encoded = encode_numerical_feature(full, "full", train_ds)
# part_encoded = encode_numerical_feature(part, "part", train_ds)
ppm_encoded = encode_numerical_feature(ppm, "ppm", train_ds)
# canx_encoded = encode_numerical_feature(canx, "canx", train_ds)
# ontime_encoded = encode_numerical_feature(ontime, "ontime", train_ds)
# righttime_encoded = encode_numerical_feature(righttime, "righttime", train_ds)
# originontime_encoded = encode_numerical_feature(originontime, "originontime", train_ds)
# aol_encoded = encode_numerical_feature(aol, "aol", train_ds)
# recstops_encoded = encode_numerical_feature(recstops, "recstops", train_ds)
# ontimeotm_encoded = encode_numerical_feature(ontimeotm, "ontimeotm", train_ds)

all_features = layers.concatenate([

    planned_encoded,
    # full_encoded,
    # part_encoded,
    ppm_encoded,
    # canx_encoded,
    # ontime_encoded,
    # righttime_encoded,
    # originontime_encoded,
    # aol_encoded,
    # recstops_encoded,
    # ontimeotm_encoded
])

#%%
x = layers.Dense(10, activation="relu")(all_features)
x = layers.Dense(10, activation="sigmoid")(all_features)
# x = layers.Dropout(0.5)(x)
# x = layers.Dense(128, activation = 'relu')(x)
# x = layers.Dense(128, activation = 'relu')(x)
# x = layers.Dense(128, activation = 'relu')(x)
# x = layers.Dense(128, activation = 'relu')(x)

output = layers.Dense(1, activation="relu")(x)

model = keras.Model(all_inputs, output)

opt = SGD(lr=0.1, momentum=0.2, decay=0.01)

model.compile(optimizer='sgd',  loss='binary_crossentropy',
              metrics=["accuracy"])


#%%
model.fit(train_ds, epochs=10, validation_data=val_ds)

#%% Inference

sample = {
    "planned": 0,
    "ppm": 0,
}

input_dict = {name: tf.convert_to_tensor(
    [value]) for name, value in sample.items()}
predictions = model.predict(input_dict)[0][0]


print(f'{(predictions)}')

#%%

keras.utils.vis_utils.pydot = pyd


#Visualize Model

def visualize_model(model):
  return SVG(model_to_dot(model).create(prog='C:\\Users\\Tfarhy\\OneDrive - Network Rail\\2021.01.21_Keras demos\\keras\\Graphviz\\bin\\dot.exe', format='svg'))


#create your model
#then call the function on your model
visualize_model(model)
#keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
