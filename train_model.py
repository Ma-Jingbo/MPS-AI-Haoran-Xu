# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#model = load_model('model.h5')

readings = Input(shape=(3, ))
x = Dense(8, activation="linear", kernel_initializer="glorot_uniform")(readings)
x = Dense(32, activation="relu", kernel_initializer="glorot_uniform")(x)
x = Dense(8, activation="relu", kernel_initializer="glorot_uniform")(x)
#x = Dense(3, activation="relu", kernel_initializer="glorot_uniform")(x)
benzene = Dense(3, activation="linear", kernel_initializer="glorot_uniform")(x)

model = Model(inputs=[readings], outputs=[benzene])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

NUM_EPOCHS = 500
BATCH_SIZE = 32

input_up = np.array([20, 500, 973])
output_up = np.array([20, 2, 5000])

folder = "E:\\ANN"
Xtrain = pd.read_csv(os.path.join(folder, 'Xtrain1.csv'))
Ytrain = pd.read_csv(os.path.join(folder, 'Ytrain1.csv'))
for ii in np.arange(3):
    Xtrain.iloc[:, ii] = Xtrain.iloc[:, ii]/input_up[ii]
for ii in np.arange(3):
    Ytrain.iloc[:, ii] = Ytrain.iloc[:, ii]/output_up[ii]
history = model.fit(Xtrain, Ytrain,
                    batch_size=BATCH_SIZE, 
                    epochs=NUM_EPOCHS, 
                    validation_split=0.2)

model.save('E:\\ANN\\model3.h5')