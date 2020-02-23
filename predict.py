# -*- coding: utf-8 -*-
"""

"""
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import keras

input_up = np.array([20, 500, 973])
output_up = np.array([20, 2, 5000])

folder = "C:\\Users\\xuhao\\Desktop"
input_data = pd.read_csv(os.path.join(folder, 'input1.csv'))
for ii in np.arange(3):
    input_data.iloc[:, ii] = input_data.iloc[:, ii]/input_up[ii]

model = load_model(os.path.join(folder, 'model.h5'))
output_data = model.predict(input_data)
output_data = pd.DataFrame(output_data)
for ii in np.arange(3):
    output_data.iloc[:, ii] = output_data.iloc[:, ii]*output_up[ii]
    
output_data.to_csv(os.path.join(folder, 'output1.csv'))
