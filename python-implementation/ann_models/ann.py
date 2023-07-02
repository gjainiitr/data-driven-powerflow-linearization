#importing libraries
import pandas as pd
import numpy as np
import sklearn
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from ann_train import *
from ann_test import *
import pickle

P = pd.read_csv('data/30P.csv', header = None)
Q = pd.read_csv('data/30Q.csv', header = None)
V = pd.read_csv('data/30V.csv', header = None)
Va = pd.read_csv('data/30Va.csv', header = None)

X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = P.join(Q, lsuffix='_P', rsuffix='_Q')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=(0.2), random_state=42)
train_eg = X_train.shape[0]
num_bus = P.shape[1]
Va_train = X_train.iloc[:, 0:num_bus]
V_train = X_train.iloc[:, num_bus:2*num_bus]
P_train = y_train.iloc[:, 0:num_bus]
Q_train = y_train.iloc[:, num_bus:2*num_bus]

Va_test = X_test.iloc[:, 0:num_bus]
V_test = X_test.iloc[:, num_bus:2*num_bus]
P_test = y_test.iloc[:, 0:num_bus]
Q_test = y_test.iloc[:, num_bus:2*num_bus]

test_eg = X_test.shape[0]

ann_train(P_train, Q_train, V_train, Va_train, train_eg, num_bus)
[P_mape, Q_mape, V_mae, Va_mae] = ann_test(P_test, Q_test, V_test, Va_test, test_eg, num_bus)

print("P_mape: ",P_mape)
print("Q_mape: ",Q_mape)
print("V_mae: ",V_mae)
print("Va_mae: ",Va_mae)













################## Check below code


from keras.models import Sequential # Used to build model
from keras.layers import Dense # Type of layer
from keras.optimizers import Adam # Optimization technique
from keras.layers import Dropout # For tuning the neural network
from keras import regularizers # For regularization
mape = tf.keras.losses.MeanAbsolutePercentageError()
m = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)

#initializing neural network
model1 = Sequential()
model1.add(Dense(num_bus, input_dim=num_bus*2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model1.add(Dense(num_bus*2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
adam = Adam(lr=0.001)
model1.compile(loss=mape, optimizer='adam', metrics=[m])

model1.summary()


history1=model1.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=10, batch_size=32) 
#Saving the model
file_path="ann_models\ann_saved_models\ann_model.hdf5"
if os.path.exists(file_path):
        os.remove(file_path)
model1.save_weights(file_path)