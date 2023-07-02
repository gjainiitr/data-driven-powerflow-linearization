import pandas as pd
import numpy as np
from SVM_train import *
from SVM_test import *
import pickle

P = pd.read_csv('data/118P.csv', header = None)
Q = pd.read_csv('data/118Q.csv', header = None)
V = pd.read_csv('data/118V.csv', header = None)
Va = pd.read_csv('data/118Va.csv', header = None)

X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = P.join(Q, lsuffix='_P', rsuffix='_Q')

num_train = X.shape[0]
num_bus = P.shape[1]

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=(0.2), random_state=42)
train_eg = X_train.shape[0]

Va_train = X_train.iloc[:, 0:num_bus]
V_train = X_train.iloc[:, num_bus:2*num_bus]
P_train = y_train.iloc[:, 0:num_bus]
Q_train = y_train.iloc[:, num_bus:2*num_bus]

Va_test = X_test.iloc[:, 0:num_bus]
V_test = X_test.iloc[:, num_bus:2*num_bus]
P_test = y_test.iloc[:, 0:num_bus]
Q_test = y_test.iloc[:, num_bus:2*num_bus]



# print(P_train.shape)
# print(Q_train.shape)
# print(V_train.shape)
# print(Va_train.shape)

test_eg = X_test.shape[0]

SVM_train(P_train, Q_train, V_train, Va_train, train_eg, num_bus)
[P_mape, Q_mape, V_mae, Va_mae] = SVM_test(P_test, Q_test, V_test, Va_test, test_eg, num_bus)

print("P_mape: ",P_mape)
print("Q_mape: ",Q_mape)
print("V_mae: ",V_mae)
print("Va_mae: ",Va_mae)