def ann_test(P, Q, V, Va, num_train, num_bus):
    import numpy as np
    import pandas as pd
    import pickle
    import os
    import keras
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import mean_absolute_error
    from keras.models import load_model
    
    from keras.models import Sequential # Used to build model
    from keras.layers import Dense # Type of layer
    from keras.optimizers import Adam # Optimization technique
    from keras.layers import Dropout # For tuning the neural network
    from keras import regularizers # For regularization
    

    # X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    # Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    
    # Finding P_mape
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = P
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    
    # Load the model
    file_path="ann_saved_models\p.h5"
    model = load_model(file_path)
    
    Y_pred = model.predict(X)
    # Y_pred = pd.DataFrame(Y_pred)
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    Y_temp = Y.to_numpy()
    error = np.zeros(num_test)
    for i in range(num_test):
        y_pred = Y_pred[i,:]
        y_act = Y_temp[i,:]
        error[i] = mean_absolute_percentage_error(y_act, y_pred)
    P_mape = error.mean()


    # Finding Q_mape
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = Q
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape

    # Load ANN Model
    file_path="ann_saved_models\q.h5"
    model = load_model(file_path)

    # Test on data
    Y_pred = model.predict(X)
    # Y_pred = pd.DataFrame(Y_pred)
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    Y_temp = Y.to_numpy()
    error = np.zeros(num_test)
    for i in range(num_test):
        y_pred = Y_pred[i,:]
        y_act = Y_temp[i,:]
        error[i] = mean_absolute_percentage_error(y_act, y_pred)
    Q_mape = error.mean()


    # Finding V_mae
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = V
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape

    # Load ANN Model
    file_path="ann_saved_models\v.h5"
    model = load_model(file_path)

    # Test on data
    Y_pred = model.predict(X)
    # Y_pred = pd.DataFrame(Y_pred)
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    Y_temp = Y.to_numpy()
    error = np.zeros(num_test)
    for i in range(num_test):
        y_pred = Y_pred[i,:]
        y_act = Y_temp[i,:]
        error[i] = mean_absolute_error(y_act, y_pred)
    V_mae = error.mean()

    # Finding Va_mae
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = Va
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape

    # Load ANN Model
    file_path="ann_saved_models\va.h5"
    model = load_model(file_path)

    # Test on data
    Y_pred = model.predict(X)
    # Y_pred = pd.DataFrame(Y_pred)
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    Y_temp = Y.to_numpy()
    error = np.zeros(num_test)
    for i in range(num_test):
        y_pred = Y_pred[i,:]
        y_act = Y_temp[i,:]
        error[i] = mean_absolute_error(y_act, y_pred)
    Va_mae = error.mean()


    return [P_mape, Q_mape, V_mae, Va_mae]

