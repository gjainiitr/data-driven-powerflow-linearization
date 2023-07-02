# Might have to write custom MAPE Loss and metrics function for Keras
# Finish doing Hyper-parameter tuning for ANN_train file (ANN model)


def ann_train(P, Q, V, Va, num_train, num_bus):
    import numpy as np
    import pandas as pd
    import pickle
    import os
    import keras
    import tensorflow as tf
    

    # X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    # Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    
    # Training for P
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = P
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    
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
    model1.add(Dense(num_bus, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    adam = Adam(lr=0.001)
    model1.compile(loss=mape, optimizer='adam', metrics=[m])
    # model1.summary()
    history1=model1.fit(X, Y,epochs=10, batch_size=32) 
    #Saving the model
    file_path="ann_saved_models\p.h5"
    if os.path.exists(file_path):
            os.remove(file_path)

    model1.save(file_path)


    # Training for Q
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = Q
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    m = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)

    #initializing neural network
    model1 = Sequential()
    model1.add(Dense(num_bus, input_dim=num_bus*2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model1.add(Dense(num_bus, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    adam = Adam(lr=0.001)
    model1.compile(loss=mape, optimizer='adam', metrics=[m])
    # model1.summary()
    history1=model1.fit(X, Y,epochs=10, batch_size=32) 
    #Saving the model
    file_path="ann_saved_models\q.h5"
    if os.path.exists(file_path):
            os.remove(file_path)
    model1.save(file_path)

    # Training for V
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = V
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape

    mae = tf.keras.losses.MeanAbsoluteError()
    m = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)

    #initializing neural network
    model1 = Sequential()
    model1.add(Dense(num_bus, input_dim=num_bus*2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model1.add(Dense(num_bus, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    adam = Adam(lr=0.001)
    model1.compile(loss=mae, optimizer='adam', metrics=[m])
    # model1.summary()
    history1=model1.fit(X, Y,epochs=10, batch_size=32) 
    #Saving the model
    file_path="ann_saved_models\v.h5"
    if os.path.exists(file_path):
            os.remove(file_path)
    model1.save(file_path)

    # Training for Va
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = Va
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    

    mae = tf.keras.losses.MeanAbsoluteError()
    m = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)

    #initializing neural network
    model1 = Sequential()
    model1.add(Dense(num_bus, input_dim=num_bus*2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model1.add(Dense(num_bus, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    adam = Adam(lr=0.001)
    model1.compile(loss=mae, optimizer='adam', metrics=[m])
    # model1.summary()
    history1=model1.fit(X, Y,epochs=10, batch_size=32) 
    #Saving the model
    file_path="ann_saved_models\va.h5"
    if os.path.exists(file_path):
            os.remove(file_path)
    model1.save(file_path)
