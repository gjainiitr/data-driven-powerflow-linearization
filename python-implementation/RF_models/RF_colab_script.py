def RF_eval(P, Q, V, Va, num_train, num_bus):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_absolute_percentage_error
    # import pickle
    import os

    # X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    # Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = P.join(Q, lsuffix='_P', rsuffix='_Q')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=(0.2), random_state=42)
    train_eg = X_train.shape[0]
    num_bus = X_test.shape[1]
    Va_train = X_train.iloc[:, 0:num_bus]
    V_train = X_train.iloc[:, num_bus:2*num_bus]
    P_train = y_train.iloc[:, 0:num_bus]
    Q_train = y_train.iloc[:, num_bus:2*num_bus]

    Va_test = X_test.iloc[:, 0:num_bus]
    V_test = X_test.iloc[:, num_bus:2*num_bus]
    P_test = y_test.iloc[:, 0:num_bus]
    Q_test = y_test.iloc[:, num_bus:2*num_bus]

    test_eg = X_test.shape[0]    





    # Training for P
    X = Va_train.join(V_train, lsuffix='_Va', rsuffix='_V')
    Y = P_train
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    clf = RandomForestRegressor(criterion = 'mae')
    clf.fit(X,Y)
    X = Va_test.join(V_test, lsuffix='_Va', rsuffix='_V')
    Y = P_test
    Y_pred = clf.predict(X)
    Y_pred = pd.DataFrame(Y_pred)
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    error = np.zeros(X_row)
    for i in range(X_row):
        y_pred = Y_pred.iloc[i,:]
        y_act = Y.iloc[i,:]
        error[i] = mean_absolute_percentage_error(y_act, y_pred)
    P_mape = error.mean()

    # Training for Q
    X = Va_train.join(V_train, lsuffix='_Va', rsuffix='_V')
    Y = Q_train
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    clf = RandomForestRegressor(criterion = 'mae')
    clf.fit(X,Y)
    X = Va_test.join(V_test, lsuffix='_Va', rsuffix='_V')
    Y = Q_test
    Y_pred = clf.predict(X)
    Y_pred = pd.DataFrame(Y_pred)
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    error = np.zeros(X_row)
    for i in range(X_row):
        y_pred = Y_pred.iloc[i,:]
        y_act = Y.iloc[i,:]
        error[i] = mean_absolute_percentage_error(y_act, y_pred)
    Q_mape = error.mean()

    # Training for V
    X = P_train.join(Q_train, lsuffix='_P', rsuffix='_Q')
    Y = V_train
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape    
    clf = RandomForestRegressor(criterion = 'mae')
    clf.fit(X, Y)
    X = P_test.join(Q_test, lsuffix='_P', rsuffix='_Q')
    Y = V_test
    Y_pred = clf.predict(X)
    Y_pred = pd.DataFrame(Y_pred)
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    error = np.zeros(X_row)
    for i in range(X_row):
        y_pred = Y_pred.iloc[i,:]
        y_act = Y.iloc[i,:]
        error[i] = mean_absolute_error(y_act, y_pred)
    V_mae = error.mean()



    # Training for Va
    # Training for V
    X = P_train.join(Q_train, lsuffix='_P', rsuffix='_Q')
    Y = Va_train
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape    
    clf = RandomForestRegressor(criterion = 'mae')
    clf.fit(X, Y)
    X = P_test.join(Q_test, lsuffix='_P', rsuffix='_Q')
    Y = Va_test
    Y_pred = clf.predict(X)
    Y_pred = pd.DataFrame(Y_pred)
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    error = np.zeros(X_row)
    for i in range(X_row):
        y_pred = Y_pred.iloc[i,:]
        y_act = Y.iloc[i,:]
        error[i] = mean_absolute_error(y_act, y_pred)
    Va_mae = error.mean()

    return [P_mape, Q_mape, V_mae, Va_mae]
