def SVM_test(P, Q, V, Va, num_test, num_bus):
    import numpy as np
    from sklearn.svm import SVR
    import pandas as pd
    import pickle
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import mean_absolute_error

    # Finding P_mape
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = P
    Y_pred = np.zeros((num_test, num_bus))
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    error = np.zeros(num_test)
    for i in range(0,Y_col):        
        pkl_filename = "svm_models/"+ "p_"+ str(i) +".pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        y = model.predict(X)
        Y_pred[:,i] = y
    Y_pred = pd.DataFrame(Y_pred)
    for i in range(num_test):
        y_pred = Y_pred.iloc[i,:]
        y_act = Y.iloc[i,:]
        error[i] = mean_absolute_percentage_error(y_act, y_pred)
    P_mape = error.mean()

    # Finding Q_mape
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = Q
    Y_pred = np.zeros((num_test, num_bus))
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    error = np.zeros(num_test)
    for i in range(0,Y_col):        
        pkl_filename = "svm_models/"+ "q_"+ str(i) +".pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        y = model.predict(X)
        Y_pred[:,i] = y
    Y_pred = pd.DataFrame(Y_pred)
    for i in range(num_test):
        y_pred = Y_pred.iloc[i,:]
        y_act = Y.iloc[i,:]
        error[i] = mean_absolute_percentage_error(y_act, y_pred)
    Q_mape = error.mean()

    # Finding V_mae
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = V
    Y_pred = np.zeros((num_test, num_bus))
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    error = np.zeros(num_test)
    for i in range(0,Y_col):        
        pkl_filename = "svm_models/"+ "v_"+ str(i) +".pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        y = model.predict(X)
        Y_pred[:,i] = y
    Y_pred = pd.DataFrame(Y_pred)
    for i in range(num_test):
        y_pred = Y_pred.iloc[i,:]
        y_act = Y.iloc[i,:]
        error[i] = mean_absolute_error(y_act, y_pred)
    V_mae = error.mean()

    # Finding Va_mae
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = Va
    Y_pred = np.zeros((num_test, num_bus))
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    error = np.zeros(num_test)
    for i in range(0,Y_col):        
        pkl_filename = "svm_models/"+ "va_"+ str(i) +".pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        y = model.predict(X)
        Y_pred[:,i] = y
    Y_pred = pd.DataFrame(Y_pred)
    for i in range(num_test):
        y_pred = Y_pred.iloc[i,:]
        y_act = Y.iloc[i,:]
        error[i] = mean_absolute_error(y_act, y_pred)
    Va_mae = error.mean()

    return [P_mape, Q_mape, V_mae, Va_mae]      