def RF_train(P, Q, V, Va, num_train, num_bus):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    import pickle
    import os

    # X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    # Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    
    # Training for P
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = P
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    clf = RandomForestRegressor(criterion = 'mae')
    clf.fit(X,Y.to_numpy())
    pkl_filename = "RF_models/p.pkl"
    if os.path.exists(pkl_filename):
        os.remove(pkl_filename)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

    # Training for Q
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = Q
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    clf = RandomForestRegressor(criterion = 'mae')
    clf.fit(X,Y.to_numpy())
    pkl_filename = "RF_models/q.pkl"
    if os.path.exists(pkl_filename):
        os.remove(pkl_filename)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

    # Training for V
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = V
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape    
    clf = RandomForestRegressor(criterion = 'mae')
    clf.fit(X, Y.to_numpy())
    pkl_filename = "RF_models/v.pkl"
    if os.path.exists(pkl_filename):
        os.remove(pkl_filename)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

    # Training for Va
    # Training for V
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = Va
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape    
    clf = RandomForestRegressor(criterion = 'mae')
    clf.fit(X, Y.to_numpy())
    pkl_filename = "RF_models/va.pkl"
    if os.path.exists(pkl_filename):
        os.remove(pkl_filename)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)