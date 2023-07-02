def SVM_train(P, Q, V, Va, num_train, num_bus):
    import numpy as np
    from sklearn.svm import SVR
    import pandas as pd
    import pickle
 
    # X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    # Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    
    # Training for P
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = P
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape

    import os
    
    # Y_col = num_bus
    for i in range(0,Y_col):
        y = Y.iloc[:,i]
        clf = SVR(kernel='linear')                        
        clf.fit(X, y)
        pkl_filename = "svm_models/"+ "p_"+ str(i) +".pkl"
        if os.path.exists(pkl_filename):
            os.remove(pkl_filename)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)

    # Training for Q
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = Q
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape    
    for i in range(0,Y_col):
        # if judge_Y[i]:
        y = Y.iloc[:,i]
        clf = SVR(kernel='linear')
        clf.fit(X, y)
        pkl_filename = "svm_models/"+ "q_"+ str(i) +".pkl"
        if os.path.exists(pkl_filename):
            os.remove(pkl_filename)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)
    
    # Training for V
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = V
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape    
    for i in range(0,Y_col):
        # if judge_Y[i]:
        y = Y.iloc[:,i]
        clf = SVR(kernel='linear')
        clf.fit(X, y)
        pkl_filename = "svm_models/"+ "v_"+ str(i) +".pkl"
        if os.path.exists(pkl_filename):
            os.remove(pkl_filename)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)


    # Training for Xva
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = Va
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape    
    for i in range(0,Y_col):
        # if judge_Y[i]:
        y = Y.iloc[:,i]
        clf = SVR(kernel='linear')
        clf.fit(X, y)
        pkl_filename = "svm_models/"+ "va_"+ str(i) +".pkl"
        if os.path.exists(pkl_filename):
            os.remove(pkl_filename)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)
    