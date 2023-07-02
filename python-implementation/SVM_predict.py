# Type is a string
def SVM_predict(x1, x2, num_train, num_bus, type, isForward):
    import numpy as np
    from sklearn.svm import SVR
    import pandas as pd
    import pickle
    type = type + '_'
    Y = np.zeros((num_test, num_bus))    
    if isForward:
        X = x1.join(x2, lsuffix='_Va', rsuffix='_V')
    else:
        X = x1.join(x2, lsuffix='_P', rsuffix='_Q')    
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    for i in range(0,Y_col):        
        pkl_filename = "svm_models/"+ type + str(i) +".pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        y = model.predict(X)
        Y[:,i] = y
    Y = pd.DataFrame(Y)
    return Y