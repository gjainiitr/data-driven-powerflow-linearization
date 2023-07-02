# output is a string in lower_case
def RF_predict(x1, x2, num_train, num_bus, output, isForward):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    import pickle
    # type = type + '_'
    Y = np.zeros((num_test, num_bus))    
    if isForward:
        X = x1.join(x2, lsuffix='_Va', rsuffix='_V')
    else:
        X = x1.join(x2, lsuffix='_P', rsuffix='_Q')    
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
            
    pkl_filename = "RF_models/"+ type + ".pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
    y = model.predict(X)
    y = pd.DataFrame(y)
    
    return y