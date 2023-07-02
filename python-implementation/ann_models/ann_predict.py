# Type is a string in small letters
def SVM_predict(x1, x2, num_train, num_bus, type, isForward):
    import numpy as np
    from keras.models import Sequential # Used to build model
    from keras.layers import Dense # Type of layer
    from keras.optimizers import Adam # Optimization technique
    from keras.layers import Dropout # For tuning the neural network
    from keras import regularizers # For regularization
    from keras.models import load_model
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
            
    filename = "ann_saved_models/"+ type +".h5"
    model = load_model(file_path)
    Y = model.predict(X)
    Y = pd.DataFrame(Y)
    return Y