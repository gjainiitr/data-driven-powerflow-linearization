import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression

def blrForward():
    x = pd.read_csv('x.csv', header=None)
    y = pd.read_csv('y.csv', header=None)
    x_row,x_col = x.shape
    y_row,y_col = y.shape
    x_coef = np.zeros((y_col,x_col+1))
    
    isColNotZero = ~(pd.DataFrame.sum(y,axis=0) == np.zeros(y_col))
    
    for i in range(0,y_col):
        if (isColNotZero[i]==1):
            y_temp = y[[i]]
            clf = ARDRegression()
            clf.fit(x,y_temp.values.ravel())
            coef = clf.coef_.T
            x_coef[i, :] = np.hstack((coef,clf.intercept_))
    
    x_coef = pd.DataFrame(x_coef)
    x_coef.to_csv('blr_coef_data.csv') 
% This is a comment created to test ShiftEdit