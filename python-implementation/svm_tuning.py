import numpy as np
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import GridSearchCV 

param_grid = {'C': [0.1, 1, 10, 100, 1000], 
			'gamma': [50, 25, 10, 1, 0.1, 0.01, 0.001, 0.0001], 
			'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']} 


# Training for P
X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = P
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3, scoring='neg_mean_absolute_percentage_error')
grid.fit(X,Y.to_numpy()[:,0])
print("For P[0]: ",grid.best_params_)

X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = P
X_row,X_col = X.shape
Y_row,Y_col = Y.shapeR
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3, scoring='neg_mean_absolute_percentage_error')
grid.fit(X,Y.to_numpy()[:,1])
print("For P[1]: ",grid.best_params_)

# Training for Q
X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = Q
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3, scoring='neg_mean_absolute_percentage_error')
grid.fit(X,Y.to_numpy()[:,0])
print("For Q[0]: ",grid.best_params_)

X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = Q
X_row,X_col = X.shape
Y_row,Y_col = Y.shapeR
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3, scoring='neg_mean_absolute_percentage_error')
grid.fit(X,Y.to_numpy()[:,1])
print("For Q[1]: ",grid.best_params_)


# Training for V
X = P.join(Q, lsuffix='_P', rsuffix='_Q')
Y = V
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3, scoring='neg_mean_absolute_error')
grid.fit(X,Y.to_numpy()[:,0])
print("For V[0]: ",grid.best_params_)

X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = V
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3, scoring='neg_mean_absolute_error')
grid.fit(X,Y.to_numpy()[:,1])
print("For V[1]: ",grid.best_params_)


# Training for Va
X = P.join(Q, lsuffix='_P', rsuffix='_Q')
Y = Va
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3, scoring='neg_mean_absolute_error')
grid.fit(X,Y.to_numpy()[:,0])
print("For Va[0]: ",grid.best_params_)

X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = Va
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3, scoring='neg_mean_absolute_error')
grid.fit(X,Y.to_numpy()[:,1])
print("For Va[1]: ",grid.best_params_)
