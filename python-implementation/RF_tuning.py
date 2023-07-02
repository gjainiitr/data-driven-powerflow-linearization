# Might have to create custom function for scoring neg_mean_absolute_percentage_error

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import pickle
import os

# Training for P
X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = P
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
clf = RandomForestRegressor(criterion = 'mae')
    # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 50, num = 1)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 10, scoring='neg_mean_absolute_percentage_error')
# Fit the random search model
rf_random.fit(X,Y.to_numpy())

print("For P: ",rf_random.best_params_)


# Training for Q
X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = Q
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
clf = RandomForestRegressor(criterion = 'mae')
    # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 50, num = 1)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 10, , scoring='neg_mean_absolute_percentage_error')
# Fit the random search model
rf_random.fit(X,Y.to_numpy())

print("For Q: ",rf_random.best_params_)

# Training for V
X = P.join(Q, lsuffix='_P', rsuffix='_Q')
Y = V
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
clf = RandomForestRegressor(criterion = 'mae')
    # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 50, num = 1)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 10, scoring='neg_mean_absolute_error')
# Fit the random search model
rf_random.fit(X,Y.to_numpy())

print("For V: ",rf_random.best_params_)

# Training for Va
X = P.join(Q, lsuffix='_P', rsuffix='_Q')
Y = Va
X_row,X_col = X.shape
Y_row,Y_col = Y.shape
clf = RandomForestRegressor(criterion = 'mae')
    # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 50, num = 1)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 10, scoring='neg_mean_absolute_error')
# Fit the random search model
rf_random.fit(X,Y.to_numpy())

print("For Va: ",rf_random.best_params_)