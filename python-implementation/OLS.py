'''
Goal: Perform OLS
Input - P, Q, V, Va, num_train, num_bus
Output - Xp, Xq, P_mape, Q_mape, Xv, Xva, V_mae, Va_mae

'''
import numpy as np
import pandas as pd

def OLS(P, Q, V, Va, num_train, num_bus):
    import pandas as pd
    import numpy as np
    import sklearn
    
    Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    ols = LinearRegression()
    scores = cross_val_score(ols, X, P, cv=5, scoring="neg_mean_absolute_percentage_error")
    P_mape = scores.mean()

    ols = LinearRegression()
    scores = cross_val_score(ols, X, Q, cv=5, scoring="neg_mean_absolute_percentage_error")
    Q_mape = scores.mean()

    ols2 = LinearRegression()
    scores = cross_val_score(ols2, Y, V, cv=5, scoring="neg_mean_absolute_error")
    V_mae = scores.mean()

    ols2 = LinearRegression()
    scores = cross_val_score(ols2, Y, Va, cv=5, scoring="neg_mean_absolute_error")
    Va_mae = scores.mean()

    # Find Xp, Xq, Xv, Xva - Try out on your own
    ols3 = LinearRegression()
    ols3.fit(X,P)
    arr1 = np.array(ols3.coef_)
    arr2 = np.array(ols3.intercept_)
    Xp = np.column_stack((arr1,arr2))

    ols4 = LinearRegression()
    ols4.fit(X,Q)
    arr1 = np.array(ols4.coef_)
    arr2 = np.array(ols4.intercept_)
    Xq = np.column_stack((arr1,arr2))

    ols5 = LinearRegression()
    ols5.fit(Y,V)
    arr1 = np.array(ols5.coef_)
    arr2 = np.array(ols5.intercept_)
    Xv = np.column_stack((arr1,arr2))

    ols6 = LinearRegression()
    ols6.fit(Y,Va)
    arr1 = np.array(ols6.coef_)
    arr2 = np.array(ols6.intercept_)
    Xva = np.column_stack((arr1,arr2))

    return [Xp, Xq, Xv, Xva, P_mape, Q_mape, V_mae, Va_mae]

P = pd.read_csv('data/118P.csv', header = None)
Q = pd.read_csv('data/118Q.csv', header = None)
V = pd.read_csv('data/118V.csv', header = None)
Va = pd.read_csv('data/118Va.csv', header = None)
num_train = P.shape[0]
num_bus = P.shape[1]

[Xp, Xq, Xv, Xva, P_mape, Q_mape, V_mae, Va_mae] = OLS(P, Q, V, Va, num_train, num_bus)
print(P_mape)
print(Q_mape)
print(V_mae)
print(Va_mae)
