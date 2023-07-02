'''
Goal: Perform PLS
Input - P, Q, V, Va, num_train, num_bus
Output - Xp, Xq, P_mape, Q_mape, Xv, Xva, V_mae, Va_mae

'''
import numpy as np
import pandas as pd
def PLS(P, Q, V, Va, num_train, num_bus):
    import pandas as pd
    import numpy as np
    import sklearn
    
    Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score

    pls = PLSRegression()
    scores = cross_val_score(pls, X, P, cv=5, scoring="neg_mean_absolute_percentage_error")
    P_mape = scores.mean()

    pls = PLSRegression()
    scores = cross_val_score(pls, X, Q, cv=5, scoring="neg_mean_absolute_percentage_error")
    Q_mape = scores.mean()

    pls2 = PLSRegression()
    scores = cross_val_score(pls2, Y, V, cv=5, scoring="neg_mean_absolute_error")
    V_mae = scores.mean()

    pls2 = PLSRegression()
    scores = cross_val_score(pls2, Y, Va, cv=5, scoring="neg_mean_absolute_error")
    Va_mae = scores.mean()

    # Find Xp, Xq, Xv, Xva - Try out on your own
    pls3 = PLSRegression()
    pls3.fit(X,P)
    Xp = np.array(pls3.coef_)
    # arr2 = np.array(pls3.intercept_)
    # Xp = np.column_stack((arr1,arr2))

    pls4 = PLSRegression()
    pls4.fit(X,Q)
    Xq = np.array(pls4.coef_)
    # arr2 = np.array(pls4.intercept_)
    # Xq = np.column_stack((arr1,arr2))

    pls5 = PLSRegression()
    pls5.fit(Y,V)
    Xv = np.array(pls5.coef_)
    # arr2 = np.array(pls5.intercept_)
    # Xv = np.column_stack((arr1,arr2))

    pls6 = PLSRegression()
    pls6.fit(Y,Va)
    Xva = np.array(pls6.coef_)
    # arr2 = np.array(pls6.intercept_)
    # Xv = np.column_stack((arr1,arr2))

    return [Xp, Xq, Xv, Xva, P_mape, Q_mape, V_mae, Va_mae]

P = pd.read_csv('data/118P.csv', header = None)
Q = pd.read_csv('data/118Q.csv', header = None)
V = pd.read_csv('data/118V.csv', header = None)
Va = pd.read_csv('data/118Va.csv', header = None)
num_train = P.shape[0]
num_bus = P.shape[1]

[Xp, Xq, Xv, Xva, P_mape, Q_mape, V_mae, Va_mae] = PLS(P, Q, V, Va, num_train, num_bus)
print(P_mape)
print(Q_mape)
print(V_mae)
print(Va_mae)