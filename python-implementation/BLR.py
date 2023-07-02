import numpy as np
import pandas as pd
def BLR(P, Q, V, Va, num_train, num_bus):
    import numpy as np
    from sklearn.linear_model import ARDRegression
    import pandas as pd
    threshold = 1000
    
    # Finding Xp
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    # Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = P
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    
    judge_Y = ~(pd.DataFrame.sum(Y, axis=0) == np.zeros(Y_col))
    
    X_blr = np.zeros((Y_col,X_col+1))
    sigma_blr = np.zeros((Y_col,X_col))
    
    for i in range(0,Y_col):
        if judge_Y[i]:
            y = Y.iloc[:,i]
            clf = ARDRegression()
            clf.threshold_lambda = threshold
            
            clf.fit(X, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef,clf.intercept_))
    
    Xp = pd.DataFrame(X_blr)

    # Finding Xq
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    # Y = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = Q
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    
    judge_Y = ~(pd.DataFrame.sum(Y, axis=0) == np.zeros(Y_col))
    
    X_blr = np.zeros((Y_col,X_col+1))
    sigma_blr = np.zeros((Y_col,X_col))
    
    for i in range(0,Y_col):
        if judge_Y[i]:
            y = Y.iloc[:,i]
            clf = ARDRegression()
            clf.threshold_lambda = threshold
            
            clf.fit(X, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef,clf.intercept_))
    
    Xq = pd.DataFrame(X_blr)

    # Finding Xv
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    # Y = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = V
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    
    judge_Y = ~(pd.DataFrame.sum(Y, axis=0) == np.zeros(Y_col))
    
    X_blr = np.zeros((Y_col,X_col+1))
    sigma_blr = np.zeros((Y_col,X_col))
    
    for i in range(0,Y_col):
        if judge_Y[i]:
            y = Y.iloc[:,i]
            clf = ARDRegression()
            clf.threshold_lambda = threshold
            
            clf.fit(X, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef,clf.intercept_))
    
    Xv = pd.DataFrame(X_blr)

    # Finding Xva
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    # Y = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = Va
    X_row,X_col = X.shape
    Y_row,Y_col = Y.shape
    
    judge_Y = ~(pd.DataFrame.sum(Y, axis=0) == np.zeros(Y_col))
    
    X_blr = np.zeros((Y_col,X_col+1))
    sigma_blr = np.zeros((Y_col,X_col))
    
    for i in range(0,Y_col):
        if judge_Y[i]:
            y = Y.iloc[:,i]
            clf = ARDRegression()
            clf.threshold_lambda = 1000
            
            clf.fit(X, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef,clf.intercept_))
    
    Xva = pd.DataFrame(X_blr)

    # Finding P-error - Remove errors from above
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = P
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    X_row,X_col = X_train.shape
    Y_row,Y_col = Y_train.shape
    
    judge_Y = ~(pd.DataFrame.sum(Y_train, axis=0) == np.zeros(Y_col))
    
    X_blr = np.zeros((Y_col,X_col+1))
    sigma_blr = np.zeros((Y_col,X_col))
    
    for i in range(0,Y_col):
        if judge_Y[i]:
            y = Y_train.iloc[:,i]
            clf = ARDRegression()
            clf.threshold_lambda = 1000
            
            clf.fit(X_train, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef,clf.intercept_))
    
    Xp_temp = pd.DataFrame(X_blr)
    # Xp_temp = [30 x 61]
    # X = [61 x 1]
    # Y = [30 x 1]
    
    # Adding intercept to X_test
    one = X_test.to_numpy()
    d = pd.DataFrame(np.ones((X_test.shape[0], 1)))
    two = d.to_numpy()
    X_test = pd.DataFrame(np.column_stack((one,two)))

    # checkpoint
    error = np.zeros((X_test.shape[0]))
    for i in range(X_test.shape[0]):
        x = pd.DataFrame(X_test.iloc[i,:])
        y = Xp_temp.dot(x)
        y_pred = Y_train.iloc[i,:]
        curr = mean_absolute_percentage_error(y_pred, y)
        error[i] = curr
    
    P_mape = error.mean()
    
    # Finding Q error
    X = Va.join(V, lsuffix='_Va', rsuffix='_V')
    Y = Q
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    X_row,X_col = X_train.shape
    Y_row,Y_col = Y_train.shape
    
    judge_Y = ~(pd.DataFrame.sum(Y_train, axis=0) == np.zeros(Y_col))
    
    X_blr = np.zeros((Y_col,X_col+1))
    sigma_blr = np.zeros((Y_col,X_col))
    
    for i in range(0,Y_col):
        if judge_Y[i]:
            y = Y_train.iloc[:,i]
            clf = ARDRegression()
            clf.threshold_lambda = 1000
            
            clf.fit(X_train, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef,clf.intercept_))
    
    Xq_temp = pd.DataFrame(X_blr)
    # Xp_temp = [30 x 61]
    # X = [61 x 1]
    # Y = [30 x 1]
    
    # Adding intercept to X_test
    one = X_test.to_numpy()
    d = pd.DataFrame(np.ones((X_test.shape[0], 1)))
    two = d.to_numpy()
    X_test = pd.DataFrame(np.column_stack((one,two)))

    # checkpoint
    error = np.zeros((X_test.shape[0]))
    for i in range(X_test.shape[0]):
        x = pd.DataFrame(X_test.iloc[i,:])
        y = Xq_temp.dot(x)
        y_pred = Y_train.iloc[i,:]
        curr = mean_absolute_percentage_error(y_pred, y)
        error[i] = curr
    
    Q_mape = error.mean()
    
    # Finding V_error
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = V
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    X_row,X_col = X_train.shape
    Y_row,Y_col = Y_train.shape
    
    judge_Y = ~(pd.DataFrame.sum(Y_train, axis=0) == np.zeros(Y_col))
    
    X_blr = np.zeros((Y_col,X_col+1))
    sigma_blr = np.zeros((Y_col,X_col))
    
    for i in range(0,Y_col):
        if judge_Y[i]:
            y = Y_train.iloc[:,i]
            clf = ARDRegression()
            clf.threshold_lambda = 1000
            
            clf.fit(X_train, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef,clf.intercept_))
    
    Xv_temp = pd.DataFrame(X_blr)
    # Xp_temp = [30 x 61]
    # X = [61 x 1]
    # Y = [30 x 1]
    
    # Adding intercept to X_test
    one = X_test.to_numpy()
    d = pd.DataFrame(np.ones((X_test.shape[0], 1)))
    two = d.to_numpy()
    X_test = pd.DataFrame(np.column_stack((one,two)))

    # checkpoint
    error = np.zeros((X_test.shape[0]))
    for i in range(X_test.shape[0]):
        x = pd.DataFrame(X_test.iloc[i,:])
        y = Xv_temp.dot(x)
        y_pred = Y_train.iloc[i,:]
        curr = mean_absolute_error(y_pred, y)
        error[i] = curr
    
    V_mae = error.mean()
    
    # Finding Va error
    X = P.join(Q, lsuffix='_P', rsuffix='_Q')
    Y = Va
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    X_row,X_col = X_train.shape
    Y_row,Y_col = Y_train.shape
    
    judge_Y = ~(pd.DataFrame.sum(Y_train, axis=0) == np.zeros(Y_col))
    
    X_blr = np.zeros((Y_col,X_col+1))
    sigma_blr = np.zeros((Y_col,X_col))
    
    for i in range(0,Y_col):
        if judge_Y[i]:
            y = Y_train.iloc[:,i]
            clf = ARDRegression()
            clf.threshold_lambda = 1000
            
            clf.fit(X_train, y)
            coef = clf.coef_.T
            X_blr[i, :] = np.hstack((coef,clf.intercept_))
    
    Xva_temp = pd.DataFrame(X_blr)
    # Xp_temp = [30 x 61]
    # X = [61 x 1]
    # Y = [30 x 1]
    
    # Adding intercept to X_test
    one = X_test.to_numpy()
    d = pd.DataFrame(np.ones((X_test.shape[0], 1)))
    two = d.to_numpy()
    X_test = pd.DataFrame(np.column_stack((one,two)))

    # checkpoint
    error = np.zeros((X_test.shape[0]))
    for i in range(X_test.shape[0]):
        x = pd.DataFrame(X_test.iloc[i,:])
        y = Xva_temp.dot(x)
        y_pred = Y_train.iloc[i,:]
        curr = mean_absolute_error(y_pred, y)
        error[i] = curr
    
    Va_mae = error.mean()


    return [Xp, Xq, Xv, Xva, P_mape, Q_mape, V_mae, Va_mae];

P = pd.read_csv('data/118P.csv', header = None)
Q = pd.read_csv('data/118Q.csv', header = None)
V = pd.read_csv('data/118V.csv', header = None)
Va = pd.read_csv('data/118Va.csv', header = None)
num_train = P.shape[0]
num_bus = P.shape[1]

[Xp, Xq, Xv, Xva, P_mape, Q_mape, V_mae, Va_mae] = BLR(P, Q, V, Va, num_train, num_bus)
print(P_mape)
print(Q_mape)
print(V_mae)
print(Va_mae)