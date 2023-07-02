# it is assumed that y_true and y_pred are numpy arrays, if not, we have to convert them into similar format
# y.to_numpy()

# Can be directly implemented as 'loss'
def mape(y_true, y_pred):
    row = y_true.shape[0]
    col = y_true.shape[1]
    error = np.zeros(row)
    for i in range(row):
        y1 = y_true[i,:]
        y2 = y_pred[i,:]
        error[i] = abs((y1-y2)/y1)
    return error.mean()


# Custom metric function
from sklearn.metrics import make_scorer
mape_metrics = make_scorer(mape, greater_is_better=False)
