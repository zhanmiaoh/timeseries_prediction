import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error #, root_mean_squared_error

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    # return np.sqrt(mean_squared_error(y_true, y_pred))
    return mean_squared_error(y_true, y_pred, squared=False)
    # return root_mean_squared_error(y_true, y_pred)


def mape(y_true, y_pred):
    # mean absolute percentage error
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
