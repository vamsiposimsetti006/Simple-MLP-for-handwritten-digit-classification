import numpy as np

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1.0)
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_gradient(y_true, y_pred):
    return y_pred - y_true