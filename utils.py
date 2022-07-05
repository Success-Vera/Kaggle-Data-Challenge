import numpy as np


def add_column_ones(X):
    n = X.shape[0]
    return np.hstack([X, np.ones((n, 1))])