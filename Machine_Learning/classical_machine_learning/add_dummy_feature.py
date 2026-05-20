import numpy as np

def add_dummy_feature(X, value=1.0):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if X.ndim != 2:
        raise ValueError('Expected 2D array')

    value_type = np.dtype(type(1.0))
    result = np.empty((X.shape[0], X.shape[1] + 1), dtype=value_type)
    result[:, 0] = value
    result[:, 1:] = X
    return result

# arr = np.arange(25).reshape(5, 5)
# print(arr)

# print('______ Custom ______')
# print(add_dummy_feature(arr))

# from sklearn.preprocessing import add_dummy_feature
# print('______ Original ______')

# print(add_dummy_feature(arr))
# print()