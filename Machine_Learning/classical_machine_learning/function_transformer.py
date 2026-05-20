import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class FunctionTransformer:
    def __init__(self, func=None, inverse_func=None, kw_args={}):
        if func is not None and not callable(func):
            raise TypeError('Func must be callable')
        
        self.func = func

        if inverse_func is not None and not callable(inverse_func):
            raise TypeError('Inverse_func must be callable')
        
        self.inverse_func = inverse_func

        if not isinstance(kw_args, dict):
            raise TypeError('kw_args must be dict')
        
        self.kw_args = kw_args

    @staticmethod
    def _check_data(data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        if isinstance(data, pd.Series):
            data = data.to_numpy().reshape(-1, 1)

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.ndim != 2:
            raise ValueError('Expected 2D array')

        return data

    def fit(self, X):
        return self
    
    def transform(self, X):
        X = self._check_data(X)
        return self.func(X, **self.kw_args) if self.func is not None else X
    
    def inverse_transform(self, X):
        X = self._check_data(X)
        return self.inverse_func(X, **self.kw_args) if self.inverse_func is None else X
    
    def fit_transform(self, X):
        return self.transform(X)


# ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
# print(ratio_transformer.transform(np.array([[1., 2.], [3., 4.]])))


class TransformerMixin:
    def fit_transform(self, X):
        if hasattr(self, 'fit') and hasattr(self, 'transform'):
            return self.fit(X).transform(X)
        
        raise TypeError("TransformerMixin requires fit() and transform()")
    

def check_is_fitted(estimator):
    attr_vals = [val for a, val in estimator.__dict__.items() if a.endswith('_')]

    if not attr_vals or all(not v or v is None for v in attr_vals):
        raise ValueError(f'This {estimator.__class__.__name__} instance is not fitted yet.')
    

def check_array(array):
    if isinstance(array, pd.DataFrame):
        array = array.to_numpy()

    if isinstance(array, pd.Series):
        array = array.to_numpy().reshape(-1, 1)

    if isinstance(array, csr_matrix):
        array = array.toarray()

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if array.ndim != 2:
        raise ValueError('Expected 2D array')
    
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError('Array must be numeric')
    
    if np.any(np.isnan(array) | np.isinf(array)):
        raise ValueError('Array contains NaN or inf')

    return array


# X = [[1, 2], [3, 4]]
# print(check_array(X))

# df = pd.DataFrame([[5, 6], [7, 8]])
# print(check_array(df))

# s = pd.Series([1, 2, 3])
# print(check_array(s))

# sparse_X = csr_matrix([[0, 1], [1, 0]])
# print(check_array(sparse_X))