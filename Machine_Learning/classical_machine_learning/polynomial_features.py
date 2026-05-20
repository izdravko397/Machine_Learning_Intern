from sklearn.base import check_array, check_is_fitted, TransformerMixin, BaseEstimator
from itertools import combinations_with_replacement
from add_dummy_feature import add_dummy_feature
import numpy as np

class PolynomialFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        features_inxs = np.arange(self.n_features_in_)

        combinations = []
        for i in range(1, self.degree + 1):
            for c in combinations_with_replacement(features_inxs, i):
                combinations.append(c)

        self.combinations_ = combinations
        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid features length')
        
        result = np.empty((X.shape[0], len(self.combinations_)), dtype=float)
        for i, comb in enumerate(self.combinations_):
            comb_res = X[:, comb[0]].copy()
            for inx in comb[1:]:
                comb_res *= X[:, inx]
            
            result[:, i] = comb_res

        if self.include_bias:
            result = add_dummy_feature(result)

        return result
    
    def fit_transform(self, X, y = None, **fit_params):
        return self.fit(X, y).transform(X)

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 2) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

print('-------- Custom -----------')
poly_features = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])

print('-------- Original -----------')
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])