import numpy as np
import pandas as pd
from sklearn.base import clone, check_is_fitted
from function_transformer import TransformerMixin

class SelectFromModel(TransformerMixin):
    threshold_funcs = {
        'mean': np.mean,
        'median': np.median,
    }

    def __init__(self, estimator, *, threshold='mean'):
        if not hasattr(estimator, 'fit'):
            raise TypeError(f'Invalid estimator: {type(estimator).__name__}')
        
        self.estimator = estimator

        if threshold not in self.threshold_funcs and not isinstance(threshold, (int, float)):
            raise TypeError(f'Invalid threshold: {threshold}, with type: {type(threshold).__name__}')
        
        self.threshold = threshold

    def fit(self, X, y):
        fresh_est = clone(self.estimator)
        fresh_est.fit(X, y)

        if hasattr(fresh_est, 'coef_'):
            feature_imp = getattr(fresh_est, 'coef_')
        elif hasattr(fresh_est, 'feature_importances_'):
            feature_imp = getattr(fresh_est, 'feature_importances_')
        else:
            raise AttributeError(f"'{fresh_est.__class__.__name__}' object has no attribute 'feature_importances_'/'coef_'")

        threshold = self.threshold
        if isinstance(threshold, str):
            func = self.threshold_funcs[threshold]
            threshold = func(feature_imp)

        self.feature_mask_ = feature_imp >= threshold
        self.estimator_ = fresh_est
        self.threshold_ = threshold
        self.importances_ = feature_imp

        return self

    def transform(self, X):
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            features_len = len(X.columns)
        else:
            features_len = X.shape[1]

        if features_len != len(self.feature_mask_):
            raise ValueError(f'Invalid len of features')
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_mask_]
        return X[:, self.feature_mask_]
        

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
X = [[ 0.87, -1.34,  0.31 ],
     [-2.79, -0.02, -0.85 ],
     [-1.34, -0.48, -2.55 ],
     [ 1.92,  1.48,  0.65 ]]
y = [0, 1, 0, 1]
selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
print(selector.estimator_.coef_)
print(selector.threshold_)
print(selector.get_support())
print(selector.transform(X))