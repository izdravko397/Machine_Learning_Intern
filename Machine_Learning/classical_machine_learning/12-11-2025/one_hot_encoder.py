import numpy as np
import pandas as pd

class OneHotEncoder:
    def __init__(self, handle_unknown='error'):
        if handle_unknown not in ('error', 'use_encoded_value'):
            raise ValueError("'handle_unknown' must be 'error' or 'use_encoded_value'")
        
        self.handle_unknown = handle_unknown
        self.categories_ = []

    @staticmethod
    def _check_data(data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 2:
            raise ValueError('Expected 2D array')

        return data

    def fit(self, data):
        data = self._check_data(data)
        for col in data.T:
            unique, indices = np.unique(col, return_index=True)
            cats = unique[np.argsort(indices)]
            self.categories_.append(cats)

        return self

    def transform(self, data):
        if not self.categories_:
            raise ValueError('This OneHotEncoder instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(data[0]) != len(self.categories_):
            raise ValueError(f'X has {len(data[0])} features, but OneHotEncoder is expecting {len(self.categories_)} features as input.')
        
        cols_count = sum(len(cat) for cat in self.categories_)
        trans_data = np.zeros((len(data), cols_count))
        last_col_cats = 0
        for col in range(len(data[0])):
            known_cats_count = 0
            for i, cat in enumerate(self.categories_[col]):
                mask = data[:, col] == cat
                trans_data[mask, last_col_cats + i] = 1
                known_cats_count += np.sum(mask)

            last_col_cats += len(self.categories_[col])

            if known_cats_count != len(data) and self.handle_unknown == 'error':
                raise ValueError('Unknown category')

        return trans_data

    def fit_transform(self, data):
        data = self._check_data(data)
        trans_data = np.empty((len(data), 0))
        for col in range(len(data[0])):
            unique, indices = np.unique(data[:, col], return_index=True)
            cats = unique[np.argsort(indices)]
            
            col_trans_data = np.zeros((len(data), len(cats)))
            for i, cat in enumerate(cats):
                mask = data[:, col] == cat
                col_trans_data[mask, i] = 1
            
            trans_data = np.hstack((trans_data, col_trans_data))

        return trans_data
    

def check_is_fitted(estimator):
    attr_vals = [val for a, val in estimator.__dict__.items() if a.endswith('_')]

    if not attr_vals or all(not v or v is None for v in attr_vals):
        raise ValueError(f'This {estimator.__class__.__name__} instance is not fitted yet.')

X_train = [['red'], ['green'], ['blue']]
X_test = [['red'], ['yellow'], ['blue']]

print(OneHotEncoder().fit_transform(X_train))

print("=== 1 ===")
enc1 = OneHotEncoder()  
enc1.fit(X_train)
check_is_fitted(enc1)

print(enc1.transform(X_train))
try:
    print(enc1.transform(X_test))
except ValueError as e:
    print("Error:", e)

print("=== 2 ===")

enc2 = OneHotEncoder(handle_unknown='use_encoded_value')
enc2.fit(X_train)
print(enc2.transform(X_test))