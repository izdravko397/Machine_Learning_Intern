import numpy as np
import pandas as pd

class OrdinalEncoder:
    def __init__(self, handle_unknown='error', unknown_value=np.nan):
        if handle_unknown not in ('error', 'use_encoded_value'):
            raise ValueError("'handle_unknown' must be 'error' or 'use_encoded_value'")
        
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
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
            raise ValueError('This OrdinalEncoder instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(data[0]) != len(self.categories_):
            raise ValueError(f'X has {len(data[0])} features, but OrdinalEncoder is expecting {len(self.categories_)} features as input.')
        
        trans_data = np.full_like(data, np.nan, dtype=float)
        for col in range(len(data[0])):
            for i, cat in enumerate(self.categories_[col]):
                mask = data[:, col] == cat
                trans_data[mask, col] = float(i)

            nan_mask = np.isnan(trans_data[:, col])
            if np.any(nan_mask):
                if self.handle_unknown == 'error':
                    raise ValueError('Unknown category')
                
                trans_data[nan_mask, col] = self.unknown_value

        return trans_data

    def fit_transform(self, data):
        data = self._check_data(data)

        trans_data = np.full_like(data, np.nan, dtype=float)
        for col in range(len(data[0])):
            unique, indices = np.unique(data[:, col], return_index=True)
            cats = unique[np.argsort(indices)]

            for i, cat in enumerate(cats):
                mask = data[:, col] == cat
                trans_data[mask, col] = float(i)

        return trans_data


X_train = [['red'], ['green'], ['blue']]
X_test = [['red'], ['yellow'], ['blue']]

print(OrdinalEncoder().fit_transform(X_train))

print("=== 1 ===")
enc1 = OrdinalEncoder()  
enc1.fit(X_train)
print(enc1.transform(X_train))
try:
    print(enc1.transform(X_test))
except ValueError as e:
    print("Error:", e)

print("=== 2 ===")

enc2 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
enc2.fit(X_train)
print(enc2.transform(X_test))
print("=== 3 ===")

enc3 = OrdinalEncoder(handle_unknown='use_encoded_value') 
enc3.fit(X_train)
print(enc3.transform(X_test))