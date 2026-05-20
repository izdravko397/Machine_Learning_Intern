import numpy as np
import pandas as pd

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        if not (isinstance(feature_range, (list, tuple)) and len(feature_range) == 2 
                                                and feature_range[0] < feature_range[1]):
            raise ValueError("'feature_range' must be list/tuple with 2 elements")
        
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.min_ = None

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
        self.data_max_ = data.max(axis=0)
        self.data_min_ = data.min(axis=0)

        a = self.feature_range[0]
        b = self.feature_range[1]
        self.scale_ = (b - a) / (self.data_max_ - self.data_min_)
        self.min_ = a - self.data_min_ * self.scale_ 

        return self

    def transform(self, data):
        if self.data_max_ is None:
            raise ValueError('This MinMaxScaler instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(data[0]) != len(self.data_max_):
            raise ValueError(f'X has {len(data[0])} features, but MinMaxScaler is expecting {len(self.data_max_)} features as input.')
        
        data_copy = data.copy().astype(float)
        data_copy *= self.scale_
        data_copy += self.min_

        return data_copy

    def fit_transform(self, data):
        data = self._check_data(data)
        self.data_max_ = data.max(axis=0)
        self.data_min_ = data.min(axis=0)

        a = self.feature_range[0]
        b = self.feature_range[1]
        self.scale_ = (b - a) / (self.data_max_ - self.data_min_)
        self.min_ = a - self.data_min_ * self.scale_ 

        data_copy = data.copy().astype(float)
        data_copy *= self.scale_
        data_copy += self.min_

        return data_copy
    
    def inverse_transform(self, data):
        if self.data_max_ is None:
            raise ValueError('This MinMaxScaler instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(data[0]) != len(self.data_max_):
            raise ValueError(f'X has {len(data[0])} features, but MinMaxScaler is expecting {len(self.data_max_)} features as input.')

        data = self._check_data(data)
        data_copy = data.copy().astype(float)
        return (data_copy - self.min_) / self.scale_
    


# X = np.array([[1, 10],
#               [2, 20],
#               [3, 30]])
# print(X)

# print('______ Custom ______')
# print('=== (0, 1) ===')
# scaler = MinMaxScaler()
# scaler.fit(X)
# scaled_data = scaler.transform(X)
# print(scaled_data)
# print(scaler.inverse_transform(scaled_data))

# print('=== (-1, 1) ===')
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler.fit(X)
# scaled_data = scaler.transform(X)
# print(scaled_data)
# print(scaler.inverse_transform(scaled_data))

# print('=== fit_transform (-2, 2) ===')
# print(MinMaxScaler(feature_range=(-2, 2)).fit_transform(X))


# print('______ Original ______')
# from sklearn.preprocessing import MinMaxScaler

# print('=== (0, 1) ===')
# scaler = MinMaxScaler()
# scaler.fit(X)
# print(scaler.transform(X))

# print('=== (-1, 1) ===')
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler.fit(X)
# print(scaler.transform(X))

# print('=== fit_transform (-2, 2) ===')
# print(MinMaxScaler(feature_range=(-2, 2)).fit_transform(X))
