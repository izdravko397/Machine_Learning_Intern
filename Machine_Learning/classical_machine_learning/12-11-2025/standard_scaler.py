import numpy as np
import pandas as pd

class StandardScaler:
    def __init__(self):
        self.scale_ = None
        self.mean_ = None

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
        if isinstance(data, pd.DataFrame):
            self.feature_names_in_ = data.columns.to_numpy()
        else:
            self.feature_names_in_ = np.array([f'x{i}' for i in range(len(data[0]))])

        data = self._check_data(data)
        self.n_features_in_ = len(data[0])
        self.scale_ = np.std(data, axis=0)
        self.mean_ = np.mean(data, axis=0)

        return self

    def transform(self, data):
        if self.scale_ is None:
            raise ValueError('This StandardScaler instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(data[0]) != len(self.scale_):
            raise ValueError(f'X has {len(data[0])} features, but StandardScaler is expecting {len(self.scale_)} features as input.')
        
        data_copy = data.copy().astype(float)
        data_copy -= self.mean_
        data_copy /= self.scale_

        return data_copy

    def fit_transform(self, data):
        data = self._check_data(data)
        self.scale_ = np.std(data, axis=0)
        self.mean_ = np.mean(data, axis=0)

        data_copy = data.copy().astype(float)
        data_copy -= self.mean_
        data_copy /= self.scale_

        return data_copy
    
    def inverse_transform(self, data):
        if self.scale_ is None:
            raise ValueError('This StandardScaler instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(data[0]) != len(self.scale_):
            raise ValueError(f'X has {len(data[0])} features, but StandardScaler is expecting {len(self.scale_)} features as input.')
        
        data_copy = data.copy().astype(float)
        return data_copy * self.scale_ + self.mean_
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        
        if self.n_features_in_ != len(input_features):
            raise ValueError('Invalid input_features length')
        
        return input_features
    

# X = np.array([[-100, -1000],
#               [1, 10],
#               [2, 20],
#               [3, 30],
#               [100, 1000]])
# print(X)

# print('______ Custom ______')
# scaler = StandardScaler()
# scaler.fit(X)
# scaled_data = scaler.transform(X)
# print(scaled_data)

# print('=== inverse_transform ===')
# print(scaler.inverse_transform(scaled_data))

# print('=== fit_transform ===')
# print(StandardScaler().fit_transform(X))



# print('______ Original ______')
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# scaler.fit(X)
# scaled_data = scaler.transform(X)
# print(scaled_data)

# print('=== inverse_transform ===')
# print(scaler.inverse_transform(scaled_data))

# print('=== fit_transform ===')
# print(StandardScaler().fit_transform(X))
