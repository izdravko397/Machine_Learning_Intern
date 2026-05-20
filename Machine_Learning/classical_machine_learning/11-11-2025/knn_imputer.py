import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class KNNImputer:
    def __init__(self, k=1, metric='euclidean'):
        self.k = k
        self.metric = metric
        self._models =[]

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
        train_non_nan_mask = ~(np.isnan(data).sum(axis=1).astype(bool))

        for col in data.T:
            model = KNeighborsRegressor(self.k, metric=self.metric)

            X = np.where(train_non_nan_mask)[0].reshape(-1, 1)
            y = col[train_non_nan_mask].reshape(-1, 1)
            model.fit(X, y)
            self._models.append(model)

    def transform(self, data):
        if not self._models:
            raise ValueError('This KNNImputer instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(self._models) != len(data[0]):
            raise ValueError(f'X has {len(data[0])} features, but KNNImputer is expecting {len(self._models)} features as input.')
        
        data_copy = data.copy()
        for col, model in enumerate(self._models):
            nan_mask = np.isnan(data_copy[:, col])
            X = np.where(nan_mask)[0].reshape(-1, 1)
            predicted_vals = model.predict(X).flatten()

            data_copy[nan_mask, col] = predicted_vals

        return data_copy
    

    def fit_transform(self, data):
        data = self._check_data(data)
        # train_non_nan_mask = ~(np.isnan(data).sum(axis=1).astype(bool))
        train_non_nan_mask = np.isnan(data).sum(axis=1) == 0

        data_copy = data.copy()
        for col in range(len(data[0])):
            col_data = data_copy[:, col]
            nan_mask = np.isnan(col_data)

            if not np.any(nan_mask):
                continue

            model = KNeighborsRegressor(self.k, metric=self.metric)
            X_train = np.where(train_non_nan_mask)[0].reshape(-1, 1)
            y_train = col_data[train_non_nan_mask].reshape(-1, 1)
            model.fit(X_train, y_train)

            X_new = np.where(nan_mask)[0].reshape(-1, 1)
            predicted_vals = model.predict(X_new).flatten()

            data_copy[nan_mask, col] = predicted_vals

        return data_copy
            
                

data = np.array([
    [1, 10],
    [2, np.nan],
    [2, 15],
    [np.nan, 10]
])
print(data)

imp = KNNImputer(2)
imp.fit(data)
X_filled = imp.transform(data)
print(X_filled)

print('___________ fit_transform ___________')
print(KNNImputer(2).fit_transform(data))

print('________ Original ________')
from sklearn.impute import KNNImputer
imp = KNNImputer(n_neighbors=2, metric='nan_euclidean')
imp.fit(data)
X_filled = imp.transform(data)
print(X_filled)