import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from simple_imputer import SimpleImputer

class IterativeImputer:
    def __init__(self, initial_strategy: str='mean', max_iter: int=10):
        self.initial_strategy = initial_strategy
        self.max_iter = max_iter
        self.models = []

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

        data_copy = data.copy()
        imputed_data = SimpleImputer(self.initial_strategy).fit_transform(data_copy)
        self.models = [LinearRegression() for _ in range(len(data_copy[0]))]

        cols_nan_count = np.isnan(data_copy).sum(axis=0)
        col_inxs = np.arange(len(data_copy[0]))
        iter_col_inx_ordered = col_inxs[np.argsort(cols_nan_count)[::-1]]

        cols_nan_mask = []
        for it in range(self.max_iter):
            for i, col in enumerate(iter_col_inx_ordered):
                if it == 0:
                    cols_nan_mask.append(np.isnan(data_copy[:, col]))

                X_col_inxs = iter_col_inx_ordered[iter_col_inx_ordered != col]
                X_train = imputed_data[:, X_col_inxs]
                y_train = imputed_data[:, [col]]

                model = self.models[i]
                model.fit(X_train, y_train)

                nan_mask = cols_nan_mask[i]
                imputed_data[nan_mask, col] = model.predict(X_train).flatten()[nan_mask]

        return self

    def transform(self, data):
        if not self.models:
            raise ValueError('This IterativeImputer instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(self.models) != len(data[0]):
            raise ValueError(f'X has {len(data[0])} features, but KNNImputer is expecting {len(self.models)} features as input.')
        
        data_copy = data.copy()
        imputed_data = SimpleImputer(self.initial_strategy).fit_transform(data_copy)
        
        cols_nan_count = np.isnan(data_copy).sum(axis=0)
        col_inxs = np.arange(len(data_copy[0]))
        iter_col_inx_ordered = col_inxs[np.argsort(cols_nan_count)[::-1]]

        for i, col in enumerate(iter_col_inx_ordered):
            nan_mask = np.isnan(data_copy[:, col])
            if not np.any(nan_mask):
                break

            X_col_inxs = iter_col_inx_ordered[iter_col_inx_ordered != col]
            X_new = imputed_data[:, X_col_inxs]

            model = self.models[i]
            predicted_val = model.predict(X_new).flatten()

            data_copy[nan_mask, col] = predicted_val[nan_mask]

        return data_copy

    def fit_transform(self, data):
        data = self._check_data(data)

        data_copy = data.copy()
        train_data = SimpleImputer(self.initial_strategy).fit_transform(data_copy)

        cols_nan_count = np.isnan(data_copy).sum(axis=0)
        col_inxs = np.arange(len(data_copy[0]))
        iter_col_inx_ordered = col_inxs[np.argsort(cols_nan_count)[::-1]]

        self.models = []
        cols_nan_mask = []
        for it in range(self.max_iter):
            for i, col in enumerate(iter_col_inx_ordered):
                if cols_nan_count[col] == 0:
                    break

                if it == 0:
                    cols_nan_mask.append(np.isnan(data_copy[:, col]))
                    self.models.append(LinearRegression())

                X_col_inxs = iter_col_inx_ordered[iter_col_inx_ordered != col]
                X_train = train_data[:, X_col_inxs]
                y_train = train_data[:, [col]]

                model = self.models[i]
                model.fit(X_train, y_train)

                nan_mask = cols_nan_mask[i]
                train_data[nan_mask, col] = model.predict(X_train).flatten()[nan_mask]

        return train_data


        # imputed_data = SimpleImputer(self.initial_strategy).fit_transform(data_copy)
        # for i, col in enumerate(iter_col_inx_ordered):
        #     if cols_nan_count[col] == 0:
        #         break

        #     X_col_inxs = iter_col_inx_ordered[iter_col_inx_ordered != col]
        #     X_new = imputed_data[:, X_col_inxs]

        #     model = self.models[i]
        #     predicted_val = model.predict(X_new).flatten()

        #     nan_mask = cols_nan_mask[i]
        #     data_copy[nan_mask, col] = predicted_val[nan_mask]

        # return data_copy


                

# data = np.array([
#     [1.0, 10.0, np.nan],
#     [2.0, np.nan, 30.0],
#     [2.0, 15.0, 35.0],
#     [np.nan, 10.0, 40.0]
# ])

# data = np.array([
#     [1.0, 2.0, np.nan],
    # [np.nan, 4.0, 7.0],
    # [3.0, np.nan, 9.0],
    # [4.0, 8.0, np.nan],
    # [np.nan, np.nan, 15.0]
# ])

data = np.array([[1, 2 ,np.nan],
                 [np.nan, 3 ,4],
                 [np.nan, np.nan ,5]])


print(data)
print('_______ Custom _______')
imp = IterativeImputer(initial_strategy='mean', max_iter=10)
imp.fit(data)
data_filled = imp.transform(data)
print(data_filled)
print('_______ Custom fit_transform _______')
print(IterativeImputer(initial_strategy='mean', max_iter=10).fit_transform(data))

print('_______ Original _______')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(initial_strategy='mean', max_iter=10, estimator=LinearRegression())
imp.fit(data)
data_filled = imp.transform(data)
print(data_filled)