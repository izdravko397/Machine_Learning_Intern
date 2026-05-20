import numpy as np
import pandas as pd
from collections import Counter



class SimpleImputer:
    def __init__(self, strategy='mean', fill_value=None):
        if strategy not in self._strategies:
            raise ValueError(f'Invalid startegy: {strategy}')
        
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    @property
    def _strategies(self):
        options = {
        'mean':     lambda x: np.nanmean(x, axis=0),
        'median':   lambda x: np.nanmedian(x, axis=0),
        'constant': lambda x: np.repeat([self.fill_value], len(x[0])),
        'most_frequent': self._most_frequent_strategy,
        }

        return options
    
    @staticmethod
    def _most_frequent_strategy(x):
        statistics = np.empty(len(x[0]))
        for i, col in enumerate(x.T):
            counts = Counter(col)
            statistics[i] = counts.most_common(1)[0][0]

        return statistics

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
        func = self._strategies[self.strategy]
        self.statistics_ = func(data)

    def transform(self, data):
        if self.statistics_ is None:
            raise ValueError('This SimpleImputer instance is not fitted yet.')
        
        data = self._check_data(data)
        if len(self.statistics_) != len(data[0]):
            raise ValueError(f'X has {len(data[0])} features, but SimpleImputer is expecting {len(self.statistics_)} features as input.')
        
        data_copy = data.copy()
        data_copy = np.where(np.isnan(data_copy), self.statistics_, data_copy)

        return data_copy
    
    def fit_transform(self, data):
        data = self._check_data(data)
        func = self._strategies[self.strategy]
        filling_values = func(data)

        data_copy = data.copy()
        data_copy = np.where(np.isnan(data_copy), filling_values, data_copy)

        return data_copy

# train = np.array([
#     [1, 2],
#     [3, np.nan],
#     [5, 6]
# ])

# imp = SimpleImputer(strategy='mean')
# # print(imp.statistics_)

# imp.fit(train)
# print(imp.statistics_)

# test = np.array([
#     [7, np.nan],
#     [np.nan, 10]
# ])

# X_filled = imp.transform(test)
# print(X_filled)
# print(test)




# train = np.array([
#     [1, 2],
#     [3, np.nan],
#     [5, 6]
# ])

# imp = SimpleImputer(strategy="constant", fill_value=888)
# # print(imp.statistics_)

# imp.fit(train)
# print(imp.statistics_)

# test = np.array([
#     [7, np.nan],
#     [np.nan, 10]
# ])

# X_filled = imp.transform(test)
# print(X_filled)
# print(test)



# data = np.array([
#     [1, 10],
#     [2, np.nan],
#     [2, 15],
#     [np.nan, 10]
# ])
# # imp = SimpleImputer(strategy='most_frequent')
# # imp.fit(data)
# # print("Statistics:", imp.statistics_)
# # X_filled = imp.transform(data)
# # print(X_filled)




# print(SimpleImputer().fit_transform(data))