import numpy as np
import pandas as pd
from standard_scaler import StandardScaler
from min_max_scaler import MinMaxScaler

class TransformedTargetRegressor:
    def __init__(self, regressor, transformer):
        self.regressor = regressor
        self.transformer = transformer

        self.regressor_ = None
        self.transformer_ = None

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

    def fit(self, X, y):
        X = self._check_data(X)
        y = self._check_data(y)

        scaled_y = self.transformer.fit_transform(y)
        self.transformer_ = self.transformer

        self.regressor.fit(X, scaled_y)
        self.regressor_ = self.regressor
        
        return self
    
    def predict(self, X):
        if self.regressor_ is None:
            raise ValueError('This TransformedTargetRegressor instance is not fitted yet.')
        
        X = self._check_data(X)
        scaled_predict = self.regressor_.predict(X)
        predict = self.transformer_.inverse_transform(scaled_predict)
        
        return predict
    
    def score(self, X, y):
        predicts = self.predict(X).flatten()
        y = self._check_data(y).flatten()

        y_mean = np.mean(y)
        predict_sq_sum = np.sum(np.square(y - predicts))
        var_sq_sum = np.sum(np.square(y - y_mean))

        return 1 - predict_sq_sum / var_sq_sum

    

from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])

reg = TransformedTargetRegressor(
    regressor=LinearRegression(),
    transformer=StandardScaler()
)

reg.fit(X, y)
y_pred = reg.predict(np.array([[6], [7]]))
print("Custom Predictions:\n", y_pred)
print(reg.score(X, y))


from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

reg = TransformedTargetRegressor(
    regressor=LinearRegression(),
    transformer=StandardScaler()
)

reg.fit(X, y)
y_pred = reg.predict(np.array([[6], [7]]))
print("Original Predictions:\n", y_pred)
print(reg.score(X, y))