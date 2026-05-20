from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import numpy as np

class EarlyStoppingEstimator:
    def __init__(self, estimator, early_stop_iters, max_iter=1000):
        if not hasattr(estimator, 'partial_fit'):
            raise TypeError('estimator must have partial_fit')
        
        self.estimator = estimator
        self.early_stop_iters = early_stop_iters
        self.max_iter = max_iter
        self.best_est_ = None

    def __getattr__(self, method):
        est = self.best_est_
        if est is None:
            est = self.estimator 

        m = getattr(est, method, None)
        if m is None:
            raise AttributeError(f'{est.__class__.__name__} has not {method}')
        
        return m
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        best_est = None
        best_mse = np.inf
        counter = 0

        for i in range(self.max_iter):
            self.estimator.partial_fit(X_train, y_train)
            y_pred = self.estimator.predict(X_test)
            current_mse = MSE(y_test, y_pred)

            if current_mse < best_mse:
                best_mse = current_mse
                best_est = deepcopy(self.estimator)
                counter = 0
                continue

            counter += 1
            if counter == self.early_stop_iters:
                break
        
        print(i)
        best_est.fit(X, y)
        self.best_est_ = best_est
        return self
    

np.random.seed(42) # to make this code example reproducible
m = 100 # number of instances
X = 2 * np.random.rand(m, 1) # column vector
y = 4 + 3 * X + np.random.randn(m, 1) # column vector

from sklearn.linear_model import SGDRegressor
sdg = SGDRegressor()

est = EarlyStoppingEstimator(sdg, 20)
print(est.alpha)
est.fit(X, y.ravel())
pred = est.predict(X)
print('Best mse:', MSE(y, pred))