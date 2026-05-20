import numpy as np
from sklearn.base import check_array, check_is_fitted
from sklearn.preprocessing import add_dummy_feature
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as RMSE

class NormalEquation:
    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y)

        X_b = add_dummy_feature(X)
        X_T = X_b.T
        theta_best = np.linalg.inv(X_T @ X_b) @ X_T @ y

        self.n_features_in_ = X.shape[1]
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]
        self.theta_ = theta_best
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid features length')
        
        X_b = add_dummy_feature(X)
        return X_b @ self.theta_


class BatchGD:
    def __init__(self, eta=0.1, n_epochs=1000, tol=1e-5, n_iter_no_change=5):
        self.eta = eta
        self.n_epochs = n_epochs
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.gradients_func = lambda X_b, theta, y, m: 2 / m * X_b.T @ (X_b @ theta - y)

    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y)

        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
        scaled_X = self.x_scaler_.fit_transform(X)
        scaled_y = self.y_scaler_.fit_transform(y)
        
        X_b = add_dummy_feature(scaled_X)
        m = X_b.shape[0]
        theta = np.random.randn(X_b.shape[1], 1)

        c = 0
        iter_no_change_counter = 0
        for _ in range(self.n_epochs):
            gradients = self.gradients_func(X_b, theta, scaled_y, m)
            delta = self.eta * gradients
            theta -= delta

            l2_delta = np.linalg.norm(delta, ord=2)
            if np.abs(l2_delta) <= self.tol:
                iter_no_change_counter += 1

                if iter_no_change_counter == self.n_iter_no_change:
                    break
            else:
                iter_no_change_counter = 0

            c += 1

        print(c)
        self.n_features_in_ = X.shape[1]
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.theta_ = theta
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid features length')
        
        scaled_X = self.x_scaler_.transform(X)
        X_b = add_dummy_feature(scaled_X)
        return self.y_scaler_.inverse_transform(X_b @ self.theta_)
    

class SGD(BatchGD):
    def __init__(self, n_epochs=1000, tol=1e-5, eta0=0.01, power_t=0.25, penalty='l2',
                learning_rate='invscaling', n_iter_no_change=5, alpha=0.0001):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.tol = tol
        self.eta0 = eta0
        self.power_t = power_t
        self.n_iter_no_change = n_iter_no_change

        assert penalty in ('l1', 'l2', None), 'Invalid penalty'
        self.penalty = penalty

        self.learning_schedule = self.learning_rate_map().get(learning_rate) 
        assert self.learning_schedule is not None, 'Invalid learning_rate func'
        self.learning_rate = learning_rate
        
        self.gradients_func = lambda X_b, theta, y: 2 * X_b.T @ (X_b @ theta - y)

    def learning_rate_map(self): 
        map = {
            'constant': lambda t: self.eta0,
            'optimal': lambda t: 1.0 / (self.alpha * (t + self.eta0)),
            'invscaling': lambda t: self.eta0 / np.pow(t, self.power_t),
        }

        return map
    
    def _get_delta(self, eta, gradients, theta):
        theta_norm = theta.copy()
        if self.penalty == 'l2':
            theta_norm[1:] *= self.alpha
            return eta * (gradients + theta_norm)
        if self.penalty == 'l1':
            theta_norm[1:] = (2 * self.alpha * np.sign(theta))[1:]
            return eta * gradients + theta_norm

        return eta * gradients

    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y)
        
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
        scaled_X = self.x_scaler_.fit_transform(X)
        scaled_y = self.y_scaler_.fit_transform(y)

        X_b = add_dummy_feature(scaled_X)
        m = X_b.shape[0]
        theta = np.random.randn(X_b.shape[1], 1)
        
        c = 0
        to_break = False
        iter_no_change_counter = 0
        for epoch in range(self.n_epochs):
            for iteration in range(m):
                random_index = np.random.randint(m)
                xi = X_b[random_index : random_index + 1]
                yi = scaled_y[random_index : random_index + 1]

                gradients = self.gradients_func(xi, theta, yi)
                eta = self.learning_schedule(epoch * m + iteration + 1)
                delta = self._get_delta(eta, gradients, theta)
                theta -= delta

                l2_delta = np.linalg.norm(delta, ord=2)
                if np.abs(l2_delta) <= self.tol:
                    iter_no_change_counter += 1

                    if iter_no_change_counter == self.n_iter_no_change:
                        to_break = True
                        break
                else:
                    iter_no_change_counter = 0

            if to_break:
                break

            c += 1

        print(c)
        self.n_features_in_ = X.shape[1]
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.theta_ = theta
        return self

    def partial_fit(self, X, y):
        X = check_array(X)
        y = check_array(y)

        x_scaler_ = self.x_scaler_ if hasattr(self, 'x_scaler_') else StandardScaler()
        y_scaler_ = self.y_scaler_ if hasattr(self, 'y_scaler_') else StandardScaler()

        x_scaler_.partial_fit(X)
        y_scaler_.partial_fit(y)
        
        scaled_X = x_scaler_.transform(X)
        scaled_y = y_scaler_.transform(y)

        X_b = add_dummy_feature(scaled_X)
        m = X_b.shape[0]

        theta = getattr(self, 'theta_', None)
        if theta is None:
            theta = np.random.randn(X_b.shape[1], 1)
        
        for iteration in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index : random_index + 1]
            yi = scaled_y[random_index : random_index + 1]

            gradients = self.gradients_func(xi, theta, yi)
            eta = self.learning_schedule(iteration + 1)
            delta = self._get_delta(eta, gradients, theta)
            theta -= delta

        self.n_features_in_ = X.shape[1]
        self.x_scaler_ = x_scaler_
        self.y_scaler_ = y_scaler_
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.theta_ = theta
        return self

class MiniBatchGD(BatchGD):
    def __init__(self, n_epochs=1000, tol=0.001, eta0=0.01, power_t=0.25, batch_size=32,
                         learning_rate='invscaling', n_iter_no_change=5, alpha=0.0001):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.tol = tol
        self.eta0 = eta0
        self.power_t = power_t
        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change

        self.learning_schedule = self.learning_rate_map().get(learning_rate) 
        assert self.learning_schedule is not None, 'Invalid learning_rate func'
        self.learning_rate = learning_rate
        
        self.gradients_func = lambda X_b, theta, y, m: 2 / m * X_b.T @ (X_b @ theta - y)

    def learning_rate_map(self): 
        map = {
            'constant': lambda t: self.eta0,
            'optimal': lambda t: 1.0 / (self.alpha * (t + self.eta0)),
            'invscaling': lambda t: self.eta0 / np.pow(t, self.power_t),
        }

        return map

    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y)
        
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
        scaled_X = self.x_scaler_.fit_transform(X)
        scaled_y = self.y_scaler_.fit_transform(y)

        X_b = add_dummy_feature(scaled_X)
        m = X_b.shape[0]
        theta = np.random.randn(X_b.shape[1], 1)
        
        c = 0
        to_break = False
        iter_no_change_counter = 0
        shuffle_inxs = np.random.permutation(m)
        for epoch in range(self.n_epochs):
            X_b = X_b[shuffle_inxs]
            scaled_y = scaled_y[shuffle_inxs]

            for iteration in range(0, m, self.batch_size):
                xi = X_b[iteration :iteration + self.batch_size]
                yi = scaled_y[iteration :iteration + self.batch_size]

                gradients = self.gradients_func(xi, theta, yi, len(xi))
                eta = self.learning_schedule(epoch * m + iteration + 1)
                delta = eta * gradients
                theta -= delta

                l2_delta = np.linalg.norm(delta, ord=2)
                if np.abs(l2_delta) <= self.tol:
                    iter_no_change_counter += 1

                    if iter_no_change_counter == self.n_iter_no_change:
                        to_break = True
                        break
                else:
                    iter_no_change_counter = 0

            if to_break:
                break

            np.random.shuffle(shuffle_inxs)
            c += 1

        print(c)
        self.n_features_in_ = X.shape[1]
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.theta_ = theta
        return self 


# np.random.seed(42) # to make this code example reproducible
# m = 100 # number of instances
# X = 2 * np.random.rand(m, 1) # column vector
# y = 4 + 3 * X + np.random.randn(m, 1) # column vector


# print('________ Original ________')
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# pred = model.predict(X)
# print(pred[:2])
# print('RMSE:', RMSE(y, pred))
# print('\n')

# print('________ NormalEquation ________')
# model = NormalEquation()
# model.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# pred = model.predict(X)
# print(pred[:5])
# print('RMSE:', RMSE(y, pred))
# print('\n')

# print('________ BatchGD ________')
# model = BatchGD()
# model.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# pred = model.predict(X)
# print(pred[:5])
# print('RMSE:', RMSE(y, pred))
# print('\n')


# print('________ SGD penalty=l2 ________')
# model = SGD(n_epochs=1000, n_iter_no_change=30, tol=0.001)
# model.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# pred = model.predict(X)
# print(pred[:2])
# print('RMSE:', RMSE(y, pred))
# print('\n')

# print('________ SGD penalty=l1 ________')
# model = SGD(n_epochs=1000, n_iter_no_change=30, tol=0.001, penalty='l1')
# model.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# pred = model.predict(X)
# print(pred[:2])
# print('RMSE:', RMSE(y, pred))
# print('\n')

# print('________ SGD penalty=None ________')
# model = SGD(n_epochs=1000, n_iter_no_change=30, tol=0.001, penalty=None)
# model.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# pred = model.predict(X)
# print(pred[:2])
# print('RMSE:', RMSE(y, pred))
# print('\n')


# print('_____ SGD partial_fit _____')
# minibatch_size = 10
# model = SGD()
# for i in range(0, X.shape[0], minibatch_size):
#     model.partial_fit(X, y)


# print(model.intercept_)
# print(model.coef_)
# pred = model.predict(X)
# print(pred[:5])
# print('RMSE:', RMSE(y, pred))
# print('\n')


# print('________ MiniBatchGD ________')
# model = MiniBatchGD(n_epochs=1000, tol=0.0001, n_iter_no_change=3)
# model.fit(X, y)
# print(model.intercept_)
# print(model.coef_)
# pred = model.predict(X)
# print(pred[:5])
# print('RMSE:', RMSE(y, pred))
# print('\n')
