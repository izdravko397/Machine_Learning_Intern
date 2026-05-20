from sklearn.base import check_array
from sklearn.preprocessing import add_dummy_feature, StandardScaler, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


class CustomLogisticRegression:
    def __init__(self, n_epochs=100, tol=0.0001, eta0=0.01, power_t=0.25, penalty='l2', threshold=0.5,
            learning_rate='invscaling', n_iter_no_change=5, alpha=0.0001, random_state=None, r=0.5):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.tol = tol
        self.eta0 = eta0
        self.power_t = power_t
        self.n_iter_no_change = n_iter_no_change
        self.threshold = threshold
        self.r = r

        assert penalty in ('l1', 'l2', 'elasticnet', None), 'Invalid penalty'
        self.penalty = penalty

        self.learning_schedule = self.learning_rate_map().get(learning_rate) 
        assert self.learning_schedule is not None, 'Invalid learning_rate func'
        self.learning_rate = learning_rate
        self.random_state = random_state
        


    def learning_rate_map(self): 
        map = {
            'constant': lambda t: self.eta0,
            'optimal': lambda t: 1.0 / (self.alpha * (t + self.eta0)),
            'invscaling': lambda t: self.eta0 / np.pow(t, self.power_t),
        }
        return map
    
    def _funcs(self):
        map = {
            'ridge':        lambda theta: theta * self.alpha,
            'lasso':        lambda theta: 2 * self.alpha * np.sign(theta),
            'log_func':     lambda reg_score: 1 / (1 + np.exp(-reg_score)),
            'softmax_func': lambda reg_score: self._softmax_func(reg_score)
        }
        return map
    
    @staticmethod
    def _softmax_func(reg_score):
        classes_exp = np.exp(reg_score)
        return classes_exp / classes_exp.sum(axis=1, keepdims=True)

    def _get_delta(self, eta, gradients, theta):
        theta_norm = theta.copy()
        theta_norm[:, 0] = 0

        regularization = 0
        match self.penalty:
            case 'l2': regularization = self._funcs()['ridge'](theta_norm)
            case 'l1': regularization = self._funcs()['lasso'](theta_norm)
            case 'elasticnet':
                ridge = self._funcs()['ridge'](theta_norm)
                lasso = self._funcs()['lasso'](theta_norm)
                regularization = self.r * lasso + (1 - self.r) * ridge
            
        return eta * (gradients + regularization)

    def fit(self, X, y):
        X = check_array(X)
        if y.ndim == 2:
            y = y.ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        one_hot_y = OneHotEncoder().fit_transform(y.reshape(-1, 1))

        self.X_scaler_ = StandardScaler()
        X_scaled = self.X_scaler_.fit_transform(X)
        X_b = add_dummy_feature(X_scaled)
        m = X_b.shape[0]
        theta = np.random.randn(len(self.classes_), X_b.shape[1])

        to_break = False
        iter_no_change_counter = 0
        for epoch in range(self.n_epochs):
            for iteration in range(m):
                random_index = np.random.randint(m)
                xi = X_b[random_index : random_index + 1]
                yi = one_hot_y[random_index : random_index + 1]

                p = self._funcs()['softmax_func'](xi @ theta.T)
                gradients = (p - yi).T @ xi
                eta = self.learning_schedule(epoch * m + iteration + 1)
                delta = self._get_delta(eta, gradients, theta)
                theta -= delta

                l2_delta = np.linalg.norm(delta, ord=2, axis=1)
                if np.all(np.abs(l2_delta) <= self.tol):
                    iter_no_change_counter += 1

                    if iter_no_change_counter == self.n_iter_no_change:
                        to_break = True
                        break
                else:
                    iter_no_change_counter = 0

            if to_break:
                break

        self.n_iter_ = epoch + 1
        self.n_features_in_ = X.shape[1]
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.theta_ = theta
        return self

    def predict(self, X):
        p = self.predict_proba(X).argmax(axis=1)
        return self.classes_[p]
    
    def predict_proba(self, X):
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid features length')
        
        X_b = self.X_scaler_.transform(X)
        X_b = add_dummy_feature(X_b) 
        return self._funcs()['softmax_func'](X_b @ self.theta_.T)


np.random.seed(42) # to make this code example reproducible
m = 100 # number of instances
X = 2 * np.random.rand(m, 1) # column vector
y = 4 + 3 * X + np.random.randn(m, 1) # column vector
y_bin = (y >= 7).ravel()


print('---------- Custom ----------')
model = CustomLogisticRegression(random_state=42, penalty=None, n_iter_no_change=6)
model.fit(X, y_bin)
pred = model.predict(X)
print(pred[:5])
print(model.n_iter_)
print('F1:', f1_score(y_bin, pred))
print(confusion_matrix(y_bin, pred))

print('---------- Original ----------')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
model.fit(X, y_bin)
pred = model.predict(X)
print(pred[:5])
print(model.n_iter_)
print('F1:', f1_score(y_bin, pred))
print(confusion_matrix(y_bin, pred))

# plt.scatter(X, y)
# plt.show()

# y = y.ravel()
# y_multi = y.copy()
# y_multi[y >= 9] = 3
# y_multi[(y < 9) & (y > 6)] = 2
# y_multi[y <= 6] = 1

# print('---------- Custom ----------')
# model = CustomLogisticRegression(random_state=42, penalty=None, n_epochs=300, tol=0.0003)
# model.fit(X, y_multi)
# print(model.n_iter_)
# pred = model.predict(X)
# print('accuracy:', accuracy_score(y_multi, pred))
# print('F1:', f1_score(y_multi, pred, average='macro'))
# print(confusion_matrix(y_multi, pred))
# print(model.predict_proba([X[50]]), y_multi[50])

# print('---------- Original ----------')
# model = LogisticRegression(solver='sag', random_state=42)
# model.fit(X, y_multi)
# print(model.n_iter_)
# pred = model.predict(X)
# print('accuracy:', accuracy_score(y_multi, pred))
# print('F1:', f1_score(y_multi, pred, average='macro'))
# print(confusion_matrix(y_multi, pred))
# print(model.predict_proba([X[50]]), y_multi[50])
