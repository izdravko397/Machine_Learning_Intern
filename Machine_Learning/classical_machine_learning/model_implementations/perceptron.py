from sklearn.base import check_is_fitted, check_array
from sklearn.preprocessing import add_dummy_feature, OneHotEncoder
import numpy as np
 
class Perceptron:
    def __init__(self, alpha=0.0001, max_iter=1000, tol=0.001, eta0=1.0, 
                random_state=None, n_iter_no_change=5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.random_state = random_state
        self.n_iter_no_change = n_iter_no_change

    def fit(self, X, y):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape
        self.classes_ = np.unique(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        X_b = add_dummy_feature(X)
        n_tlu = len(self.classes_) if len(self.classes_) > 2 else 1
        theta = np.random.randn(n_tlu, X_b.shape[1])

        y_t = np.asarray(y.copy()).reshape(-1, 1)
        if np.issubdtype(y_t.dtype, np.bool_):
            y_t = y_t.astype(int)

        if n_tlu != 1:
            self.onehot = OneHotEncoder(sparse_output=False)
            y_t = self.onehot.fit_transform(y_t)

        counter = 0
        for epoch in range(self.max_iter):
            old_theta = theta.copy()

            for i in range(self.n_samples_):
                inx = np.random.randint(self.n_samples_)
                xi = X_b[inx : inx + 1]
                yi = y_t[inx : inx + 1]

                pred = xi @ theta.T
                if n_tlu == 1:
                    pred = pred >= 0
                    theta += self.eta0 * (yi - pred) * xi
                else:
                    pred = pred.argmax(axis=1).reshape(-1, 1)
                    pred_onehot = self.onehot.transform(pred)

                    for k in range(n_tlu):
                        theta[k] += (self.eta0 * (yi[0, k] - pred_onehot[0, k]) * xi).ravel()
                    
            if np.linalg.norm(old_theta - theta, ord=2) < self.tol:
                counter += 1
                if counter == self.n_iter_no_change:
                    break
            else:
                counter = 0

        self.n_iter_ = epoch + 1
        self.theta_ = theta
        self.n_tlu_ = n_tlu
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_b = add_dummy_feature(X)

        raw_res = X_b @ self.theta_.T
        if self.n_tlu_ == 1:
            return (raw_res >= 0).astype(int).ravel()
        return raw_res.argmax(axis=1)
        



class MLPBase:
    def __init__(self, kind, hidden_layer_sizes=(100,), activation='relu', *, 
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.01, 
                 power_t=0.5, max_iter=200, random_state=None, tol=0.01, n_iter_no_change=10):
        self.kind = kind
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change

        self._sigmoid = lambda z: 1 / (1 + np.exp(-z))
        self.last_epoch = 0

    def _learning_rate(self, t, lr):
        match self.learning_rate:
            case 'constant': return self.learning_rate_init
            case 'invscaling': 
                return self.learning_rate_init / np.pow(t, self.power_t)
            case 'adaptive':
                if len(self.loss_curve_) < 3 or self.last_epoch == self.n_iter_:
                    return self.learning_rate_init
                
                res = lr
                diff1 = self.loss_curve_[-2] - self.loss_curve_[-1]
                diff2 = self.loss_curve_[-3] - self.loss_curve_[-2]
                diffs = np.array([diff1, diff2])
                if np.all(diffs < 0) or np.all(diffs < self.tol):
                    res /= 5

                self.last_epoch = self.n_iter_
                return res
        
    @staticmethod
    def _softmax_func(z):
        classes_exp = np.exp(z)
        return classes_exp / classes_exp.sum(axis=1, keepdims=True)

    def _get_active_res(self, raw_res):
        match self.activation:
            case 'relu': 
                res = raw_res.copy()
                res[res < 0] = 0
                return res
            case 'tanh': return 2 * self._sigmoid(2 * raw_res) - 1
            case 'logistic': return self._sigmoid(raw_res)

    def _get_active_prim(self, active_res):
        match self.activation:
            case 'relu': return (active_res > 0).astype(int)
            case 'tanh': return 1 - active_res ** 2
            case 'logistic': return active_res * (1 - active_res)

    def _backpropagation(self, xi, yi, lr):
        # forward
        last_input = xi.copy()
        layers_active_outputs = []

        for theta in self.theta_list[:-1]:
            raw_res = last_input @ theta.T
            active_output = self._get_active_res(raw_res)
            layers_active_outputs.append(active_output)
            last_input = active_output

        output_layer = self.theta_list[-1]
        raw_output = last_input @ output_layer.T

        # loss
        yi_pred = raw_output
        if self.kind == 'classifier':
            if self.n_output_neurons == 1:
                yi_pred = self._sigmoid(raw_output)
                loss = -np.mean(yi * np.log(yi_pred) + (1 - yi) * np.log(1 - yi_pred))
            else:
                yi_pred = self._softmax_func(raw_output)
                loss = -np.mean((yi * np.log(yi_pred)).sum(axis=1))
        else:
            loss = ((yi - yi_pred) ** 2).mean()

        # backward
        last_b = (yi_pred - yi) 
        gradients = [(last_b.T @ layers_active_outputs[-1]) / xi.shape[0]]
        for i in range(len(self.theta_list) -2, -1, -1):
            last_b = (last_b @ self.theta_list[i + 1]) * self._get_active_prim(layers_active_outputs[i])
            prev_layer_pred = layers_active_outputs[i - 1] if i - 1 >= 0 else xi
            gradients.append((last_b.T @ prev_layer_pred) / xi.shape[0])

        # Update
        gradients.reverse()
        for i in range(len(self.theta_list)):
            self.theta_list[i] -= lr * gradients[i]

        return loss

    def fit(self, X, y):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape

        batch_size = self.batch_size
        if batch_size == 'auto':
            batch_size = np.min([200, self.n_samples_])

        n_batches = self.n_samples_ // batch_size
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        y_t = np.asarray(y.copy()).reshape(-1, 1)
        X_b = add_dummy_feature(X)

        if self.kind == 'classifier':
            self.classes_ = np.unique(y)
            self.n_output_neurons = len(self.classes_) if len(self.classes_) > 2 else 1
        else:
            self.n_output_neurons = y_t.shape[1]
        
        if np.issubdtype(y_t.dtype, np.bool_):
            y_t = y_t.astype(int)

        if self.n_output_neurons != 1 and self.kind == 'classifier':
            self.onehot = OneHotEncoder(sparse_output=False)
            y_t = self.onehot.fit_transform(y_t)

        self.theta_list = []
        last_layer_neurons = X_b.shape[1]
        for n_neurons in self.hidden_layer_sizes:
            params = np.random.randn(n_neurons, last_layer_neurons) * np.sqrt(2 / last_layer_neurons)
            self.theta_list.append(params)
            last_layer_neurons = n_neurons
        
        self.theta_list.append(np.random.randn(self.n_output_neurons, last_layer_neurons))

        counter = 0
        last_loss = np.inf
        lr = self.learning_rate_init
        self.loss_curve_ = []
        for epoch in range(1, self.max_iter + 1):
            mean_loss = []
            for i in range(1, n_batches + 1):
                batch_inxs = np.random.choice(self.n_samples_, batch_size, False)
                xi, yi = X_b[batch_inxs], y_t[batch_inxs]
                lr = self._learning_rate(epoch * i, lr)
                loss = self._backpropagation(xi, yi, lr)
                mean_loss.append(loss)
            
            mean_loss = np.mean(mean_loss)
            self.loss_curve_.append(mean_loss)
            self.n_iter_ = epoch

            if np.abs(last_loss - mean_loss) <= self.tol:
                counter += 1
                if self.n_iter_no_change == counter:
                    break
            else:
                counter = 0

            last_loss = mean_loss

        self.loss_ = self.loss_curve_[-1]
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_b = add_dummy_feature(X)

        last_input = X_b.copy()
        for theta in self.theta_list[:-1]:
            raw_res = last_input @ theta.T
            active_output = self._get_active_res(raw_res)
            last_input = active_output

        output_layer = self.theta_list[-1]
        raw_output = last_input @ output_layer.T

        if self.kind == 'regressor':
            return raw_output
        
        if self.n_output_neurons == 1:
            yi_pred = self._sigmoid(raw_output)
            return (yi_pred >= 0.5).astype(int)
        return self._softmax_func(raw_output).argmax(axis=1)
    


class MLPClassifier(MLPBase):
    def __init__(self, hidden_layer_sizes=(100, ), activation='relu', *, batch_size='auto', 
                 learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                 random_state=None, tol=0.0001, n_iter_no_change=10):
        super().__init__('classifier', hidden_layer_sizes, activation, batch_size=batch_size, 
                         learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, 
                         max_iter=max_iter, random_state=random_state, tol=tol, n_iter_no_change=n_iter_no_change)
        

class MLPRegressor(MLPBase):
    def __init__(self, hidden_layer_sizes=(100, ), activation='relu', *, batch_size='auto', 
                 learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                 random_state=None, tol=0.0001, n_iter_no_change=10):
        super().__init__('regressor', hidden_layer_sizes, activation, batch_size=batch_size, 
                         learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, 
                         max_iter=max_iter, random_state=random_state, tol=tol, n_iter_no_change=n_iter_no_change)