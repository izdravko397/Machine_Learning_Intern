from collections import defaultdict
from io import StringIO
import numpy as np
from abc import ABC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import check_array, check_is_fitted
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, root_mean_squared_error
from tqdm import tqdm
# from scipy.special import erf
from math import erf
import joblib
import numba

@numba.jit
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class History:
    def __init__(self, history):
        self.history = history

class Model:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self._weights = []
        self._stop_training = False

    def _check_is_builded(self, layers):
        for lr in layers:
            while True:
                if not lr.is_build:
                    raise ValueError(f'{lr.name} ({lr.__class__.__name__}) is not built')
                
                if lr.next_layer is None:
                    break

                lr = lr.next_layer
                if isinstance(lr, list):
                    return self._active_res_tonone(lr)

    def _get_learning_rate(self):
        match self.optimizer_:
            case 'sgd': return SGD()
            case _:
                if not isinstance(self.optimizer_, OptimizersBase):
                    raise ValueError('Invalid optimizer')
                
                return self.optimizer_
    
    @staticmethod
    @numba.jit
    def _get_loss(loss, y_true, y_pred):
        if loss in ("sparse_categorical_crossentropy", "categorical_crossentropy"):
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        if loss == "binary_crossentropy":
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        if loss == "mse":
            return ((y_true - y_pred) ** 2).mean()
        else:
            raise ValueError(f'Invalid loss {loss}')
    
    @staticmethod
    @numba.njit
    def _get_scores(metrics, out_activation, y_true, y_pred):
        n = y_true.shape[-1]
        y_cls = np.zeros(n, dtype=np.int64)

        if 'accuracy' in metrics:        
            for i in range(n):
                if out_activation == 'softmax':
                    y_cls[i] = np.argmax(y_pred[i])
                else:
                    y_cls[i] = 1 if y_pred[i] >= 0.5 else 0

        scores = np.empty(len(metrics), dtype=np.float32)
        for i in range(len(metrics)):
            metric = metrics[i]
            if metric == 'accuracy':
                correct = 0
                for j in range(n):
                    if y_true[j] == y_cls[j]:
                        correct += 1
                score = correct / n
            elif metric == 'rmse':
                score = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            scores[i] = score

        return scores

    def compile(self, optimizer, loss, metrics):
        self.optimizer_ = optimizer
        self.loss_ = loss
        self.metrics_ = metrics

    def _backward(self, layer, delta):
        delta = layer.backward(delta)
        prev = layer.prev_layer

        if prev is None:
            return
        
        if hasattr(prev, '__iter__'):
            for pl, dlt in zip(prev, delta):
                self._backward(pl, dlt)
        else:
            self._backward(prev, delta)

    def _forward(self, layers, X=None, training=True):
        X = [None] * len(layers) if X is None else X
        for lr, x in zip(layers, X):
            while True:
                output = lr.forward(x, training)

                if lr.next_layer is None or output:
                    break

                lr = lr.next_layer
                if isinstance(lr, list):
                    return self._forward(lr)
    
    def _update(self, layers, lr_rate, itr):
        for lr in layers:
            while True:
                output = lr.step(lr_rate, itr)

                if lr.next_layer is None or output:
                    break

                lr = lr.next_layer
                if isinstance(lr, list):
                    return self._update(lr, lr_rate)
    
    def _active_res_tonone(self, layers):
        for lr in layers:
            while True:
                lr.active_res = None
                if lr.next_layer is None:
                    break

                lr = lr.next_layer
                if isinstance(lr, list):
                    return self._active_res_tonone(lr)

    def _backpropagation(self, xi, yi, yi_true, lr, itr):
        # forward
        self._forward(self.inputs, xi, training=True)

        # loss & scores
        loss = self._get_loss(self.loss_, yi, self.outputs.active_res)
        scores = self._get_scores(self.metrics_, self.outputs.activation, yi_true, self.outputs.active_res)

        # backward
        delta = (self.outputs.active_res - yi)
        self._backward(self.outputs, delta)

        # Update
        if isinstance(lr, OptimizersBase):
            lr.learning_rate = lr.learning_schedule.get_learning_rate(itr)

        self._update(self.inputs, lr, itr)

        return loss, scores
    
    def _execute_callbacks(self, callbacks, epoch=None, logs=None, mode='set_model_instance'):
        new_lr = None
        for callback in callbacks:
            if isinstance(callback, LearningRateScheduler) and mode == 'on_epoch_begin':
                new_lr =  callback.on_epoch_begin(epoch, logs)
                continue

            match mode:
                case 'set_model_instance': callback.model = self
                case 'on_train_begin':     callback.on_train_begin(logs)
                case 'on_train_end':       callback.on_train_end(logs)
                case 'on_epoch_begin':     callback.on_epoch_begin(epoch, logs)
                case 'on_epoch_end':       callback.on_epoch_end(epoch, logs)
                case 'on_batch_begin':     callback.on_batch_begin(epoch, logs)
                case 'on_batch_end':       callback.on_batch_end(epoch, logs)

        return new_lr

    def fit(self, X, y, epochs=1, batch_size=32, validation_data=None, validation_ratio=0.1, callbacks=[]):
        self._check_is_builded(self.inputs)

        if not isinstance(callbacks, list):
            raise TypeError(f'callbacks must be list not: {type(callbacks)}')
        
        self._execute_callbacks(callbacks)

        if not isinstance(X, tuple):
            X = (X,)

        self.n_samples, *self.n_features = X[0].shape
        self.n_batches = int(np.ceil(self.n_samples / batch_size))

        if validation_data is None:
            val_data_len = self.n_samples * validation_ratio
            val_data_inxs = np.random.choice(self.n_samples, val_data_len, False) 
            validation_data = (tuple([x[val_data_inxs] for x in X]), y[val_data_inxs])
            not_val_inx = ~np.isin(np.arange(self.n_samples), val_data_inxs)
            X = X[not_val_inx]
            y = y[not_val_inx]
        else:
            if not isinstance(validation_data[0], tuple):
                validation_data = ((validation_data[0],), *validation_data[1:])

        y_t = np.asarray(y).reshape(-1, 1)
        y_val_t = validation_data[1].reshape(-1, 1)
        if np.issubdtype(y_t.dtype, np.bool_):
            y_t = y_t.astype(int)

        if self.loss_ == "sparse_categorical_crossentropy":
            onehot = OneHotEncoder(sparse_output=False)
            y_t = onehot.fit_transform(y_t)
            y_val_t = onehot.transform(y_val_t)

        loss_curve = []
        loss_val_curve = []
        score_curve = []
        score_val_curve = []
        self._execute_callbacks(callbacks, mode='on_train_begin')
        val_metrics_names = ['val_' + m for m in self.metrics_]
        for epoch in range(epochs):
            new_lr = self._execute_callbacks(callbacks, epoch+1, mode='on_epoch_begin')
            print(f'Epoch {epoch+1}/{epochs}') 
            mean_loss = []
            mean_scores = []

            for i in tqdm(range(1, self.n_batches + 1)):
                batch_inxs = np.random.choice(self.n_samples, batch_size, False)
                xi, yi, yi_true = tuple([x[batch_inxs] for x in X]), y_t[batch_inxs], y[batch_inxs]
                lr = self._get_learning_rate() if new_lr is None else new_lr

                self._execute_callbacks(callbacks, i, mode='on_batch_begin') 
                
                itr = epoch * self.n_batches + i
                loss, metrics = self._backpropagation(xi, yi, yi_true, lr, itr)
                mean_loss.append(loss)
                mean_scores.append(metrics)

                if callbacks:
                    logs = {'loss': loss, 'learning_rate': lr}
                    logs.update(dict(zip(self.metrics_, metrics)))
                    self._execute_callbacks(callbacks, i, logs, mode='on_batch_end')

            mean_loss = np.mean(mean_loss)
            mean_scores = np.array(mean_scores).mean(axis=1)

            self._forward(self.inputs, validation_data[0])
            val_loss = self._get_loss(self.loss_, y_val_t, self.outputs.active_res)
            val_scores = self._get_scores(self.metrics_, self.outputs.activation, validation_data[1], self.outputs.active_res)
            self._active_res_tonone(self.inputs)

            loss_curve.append(mean_loss)
            loss_val_curve.append(val_loss)
            score_curve.append(mean_scores)
            score_val_curve.append(val_scores)

            
            train_stats = f'    - loss: {mean_loss}'
            for i, m in enumerate(self.metrics_):
                train_stats += f' - {m}: {mean_scores[i]}'
            print(train_stats)

            val_stats = f'    - val_loss: {val_loss}'
            for i, m in enumerate(self.metrics_):
                val_stats += f' - {'val_' + m}: {val_scores[i]}'
            print(val_stats)

            print('Learning Rate:', lr.learning_rate)
            if callbacks:
                logs = {'loss': mean_loss, 'val_loss': val_loss}
                logs.update(dict(zip(self.metrics_, mean_scores)))
                logs.update(dict(zip(val_metrics_names, val_scores)))
                logs['learning_rate'] = lr
                self._execute_callbacks(callbacks, epoch+1, logs, mode='on_epoch_end')
                
                if self._stop_training:
                    break

        score_curve = np.array(score_curve)
        score_val_curve = np.array(score_val_curve)
        curves = {'loss': loss_curve, 'val_loss': loss_val_curve}

        for i, m in enumerate(self.metrics_):
            curves[m] = score_curve[:, i]
            curves['val_' + m] = score_val_curve[:, i]

        self.curves_ = curves
        self._execute_callbacks(callbacks, mode='on_train_end')

        return History(curves)
    
    def stop_training(self):
        self._stop_training = True

    def predict(self, X):
        check_is_fitted(self)

        if not isinstance(X, tuple):
            X = (X,)

        self._forward(self.inputs, X, training=False)
        res = self.outputs.active_res
        self._active_res_tonone(self.inputs)
        return res
    
    def save(self, filename):
        joblib.dump(self, filename)

    def _reset_visited_param(self, layers=None):
        if layers is None:
            layers = self.inputs

        for lr in layers:
            while True:
                if hasattr(lr, '_visited'):
                    lr._visited = False

                lr = lr.next_layer
                if lr is None:
                    break

                if isinstance(lr, list):
                    return self._reset_visited_param(lr)

    def get_weights(self, layers=None):
        if layers is None:
            layers = self.inputs

        for lr in layers:
            while True:
                if hasattr(lr, '_visited') and not lr._visited:
                    self._weights.append(lr.get_weights())
                    lr._visited = True

                lr = lr.next_layer
                if lr is None:
                    break

                if isinstance(lr, list):
                    return self.get_weights(lr)
                
        weights = self._weights
        self._weights = []
        self._reset_visited_param()
        return weights
    
    def set_weights(self, weights, layers=None):
        if layers is None:
            layers = self.inputs

        for lr in layers:
            while True:
                if hasattr(lr, '_visited') and not lr._visited:
                    lr.set_weights(*weights.pop(0))
                    lr._visited = True

                lr = lr.next_layer
                if lr is None:
                    break

                if isinstance(lr, list):
                    return self.get_weights(lr)

        self._reset_visited_param()



class Input:
    def __init__(self, shape, name=None):
        self.name = self.__class__.__name__.lower() if name is None else name
        self.is_build = True
        shape = tuple(shape)
        self.shape = shape
        self.output = shape
        self.trainable = False
        self.prev_layer = None

    def forward(self, X, *_):
        X = np.asarray(X)

        if X.shape[1:] != self.shape:
            raise ValueError(f'Invaluid input shape in layer: {self.name}')
        
        self.active_res = X

    def backward(self, grad=None):
        self.grad = grad

    def step(self, *_):
        pass


class BatchNormalization:
    def __init__(self, momentum=0.99, epsilon=0.001, beta_initializer='zeros', gamma_initializer='ones', 
                 moving_mean_initializer='zeros', moving_variance_initializer='ones', name=None):
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.next_layer = None
        self.prev_layer = None
        self.active_res = None
        self.is_build = False
        self.trainable = True
        self.name = self.__class__.__name__.lower() if name is None else name 

    def __call__(self, tuner, *args, **kwds):
        self.prev_layer = tuner
        self.input  = tuner.output
        self.output = tuner.output
        tuner.next_layer = self
        self.is_build = True

        self.mean_ = np.zeros(self.input) if self.moving_mean_initializer == 'zeros' \
                                else np.full(self.input, self.moving_mean_initializer)
        
        self.beta_ = np.zeros(self.input) if self.beta_initializer == 'zeros' \
                                else np.full(self.input, self.beta_initializer)
        
        self.variance_ = np.ones(self.input) if self.moving_variance_initializer == 'ones' \
                                else np.full(self.input, self.moving_variance_initializer)

        self.gama_ = np.ones(self.input) if self.gamma_initializer == 'ones' \
                                else np.full(self.input, self.gamma_initializer)
        
        return self

    @numba.jit
    def forward(self, X=None):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')
        
        X = self.prev_layer.active_res
        self.X = X

        curr_mean = np.mean(X, axis=0)
        curr_var  = np.var(X , axis=0)

        self.variance_ = self.variance_ * self.momentum + curr_var * (1 - self.momentum)
        self.mean_     = self.mean_ * self.momentum + curr_mean * (1 - self.momentum)

        self.X_hat = (X - self.mean_) / np.sqrt(self.variance_ + self.epsilon)
        self.active_res = self.X_hat * self.gama_ + self.beta_

    @numba.jit
    def backward(self, grad):
        self.gamma_grad = np.mean(grad * self.X_hat, axis=0)
        self.beta_grad  = np.mean(grad, axis=0)
        self.active_res = None

        dx = grad * self.gama_
        b_size = self.X.shape[0]
        return (1. / b_size) * (1. / np.sqrt(self.variance_ + self.epsilon)) * (b_size * dx - np.mean(dx, axis=0)
                                - self.X_hat * np.mean(dx * self.X_hat, axis=0))
    
    @numba.jit
    def step(self, learning_rate, *_):
        if not self.trainable:
            return 
        
        lr = learning_rate
        c_value, c_norm = None, None
        if isinstance(lr, SGD):
            lr = learning_rate.learning_rate
            c_value, c_norm = learning_rate.clipvalue, learning_rate.clipnorm

        if c_value is not None:
            self.gamma_grad = np.clip(self.gamma_grad, -c_value, c_value)
            self.beta_grad = np.clip(self.beta_grad, -c_value, c_value)

        elif c_norm is not None:
            total_norm = np.sqrt(np.sum(self.gamma_grad ** 2) +
                                 np.sum(self.beta_grad ** 2))

            scale = c_norm / max(total_norm, c_norm)

            self.gamma_grad *= scale
            self.beta_grad  *= scale

        self.gama_ -= lr * self.gamma_grad
        self.beta_ -= lr * self.beta_grad

class Dropout:
    def __init__(self, rate, name=None):
        self.rate = rate
        self.input = None
        self.output = None
        self.prev_layer = None
        self.next_layer = None
        self.is_build = False
        self.active_res = None
        self.trainable = False
        self.mask = None
        self.name = self.__class__.__name__.lower() if name is None else name

    def __call__(self, tensor, *args, **kwds):
        self.input  = tensor.output
        self.output = tensor.output
        self.prev_layer = tensor
        tensor.next_layer = self
        self.is_build = True

        return self
    
    def forward(self, X=None, training=True):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')
        
        X = self.prev_layer.active_res
        if not training:
            self.active_res = X
            return
        
        self.mask = (np.random.rand(*X.shape) > self.rate) / (1 - self.rate)
        self.active_res = X * self.mask

    def backward(self, grad):
        return grad * self.mask
    
    def step(self, *_):
        pass

class AlphaDropout(Dropout):
    def __init__(self, rate, name=None):
        super().__init__(rate, name)
        alpha = 1.67
        scale = 1.05
        self.alpha_p = -scale * alpha

    def forward(self, X=None, training=True):
        X = self.prev_layer.active_res

        if not training:
            self.active_res = X
            return 

        keep_prob = 1 - self.rate
        self.mask = (np.random.rand(*X.shape) < keep_prob)
        X_const = self.mask * X + (~self.mask) * self.alpha_p
        self.scale = 1.0 / np.sqrt(keep_prob * (1 + self.rate * self.alpha_p**2))
        bias = -self.scale * self.rate * self.alpha_p

        self.active_res = self.scale * X_const + bias
    
    def backward(self, grad):
        grad *= self.scale * self.mask
        return grad

class Dense:
    def __init__(self, units, activation=None, kernel_initializer='glorot_uniform', 
                 kernel_regularizer=None, kernel_constraint=None, bias_constraint=None, name=None):
        self.units = units
        self.activation = activation
        self.output = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.next_layer = None
        self.is_build = False
        self.active_res = None
        self._visited = False
        self.trainable = True
        self._opt_w = None
        self._opt_b = None
        self.name = self.__class__.__name__.lower() if name is None else name

        if self.kernel_regularizer is not None:
            if not isinstance(self.kernel_regularizer, RegularizationBase):
                raise TypeError('Invalid regularizer')
            
            self.kernel_regularizer.layer = self

    def _get_fan_avg(self):
        return (self.input + self.output) / 2

    def __call__(self, tensor, *args, **kwds):
        if isinstance(tensor, ActivationFuncBase):
            self.activation = tensor
            tensor = tensor.prev_layer

        self.input = tensor.output[0] if hasattr(tensor.output, '__iter__') else tensor.output
        self.prev_layer = tensor
        tensor.next_layer = self
        self.is_build = True
        self.bias = np.full((1, self.units),  0.1)
        self._weights_init()

        if tensor.name.startswith('dense'):
            num = 1
            if tensor.name[-1].isdigit():
                num += int(tensor.name[-1])

            self.name += f'_{num}'

        return self

    def _weights_init(self):
        match self.kernel_initializer:
            case 'glorot_uniform':
                r = np.sqrt(3 / self._get_fan_avg())
                self.weights = np.random.uniform(-r, r, size=(self.units, self.input))
            case 'glorot_normal':
                self.weights = np.random.normal(0, np.sqrt(1/self._get_fan_avg()), 
                                                size=(self.units, self.input))
            case 'he_uniform':
                r = np.sqrt(2 / self.input)
                self.weights = np.random.uniform(-r, r, size=(self.units, self.input))
            case 'he_normal':
                self.weights = np.random.normal(0, np.sqrt(2 / self.input), 
                                                size=(self.units, self.input))
                
            case _: 
                if not isinstance(self.kernel_initializer, VarianceScaling):
                    raise ValueError(f'Invalid kernel_initializer: {self.kernel_initializer}')

                fans = {
                    'fan_in':  self.input, 
                    'fan_out': self.output, 
                    'fan_avg': self._get_fan_avg()
                    }
                
                fan = fans[self.kernel_initializer.mode]
                match self.kernel_initializer.distribution:
                    case 'normal':
                        std = np.sqrt(self.kernel_initializer.scale / fan)
                        self.weights = np.random.normal(0, std, size=(self.units, self.input))
                    case 'uniform':
                        x = np.sqrt(self.kernel_initializer.scale / fan)
                        self.weights = np.random.uniform(-x, x, size=(self.units, self.input))
    
    def get_weights(self):
        return [self.weights, self.bias]
    
    def set_weights(self, weights, bias):
        if self.weights.shape != weights.shape or \
            self.bias.shape != bias.shape:
            raise ValueError('Invalid shape on weights or bias')
        
        self.weights = weights
        self.bias = bias
    
    @staticmethod
    @numba.njit
    def _get_active_res(activation, raw_res):
        if activation == 'relu':
            flat = raw_res.ravel()
            for i in range(flat.size):
                if flat[i] < 0:
                    flat[i] = 0
            return flat.reshape(raw_res.shape)
        elif activation == 'elu':
            alpha = 1.0
            return np.where(raw_res < 0, alpha * (np.exp(raw_res) - 1), raw_res)
        elif activation == 'selu':
            alpha = 1.67
            return 1.05 * np.where(raw_res < 0, alpha * (np.exp(raw_res) - 1), raw_res)
        elif activation == 'swish':
            return raw_res * sigmoid(raw_res)   
        elif activation == 'tanh':
            return 2 * sigmoid(2 * raw_res) - 1
        elif activation == 'logistic':
            return sigmoid(raw_res)
        elif activation == 'softmax':
            exps = np.exp(raw_res)
            s = np.empty_like(raw_res)
            for i in range(raw_res.shape[0]):
                s[i, :] = exps[i, :] / np.sum(exps[i, :])
            
            return s
        else:
            return raw_res

    @staticmethod
    @numba.njit
    def _get_active_prim(activation, active_res):
        if activation == 'relu':
            flat = active_res.ravel()
            res = np.empty_like(flat)
            for i in range(len(flat)):
                v = 1 if flat[i] > 0 else flat[i]
                res[i] = v

            return res.reshape(active_res.shape).astype(np.float64)
        elif activation == 'elu':
            alpha = 1.0
            return np.where(active_res < 0, alpha * np.exp(active_res), 1.0)
        elif activation == 'selu':
            alpha = 1.67
            return np.where(active_res > 0, alpha, alpha * 1.05 * np.exp(active_res))
        elif activation == 'swish':
            s = sigmoid(active_res)
            return s * (1 + active_res * (1 - s))
        elif activation == 'tanh':
            return 1 - active_res ** 2
        elif activation == 'logistic':
            return active_res * (1 - active_res)
        else:
            return np.ones_like(active_res, dtype=np.float64)
    
    @staticmethod
    @numba.jit
    def get_raw_res(X, weights, bias):
        return X @ weights.T + bias
    
    def forward(self, X=None, *_):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')
        
        X = self.prev_layer.active_res
        self.X = X
        self.raw_res = self.get_raw_res(X, self.weights, self.bias)
        self.active_res = self._get_active_res(self.activation, self.raw_res)
    
    @staticmethod
    @numba.jit
    def get_delta(grad, active_prim):
        return grad * active_prim

    @staticmethod
    @numba.jit
    def get_delta_w(delta, X):
        return delta.T @ X / X.shape[0]
    
    @staticmethod
    @numba.jit
    def get_delta_b(delta):
        res = np.empty(delta.shape[1])
        for col in range(delta.shape[1]):
            res[col] = np.mean(delta[:, col])

        return res.reshape(-1, 1)
    
    @staticmethod
    @numba.jit
    def get_grad(delta, weights):
        return delta @ weights
    
    def backward(self, grad):
        if isinstance(self.activation, (PReLU, Swish)):
            self.activation.backward(grad, self.raw_res)

        delta = grad
        if self.activation != 'softmax':
            delta = self.get_delta(grad, self._get_active_prim(self.activation, self.active_res))
        
        self.active_res = None
        self.delta_weight = self.get_delta_w(delta, self.X)
        self.delta_bias = self.get_delta_b(delta)

        if self._opt_w is None:
            self._opt_w = np.zeros_like(self.delta_weight)
            self._opt_b = np.zeros_like(self.delta_bias)

        return self.get_grad(delta, self.weights)
    
    def step(self, learning_rate, itr):
        if not self.trainable:
            return 
        
        if self.kernel_regularizer is not None:
            self.kernel_regularizer.update()

        if self.kernel_constraint is not None:
            self.weights = self.kernel_constraint.update(self.weights)

        if self.bias_constraint is not None:
            self.bias = self.bias_constraint.update(self.bias)
        
        self.itr = itr
        if isinstance(self.activation, (PReLU, Swish)):
            self.activation.step(learning_rate)

        if isinstance(learning_rate, SGD):
            self._SGD(learning_rate)
        
        elif isinstance(learning_rate, RMSProp):
            self._RMSProp(learning_rate)

        elif isinstance(learning_rate, Adam):
            self._Adam(learning_rate)

        else:
            self.weights -= learning_rate * self.delta_weight
            self.bias    -= learning_rate * self.delta_bias

    def _clip_delta(self, c_value, c_norm):
        if c_value is not None:
            self.delta_weight = np.clip(self.delta_weight, -c_value, c_value)
            self.delta_bias   = np.clip(self.delta_bias, -c_value, c_value)

        elif c_norm is not None:
            total_norm = np.sqrt(np.sum(self.delta_weight ** 2) +
                                 np.sum(self.delta_bias ** 2))

            scale = c_norm / max(total_norm, c_norm)

            self.delta_weight *= scale
            self.delta_bias   *= scale
        
    @staticmethod
    @numba.jit
    def _sgd_calc(decay, weight_decay, momentum, c, itr, delta_weight, delta_bias, 
                  nesterov, learning_rate, weights, bias, _opt_w, _opt_b):
        if decay is not None:
            learning_rate /= 1 + (decay / itr) ** c

        if weight_decay is not None:
            weights -= learning_rate * weight_decay * weights
            bias    -= learning_rate * weight_decay * bias

        _opt_w = momentum * _opt_w - learning_rate * delta_weight
        _opt_b = momentum * _opt_b - learning_rate * delta_bias

        if nesterov:
            weights += momentum * _opt_w - learning_rate * delta_weight
            bias    += momentum * _opt_b - learning_rate * delta_bias

        else:
            weights += _opt_w
            bias    += _opt_b

        return _opt_w, _opt_b

    def _SGD(self, lr): 
        self._clip_delta(lr.clipvalue, lr.clipnorm)
        self._opt_w, self._opt_b = self._sgd_calc(lr.decay, lr.weight_decay, lr.momentum, lr.c, 
                       self.itr, self.delta_weight, self.delta_bias, lr.nesterov, lr.learning_rate, 
                       self.weights, self.bias, self._opt_w, self._opt_b)        

    def _RMSProp(self, lr):
        self._clip_delta(lr.clipvalue, lr.clipnorm)

        self._opt_w = lr.rho * self._opt_w + (1 - lr.rho) * self.delta_weight ** 2
        self._opt_b = lr.rho * self._opt_b + (1 - lr.rho) * self.delta_bias   ** 2

        self.weights -= lr.learning_rate * self.delta_weight / np.sqrt(self._opt_w + lr.epsilon) 
        self.bias    -= lr.learning_rate * self.delta_bias   / np.sqrt(self._opt_b + lr.epsilon)

    def _Adam(self, lr):
        self._clip_delta(lr.clipvalue, lr.clipnorm)

        if not hasattr(self, '_opt2_w'):
            self._opt2_w = np.zeros_like(self._opt_w)
            self._opt2_b = np.zeros_like(self._opt_b)

        self._opt_w = lr.beta_1 * self._opt_w - (1 - lr.beta_1) * self.delta_weight
        self._opt_b = lr.beta_1 * self._opt_b - (1 - lr.beta_1) * self.delta_bias

        if not lr.max:
            self._opt2_w = lr.beta_2 * self._opt2_w + (1 - lr.beta_2) * self.delta_weight ** 2
            self._opt2_b = lr.beta_2 * self._opt2_b + (1 - lr.beta_2) * self.delta_bias   ** 2
        
            s_hat_w = self._opt2_w / (1 - lr.beta_2 ** self.itr)
            s_hat_b = self._opt2_b / (1 - lr.beta_2 ** self.itr)
   
            denominator_w = np.sqrt(s_hat_w + lr.epsilon)
            denominator_b = np.sqrt(s_hat_b + lr.epsilon)
        else:
            self._opt2_w = np.maximum(lr.beta_2 * self._opt2_w, np.abs(self.delta_weight))
            self._opt2_b = np.maximum(lr.beta_2 * self._opt2_b, np.abs(self.delta_bias))

            denominator_w = self._opt2_w + lr.epsilon
            denominator_b = self._opt2_b + lr.epsilon

        m_hat_w = self._opt_w / (1 - lr.beta_1 ** self.itr)
        m_hat_b = self._opt_b / (1 - lr.beta_1 ** self.itr)

        self.weights += lr.learning_rate * m_hat_w / denominator_w
        self.bias    += lr.learning_rate * m_hat_b / denominator_b

        if lr.weight_decay is not None:
            self.weights -= lr.learning_rate * lr.weight_decay * self.weights
            self.bias    -= lr.learning_rate * lr.weight_decay * self.bias

# ----------------------------- Optimizers ----------------------------

class OptimizersBase:
    def __init__(self, learning_rate=0.01, clipvalue=None, clipnorm=None):
        if isinstance(learning_rate, ExponentialDecay):
            self.learning_schedule = learning_rate
            self.learning_rate = learning_rate.initial_learning_rate
        else:
            self.learning_schedule = self
            self.learning_rate = learning_rate

        self.clipvalue = clipvalue
        self.clipnorm = clipnorm

    def get_learning_rate(self, itr):
        return self.learning_rate

class SGD(OptimizersBase):
    def __init__(self, learning_rate=0.01, clipvalue=None, clipnorm=None, 
                 momentum=0.0, nesterov=False, weight_decay=None, decay=None, c=1):
        super().__init__(learning_rate, clipvalue, clipnorm)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.decay = decay
        self.c = c

class RMSProp(OptimizersBase):
    def __init__(self, learning_rate=0.001, clipvalue=None, clipnorm=None, rho=0.9, epsilon=1e-07):
        super().__init__(learning_rate, clipvalue, clipnorm)
        self.rho = rho
        self.epsilon = epsilon

class Adam(OptimizersBase):
    def __init__(self, learning_rate=0.001, clipvalue=None, clipnorm=None, beta_1=0.9, 
                 beta_2=0.999, epsilon=1e-07, max=False, weight_decay=None, decay=None):
        super().__init__(learning_rate, clipvalue, clipnorm)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.max = max
        self.weight_decay = weight_decay
        self.decay = decay

class ExponentialDecay:
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def get_learning_rate(self, itr):
        return self.initial_learning_rate * (self.decay_rate ** (itr / self.decay_steps))

# ------------------- Regularization -------------------
class RegularizationBase(ABC):
    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2
        self.layer = None

    def update(self):
        if self.l2 > 0:
            self.layer.delta_weight += 2 * self.l2 * self.layer.weights
            self.layer.delta_bias   += 2 * self.l2 * self.layer.bias
        
        if self.l1 > 0:
            self.layer.delta_weight += self.l1 * np.sign(self.layer.weights)
            self.layer.delta_bias   += self.l1 * np.sign(self.layer.bias)

class l1(RegularizationBase):
    def __init__(self, l1=0.01):
        super().__init__(l1=l1)

class l2(RegularizationBase):
    def __init__(self, l2=0.01):
        super().__init__(l2=l2)

class l1_l2(RegularizationBase):
    def __init__(self, l1=0.01, l2=0.01):
        super().__init__(l1=l1, l2=l2)

class MaxNorm:
    def __init__(self, r=2):
        self.r = r

    def update(self, params):
        return params * self.r / np.linalg.norm(params, ord=2)
        

# -------------------  -------------------

class Concatenate:
    def __init__(self, name=None):
        self.name = 'concatenate' if name is None else name
        self.is_build = False
        self.trainable = False

    def __call__(self, tensors):
        if not isinstance(tensors, list) or len(tensors) < 2:
            raise ValueError('Concat expects sequence of tensors')
        
        self.output = 0
        self.is_build = True
        self.prev_layer = tensors
        self.outputs = []

        for tensor in tensors:
            if tensor.next_layer is not None:
                tensor.next_layer = [tensor.next_layer, self]
            else:
                tensor.next_layer = self

            self.output += tensor.output
            self.outputs.append(tensor.output)

        return self
    
    def forward(self, x=None, *_):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')

        X = []        
        for prv_l in self.prev_layer:
            act_res = prv_l.active_res
            if act_res is None:
                return True
            
            X.append(act_res)

        self.active_res = np.c_[*X]

    def backward(self, grad):
        res = []
        for i, output in enumerate(self.outputs):
            start = i if not i else self.outputs[i-1]
            res.append(grad[:, start : start + output])

        return res

    def step(self, *_):
        pass

class Normalization:
    def __init__(self, name=None):
        self.name = 'normalization' if name is None else name
        self.is_build = False
        self.is_adapt = False
        self.active_res = None
        self.next_layer = None
        self.adaptive = False
        self.trainable = False

    def __call__(self, tensor, *args, **kwds):
        self.is_build = True
        self.output = tensor.output[0]
        self.input = tensor.output
        tensor.next_layer = self
        self.prev_layer = tensor
        return self

    def adapt(self, X):
        self.is_adapt = True
        self.scaler = StandardScaler().fit(X)

    def forward(self, x=None, *_):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')

        X = self.prev_layer.active_res
        if not self.adaptive and not self.is_adapt:
            raise ValueError(f'This instance {self.name} is not adapted yet')
        else:
            if not hasattr(self, 'scaler'):
                self.scaler = StandardScaler()
            self.scaler.partial_fit(X)

        self.active_res = self.scaler.transform(X)

    def backward(self, grad=None):
        if grad is None:
            grad = self.next_layer.grad

        self.active_res = None
        self.grad = grad

    def step(self, *_):
        pass

class Faltten:
    def __init__(self, name=None):
        self.name = 'concatenate' if name is None else name
        self.is_build = False
        self.trainable = False

    def __call__(self, tensor):
        tensor.next_layer = self
        self.prev_layer = tensor
        self.input = tensor.output
        self.output = (np.prod(tensor.output), )
        self.is_build = True

        return self
    
    def forward(self, x=None, *_):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')

        self.active_res = self.prev_layer.active_res.reshape(-1, *self.output)

    def backward(self, grad):
        return grad

    def step(self, *_):
        pass

class Sequential(Model):
    def __init__(self, layers=None):
        self._names_counts = defaultdict(int)
        self._names = {}
        self.layers = []

        if layers is None:
            layers = []

        for layer in layers:
            layer_name = layer.__class__.__name__.lower()
            self._add_name(layer_name, layer)

    def _add_name(self, name, obj):
        self._names_counts[name] += 1
        new_name = name
        if (count := self._names_counts[name]) > 1:
            new_name = name + f'_{count - 1}'
            self._names_counts[new_name]

        self._names[new_name] = obj
        self.layers.append(obj)
            
    def add(self, layer):
        self._add_name(layer.__class__.__name__.lower(), layer)

    def get_layer(self, name=None, index=None):
        if (name is None and index is None) or \
           (name is not None and index is not None):
            raise ValueError('Must inpute name or index')

        if name is not None:
            return self._names[name]
        return self.layers[index]
    
    def compile(self, optimizer, loss, metrics):
        self.optimizer_ = optimizer
        self.loss_ = loss
        self.metrics_ = metrics

        for i in range(1, len(self.layers)):
            self.layers[i] = self.layers[i](self.layers[i-1])
            if isinstance(self.layers[i], Normalization):
                self.layers[i].adaptive = True

        super().__init__([self.layers[0]], self.layers[-1])
    
    def summary(self):
        buffer = StringIO()
        width = 100
        buffer.write(f'Model: "{self.__class__.__name__.lower()}"\n{'-' * width}\n')
        buffer.write(f'{'Layer (type)':<40}{'Output Shape':<40}{'Param #':<20}\n{'=' * width}\n')
        
        params = []
        last_layer_neurons = 0
        for i, name in enumerate(self._names):
            layer = self.layers[i]

            if isinstance(layer, Input):
                n_neurons = ', '.join(str(x) for x in layer.shape)
                param = 0
            else:
                n_neurons = layer.output
                param = (n_neurons * last_layer_neurons) + n_neurons if i != 0 else 0
                last_layer_neurons = n_neurons

            params.append(param)
            buffer.write(f'{f'{name} ({layer.__class__.__name__})':<40}')
            buffer.write(f'{f'(None, {n_neurons})':<40}')
            buffer.write(f'{param:<20}\n')
        
        buffer.write(f'{'=' * width}\n')
        params_sum = np.sum(params)
        buffer.write(f'Total params: {params_sum}\n')
        buffer.write(f'Trainable params: {params_sum}\n')
        buffer.write(f'Non-trainable params: 0\n')
        buffer.write('-' * width)
        
        buffer.seek(0)
        print(buffer.read())

def load_model(filename):
    return joblib.load(filename)

class Callback:
    def __init__(self):
        self.model = None

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

class LearningRateScheduler(Callback):
    def __init__(self, func):
        super().__init__()
        
        if not callable(func):
            raise TypeError('func must be callable')

        self.func = func

    def on_epoch_begin(self, epoch, logs=None):
        return self.func(epoch)

class ReduceLROnPlateau(Callback):
    def __init__(self, factor=0.1, patience=10):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self._best_val_loss = np.inf
        self._counter = 0 

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        
        if val_loss is None:
            raise ValueError('ReduceLROnPlateau expect val loss')
        
        if val_loss > self._best_val_loss:
            self._counter += 1

            if self._counter == self.patience:
                logs['learning_rate'].learning_rate *= self.factor
                self._counter = 0
                print('da')

        else:
            self._best_val_loss = val_loss
            self._counter = 0

class EarlyStopping(Callback):
    def __init__(self, patience, restore_best_weights=False, min_delta=0.001):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.min_delta = min_delta
        self._best_loss = np.inf
        self._best_weights = None
        self._counter = 0
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        new_loss = logs['val_loss']

        if self._best_loss - new_loss < self.min_delta:
            self._counter += 1
            if self._counter == self.patience:
                if self.restore_best_weights:
                    self.model.set_weights(self._best_weights)
                self.model.stop_training()
        else:
            self._counter = 0
            self._best_loss = new_loss

            if self.restore_best_weights:
                self._best_weights = self.model.get_weights()

class ModelCheckpoint(Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath)

class OneCycleScheduler(Callback):
    def __init__(
        self, max_lr, total_iters, pct_start=0.5, div_factor=10, final_div_factor=1e4):
        super().__init__()

        self.max_lr = max_lr
        self.total_iters = total_iters
        self.pct_start = pct_start

        self.start_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

        self.step = 0
        self.up_steps = int(total_iters * pct_start)
        self.down_steps = total_iters - self.up_steps

    def on_train_begin(self, logs=None):
        self.model.optimizer_.learning_rate = self.start_lr

    def on_batch_begin(self, batch, logs=None):
        if self.step < self.up_steps:
            lr = self.start_lr + (self.max_lr - self.start_lr) * (self.step / self.up_steps)
        else:
            k = self.step - self.up_steps
            lr = self.max_lr - (self.max_lr - self.final_lr) * (k / self.down_steps)

        self.model.optimizer_.learning_rate = lr
        self.step += 1

class VarianceScaling:
    def __init__(self, scale=1.0, mode='fan_in', distribution='normal'):
        if mode not in ("fan_in", "fan_out", "fan_avg"):
            raise ValueError(f'Invalid mode: {mode}')
        
        if distribution not in ("normal", "uniform"):
            raise ValueError(f'Invalid distribution: {distribution}')
        
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

class ActivationFuncBase(ABC):
    def __call__(self, tensor):
        if not isinstance(tensor, Dense):
            raise ValueError('prev layer must be Dense')
        
        self.prev_layer = tensor
        return self

class LeakyReLU(ActivationFuncBase):
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.trainable = False

class PReLU(ActivationFuncBase):
    def __init__(self, alpha_initializer=0.25):
        self.alpha = alpha_initializer
        self.alpha_grad = None
        self.trainable = True

    def backward(self, grad, raw_res):
        alpha_delta = raw_res.copy() 
        alpha_delta[alpha_delta > 0] = 0
        self.alpha_grad = np.mean(grad * alpha_delta)

    def step(self, lr, *_):
        if not self.trainable:
            return 
        
        self.alpha -= lr * self.alpha_grad

class Swish(ActivationFuncBase):
    def __init__(self, beta_initializer=1):
        self.beta = beta_initializer
        self.beta_grad = None
        self.trainable = True

    def backward(self, grad, raw_res):
        s = sigmoid(self.beta * raw_res)
        self.beta_grad = np.mean(grad * raw_res**2 * s * (1 - s))

    def step(self, lr, *_):
        if not self.trainable:
            return 
        
        self.beta -= lr * self.beta_grad

class MCModel(Model):
    def predict(self, X, n_predictions=100):
        check_is_fitted(self)

        if not isinstance(X, tuple):
            X = (X,)


        self._forward(self.inputs, X, training=True)
        res = self.outputs.active_res
        self._active_res_tonone(self.inputs)

        preidctions = np.empty_like(res, shape=(n_predictions, *res.shape))
        preidctions[0] = res

        for i in range(1, n_predictions):
            self._forward(self.inputs, X, training=True)
            preidctions[i] = self.outputs.active_res
            self._active_res_tonone(self.inputs)

        return preidctions.mean(axis=0).round(2)


# ------------------ MCModel ------------------
# import tensorflow as tf

# fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
# (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
# X_train, y_train = X_train_full[:20_000], y_train_full[:20_000]
# X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# inp = Input(X_train.shape[1:])
# flatten = Faltten()(inp)
# hd1 = Dense(300, 'relu', 'he_normal')(flatten)
# hd2 = Dense(100, 'relu', 'he_normal')(hd1)
# output = Dense(10, 'softmax')(hd2)

# model = Model([inp], output)
# model.compile('sgd', 'sparse_categorical_crossentropy', ['accuracy'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))




# inp = Input(X_train.shape[1:])
# flatten = Faltten()(inp)

# dropout1 = Dropout(0.2)(flatten)
# hd1 = Dense(300, 'relu', 'he_normal')(dropout1)

# dropout2 = Dropout(0.2)(hd1)
# hd2 = Dense(100, 'relu', 'he_normal')(dropout2)

# dropout3 = Dropout(0.2)(hd2)
# output = Dense(10, 'softmax')(dropout3)

# model = MCModel([inp], output)
# model.compile('sgd', 'sparse_categorical_crossentropy', ['accuracy'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# print(model.predict(X_test))









# -------------------------- Tests --------------------------
# np.random.seed(13)
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split

# X, y = fetch_california_housing(return_X_y=True)
# X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True) 
# X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, random_state=42, test_size=0.1, shuffle=True) 


# ------------------- topology 1 -------------------
# v_scaling = VarianceScaling(2, distribution='normal')

# prelu = PReLU(0.025)
# lrelu = LeakyReLU(0.025)

# normalization_layer = Normalization()
# hidden_layer1 = Dense(300, activation=None, kernel_initializer='he_normal')
# hidden_layer2 = Dense(100, activation=None, kernel_initializer='he_normal')
# output_layer = Dense(1, kernel_initializer='he_normal')

# normalization_layer.adapt(X_train)

# input_ = Input(shape=X_train.shape[1:])
# normalized = normalization_layer(input_)
# hidden1 = hidden_layer1(normalized)
# prelu = prelu(hidden1)

# hidden2 = hidden_layer2(prelu)
# prelu2 = PReLU(0.025)(hidden2)

# output = output_layer(prelu2)

# model = Model(inputs=[input_], outputs=output)

# optimizer = SGD(0.0005)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])

# class CustomCallback(Callback):
#     def on_epoch_end(self, epoch=None, logs=None):
#         ratio = logs["val_loss"] / logs["loss"]
#         print(f'Ep: {epoch} -> {ratio}')

# checkpoint = ModelCheckpoint('custom_keras_model.pkl')
# early_stop = EarlyStopping(5, restore_best_weights=True, min_delta=0.01)
# stats = CustomCallback()
# # model.fit(X_train, y_train, epochs=40, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stop, stats])


# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))


# print(model.predict(X_test[:3]))

# loaded_model = load_model('custom_keras_model.pkl')
# print(loaded_model.predict(X_test[:3]))



# ------------------------ "elu": 0.62, "gelu": 0.64 и "swish":0.67 "swish_beta": 0.67
# input_ = Input(shape=X_train.shape[1:])

# normalization_layer = Normalization()
# normalization_layer.adapt(X_train)
# normalized = normalization_layer(input_)

# swish = Swish(0.6)

# layer1 = Dense(300, activation=swish, kernel_initializer='he_normal')(normalized)
# layer2 = Dense(100, activation=swish, kernel_initializer='he_normal')(layer1)
# output = Dense(1, kernel_initializer='he_normal')(layer2)
# model = Model(inputs=[input_], outputs=output)

# optimizer = SGD(0.001)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
# print(swish.beta)

# ----------------------- BatchNormalization : 0.56
# model = Sequential([
#     Input(X_train.shape[1:]),
#     Normalization(),

#     BatchNormalization(),
#     Dense(300, activation="relu", kernel_initializer="he_normal"),

#     BatchNormalization(),
#     Dense(100, activation="relu", kernel_initializer="he_normal"),

#     BatchNormalization(),
#     Dense(1)
# ])

# optimizer = SGD(0.01)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# ----------------------- clipvalue : 0.59; clipnorm : 0.60
# model = Sequential([
#     Input(X_train.shape[1:]),
#     Normalization(),

#     Dense(300, activation="relu", kernel_initializer="he_normal"),
#     Dense(100, activation="relu", kernel_initializer="he_normal"),
#     Dense(1)
# ])

# optimizer = SGD(0.01, clipnorm=1)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))


# ----------------------- lock layers
# model = Sequential([
#     Input(X_train.shape[1:]),
#     Normalization(),

#     Dense(300, activation="relu", kernel_initializer="he_normal"),
#     Dense(100, activation="relu", kernel_initializer="he_normal"),
#     Dense(1)
# ])

# for layer in model.layers:
#     layer.trainable = False 

# optimizer = SGD(0.01, clipnorm=1)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# ----------------------- momentum :0.574
# model = Sequential([
#     Input(X_train.shape[1:]),
#     Normalization(),

#     Dense(300, activation="relu", kernel_initializer="he_normal"),
#     Dense(100, activation="relu", kernel_initializer="he_normal"),
#     Dense(1)
# ])

# optimizer = SGD(0.01, momentum=0.9)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# ----------------------- momentum + nesterov :0.58
# model = Sequential([
#     Input(X_train.shape[1:]),
#     Normalization(),

#     Dense(300, activation="relu", kernel_initializer="he_normal"),
#     Dense(100, activation="relu", kernel_initializer="he_normal"),
#     Dense(1)
# ])

# optimizer = SGD(0.005, momentum=0.9, nesterov=True)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# ----------------------- RMSProp: 0.56
# model = Sequential([
#     Input(X_train.shape[1:]),
#     Normalization(),

#     Dense(300, activation="relu", kernel_initializer="he_normal"),
#     Dense(100, activation="relu", kernel_initializer="he_normal"),
#     Dense(1)
# ])

# optimizer = RMSProp()
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# model.save('rmsprop_opt_model.pkl')
# load_m = load_model('rmsprop_opt_model.pkl')
# print(load_m.predict(X_test[:3]))


# ----------------------- Adam: 0.58; Adamax: 0.59; adamW : 0.65; sgd weight decay: 0.75; sgd decay: 0.59
# model = Sequential([
#     Input(X_train.shape[1:]),
#     Normalization(),

#     Dense(300, activation="relu", kernel_initializer="he_normal"),
#     Dense(100, activation="relu", kernel_initializer="he_normal"),
#     Dense(1)
# ])

# # optimizer = Adam(weight_decay=0.99)
# optimizer = SGD(decay=1e-4)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))


# ----------------------- LearningRateScheduler; ReduceLROnPlateau; 
# model = Sequential([
#     Input(X_train.shape[1:]),
#     Normalization(),

#     Dense(300, activation="relu", kernel_initializer="he_normal"),
#     Dense(100, activation="relu", kernel_initializer="he_normal"),
#     Dense(1)
# ])


# def exponential_decay(lr0, s):
#     def exponential_decay_fn(epoch):
#         lr = lr0 * 1e-2 ** (epoch / s)
#         print(lr)
#         return lr
#     return exponential_decay_fn

# exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
# lr_scheduler = LearningRateScheduler(exponential_decay_fn)

# lr_scheduler = ReduceLROnPlateau(patience=2)

# import math
# batch_size = 32
# n_epochs = 20
# n_steps = n_epochs * math.ceil(len(X_train) / batch_size)
# lr_scheduler = ExponentialDecay(initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1)

# optimizer = SGD(learning_rate=lr_scheduler)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val)) # , callbacks=[lr_scheduler]

# ----------------------- OneCycleScheduler; 

# optimizer = SGD()
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),  callbacks=[OneCycleScheduler(0.1, n_steps)])



# ------------------------------ l1, l2, l1_l2
# norm = Normalization()
# norm.adapt(X_train)

# model = Sequential([
#     Input(X_train.shape[1:]),
#     norm,
#     Dense(100, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l1_l2(0.02, 0.02)),
#     Dense(100, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l1_l2(0.02, 0.02)),
#     Dense(1, kernel_regularizer=l1_l2(0.02, 0.02))
# ])

# optimizer = SGD()
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# ------------------------------ DROPOUT ------------------------
# norm = Normalization()
# norm.adapt(X_train)

# model = Sequential([
#     Input(X_train.shape[1:]),
#     norm,
#     # Dropout(0.1),
#     Dense(300, activation='relu', kernel_initializer='he_normal'),
    
#     # Dropout(0.1),
#     Dense(200, activation='relu', kernel_initializer='he_normal'),

#     Dropout(0.3),
#     Dense(1)
# ])

# optimizer = SGD()
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# --------------------------------------------- AlphaDROPOUT

# norm = Normalization()
# norm.adapt(X_train)

# model = Sequential([
#     Input(X_train.shape[1:]),
#     norm,
#     AlphaDropout(0.1),
#     Dense(300, activation='selu', kernel_initializer=VarianceScaling(1, 'fan_in')),
    
#     AlphaDropout(0.1),
#     Dense(200, activation='selu', kernel_initializer=VarianceScaling(1, 'fan_in')),

#     AlphaDropout(0.1),
#     Dense(1)
# ])

# optimizer = SGD(0.01)
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# ------------------------------ MaxNorm ------------------------
# norm = Normalization()
# norm.adapt(X_train)

# model = Sequential([
#     Input(X_train.shape[1:]),
#     norm,
#     Dense(300, activation='relu', kernel_initializer='he_normal', 
#         kernel_constraint=MaxNorm(2), bias_constraint=MaxNorm(2)),

#     Dense(200, activation='relu', kernel_initializer='he_normal', 
#         kernel_constraint=MaxNorm(2), bias_constraint=MaxNorm(2)),

#     Dense(1)
# ])

# optimizer = SGD()
# model.compile(loss='mse', optimizer=optimizer, metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))


# ------------------- topology 2 -------------------
# normalization_layer = Normalization()
# hidden_layer1 = Dense(300, activation="relu", name='dense1')
# hidden_layer2 = Dense(100, activation="relu", name='dense2')
# concat_layer = Concatenate(name='concat')
# output_layer = Dense(1, name='dense3')

# normalization_layer.adapt(X_train)

# input_ = Input(shape=X_train.shape[1:])
# normalized = normalization_layer(input_)
# hidden1 = hidden_layer1(normalized)
# hidden2 = hidden_layer2(hidden1)
# concat = concat_layer([normalized, hidden2])
# output = output_layer(concat)

# model = Model(inputs=[input_], outputs=output)
# model.compile(loss='mse', optimizer=SGD(0.001), metrics=['rmse'])

# model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

# ------------------- topology 3 -------------------

# input_wide = Input(shape=[5])
# input_deep = Input(shape=[6])
# norm_layer_wide = Normalization()
# norm_layer_deep = Normalization()
# norm_wide = norm_layer_wide(input_wide)
# norm_deep = norm_layer_deep(input_deep)
# hidden1 = Dense(30, activation="relu")(norm_deep)
# hidden2 = Dense(30, activation="relu")(hidden1)
# concat = Concatenate()([norm_wide, hidden2])
# output = Dense(1)(concat)

# model = Model(inputs=[input_wide, input_deep], outputs=output)
# model.compile(loss="mse", optimizer='sgd', metrics=["rmse"])

# X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
# X_valid_wide, X_valid_deep = X_val[:, :5], X_val[:, 2:]

# norm_layer_wide.adapt(X_train_wide)
# norm_layer_deep.adapt(X_train_deep)
# history = model.fit((X_train_wide, X_train_deep), y_train, epochs=20,
#                                     validation_data=((X_valid_wide, X_valid_deep), y_val))


#--------------------------------------------------------------------------
# from sklearn.datasets import make_moons
# X, y = make_moons(5000, noise=0.2, random_state=42)

# X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.1)
# X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, random_state=42, shuffle=True, test_size=0.15)

# normalization_layer = Normalization()
# hidden_layer1 = Dense(300, activation="relu")
# hidden_layer2 = Dense(100, activation="relu")
# output_layer = Dense(1, activation='logistic')

# normalization_layer.adapt(X_train)

# input_ = Input(shape=X_train.shape[1:])
# normalized = normalization_layer(input_)
# hidden1 = hidden_layer1(normalized)
# hidden2 = hidden_layer2(hidden1)
# output = output_layer(hidden2)

# model = Model(inputs=[input_], outputs=output)
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))



# --------------------- Seqential ---------------------

# seq = Sequential()
# seq.add(Input(X_train.shape[1:]))
# seq.add(Normalization())
# seq.add(Dense(300, activation='relu'))
# seq.add(Dense(100, activation='relu'))
# seq.add(Dense(1))

# seq.compile(SGD(0.001), 'mse', ['rmse'])
# seq.summary()
# seq.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))