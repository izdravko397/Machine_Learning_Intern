from collections import defaultdict
from io import StringIO
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import check_array, check_is_fitted
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, root_mean_squared_error, precision_score, mean_squared_error
from tqdm import tqdm
from scipy.special import erf
import joblib
import pickle
from copy import deepcopy
import tensorflow as tf
from functools import partial
from tf_data import Dataset



one = tf.constant(1.0)
selu_alpha = tf.constant(1.67)
selu_const = tf.constant(1.05)

# -------- activations --------

@tf.function
def sigmoid(z): 
    return 1 / (1 + tf.exp(-tf.clip_by_value(z, -500, 500)))

# @tf.function
# def softmax(z):
#     classes_exp = tf.exp(z)
#     return classes_exp / tf.reduce_sum(classes_exp, axis=1, keepdims=True)

@tf.function(jit_compile=True)
def softmax(z):
    z_max = tf.reduce_max(z, axis=-1, keepdims=True)  
    exp_z = tf.exp(z - z_max)
    return exp_z / tf.reduce_sum(exp_z, axis=-1, keepdims=True)

@tf.custom_gradient
def relu_fn(raw_res):
    res = tf.where(raw_res > 0, raw_res, 0.0)

    def relu_grad(grad):
        return grad * tf.cast(raw_res > 0, tf.float32)

    return res, relu_grad

@tf.function(jit_compile=True)
def relu(raw_res):
    return relu_fn(raw_res)

@tf.function(jit_compile=True)
def elu(raw_res): 
    return tf.where(raw_res < 0, one * (tf.exp(raw_res) - 1), raw_res)

@tf.function(jit_compile=True)
def selu(raw_res): 
    return selu_const * tf.where(raw_res < 0, selu_alpha * (tf.exp(raw_res) - 1), raw_res)

@tf.function(jit_compile=True)
def gelu(raw_res): 
    return raw_res * 0.5 * (1 + erf(raw_res / tf.sqrt(2)))

@tf.function(jit_compile=True)
def swish(raw_res): 
    return raw_res * sigmoid(raw_res)

@tf.function(jit_compile=True)
def tanh(raw_res): 
    return 2 * sigmoid(2 * raw_res) - 1

# -------- derivatives --------

@tf.function
def relu_prim(active_res): 
    return tf.cast(active_res > 0, tf.float32)

@tf.function
def elu_prim(active_res): 
    return tf.where(active_res < 0, one * tf.exp(active_res), one)

@tf.function
def selu_prim(active_res): 
    return tf.where(active_res > 0, selu_alpha, selu_alpha * selu_const * tf.exp(active_res))

@tf.function
def gelu_prim(active_res): 
    return (tf.constant(0.5) * (one + erf(active_res / tf.sqrt(2))) + active_res * 
            (tf.exp(tf.constant(-0.5) * tf.square(active_res)) / tf.sqrt(2*tf.constant(np.pi))))

@tf.function
def swish_prim(active_res): 
    return sigmoid(active_res) * (one + active_res * (one - sigmoid(active_res)))

@tf.function
def tanh_prim(active_res): 
    return one - active_res ** 2

@tf.function
def sigmoid_prim(active_res): 
    return active_res * (one - active_res)

# ------------------- losses -------------------
@tf.function
def crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    return -tf.reduce_mean(tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.math.log(y_pred), axis=1))

@tf.function
def binary_crossentropy(y_true, y_pred):
    return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))


class History:
    def __init__(self, history={}, model=None):
        self.history = history or defaultdict(list)
        self.model   = model

    def update(self, losses, scores, val_losses, val_scores):
        self.history['loss'].extend(losses)
        self.history['val_loss'].extend(val_losses)

        scores     = tf.convert_to_tensor(scores)
        val_scores = tf.convert_to_tensor(val_scores)

        for i, (m, val_m) in enumerate(zip(self.model.metrics_names, self.model.val_metrics_names)):
            self.history[m].extend(scores[:, i])
            self.history[val_m].extend(val_scores[:, i])

class Model:
    def __init__(self, inputs=None, outputs=None):
        self.inputs = inputs
        self.outputs = outputs
        self._weights = []
        self._stop_training = False
        self._trainable_variables = []
        self._additional_loss = 0
        self._additional_metrics = set()
        self.losses, self.val_losses = [], []
        self.scores, self.val_scores = [], []
        self.history = History(model=self)

    def _get_optimizer(self, optimizer=None):
        optimizer = optimizer if optimizer is not None else self.optimizer_
        match optimizer:
            case 'sgd': return SGD()
            case 'adam': return Adam()
            case _:
                if not isinstance(optimizer, OptimizersBase):
                    raise ValueError('Invalid optimizer')
                
                return optimizer

    def add_loss(self, value):
        self._additional_loss += value

    def add_metric(self, metric):
        if not isinstance(metric, Metric):
            raise TypeError(f'Invalid metric: {type(metric)}')
        
        if id(metric) not in self._additional_metrics:
            self.metrics_.append(metric)
            self._additional_metrics.add(id(metric))

    def _get_loss(self, y_true, y_pred):
        match self.loss_:
            case "sparse_categorical_crossentropy" | "categorical_crossentropy": 
                loss = crossentropy(y_true, y_pred)
            case "binary_crossentropy": 
                loss = binary_crossentropy(tf.cast(y_true, tf.float32), y_pred)
            case "mse": loss = self.mse(y_true, y_pred)
            case 'mae': loss = self.mae(y_true, y_pred)
            case _:
                if callable(self.loss_):
                    loss = self.loss_(y_true, y_pred)
                
                raise ValueError(f'Invalid loss {self.loss_}')
        
        total_loss = loss + self._additional_loss
        self._additional_loss = 0
        return total_loss

    def _prob_to_cls(self, y_pred):
        if self.loss_ == 'sparse_categorical_crossentropy':
            out = tf.argmax(y_pred, axis=-1)
            if len(out.shape) == 1:
                return tf.reshape(out, (-1, 1))
            return out
        return tf.where(y_pred >= 0.5, 1, 0)
            
    def _get_scores(self, y_true, y_pred, metrics=None):
        scores = []
        metrics = metrics or self.metrics_

        for metric in metrics:
            score = None
            match metric:
                # case 'accuracy': score = accuracy_score(y_true, self._prob_to_cls(y_pred))
                case 'accuracy': score = Accuracy.one_shot(y_true, self._prob_to_cls(y_pred))
                case 'f1_macro': score = f1_score(y_true, self._prob_to_cls(y_pred), average='macro')
                case 'auc':  score = roc_auc_score(y_true, self._prob_to_cls(y_pred))
                case 'f1':   score = f1_score(y_true, self._prob_to_cls(y_pred))
                case 'precision': score = precision_score(y_true, self._prob_to_cls(y_pred))
                case 'mse'  | 'meansquarederror': score = mean_squared_error(y_true, y_pred)
                case 'mae'  | 'meanabsoluteerror': score = tf.reduce_mean(tf.abs(y_true - y_pred))
                case 'mape' | 'meanabsolutepercentageerror': score = tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))
                case 'rmse' | 'rootmeansquarederror': score = tf.sqrt(self.mse(y_true, y_pred))
                case _: 
                    if id(metric) in self._additional_metrics:
                        continue 

                    preds = self._prob_to_cls(y_pred) \
                            if self.loss_ in ("sparse_categorical_crossentropy", "binary_crossentropy") \
                            else y_pred
                    metric.update_state(y_true, preds)

            scores.append(score)
        return scores
            
    def compile(self, optimizer, loss, metrics):
        self.optimizer_ = self._get_optimizer(optimizer)
        self.loss_ = loss

        if self.loss_ == 'mse':
            self.mse = tf.keras.losses.MeanSquaredError()

        elif self.loss_ == 'mae':
            self.mae = tf.keras.losses.MeanAbsoluteError()

        self.metrics_ = []
        for metric in metrics:
            match metric:
                case 'accuracy' | 'Accuracy':    self.metrics_.append(Accuracy())
                case 'mse' | 'MeanSquaredError': self.metrics_.append(MeanSquaredError())
                case 'mae' | 'MeanAbsoluteError': self.metrics_.append(MeanAbsoluteError())
                case 'mape' | 'MeanAbsolutePercentageError': self.metrics_.append(MeanAbsolutePercentageError())
                case 'rmse' | 'RootMeanSquaredError': self.metrics_.append(RootMeanSquaredError())
                case 'huber' | 'Huber': self.metrics_.append(HuberMetric())
                case 'precision' | 'Rrecision': self.metrics_.append(Precision())

        self.metrics_names     = [m.name for m in self.metrics_]
        self.val_metrics_names = ['val_' + m.name for m in self.metrics_]
        self.n_samples, self.n_features = None, [None]

        if self.outputs is not None and loss == 'sparse_categorical_crossentropy':
            self.n_classes = self.outputs.units

    def _get_trainable_variables(self, layers=None):
        layers = layers if layers is not None else self.inputs 

        if isinstance(self, Sequential):
            for value in self.__dict__.values():
                if not isinstance(value, list):
                    value = [value]

                for element in value:
                    if hasattr(element, '_trainable_variables'):
                        self._trainable_variables.extend(element._trainable_variables)
            return

        self.layers = set()
        for layer in layers:
            
            while True:
                self.layers.add(layer)
                if not layer._visited:
                    self._trainable_variables.extend(layer._trainable_variables)

                layer._visited = True
                layer = layer.next_layer

                if layer is None:
                    break

                if isinstance(layer, list):
                    self._get_trainable_variables(layer)

        self._reset_visited_param()

    def call(self, X, layers=None, training=False):
        layers = layers if layers is not None else self.inputs 
        X = X if isinstance(X, (tuple, dict)) else (X, )

        if isinstance(X, tuple):
            for layer, x in zip(layers, X):
                while True:
                    if isinstance(x, tuple) and layer.supports_masking:
                        inp, mask = x
                        x = layer.call(inp, mask=mask, training=training)
                    else:
                        x = layer.call(x, training=training)

                    if layer.next_layer is None or \
                        (isinstance(x, bool) and x == True):
                        break

                    layer = layer.next_layer
                    if isinstance(layer, list):
                        self.call(x, layers=layer, training=training)
            return x

        for name, x in X.items():
            for layer in self.inputs:
                if layer.name == name:
                    break
            else:
                raise ValueError(f'Not exist layer with name: {name}')

            while True:
                x = layer.call(x, training=training)
                layer = layer.next_layer

                if layer is None or \
                    (isinstance(x, bool) and x == True):
                    break

        return x

    @tf.function
    def _backpropagation(self, xi, yi, yi_true, itr):
        with tf.GradientTape() as tape:
            y_pred = self.call(xi, training=True)

            if not self._trainable_variables:
                self._get_trainable_variables()

            loss = self._get_loss(yi, y_pred)

        grads = tape.gradient(loss, self._trainable_variables)
        self.optimizer_.apply_gradients(zip(self._trainable_variables, grads), itr=itr)
        self._get_scores(yi_true, y_pred)
        return loss
    
    def _execute_callbacks(self, callbacks, epoch=None, logs=None, mode='set_model_instance'):
        for callback in callbacks:
            match mode:
                case 'set_model_instance': callback.model = self
                case 'on_train_begin':     callback.on_train_begin(logs)
                case 'on_train_end':       callback.on_train_end(logs)
                case 'on_epoch_begin':     callback.on_epoch_begin(epoch, logs)
                case 'on_epoch_end':       callback.on_epoch_end(epoch, logs)
                case 'on_batch_begin':     callback.on_batch_begin(epoch, logs)
                case 'on_batch_end':       callback.on_batch_end(epoch, logs)

    def build(self, batch_input_shape):
        pass

    def _reset_metrics(self):
        for metric in self.metrics_:
            metric.reset_states()

    def _check_sparse_target_data(self, target):
        if self.loss_ != "sparse_categorical_crossentropy":
            return target
        
        one_hot = tf.one_hot(tf.cast(target, tf.int32), depth=self.n_classes)
        return tf.reshape(one_hot, [*target.shape, self.n_classes])

    def _data_preprocessing(self, X, y, validation_data, validation_ratio, batch_size):
        if not isinstance(X, (tuple, dict)):
            X = (X,)

        if isinstance(X, tuple):
            X = tuple(tf.convert_to_tensor(x) for x in X)
            first_item = X[0]

        elif isinstance(X, dict):
            X = dict((key, tf.convert_to_tensor(x)) for key, x in X.items())
            X_itr = iter(X.values())
            first_item = next(X_itr)

        y = tf.convert_to_tensor(y)

        if len(y.shape) == 1:
            y = tf.reshape(y, shape=(-1, 1))
        
        self.n_samples, *self.n_features = first_item.shape
        self.inxs = tf.range(self.n_samples)
        self.n_batches = tf.cast(tf.math.ceil(self.n_samples / batch_size), tf.int32)
        
        if validation_data is None:
            val_data_len = tf.constant(self.n_samples * validation_ratio)
            shuffle_inxs = tf.random.shuffle(self.inxs)
            valid_ixns   = shuffle_inxs[:val_data_len]
            not_val_inx  = shuffle_inxs[val_data_len:]

            if isinstance(X, tuple):
                X_valid = tuple([x[valid_ixns] for x in X])
            elif isinstance(X, dict):
                X_valid = dict((key, x[valid_ixns]) for key, x in X.items())

            validation_data = (X_valid, y[valid_ixns])
            X = X[not_val_inx]
            y = y[not_val_inx]

        elif not isinstance(validation_data[0], (tuple, dict)):
            validation_data = ((validation_data[0],), *validation_data[1:])

        y_t     = tf.identity(y)
        y_val_t = tf.identity(validation_data[1])

        if len(tf.shape(y_val_t)) == 1:
            y_val_t = tf.reshape(y_val_t, [-1, 1])

        y_t     = self._check_sparse_target_data(y_t) 
        y_val_t = self._check_sparse_target_data(y_val_t)

        return X, y, validation_data, y_t, y_val_t
    
    def _set_callbacks(self, callbacks):
        if not isinstance(callbacks, list):
            raise TypeError(f'callbacks must be list not: {type(callbacks)}')
        
        self._execute_callbacks(callbacks)

    def _print_update_stats(self, valid_data, y_val_t, loss, scores):
        if y_val_t is None:
            mean_val_loss, mean_val_scores = [], [] 
            for X_valid, y_valid in valid_data:
                y_val_t = self._check_sparse_target_data(y_valid)
                output  = self.call(X_valid, training=False)
                mean_val_loss.append(self._get_loss(y_val_t, output))  
                mean_val_scores.append(self._get_scores(y_valid, output, self.metrics_names)) 

            val_loss   = tf.reduce_mean(mean_val_loss)
            val_scores = tf.reduce_mean(mean_val_scores, axis=0)

        else:
            X_valid = valid_data[0]
            y_valid = valid_data[1]

            if not isinstance(X_valid, (list, tuple)):
                X_valid = [X_valid]
            
            batches = len(X_valid[0]) // self.batch_size
            val_losses = []
            val_scores = []
            for i in range(batches):
                start = i * self.batch_size
                end = start + self.batch_size
                batch = tuple(item[start:end] for item in X_valid)
                
                batch_y  = y_valid[start:end]
                batch_yt = y_val_t[start:end]

                output = self.call(batch, training=False)
                val_losses.append(self._get_loss(batch_yt, output))
                val_scores.append(self._get_scores(batch_y, output, self.metrics_names))

            val_loss = tf.reduce_mean(val_losses)
            val_scores = tf.reduce_mean(val_scores, axis=0)

        self.losses.append(loss)
        self.scores.append(scores)
        self.val_losses.append(val_loss)
        self.val_scores.append(val_scores)
        
        train_stats = f'    - loss: {loss:.3f}'
        for i, m in enumerate(self.metrics_):
            train_stats += f' - {m.name}: {scores[i]:.3f}'
        print(train_stats)

        val_stats = f'    - val_loss: {val_loss:.3f}'
        for i, m in enumerate(self.val_metrics_names):
            val_stats += f' - {m}: {val_scores[i]:.3f}'
        print(val_stats)

        print('    - Learning Rate:', self.optimizer_.learning_rate)

        return val_loss, val_scores
    
    def _get_random_batch(self, batch_size, X, y, y_t):
        batch_inxs = tf.random.shuffle(self.inxs)[:batch_size]

        if isinstance(X, tuple):
            xi = tuple(tf.gather(x, batch_inxs) for x in X)
        else:
            xi = dict((key, tf.gather(x, batch_inxs)) for key, x in X.items())

        yi = tf.gather(y_t, batch_inxs)
        yi_true = tf.gather(y, batch_inxs)

        return xi, yi, yi_true 

    def fit(self, X, y=None, epochs=1, batch_size=32, validation_data=None,
            validation_ratio=0.1, callbacks=[]):

        self.batch_size = batch_size
        y_val_t = None
        if not isinstance(X, Dataset):
            X, y, validation_data, y_t, y_val_t = self._data_preprocessing(X, y, 
                                                                           validation_data, 
                                                                           validation_ratio, 
                                                                           batch_size)

        self.build([self.n_samples, *self.n_features])
        self._set_callbacks(callbacks)
        self._get_trainable_variables()
        self._execute_callbacks(callbacks, mode='on_train_begin')
        
        itr = tf.Variable(0., dtype=tf.float32)
        for epoch in range(epochs):
            self._execute_callbacks(callbacks, epoch + 1, mode='on_epoch_begin')
            print(f'Epoch {epoch + 1}/{epochs}') 
            mean_loss = []

            data_iter = enumerate(X, start=1) if isinstance(X, Dataset) else range(1, self.n_batches + 1)
            for i in tqdm(data_iter):
                itr.assign_add(1.)

                if isinstance(X, Dataset):
                    i, batch_data = i
                    xi, yi_true = batch_data 
                    yi = self._check_sparse_target_data(yi_true)
                else:      
                    xi, yi, yi_true = self._get_random_batch(batch_size, X, y, y_t)
                
                self._execute_callbacks(callbacks, i, mode='on_batch_begin') 
                loss = self._backpropagation(xi, yi, yi_true, itr)
                mean_loss.append(loss)

                if callbacks:
                    logs = {'loss': loss}
                    logs.update({m.name: m.result() for m in self.metrics_})
                    self._execute_callbacks(callbacks, i, logs, mode='on_batch_end')

            mean_loss   = tf.reduce_mean(mean_loss)
            mean_scores = [m.result() for m in self.metrics_]
            self._reset_metrics()
            val_loss, val_scores = self._print_update_stats(validation_data, y_val_t, 
                                                            mean_loss, mean_scores)

            if callbacks:
                logs = {'loss': mean_loss, 'val_loss': val_loss}
                logs.update(dict(zip(self.metrics_names, mean_scores)))
                logs.update(dict(zip(self.val_metrics_names, val_scores)))
                self._execute_callbacks(callbacks, epoch + 1, logs, mode='on_epoch_end')
                
                if self._stop_training:
                    break

        self._execute_callbacks(callbacks, mode='on_train_end')
        self.history.update(self.losses, self.scores, self.val_losses, self.val_scores)
        return self.history 
    
    def reset_states(self):
        for layer in self.layers:
            layer.reset_states()
    
    def stop_training(self):
        self._stop_training = True

    def predict(self, X): 
        check_is_fitted(self)

        if not isinstance(X, Dataset):
            return self.call(X, training=False)

        preds = [self.call(X_test, training=False) for X_test, *_ in X]
        return tf.concat(preds, axis=0)

    def evaluate(self, X, y=None):
        check_is_fitted(self)
        metric_names = [m.name for m in self.metrics_]
        
        if not isinstance(X, Dataset):
            y   = tf.reshape(y, [-1, 1]) 
            y_t = tf.identity(y)

            if self.loss_ == "sparse_categorical_crossentropy":
                y_t = tf.reshape(tf.one_hot(tf.cast(y, tf.int32), depth=self.n_classes), 
                                                                        [-1, self.n_classes])

            pred = self.predict(X)
            loss   = self._get_loss(y_t, pred)
            scores = self._get_scores(y, pred, metric_names)

        else:
            mean_loss, mean_scores = [], []

            for X_test, y_test in X:
                y_t = self._check_sparse_target_data(y_test)
                pred = self.predict(X_test)
                mean_loss.append(self._get_loss(y_t, pred)) 
                mean_scores.append(self._get_scores(y_test, pred, metric_names))

            loss   = tf.reduce_mean(mean_loss) 
            scores = tf.reduce_mean(mean_scores, axis=0)

        score_results = ' - '.join(f'{name}: {val}' 
                    for name, val in zip(metric_names, scores))
        
        print(f'    - loss: {loss} - ' + score_results)

        return loss, *scores
    
    def _get_architecture(self, layers=None):
        if layers is None:
            self.architecture = []
            layers = self.inputs

        for lr in layers:
            while True:
                self.architecture.append({'object': lr.__class__,
                                          'dict': lr.__dict__})
                if lr.next_layer is None:
                    break

                lr = lr.next_layer
                if isinstance(lr, list):
                    return self._active_res_tonone(lr)

    def save(self, filename):
        # joblib.dump(self, filename)

        self._get_architecture()
        data = {
            'class': self.__class__.__name__,
            'architecture': self.architecture,
            'loss': self.loss_,
            'optimizer': self.optimizer_,
            'metrics': self.metrics_
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def _reset_visited_param(self, layers=None):
        layers = layers if layers is not None else self.inputs 

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
    
# ------------------------ Layers ------------------------

class Layer(ABC):
    def __init__(self, name=None):
        super().__init__()
        self.name = name or self.__class__.__name__.lower()
        self.next_layer = None
        self.prev_layer = None
        self.is_build = False
        self._visited = False
        self.trainable = True
        self.supports_masking = False 
        self._trainable_variables = []

    def __call__(self, layer):
        if not isinstance(layer, (Layer, list)):
            return self.call(layer)

        self.is_build = True
        self.prev_layer = layer

        layer_list = layer if isinstance(layer, (list, tuple)) else [layer]

        for lyr in layer_list:
            self.batch_size = lyr.batch_size
            if lyr.next_layer is None:
                lyr.next_layer = self
            else: 
                if not isinstance(lyr.next_layer, (list, tuple)):
                    lyr.next_layer = [lyr.next_layer]

                lyr.next_layer.append(self)

        return self
    
    def get_weights(self):
        return []
    
    def set_weights(self, *weights):
        pass

    def reset_states(self):
        pass

    def compute_mask(self, inputs, prev_mask):
        return prev_mask

    def _dropout(self, X, rate):
        rnd_mat = tf.random.uniform(minval=0, maxval=1, shape=X.shape)
        mask = tf.cast(rnd_mat > rate, tf.float32) / (1 - rate)
        return X * mask

    @abstractmethod
    def call(self, X=None, *_):
        pass

    def _weights_init(self, initializer, shape):
        match initializer:
            case 'glorot_uniform':
                r = tf.cast(tf.sqrt(3 / self.fan_avg), tf.float32)
                return tf.Variable(tf.random.uniform(minval=-r, maxval=r, shape=shape))
            case 'glorot_normal':
                std = np.sqrt(1 / self.fan_avg)
                return tf.Variable(tf.random.normal(mean=0, stddev=std, shape=shape))
            case 'he_uniform':
                r = np.sqrt(2 / self.fan_in)
                return tf.Variable(tf.random.uniform(minval=-r, maxval=r, shape=shape))
            case 'he_normal':
                std = np.sqrt(2 / self.fan_in)
                return tf.Variable(tf.random.normal(mean=0, stddev=std, shape=shape))
            case 'orthogonal':
                orth_init = tf.keras.initializers.Orthogonal()
                return tf.Variable(orth_init(shape=shape))
            case 'ones' :
                return tf.Variable(tf.ones(shape, dtype=tf.float32))
            case 'zeros':
                return tf.Variable(tf.zeros(shape, dtype=tf.float32))

        if not isinstance(self.kernel_initializer, VarianceScaling):
            raise ValueError(f'Invalid kernel_initializer: {self.kernel_initializer}')

        fans = {
            'fan_in':  self.fan_in, 
            'fan_out': self.fan_out, 
            'fan_avg': self.fan_avg,
            }
        
        fan = fans[self.kernel_initializer.mode]
        match self.kernel_initializer.distribution:
            case 'normal':
                std = tf.sqrt(self.kernel_initializer.scale / fan)
                return tf.Variable(tf.random.normal(mean=0, stddev=std, shape=shape))
            case 'uniform':
                x = tf.sqrt(self.kernel_initializer.scale / fan)
                return tf.Variable(tf.random.uniform(minval=-x, maxval=x, shape=shape))

    @tf.function
    def _LeakyReLU_PReLU(self, raw_res):
        return tf.where(raw_res <= 0, raw_res * self.activation.alpha, raw_res)
    
    @tf.function
    def _Swish(self, raw_res):
        return raw_res * sigmoid(self.activation.beta * raw_res)

    def _activation_func(self, activation, raw_res):
        match activation:
            case 'relu':    return relu(raw_res)
            # case 'relu':    return tf.nn.relu(raw_res)
            case "elu":     return elu(raw_res) 
            case "selu":    return selu(raw_res) 
            case "gelu":    return gelu(raw_res)
            case "swish":   return swish(raw_res)
            # case 'tanh':    return tanh(raw_res)
            case 'tanh':    return tf.nn.tanh(raw_res)
            case 'sigmoid': return sigmoid(raw_res)
            # case 'softmax': return softmax(raw_res)
            case 'softmax': return tf.nn.softmax(raw_res, axis=-1)
            case None:      return raw_res
            case _:
                if isinstance(self.activation, (LeakyReLU, PReLU)):
                    return self._LeakyReLU_PReLU(raw_res) 
                
                if isinstance(self.activation, Swish):
                    return self._Swish(raw_res)
                
                raise ValueError('Invalid activation')
    
    def __repr__(self):
        return f'Custom {self.__class__.__name__} Layer ({self.name})'

class Input(Layer):
    def __init__(self, shape=None, batch_size=None, batch_shape=None, **kwargs):
        super().__init__(**kwargs)
        shape = tuple(shape or []) 
        self.shape = shape
        self.output = shape
        self.is_build = True
        self.trainable = False
        self.prev_layer = None
        self.batch_shape = batch_shape if hasattr(batch_shape, '__iter__') else [batch_shape]
        self.batch_size = batch_size or self.batch_shape[0]

    def call(self, X, training=False):
        X = tf.convert_to_tensor(X)

        if X.shape[-1] is None and X.shape[-1] != self.shape[-1]:
            raise ValueError(f'Invaluid input shape in Input layer: expect: {self.shape}; recive: {X.shape[1:]}')
        
        return X


class Dense(Layer):
    def __init__(self, units=None, activation=None, kernel_initializer='glorot_uniform', 
                 kernel_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.output = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        
        if self.kernel_regularizer is not None:
            if not isinstance(self.kernel_regularizer, RegularizationBase):
                raise TypeError('Invalid regularizer')
            
            self.kernel_regularizer.layer = self

    def __call__(self, layer, *args, **kwds):
        super().__call__(layer)

        if isinstance(layer, ActivationFuncBase):
            self.activation = layer
            layer = layer.prev_layer

        self.input   = layer.output
        inp_neurons  = self.input[-1] if isinstance(self.input, (list, tuple)) else self.input
        self.fan_in  = inp_neurons
        self.fan_out = self.output
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        self.bias = tf.Variable(tf.zeros([self.units]))
        shape = (int(inp_neurons), int(self.units))
        self.weights = self._weights_init(self.kernel_initializer, shape)

        self._trainable_variables.extend([self.weights, self.bias])
        return self

    @tf.function(jit_compile=True)
    def calculations(self, X):
        raw_res = tf.matmul(X, self.weights) + self.bias
        return self._activation_func(self.activation, raw_res)
    
    def call(self, X, training=False):
        if isinstance(X, tuple):
            X = X[0]

        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        if not self.is_build:
            self.__call__(Input(X.shape[1:]))

        return self.calculations(X)

    def get_weights(self):
        return [self.weights, self.bias]
    
    def set_weights(self, weights, bias):
        if self.weights.shape != weights.shape or \
            self.bias.shape != bias.shape:
            raise ValueError('Invalid shape on weights or bias')
        
        self.weights.assign(weights)
        self.bias.assign(bias)


class LSTM(Layer):
    def __init__(self, 
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 return_sequences=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias,
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.supports_masking = True

    def __call__(self, layer):
        super().__call__(layer)

        if len(layer.output) != 2:
            raise ValueError('Invalid input shape SimpleRNN expect 3D Tensor')

        self.input = layer.output
        self.output = [self.input[0], self.units] if self.return_sequences else self.units
        
        self.fan_in  = self.input[1]
        self.fan_out = self.units
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        kernel_shape = (int(layer.output[1]), int(self.units))
        self.weights = self._weights_init(self.kernel_initializer, kernel_shape)
        self.forget_weights = self._weights_init(self.kernel_initializer, kernel_shape)
        self.input_weights  = self._weights_init(self.kernel_initializer, kernel_shape)
        self.output_weights = self._weights_init(self.kernel_initializer, kernel_shape)

        recurrent_shape = (int(self.units), int(self.units))
        self.recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.forget_recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.input_recurrent_weights  = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.output_recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)

        self._trainable_variables.extend([self.weights, self.recurrent_weights,
                                          self.input_weights, self.input_recurrent_weights,
                                          self.output_weights, self.output_recurrent_weights,
                                          self.forget_weights, self.forget_recurrent_weights])

        if self.use_bias: 
            bias_shape = [self.units]
            self.bias = self._weights_init(self.bias_initializer, bias_shape)
            self.forget_bias = self._weights_init(self.bias_initializer, bias_shape)
            self.input_bias  = self._weights_init(self.bias_initializer, bias_shape)
            self.output_bias = self._weights_init(self.bias_initializer, bias_shape)
            self._trainable_variables.extend([self.bias, self.forget_bias, self.input_bias, self.output_bias])

        return self
    
    def _weighted_sum(self, W, inp, rec_W, state, b, activation):
        kernel_weighted_sum    = tf.matmul(inp, W)
        recurrenr_weighted_sum = tf.matmul(state, rec_W)
        out = kernel_weighted_sum + recurrenr_weighted_sum 

        if self.use_bias:
            out += b

        return self._activation_func(activation, out)

    def call(self, X, mask=None, training=False):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        batch, time_steps = tf.shape(X)[0], tf.shape(X)[1]

        ta = tf.TensorArray(dtype=X.dtype, size=time_steps)
        short_term = tf.zeros([batch, self.units], X.dtype)
        long_term  = tf.zeros([batch, self.units], X.dtype)

        t = tf.constant(0)

        def cond(t, short_term, long_term, ta):
            return t < time_steps

        def body(t, short_term, long_term, ta):
            time_inp = X[:, t, :]

            forget_gate = self._weighted_sum(
                self.forget_weights, time_inp,
                self.forget_recurrent_weights, short_term,
                self.forget_bias, self.recurrent_activation
            )

            input_gate = self._weighted_sum(
                self.input_weights, time_inp,
                self.input_recurrent_weights, short_term,
                self.input_bias, self.recurrent_activation
            )

            output_gate = self._weighted_sum(
                self.output_weights, time_inp,
                self.output_recurrent_weights, short_term,
                self.output_bias, self.recurrent_activation
            )

            simple = self._weighted_sum(
                self.weights, time_inp,
                self.recurrent_weights, short_term,
                self.bias, self.activation
            )

            simple = simple * input_gate
            long_update  = (long_term * forget_gate) + simple
            short_update = self._activation_func(self.activation, long_update) * output_gate

            if mask is not None:
                time_mask = tf.cast(mask[:, t:t+1], long_term.dtype)
                long_term_new  = long_update  * time_mask + long_term  * (1 - time_mask)
                short_term_new = short_update * time_mask + short_term * (1 - time_mask)
            else:
                long_term_new  = long_update
                short_term_new = short_update

            if self.return_sequences:
                ta = ta.write(t, short_term_new)

            return t + 1, short_term_new, long_term_new, ta

        _, short_term, long_term, ta = tf.while_loop(
            cond,
            body,
            loop_vars=[t, short_term, long_term, ta]
        )

        # for time_st in tf.range(self.input[0]):
        #     time_inp = X[:, time_st, :]

        #     forget_gate = self._weighted_sum(self.forget_weights, time_inp,
        #                                      self.forget_recurrent_weights, short_term,
        #                                      self.forget_bias, self.recurrent_activation)
            
        #     input_gate  = self._weighted_sum(self.input_weights, time_inp,
        #                                      self.input_recurrent_weights, short_term,
        #                                      self.input_bias, self.recurrent_activation)
            
        #     output_gate = self._weighted_sum(self.output_weights, time_inp,
        #                                      self.output_recurrent_weights, short_term,
        #                                      self.output_bias, self.recurrent_activation)

        #     simple      = self._weighted_sum(self.weights, time_inp, 
        #                                      self.recurrent_weights, short_term,
        #                                      self.bias, self.activation)

        #     simple = simple * input_gate
        #     long_update  = (long_term * forget_gate) + simple
        #     short_update = self._activation_func(self.activation, long_update) * output_gate
            
        #     if mask is not None:
        #         time_mask = tf.cast(mask[:, time_st:time_st+1], long_term.dtype)
        #         long_term  = long_update  * time_mask + long_term  * (1 - time_mask)
        #         short_term = short_update * time_mask + short_term * (1 - time_mask)
        #     else:
        #         long_term  = long_update
        #         short_term = short_update

        #     if self.return_sequences:
        #         ta = ta.write(time_st, short_term)
        
        output = short_term
        if self.return_sequences:
            output = ta.stack()
            output = tf.transpose(output, [1, 0, 2])

        if mask is None or not self.next_layer.supports_masking:
            return output

        return output, self.compute_mask(X, mask)

    def get_weights(self):
        weights = [self.weights, self.recurrent_weights,
                   self.input_weights, self.input_recurrent_weights,
                   self.output_weights, self.output_recurrent_weights,
                   self.forget_weights, self.forget_recurrent_weights]
        
        if self.use_bias:
            weights.extend([self.bias, self.input_bias, 
                            self.output_bias, self.forget_bias])

        return weights

    def set_weights(self, weights, recurrent_weights,
                   input_weights, input_recurrent_weights,
                   output_weights, output_recurrent_weights,
                   forget_weights, forget_recurrent_weights,
                   bias, input_bias, output_bias, forget_bias):
        
        if self.weights.shape != weights.shape or \
           self.input_weights.shape != input_weights.shape or \
           self.output_weights.shape != output_weights.shape or \
           self.forget_weights.shape != forget_weights.shape or \
           self.recurrent_weights.shape != recurrent_weights.shape or \
           self.input_recurrent_weights.shape != input_recurrent_weights.shape or \
           self.output_recurrent_weights.shape != output_recurrent_weights.shape or \
           self.forget_recurrent_weights.shape != forget_recurrent_weights.shape or \
           self.bias.shape != bias.shape or \
           self.input_bias.shape != input_bias.shape or \
           self.output_bias.shape != output_bias.shape or \
           self.forget_bias.shape != forget_bias.shape:
            raise ValueError('Invalid shape on weights or biases')

        self.weights.assign(weights)
        self.input_weights.assign(input_weights)
        self.output_weights.assign(output_weights)
        self.forget_weights.assign(forget_weights)
        self.recurrent_weights.assign(recurrent_weights)
        self.input_recurrent_weights.assign(input_recurrent_weights)
        self.output_recurrent_weights.assign(output_recurrent_weights)
        self.forget_recurrent_weights.assign(forget_recurrent_weights)
        self.bias.assign(bias)
        self.input_bias.assign(input_bias)
        self.output_bias.assign(output_bias)
        self.forget_bias.assign(forget_bias)

class GRU(Layer):
    def __init__(self, 
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 return_sequences=False,
                 stateful=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias,
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.stateful = stateful
        self.supports_masking = True

    def __call__(self, layer):
        super().__call__(layer)

        if len(layer.output) != 2:
            raise ValueError('Invalid input shape GRU expect 3D Tensor')

        self.input = layer.output
        self.output = [self.input[0], self.units] if self.return_sequences else self.units
        
        self.fan_in  = self.input[1]
        self.fan_out = self.units
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        kernel_shape = (int(layer.output[1]), int(self.units))
        self.weights   = self._weights_init(self.kernel_initializer, kernel_shape)
        self.r_weights = self._weights_init(self.kernel_initializer, kernel_shape)
        self.z_weights = self._weights_init(self.kernel_initializer, kernel_shape)

        recurrent_shape = (int(self.units), int(self.units))
        self.recurrent_weights   = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.r_recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.z_recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)

        self._trainable_variables.extend([self.weights, self.recurrent_weights,
                                          self.r_weights, self.r_recurrent_weights,
                                          self.z_weights, self.z_recurrent_weights])

        if self.use_bias: 
            bias_shape = [self.units]
            self.bias   = self._weights_init(self.bias_initializer, bias_shape)
            self.r_bias = self._weights_init(self.bias_initializer, bias_shape)
            self.z_bias = self._weights_init(self.bias_initializer, bias_shape)
            self._trainable_variables.extend([self.bias, self.r_bias, self.z_bias])

        self.hd_state = None
        if self.stateful:
            if self.batch_size is None:
                raise ValueError('RNN stateful required constant batch size')

            self.hd_state = tf.zeros([self.batch_size, self.units], tf.float32)

        return self
    
    def _weighted_sum(self, W, inp, rec_W, state, b, activation):
        kernel_weighted_sum    = tf.matmul(inp, W)
        recurrenr_weighted_sum = tf.matmul(state, rec_W)
        out = kernel_weighted_sum + recurrenr_weighted_sum 

        if self.use_bias:
            out += b

        return self._activation_func(activation, out)

    def call(self, X, mask=None, training=False):
        if isinstance(X, tf.RaggedTensor):
            X = X.to_tensor(default_value=0)
            mask = tf.reduce_any(tf.not_equal(X, 0), axis=-1)
        else:
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        batch, time_steps = tf.shape(X)[0], tf.shape(X)[1]
        
        hd_state = self.hd_state 
        if hd_state is None:
            hd_state = tf.zeros([batch, self.units], tf.float32)

        ta = tf.TensorArray(dtype=X.dtype, size=time_steps)
        for time_st in tf.range(time_steps):
            time_inp = X[:, time_st, :]
   
            r_gate = self._weighted_sum(self.r_weights, time_inp,
                                        self.r_recurrent_weights, hd_state,
                                        self.r_bias, self.recurrent_activation)
            
            z_gate = self._weighted_sum(self.z_weights, time_inp,
                                        self.z_recurrent_weights, hd_state,
                                        self.z_bias, self.recurrent_activation)
            
            simple = self._weighted_sum(self.weights, time_inp, 
                                        self.recurrent_weights, hd_state * r_gate,
                                        self.bias, self.activation)

            hd_update = (hd_state * z_gate) + (simple * (1 - z_gate))
            if mask is not None:
                time_mask = tf.cast(mask[:, time_st:time_st+1], hd_state.dtype)
                hd_state = hd_update * time_mask + hd_state * (1 - time_mask)
            else:
                hd_state = hd_update

            if self.return_sequences:
                ta = ta.write(time_st, hd_state)
        
        output = hd_state
        if self.return_sequences:
            output = ta.stack()
            output = tf.transpose(output, [1, 0, 2])

        if mask is None or self.next_layer is None or not self.next_layer.supports_masking:
            return output

        return output, self.compute_mask(X, mask)

    def reset_states(self):
        if self.hd_state is not None:
            self.hd_state = tf.zeros([self.batch_size, self.units], tf.float32)

    def get_weights(self):
        weights = [self.weights, self.recurrent_weights,
                   self.r_weights, self.r_recurrent_weights,
                   self.z_weights, self.z_recurrent_weights]
        
        if self.use_bias:
            weights.extend([self.bias, self.r_bias, self.z_bias])

        return weights

    def set_weights(self, weights, recurrent_weights,
                   r_weights, r_recurrent_weights,
                   z_weights, z_recurrent_weights,
                   bias, r_bias, z_bias):
        
        if self.weights.shape != weights.shape or \
           self.r_weights.shape != r_weights.shape or \
           self.z_weights.shape != z_weights.shape or \
           self.recurrent_weights.shape != recurrent_weights.shape or \
           self.r_recurrent_weights.shape != r_recurrent_weights.shape or \
           self.z_recurrent_weights.shape != z_recurrent_weights.shape or \
           self.bias.shape != bias.shape or \
           self.r_bias.shape != r_bias.shape or \
           self.z_bias.shape != z_bias.shape:
            raise ValueError('Invalid shape on weights or biases')

        self.weights.assign(weights)
        self.r_weights.assign(r_weights)
        self.z_weights.assign(z_weights)
        self.recurrent_weights.assign(recurrent_weights)
        self.r_recurrent_weights.assign(r_recurrent_weights)
        self.z_recurrent_weights.assign(z_recurrent_weights)
        self.bias.assign(bias)
        self.r_bias.assign(r_bias)
        self.z_bias.assign(z_bias)

class SimpleRNN(Layer):
    def __init__(self, units, activation='tanh', use_bias=True,
                 kernel_initializer='glorot_uniform', 
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 return_sequences=False,
                 name=None):
        
        super().__init__(name) 
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.supports_masking = True

    def __call__(self, layer):
        super().__call__(layer)

        if len(layer.output) != 2:
            raise ValueError('Invalid input shape SimpleRNN expect 3D Tensor')

        self.input = layer.output
        self.output = [self.input[0], self.units] if self.return_sequences else self.units
        
        self.fan_in  = self.input[1]
        self.fan_out = self.units
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        kernel_shape = (int(layer.output[1]), int(self.units))
        self.weights = self._weights_init(self.kernel_initializer, kernel_shape)

        recurrent_shape = (int(self.units), int(self.units))
        self.recurrenr_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)

        self._trainable_variables.extend([self.weights, self.recurrenr_weights])

        if self.use_bias:
            self.bias = tf.Variable(tf.zeros([self.units]))
            self._trainable_variables.append(self.bias)

        return self
    
    def _weighted_sum(self, W, inp, rec_W, state, b, activation):
        kernel_weighted_sum    = tf.matmul(inp, W)
        recurrenr_weighted_sum = tf.matmul(state, rec_W)
        out = kernel_weighted_sum + recurrenr_weighted_sum 

        if self.use_bias:
            out += b

        return self._activation_func(activation, out)

    def call(self, X, mask=None, training=False):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        time_steps = tf.shape(X)[1]

        ta = tf.TensorArray(dtype=X.dtype, size=time_steps)
        hd_state = tf.zeros([X.shape[0], self.units], tf.float32)

        for time_st in tf.range(time_steps):
            time_inp = X[:, time_st, :]

            if training and self.dropout > 0:
                time_inp = self._dropout(time_inp, self.dropout)

            if training and self.recurrent_dropout > 0:
                hd_state = self._dropout(hd_state, self.recurrent_dropout)

            hd_update = self._weighted_sum(self.weights, time_inp, 
                                           self.recurrenr_weights, hd_state, 
                                           self.bias, self.activation)
            
            if mask is not None:
                time_mask = tf.cast(mask[:, time_st:time_st+1], hd_state.dtype)
                hd_state = hd_update * time_mask + hd_state * (1 - time_mask)
            else:
                hd_state = hd_update

            if self.return_sequences:
                ta = ta.write(time_st, hd_state)
        
        output = hd_state
        if self.return_sequences:
            output = ta.stack()
            output = tf.transpose(output, [1, 0, 2])

        if mask is None or not self.next_layer.supports_masking:
            return output

        return output, self.compute_mask(X, mask)

    def get_weights(self):
        weights = [self.weights, self.recurrenr_weights]
        if self.use_bias:
            weights.append(self.bias)

        return weights
    
    def set_weights(self, weights, recurrenr_weights, bias):
        if self.weights.shape != weights.shape or \
            self.recurrenr_weights.shape != recurrenr_weights.shape or \
            self.bias.shape != bias.shape:
            raise ValueError('Invalid shape on weights or recurrenr_weights or bias')
        
        self.weights.assign(weights)
        self.recurrenr_weights.assign(recurrenr_weights)
        self.bias.assign(bias)


class Bidirectional(Layer):
    def __init__(self, 
                 layer,
                 merge_mode='concat',
                 backward_layer=None, 
                 **kwargs):
        
        super().__init__(**kwargs)

        self.layer = layer
        self.merge_mode = merge_mode
        self.backward_layer = backward_layer
        self.supports_masking = True

    def __call__(self, layer):
        super().__call__(layer)

        self.layer = self.layer(layer)
        layer.next_layer.pop()
        
        self.input  = layer.output
        self.output = self.layer.output

        if not isinstance(self.output, (list, tuple)):
            self.output = [self.output] 

        if self.backward_layer is None:
            self.backward_layer = deepcopy(self.layer)
            self.output[-1] *= 2
        else:
            self.backward_layer = self.backward_layer(layer)
            layer.next_layer.pop()
            self.output[-1] += self.backward_layer.output[-1]

        if len(layer.next_layer) == 1:
            layer.next_layer = layer.next_layer[0]

        self._trainable_variables.extend(self.layer._trainable_variables)
        self._trainable_variables.extend(self.backward_layer._trainable_variables)

        return self

    def call(self, X, mask=None, training=False):
        X = tf.convert_to_tensor(X)

        if self.layer.supports_masking:
            forward_out = self.layer.call(X, mask=mask, training=training)
        else:
            forward_out = self.layer.call(X, training=training)
        
        X_backward = tf.reverse(X, axis=[1])

        if self.backward_layer.supports_masking:
            backward_mask = mask if mask is None else tf.reverse(mask, axis=[1])
            backward_out = self.backward_layer.call(X_backward, mask=backward_mask, training=training)
        else:
            backward_out = self.backward_layer.call(X_backward, training=training)

        if self.backward_layer.return_sequences == True:
            backward_out = tf.reverse(backward_out, axis=[1])

        concat_out = tf.concat([forward_out, backward_out], axis=-1)

        if self.next_layer.supports_masking:
            return concat_out, mask

        return concat_out

class Attention(Layer):
    def __init__(self,
                 use_scale=False,
                 score_mode='dot',
                 **kwargs):
        
        super().__init__(**kwargs)
        self.use_scale  = use_scale
        self.score_mode = score_mode
        self.support_masking = True
        self.inputs = []

    @staticmethod
    def _input_unpack(inputs):
        if len(inputs) == 2:
            query, value = inputs
            key = value
        elif len(inputs) == 3:
            query, value, key = inputs
        else:
            raise ValueError('Invalid call inputs; expect [query, value, key] or [query, (value, key)]')

        return query, value, key

    def __call__(self, layer):
        super().__call__(layer)

        self.n_expected_inputs = len(layer)
        query, value, key = self._input_unpack(layer)

        self.input  = [item.output for item in (query, value, key)]
        self.output = query.output

        if self.score_mode == 'concat':
            self.dense = Dense(1)(Input([query.output[-1] + key.output[-1]]))
            self._trainable_variables.extend(self.dense._trainable_variables)

        return self

    def call(self, inputs, mask=None, training=False): # [decoder_outputs: query, encoder_outputs: value, ? :key] default value = key
        if not isinstance(inputs, list):
            inputs = [tf.cast(inputs, tf.float32)]

        self.inputs.extend(inputs)
        if len(self.inputs) != self.n_expected_inputs:
            return True
        
        query, value, key = self._input_unpack(self.inputs)

        if self.score_mode == 'dot':
            score = tf.matmul(query, key, transpose_b=True)

            if self.use_scale:
                f = tf.cast(tf.shape(query)[-1], tf.float32)
                score /= tf.sqrt(f)

        elif self.score_mode == 'concat':
            query_ext = tf.expand_dims(query, axis=2)
            key_ext   = tf.expand_dims(key, axis=1)

            query_ext = tf.tile(query_ext, [1, 1, tf.shape(key_ext)[2], 1])
            key_ext   = tf.tile(key_ext, [1, tf.shape(query_ext)[1], 1, 1])

            concat = tf.concat([query_ext, key_ext], axis=-1)

            score = self.dense.call(tf.nn.tanh(concat))
            score = tf.squeeze(score, axis=-1)

        else:
            raise NotImplemented()
        
        if mask is not None:
            if isinstance(mask, (list, tuple)):
                mask = mask[0]

            mask = tf.cast(mask, score.dtype)
            mask = mask[:, tf.newaxis, :]
            score += (1.0 - mask) * -1e9 # -1e9 = -1 000 000 000 -> softmax -> 0
            
        softmax = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(softmax, value)

        self.inputs = []
        return context

class AdditiveAttention(Attention):
    def __init__(self, use_scale=False, **kwargs):
        super().__init__(use_scale, 'concat', **kwargs)

class TimeDistributed(Layer):
    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer


    def __call__(self, layer):
        super().__call__(layer)

        self.input = layer.output
        self.layer = self.layer(Input(self.input[1:]))
        self.output  = [self.input[0]]  
        layer_output = self.layer.output 

        if isinstance(layer_output, list):
            self.output.extend(layer_output)
        else:
            self.output.append(layer_output)

        self._trainable_variables.extend(self.layer._trainable_variables)
        return self

    def call(self, X, training=False):
        batch, time = tf.shape(X)[0], tf.shape(X)[1]

        flat_shape = tf.concat([[batch * time], tf.shape(X)[2:]], axis=0)
        X_flat = tf.reshape(X, flat_shape)
        y_flat = self.layer.call(X_flat)  
        
        y_shape = tf.concat([[batch, time], tf.shape(y_flat)[1:]], axis=0)
        y = tf.reshape(y_flat, y_shape)
        return y

class RNN(Layer):
    def __init__(self, cell, return_sequences=False, dropout=0, recurrent_dropout=0, **kwargs):
        super().__init__(**kwargs)

        if not (hasattr(cell, 'state_size') and hasattr(cell, 'output_size')):
            raise TypeError('RNN Layer recive invalid cell; Cell must define a `state_size` and `output_size` attributes')
        
        self.cell = cell
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.supports_masking = True

    def __call__(self, layer):
        super().__call__(layer)

        if len(layer.output) != 2:
            raise ValueError('Invalid input shape SimpleRNN expect 3D Tensor')

        self.input = layer.output
        self.output = [self.input[0], self.cell.output_size] if self.return_sequences else self.cell.output_size
        self.cell = self.cell(layer)

        self._trainable_variables.extend(self.cell._trainable_variables)
        return self
    
    def call(self, X, mask=None, training=False):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        batches, time_steps = tf.shape(X)[0], tf.shape(X)[1]
        ta = tf.TensorArray(dtype=X.dtype, size=time_steps)

        hd_states_init = self.cell.state_size 
        if not isinstance(hd_states_init, (list, tuple)):
            hd_states_init = [hd_states_init]

        hd_state = [tf.zeros([batches, size], dtype=X.dtype) for size in hd_states_init]
        out = tf.keras.ops.empty((batches, self.cell.output_size))

        for time_st in tf.range(time_steps):
            time_inp = X[:, time_st, :]

            if training and self.dropout > 0:
                time_inp = self._dropout(time_inp, self.dropout)

            if training and self.recurrent_dropout > 0:
                hd_state = [self._dropout(hds, self.recurrent_dropout) for hds in hd_state]

            out_update, hd_update = self.cell.call(time_inp, hd_state, training)

            if mask is not None:
                time_mask = tf.cast(mask[:, time_st:time_st+1], X.dtype)
                out_update = out_update * time_mask + out * (1 - time_mask) 
                hd_state = [update * time_mask + state * (1 - time_mask) for update, state in zip(hd_update, hd_state)]
            else:
                hd_state = hd_update
                out = out_update

            if self.return_sequences:
                ta = ta.write(time_st, out)
        
        output = out
        if self.return_sequences:
            output = ta.stack()
            output = tf.transpose(output, [1, 0, 2])

        if mask is None or not self.next_layer.supports_masking:
            return output

        return output, self.compute_mask(X, mask)
    
    def get_weights(self):
        return self.cell.get_weights()
    
    def set_weights(self, *weights):
        self.cell.set_weights(*weights)

class LSTMCell(Layer):
    def __init__(self, 
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 return_sequences=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias,
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences

        self.state_size = [units, units]
        self.output_size = units

    def __call__(self, layer):
        super().__call__(layer)

        if len(layer.output) != 2:
            raise ValueError('Invalid input shape SimpleRNN expect 3D Tensor')

        self.input = layer.output
        self.output = [self.input[0], self.units] if self.return_sequences else self.units
        
        self.fan_in  = self.input[1]
        self.fan_out = self.units
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        kernel_shape = (int(layer.output[1]), int(self.units))
        self.weights = self._weights_init(self.kernel_initializer, kernel_shape)
        self.forget_weights = self._weights_init(self.kernel_initializer, kernel_shape)
        self.input_weights  = self._weights_init(self.kernel_initializer, kernel_shape)
        self.output_weights = self._weights_init(self.kernel_initializer, kernel_shape)

        recurrent_shape = (int(self.units), int(self.units))
        self.recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.forget_recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.input_recurrent_weights  = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.output_recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)

        self._trainable_variables.extend([self.weights, self.recurrent_weights,
                                          self.input_weights, self.input_recurrent_weights,
                                          self.output_weights, self.output_recurrent_weights,
                                          self.forget_weights, self.forget_recurrent_weights])

        if self.use_bias: 
            bias_shape = [self.units]
            self.bias = self._weights_init(self.bias_initializer, bias_shape)
            self.forget_bias = self._weights_init(self.bias_initializer, bias_shape)
            self.input_bias  = self._weights_init(self.bias_initializer, bias_shape)
            self.output_bias = self._weights_init(self.bias_initializer, bias_shape)
            self._trainable_variables.extend([self.bias, self.forget_bias, self.input_bias, self.output_bias])

        return self

    def _weighted_sum(self, W, inp, rec_W, state, b, activation):
        kernel_weighted_sum    = tf.matmul(inp, W)
        recurrenr_weighted_sum = tf.matmul(state, rec_W)
        out = kernel_weighted_sum + recurrenr_weighted_sum 

        if self.use_bias:
            out += b

        return self._activation_func(activation, out)

    def call(self, X, hd_state, training=False):
        if not isinstance(hd_state, list):
            raise TypeError(f'hd_state must be list not: {type(hd_state)}')

        short_term, long_term = hd_state

        forget_gate = self._weighted_sum(self.forget_weights, X,
                                         self.forget_recurrent_weights, short_term,
                                         self.forget_bias, self.recurrent_activation)
        
        input_gate  = self._weighted_sum(self.input_weights, X,
                                         self.input_recurrent_weights, short_term,
                                         self.input_bias, self.recurrent_activation)
        
        output_gate = self._weighted_sum(self.output_weights, X,
                                         self.output_recurrent_weights, short_term,
                                         self.output_bias, self.recurrent_activation)

        simple      = self._weighted_sum(self.weights, X, 
                                         self.recurrent_weights, short_term,
                                         self.bias, self.activation)

        simple = simple * input_gate
        long_term = (long_term * forget_gate) + simple
        short_term = self._activation_func(self.activation, long_term) * output_gate

        return short_term, [short_term, long_term]

    def get_weights(self):
        weights = [self.weights, self.recurrent_weights,
                   self.input_weights, self.input_recurrent_weights,
                   self.output_weights, self.output_recurrent_weights,
                   self.forget_weights, self.forget_recurrent_weights]
        
        if self.use_bias:
            weights.extend([self.bias, self.input_bias, 
                            self.output_bias, self.forget_bias])

        return weights

    def set_weights(self, weights, recurrent_weights,
                   input_weights, input_recurrent_weights,
                   output_weights, output_recurrent_weights,
                   forget_weights, forget_recurrent_weights,
                   bias, input_bias, output_bias, forget_bias):
        
        if self.weights.shape != weights.shape or \
           self.input_weights.shape != input_weights.shape or \
           self.output_weights.shape != output_weights.shape or \
           self.forget_weights.shape != forget_weights.shape or \
           self.recurrent_weights.shape != recurrent_weights.shape or \
           self.input_recurrent_weights.shape != input_recurrent_weights.shape or \
           self.output_recurrent_weights.shape != output_recurrent_weights.shape or \
           self.forget_recurrent_weights.shape != forget_recurrent_weights.shape or \
           self.bias.shape != bias.shape or \
           self.input_bias.shape != input_bias.shape or \
           self.output_bias.shape != output_bias.shape or \
           self.forget_bias.shape != forget_bias.shape:
            raise ValueError('Invalid shape on weights or biases')

        self.weights.assign(weights)
        self.input_weights.assign(input_weights)
        self.output_weights.assign(output_weights)
        self.forget_weights.assign(forget_weights)
        self.recurrent_weights.assign(recurrent_weights)
        self.input_recurrent_weights.assign(input_recurrent_weights)
        self.output_recurrent_weights.assign(output_recurrent_weights)
        self.forget_recurrent_weights.assign(forget_recurrent_weights)
        self.bias.assign(bias)
        self.input_bias.assign(input_bias)
        self.output_bias.assign(output_bias)
        self.forget_bias.assign(forget_bias)

class GRUCell(Layer):
    def __init__(self, 
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 return_sequences=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias,
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences

        self.state_size  = units
        self.output_size = units

    def __call__(self, layer):
        super().__call__(layer)

        if len(layer.output) != 2:
            raise ValueError('Invalid input shape SimpleRNN expect 3D Tensor')

        self.input = layer.output
        self.output = [self.input[0], self.units] if self.return_sequences else self.units
        
        self.fan_in  = self.input[1]
        self.fan_out = self.units
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        kernel_shape = (int(layer.output[1]), int(self.units))
        self.weights   = self._weights_init(self.kernel_initializer, kernel_shape)
        self.r_weights = self._weights_init(self.kernel_initializer, kernel_shape)
        self.z_weights = self._weights_init(self.kernel_initializer, kernel_shape)

        recurrent_shape = (int(self.units), int(self.units))
        self.recurrent_weights   = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.r_recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)
        self.z_recurrent_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)

        self._trainable_variables.extend([self.weights, self.recurrent_weights,
                                          self.r_weights, self.r_recurrent_weights,
                                          self.z_weights, self.z_recurrent_weights])

        if self.use_bias: 
            bias_shape = [self.units]
            self.bias   = self._weights_init(self.bias_initializer, bias_shape)
            self.r_bias = self._weights_init(self.bias_initializer, bias_shape)
            self.z_bias = self._weights_init(self.bias_initializer, bias_shape)
            self._trainable_variables.extend([self.bias, self.r_bias, self.z_bias])

        return self
    
    def _weighted_sum(self, W, inp, rec_W, state, b, activation):
        kernel_weighted_sum    = tf.matmul(inp, W)
        recurrenr_weighted_sum = tf.matmul(state, rec_W)
        out = kernel_weighted_sum + recurrenr_weighted_sum 

        if self.use_bias:
            out += b

        return self._activation_func(activation, out)

    def call(self, X, hd_state, training=False):
        if not isinstance(hd_state, list):
            raise TypeError(f'hd_state must be list not: {type(hd_state)}')

        hd_state = hd_state[0]

        r_gate = self._weighted_sum(self.r_weights, X,
                                    self.r_recurrent_weights, hd_state,
                                    self.r_bias, self.recurrent_activation)
        
        z_gate = self._weighted_sum(self.z_weights, X,
                                    self.z_recurrent_weights, hd_state,
                                    self.z_bias, self.recurrent_activation)
        
        simple = self._weighted_sum(self.weights, X, 
                                    self.recurrent_weights, hd_state * r_gate,
                                    self.bias, self.activation)

        hd_state = (hd_state * z_gate) + (simple * (1 - z_gate))

        return hd_state, [hd_state]

    def get_weights(self):
        weights = [self.weights, self.recurrent_weights,
                   self.r_weights, self.r_recurrent_weights,
                   self.z_weights, self.z_recurrent_weights]
        
        if self.use_bias:
            weights.extend([self.bias, self.r_bias, self.z_bias])

        return weights

    def set_weights(self, weights, recurrent_weights,
                   r_weights, r_recurrent_weights,
                   z_weights, z_recurrent_weights,
                   bias, r_bias, z_bias):
        
        if self.weights.shape != weights.shape or \
           self.r_weights.shape != r_weights.shape or \
           self.z_weights.shape != z_weights.shape or \
           self.recurrent_weights.shape != recurrent_weights.shape or \
           self.r_recurrent_weights.shape != r_recurrent_weights.shape or \
           self.z_recurrent_weights.shape != z_recurrent_weights.shape or \
           self.bias.shape != bias.shape or \
           self.r_bias.shape != r_bias.shape or \
           self.z_bias.shape != z_bias.shape:
            raise ValueError('Invalid shape on weights or biases')

        self.weights.assign(weights)
        self.r_weights.assign(r_weights)
        self.z_weights.assign(z_weights)
        self.recurrent_weights.assign(recurrent_weights)
        self.r_recurrent_weights.assign(r_recurrent_weights)
        self.z_recurrent_weights.assign(z_recurrent_weights)
        self.bias.assign(bias)
        self.r_bias.assign(r_bias)
        self.z_bias.assign(z_bias)

class SimpleRNNCell(Layer):
    def __init__(self, units, activation='tanh', use_bias=True,
                 kernel_initializer='glorot_uniform', 
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 return_sequences=False,
                 name=None):
        
        super().__init__(name) 
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.return_sequences = return_sequences

        self.state_size = units
        self.output_size = units

    def __call__(self, layer):
        super().__call__(layer)

        if len(layer.output) != 2:
            raise ValueError('Invalid input shape SimpleRNNCell expect 3D Tensor')

        self.input = layer.output
        self.output = [self.input[0], self.units] if self.return_sequences else self.units
        
        self.fan_in  = self.input[1]
        self.fan_out = self.units
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        kernel_shape = (int(layer.output[1]), int(self.units))
        self.weights = self._weights_init(self.kernel_initializer, kernel_shape)

        recurrent_shape = (int(self.units), int(self.units))
        self.recurrenr_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)

        self._trainable_variables.extend([self.weights, self.recurrenr_weights])

        if self.use_bias: 
            self.bias = self._weights_init(self.bias_initializer, [self.units])
            self._trainable_variables.append(self.bias)

        return self

    def call(self, X, hd_state, training=False):
        if not isinstance(hd_state, list):
            raise TypeError(f'hd_state must be list not: {type(hd_state)}')

        kernel_weighted_sum    = tf.matmul(X, self.weights)
        recurrenr_weighted_sum = tf.matmul(hd_state[0], self.recurrenr_weights)
        hd_state = kernel_weighted_sum + recurrenr_weighted_sum 

        if self.use_bias:
            hd_state += self.bias

        new_out = self._activation_func(self.activation, hd_state)
        return new_out, [new_out]
    
    def get_weights(self):
        weights = [self.weights, self.recurrenr_weights]
        if self.use_bias:
            weights.append(self.bias)

        return weights
    
    def set_weights(self, weights, recurrenr_weights, bias):
        if self.weights.shape != weights.shape or \
            self.recurrenr_weights.shape != recurrenr_weights.shape or \
            self.bias.shape != bias.shape:
            raise ValueError('Invalid shape on weights or recurrenr_weights or bias')
        
        self.weights.assign(weights)
        self.recurrenr_weights.assign(recurrenr_weights)
        self.bias.assign(bias)

class LNSimpleRNNCell(SimpleRNNCell):
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(units, activation=None, **kwargs)
        self.activation = activation
        self.ln = LayerNormalization()

    def __call__(self, layer):
        super().__call__(layer)
        self.ln = self.ln(Input([self.units]))
        self._trainable_variables.extend(self.ln._trainable_variables)

        return self

    def call(self, inputs, states, training=False):
        outputs, new_states = super().call(inputs, states)
        norm_outputs = self._activation_func(self.activation, self.ln.call(outputs))
        return norm_outputs, [norm_outputs]

class DropoutSimpleRNNCell(Layer):
    def __init__(self, units, activation='tanh', use_bias=True,
                 kernel_initializer='glorot_uniform', 
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 return_sequences=False,
                 name=None):
        
        super().__init__(name) 
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences

        self.state_size = units
        self.output_size = units

    def __call__(self, layer):
        super().__call__(layer)

        if len(layer.output) != 2:
            raise ValueError('Invalid input shape DropoutSimpleRNNCell expect 3D Tensor')

        self.input = layer.output
        self.output = [self.input[0], self.units] if self.return_sequences else self.units
        
        self.fan_in  = self.input[1]
        self.fan_out = self.units
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        kernel_shape = (int(layer.output[1]), int(self.units))
        self.weights = self._weights_init(self.kernel_initializer, kernel_shape)

        recurrent_shape = (int(self.units), int(self.units))
        self.recurrenr_weights = self._weights_init(self.recurrent_initializer, recurrent_shape)

        self._trainable_variables.extend([self.weights, self.recurrenr_weights])

        if self.use_bias: 
            self.bias = self._weights_init(self.bias_initializer, [self.units])
            self._trainable_variables.append(self.bias)

        return self

    def call(self, X, hd_state, training=False):
        if not isinstance(hd_state, list):
            raise TypeError(f'hd_state must be list not: {type(hd_state)}')
        
        hd_state = hd_state[0]

        if training and self.dropout > 0:
            X = self._dropout(X, self.dropout)

        if training and self.recurrent_dropout > 0:
            hd_state = self._dropout(hd_state, self.recurrent_dropout)

        kernel_weighted_sum    = tf.matmul(X, self.weights)
        recurrenr_weighted_sum = tf.matmul(hd_state, self.recurrenr_weights)
        hd_state = kernel_weighted_sum + recurrenr_weighted_sum 

        if self.use_bias:
            hd_state += self.bias

        new_out = self._activation_func(self.activation, hd_state)
        return new_out, [new_out]
    
    def get_weights(self):
        weights = [self.weights, self.recurrenr_weights]
        if self.use_bias:
            weights.append(self.bias)

        return weights
    
    def set_weights(self, weights, recurrenr_weights, bias):
        if self.weights.shape != weights.shape or \
            self.recurrenr_weights.shape != recurrenr_weights.shape or \
            self.bias.shape != bias.shape:
            raise ValueError('Invalid shape on weights or recurrenr_weights or bias')
        
        self.weights.assign(weights)
        self.recurrenr_weights.assign(recurrenr_weights)
        self.bias.assign(bias)


class InceptionModules(Layer):
    def __init__(self, n_filters, **kwargs):
        super().__init__(**kwargs)

        '''
                        Concatenate

        Conv2D 00 | Conv2D 01 | Conv2D 02 | Conv2D  03  
        None      | Conv2D 11 | Conv2D 12 | MaxPool 03  

                           Input
        '''

        default_conv = partial(Conv2D, 
                               padding='same',
                               activation='relu',
                               kernel_initializer='he_normal',
                               kernel_size=1)
        
        self.conv00 = default_conv(n_filters[0][0])

        self.conv01 = default_conv(n_filters[0][1], kernel_size=3)
        self.conv11 = default_conv(n_filters[1][1])

        self.conv02 = default_conv(n_filters[0][2], kernel_size=5)
        self.conv12 = default_conv(n_filters[1][2])

        self.conv03 = default_conv(n_filters[0][3])
        self.maxpool13 = MaxPool2D(pool_size=3, 
                                    strides=1, 
                                    padding='same')
        
        self.concat = Concatenate(axis=-1) 

    def __call__(self, layer):
        super().__call__(layer)

        next_layers = layer.next_layer

        self.input = layer.output

        out1 = self.conv00(layer)

        out2 = self.conv11(layer)
        out2 = self.conv01(out2)

        out3 = self.conv12(layer)
        out3 = self.conv02(out3)

        out4 = self.maxpool13(layer)
        out4 = self.conv03(out4)

        self.concat([out1, out2, out3, out4])

        self.output = self.concat.output

        for value in self.__dict__.values():
            if hasattr(value, '_trainable_variables'):
                self._trainable_variables.extend(value._trainable_variables)

        layer.next_layer = next_layers

        return self

    def call(self, X, *_):
        out1 = self.conv00.call(X)

        out2 = self.conv11.call(X)
        out2 = self.conv01.call(out2)

        out3 = self.conv12.call(X)
        out3 = self.conv02.call(out3)

        out4 = self.maxpool13.call(X)
        out4 = self.conv03.call(out4)

        return self.concat.call([out1, out2, out3, out4])

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', 
                 activation=None, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.padding = padding 
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)

        assert len(kernel_size) == 2, 'Invalid kernel_size'
        self.kernel_size = kernel_size

        if not isinstance(strides, tuple):
            strides = (strides, strides)
            
        assert len(strides) == 2, 'Invalid strides'
        self.strides = strides

    def _calc_output_shape(self):
        match self.padding:
            case 'valid':
                H_out = tf.cast(tf.math.floor((self.input[0] - self.kernel_size[0]) / self.strides[0]) + 1, tf.int32)
                W_out = tf.cast(tf.math.floor((self.input[1] - self.kernel_size[1]) / self.strides[1]) + 1, tf.int32)
            case 'same':
                H_out = tf.cast(tf.math.ceil(self.input[0] / self.strides[0]), tf.int32)
                W_out = tf.cast(tf.math.ceil(self.input[1] / self.strides[1]), tf.int32)
            case _:
                raise ValueError(f'Invalid padding: {self.padding}; expects: valid/same')
            
        self.output = [H_out, W_out, self.filters]

    def _calc_fans(self):
        kernel_size_prod = tf.reduce_prod(self.kernel_size)
        self.fan_in  = kernel_size_prod * self.input[-1]
        self.fan_out = kernel_size_prod * self.filters 
        self.fan_avg = (self.fan_in + self.fan_out) / 2

    def __call__(self, layer):
        super().__call__(layer)

        self.input = layer.output
        if len(self.input) < 3:
            raise ValueError('Conv2D layer expect 3D input shape')
        
        self._calc_fans()
        self._calc_output_shape()
        
        self.bias = tf.Variable(tf.zeros([self.filters]))
        shape = [*self.kernel_size, self.input[-1], self.filters]
        self.weights = self._weights_init(self.kernel_initializer, shape)
        self._trainable_variables.extend([self.weights, self.bias])
        return self

    def _calc_padding_1d(self, mode):
        inx = int(mode == 'W')  
        N, K, S = self.input[inx], self.kernel_size[inx], self.strides[inx] 
        
        out = tf.math.ceil(N / S)
        P_total = tf.maximum((out - 1) * S + K - N, 0)

        pad_left = P_total // 2
        pad_right = P_total - pad_left

        return pad_left, pad_right
    
    def _add_padding(self, X):
        pad_top, pad_bottom = self._calc_padding_1d('H')
        pad_left, pad_right = self._calc_padding_1d('W')

        paddings = [
            [0, 0],
            [pad_top, pad_bottom],
            [pad_left, pad_right],
            [0, 0]
        ]

        return tf.pad(X, paddings)

    def get_weights(self):
        return [self.weights, self.bias]

    def set_weights(self, weights, bias):
        if self.weights.shape != weights.shape or \
            self.bias.shape != bias.shape:
            raise ValueError('Invalid shape on weights or bias')
        
        self.weights.assign(weights)
        self.bias.assign(bias)
    
    def _calc(self, X):
        # ==================== Implementation 1 ====================

        out = tf.nn.conv2d(X, self.weights, self.strides, self.padding.upper())
        out = tf.nn.bias_add(out, self.bias)
        return self._activation_func(self.activation, out)

        # ==================== Implementation 2 ====================

        # H_out, W_out, C_out = self.output
        # B_in, C_in = tf.shape(X)[0], tf.shape(X)[-1]

        # out_rows = tf.TensorArray(dtype=X.dtype, size=H_out)
        # if self.padding == 'same':
        #     X = self._add_padding(X)

        # for i in tf.range(self.output[0]):
        #     row_start = i * self.strides[0]
        #     row_end   = row_start + self.kernel_size[0]

        #     out_cols = tf.TensorArray(dtype=X.dtype, size=W_out)

        #     for j in tf.range(self.output[1]):
        #         col_start = j * self.strides[1]
        #         col_end   = col_start + self.kernel_size[1]

        #         receptive_field = X[:, row_start:row_end, col_start:col_end, :]
        #         mul = receptive_field[..., tf.newaxis] * self.weights[tf.newaxis, ...]
        #         convolution = tf.reduce_sum(mul, axis=[1,2,3])  
        #         convolution += self.bias
        #         out_cols = out_cols.write(j, convolution)

        #     row = out_cols.stack()             
        #     row = tf.transpose(row, [1,0,2])   
        #     out_rows = out_rows.write(i, row)

        # out = out_rows.stack()             
        # out = tf.transpose(out, [1,0,2,3]) 
        # return self._activation_func(self.activation, out)

        # ==================== Implementation 3 ====================

        # patches = tf.image.extract_patches(
        #     images=X,
        #     sizes=[1, *self.kernel_size, 1],    
        #     strides=[1, *self.strides, 1],        
        #     rates=[1, 1, 1, 1],                                      
        #     padding=self.padding.upper()               
        # )

        # kernel_prod  = tf.reduce_prod(self.kernel_size)
        # patches_flat = tf.reshape(patches, [B_in * H_out * W_out, kernel_prod * C_in])
        # weights_flat = tf.reshape(self.weights, [kernel_prod * C_in, C_out])
        # out_flat     = tf.matmul(patches_flat, weights_flat) + self.bias
        # out          = tf.reshape(out_flat, [B_in, H_out, W_out, C_out])
        # return self._activation_func(self.activation, out)

    def call(self, X, training=True):
        if len(X.shape) < 3:
            raise ValueError('Conv2D except 3D inputs')
        
        return self._calc(X)

class Conv1D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', 
                 activation=None, kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', dilation_rate=1, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.padding = padding 
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.dilation_rate = dilation_rate

        assert isinstance(kernel_size, int), 'Invalid kernel_size'
        self.kernel_size = kernel_size

        assert isinstance(strides, int), 'Invalid strides'
        self.strides = strides

    def _calc_output_shape(self):
        match self.padding:
            case 'valid':
                L_out = tf.cast(tf.math.floor((self.input[0] - self.kernel_size) / self.strides) + 1, tf.int32)
            case 'same' | 'causal':
                L_out = tf.cast(tf.math.ceil(self.input[0] / self.strides), tf.int32)
            case _:
                raise ValueError(f'Invalid padding: {self.padding}; expects: valid/same/causal')
            
        self.output = [L_out, self.filters]

    def __call__(self, layer):
        super().__call__(layer)

        self.input = layer.output
        if len(self.input) != 2:
            raise ValueError('Conv1D layer expect 2D input shape')
        
        self.fan_in  = self.kernel_size * self.input[-1]
        self.fan_out = self.kernel_size * self.filters 
        self.fan_avg = (self.fan_in + self.fan_out) / 2

        self._calc_output_shape()
        
        shape = [self.kernel_size, self.input[-1], self.filters]
        self.weights = self._weights_init(self.kernel_initializer, shape)
        self.bias = self._weights_init(self.bias_initializer, [self.filters])

        self._trainable_variables.extend([self.weights, self.bias])
        return self   
    
    def _calc_padding_1d(self):
        T, K, S = self.input[0], self.kernel_size, self.strides 
        
        out = tf.math.ceil(T / S)
        P_total = tf.maximum((out - 1) * S + K - T, 0)

        pad_left = P_total // 2
        pad_right = P_total - pad_left

        return pad_left, pad_right
    
    def _add_padding(self, X):
        if self.padding == 'causal':
            pad = [self.kernel_size - 1, 0]
        else:
            pad_left, pad_right = self._calc_padding_1d()
            pad = [pad_left, pad_right]

        paddings = [
            [0, 0],
            pad,
            [0, 0]
        ]

        return tf.pad(X, paddings)

    def _calc(self, X):
        # ==================== Implementation 1 ====================
        # pad = self.padding

        # if self.padding == 'causal':
        #     pad = 'valid'
        #     X = self._add_padding(X)

        # out = tf.nn.conv1d(X, self.weights, self.strides, pad.upper(), dilations=self.dilation_rate)
        # out = tf.nn.bias_add(out, self.bias)
        # return self._activation_func(self.activation, out)

        # ==================== Implementation 2 ====================

        L_out, C_out = self.output
        out = tf.TensorArray(dtype=X.dtype, size=L_out)

        if self.padding in ('same', 'causal'):
            X = self._add_padding(X)

        for i in tf.range(L_out):
            start = i * self.strides
            end   = start + self.kernel_size * self.dilation_rate

            receptive_field = X[:, start : end : self.dilation_rate, :]     
            mul = receptive_field[..., tf.newaxis] * self.weights[tf.newaxis, ...] # (B, T, F_in, F_out)

            convolution = tf.reduce_sum(mul, axis=[1,2])   # (B, F_out)
            convolution += self.bias

            out = out.write(i, convolution)

        out = out.stack()              
        out = tf.transpose(out, [1,0,2])  

        return self._activation_func(self.activation, out)

    def call(self, X, training=True):
        if len(X.shape) < 3:
            raise ValueError('Conv1D except 3D inputs')
        
        return self._calc(X)

    def get_weights(self):
        return [self.weights, self.bias]

    def set_weights(self, weights, bias):
        if self.weights.shape != weights.shape or \
            self.bias.shape != bias.shape:
            raise ValueError('Invalid shape on weights or bias')
        
        self.weights.assign(weights)
        self.bias.assign(bias)

class GlobalAvgPool2D(Layer):
    def __call__(self, layer):
        super().__call__(layer)

        self.input = layer.output
        self.output = layer.output[-1]
        return self

    def call(self, X, *_):
        return tf.reduce_mean(X, axis=[1, 2])

class GlobalAvgPool1D(Layer):
    def __call__(self, layer):
        super().__call__(layer)

        self.input = layer.output
        self.output = layer.output[-1]
        return self

    def call(self, X, *_):
        return tf.reduce_mean(X, axis=1)

class MaxPool2D(Conv2D):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        Layer.__init__(self, **kwargs)
        self.padding = padding
        
        if not isinstance(pool_size, (tuple, list)):
            pool_size = (pool_size, pool_size) 

        self.pool_size = pool_size

        if strides is None:
            strides = pool_size
        elif not isinstance(strides, (tuple, list)):
            strides = (strides, strides)

        self.strides = strides
        self.kernel_size = pool_size

    def __call__(self, layer):
        Layer.__call__(self, layer)

        self.input = layer.output
        if len(self.input) < 3:
            raise ValueError('Conv2D layer expect 3D input shape')
        
        self.filters = self.input[-1]
        self._calc_output_shape()
        return self

    @tf.function(jit_compile=True) 
    def _calc(self, X):
        # ==================== Implementation 1 ====================

        return tf.nn.max_pool(X, self.kernel_size, self.strides, self.padding.upper())

        # ==================== Implementation 2 ====================

        # H_out, W_out, C_out = self.output
        # B_in, C_in = tf.shape(X)[0], tf.shape(X)[-1]

        # patches = tf.image.extract_patches(
        #     images=X,
        #     sizes=[1, *self.kernel_size, 1],    
        #     strides=[1, *self.strides, 1],        
        #     rates=[1, 1, 1, 1],                                      
        #     padding=self.padding.upper()               
        # )

        # kernel_prod  = tf.reduce_prod(self.kernel_size)
        # patches_reshaped = tf.reshape(patches, [B_in, H_out, W_out, kernel_prod, C_in])
        # return tf.reduce_max(patches_reshaped, axis=3)


class BatchNormalization(Layer):
    def __init__(self, momentum=0.99, epsilon=0.001, beta_initializer='zeros', gamma_initializer='ones', 
                 moving_mean_initializer='zeros', moving_variance_initializer='ones', **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer

    def __call__(self, layer, *args, **kwds):
        super().__call__(layer)
        self.input  = layer.output
        self.output = layer.output

        self.mean_ = tf.zeros(self.input) if self.moving_mean_initializer == 'zeros' \
                                else tf.fill(self.input, self.moving_mean_initializer)
        
        self.beta_ = tf.zeros(self.input) if self.beta_initializer == 'zeros' \
                                else tf.fill(self.input, self.beta_initializer)
        
        self.variance_ = tf.ones(self.input) if self.moving_variance_initializer == 'ones' \
                                else tf.fill(self.input, self.moving_variance_initializer)

        self.gama_ = tf.ones(self.input) if self.gamma_initializer == 'ones' \
                                else tf.fill(self.input, self.gamma_initializer)

        self.gama_ = tf.Variable(self.gama_) 
        self.beta_ = tf.Variable(self.beta_)

        self._trainable_variables.extend([self.gama_, self.beta_])
        
        return self

    def call(self, X, *_):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')
        
        curr_mean = tf.reduce_mean(X, axis=0)
        curr_var  = tf.math.reduce_variance(X , axis=0)

        self.variance_ = self.variance_ * self.momentum + curr_var * (1 - self.momentum)
        self.mean_     = self.mean_ * self.momentum + curr_mean * (1 - self.momentum)

        X_hat = (X - self.mean_) / tf.sqrt(self.variance_ + self.epsilon)
        return X_hat * self.gama_ + self.beta_

class Normalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_adapt = False
        self.trainable = False
        self.adaptive = False

    def __call__(self, layer, *args, **kwds):
        super().__call__(layer)
        self.output = layer.output[0]
        self.input = layer.output
        return self

    def adapt(self, X):
        self.is_adapt = True
        self.mean_ = tf.cast(tf.reduce_mean(X, axis=0, keepdims=True), tf.float32)
        self.std_  = tf.cast(tf.math.reduce_std(X, axis=0, keepdims=True), tf.float32)

    def call(self, X, *_):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')

        if not self.adaptive and not self.is_adapt:
            raise ValueError(f'This instance {self.name} is not adapted yet')

        eps = tf.keras.backend.epsilon()
        return (X - self.mean_) / (self.std_ + eps)

class LayerNormalization(Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def __call__(self, layer):
        super().__call__(layer)

        self.input  = layer.output
        self.output = self.input

        self.alpha = self._weights_init('ones' , [self.input[-1]])
        self.beta  = self._weights_init('zeros', [self.input[-1]])

        self._trainable_variables.extend([self.alpha, self.beta])
        return self

    def call(self, X, training=False):
        X = tf.convert_to_tensor(X, tf.float32)
        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / tf.sqrt(variance + self.eps) + self.beta

    def get_weights(self):
        return [self.alpha, self.beta]
    
    def set_weights(self, alpha, beta):
        if self.weights.shape != alpha.shape or \
            self.beta.shape != beta.shape:
            raise ValueError('Invalid shape on weights or recurrenr_weights or bias')
        
        self.alpha.assign(alpha)
        self.beta.assign(beta)
        

class Lambda(Layer):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        assert callable(func), 'Function must be callable'
        self.func = func

    def __call__(self, layer):
        super().__call__(layer)
        self.input  = layer.output
        self.output = layer.output
        return self

    def call(self, X, training=False):
        X = tf.convert_to_tensor(X)
        return self.func(X)

class Concatenate(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False
        self.axis = axis
        self.X = []

    def __call__(self, layer):
        if not isinstance(layer, (list, tuple)) or len(layer) < 2:
            raise ValueError('Concat expects sequence of layers')

        super().__call__(layer)
        
        self.input = [lyr.output for lyr in layer]

        if all(isinstance(val, (int, float)) for val in self.input):
            self.output = tf.reduce_sum(self.input)
        else:
            last_dim_sum = 0
            for shape in self.input:
                last_dim_sum += shape[-1]

            self.output = [*self.input[0][:-1], last_dim_sum]

        return self
    
    def call(self, X, *_):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')
        
        if not isinstance(X, list):
            X = [tf.cast(X, tf.float32)]

        self.X.extend(X)
        if len(self.X) != len(self.input):
            return True

        res = tf.concat(self.X, axis=self.axis)
        self.X = []
        return res

class Flatten(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def __call__(self, layer):
        super().__call__(layer)
        self.input = layer.output
        self.output = (tf.cast(tf.reduce_prod(layer.output), tf.int32), )
        return self
    
    def call(self, X, *_):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')

        return tf.cast(tf.reshape(X, (-1, *self.output)), tf.float32)

class Discretization(Layer):
    def __init__(self, bin_boundaries=None, num_bins=None, **kwargs):
        super().__init__(**kwargs)

        if bin_boundaries is None and num_bins is None:
            raise ValueError('must provide bin_boundaries or num_bins')
        
        self.bin_boundaries = bin_boundaries
        self.num_bins = num_bins

    def adapt(self, data):
        min_val = tf.reduce_min(data)
        max_val = tf.reduce_max(data)
        
        step = (max_val - min_val) / self.num_bins
        self.bin_boundaries = [min_val + tf.cast(i, tf.float32) * step for i in tf.range(1, self.num_bins)] 

    def call(self, input, training=False):
        if self.bin_boundaries is None:
            raise ValueError('provide bin_boundaries or adapt layer')
        
        self.bin_boundaries = tf.convert_to_tensor(self.bin_boundaries)
        out = tf.zeros_like(input)

        if len(self.bin_boundaries.shape) == 1:
            last_bin = self.bin_boundaries[0]
            for i, bin in enumerate(self.bin_boundaries[1:], start=1):
                mask = tf.logical_and(last_bin <= input, input < bin)
                out  = tf.where(mask, i, out)
                last_bin = bin

            mask = input >= last_bin
            out  = tf.where(mask, i+1, out)
            return out
        
        for col_i, bins in zip(tf.range(input.shape[-1]), self.bin_boundaries):
            col = tf.identity(input[:, col_i])

            last_bin = bins[0]
            for i, bin in enumerate(bins[1:], start=1):
                mask = tf.logical_and(last_bin <= col, col < bin)
                row_inxs = tf.cast(tf.where(mask), tf.int32)
                inxs = tf.concat([row_inxs, tf.fill(row_inxs.shape, col_i)], axis=1)
                out = tf.tensor_scatter_nd_update(out, inxs, tf.cast(tf.fill([inxs.shape[0]], i), tf.float32))
                last_bin = bin

            row_inxs = tf.cast(tf.where(last_bin <= col), tf.int32)
            inxs = tf.concat([row_inxs, tf.fill(row_inxs.shape, col_i)], axis=1)
            out = tf.tensor_scatter_nd_update(out, inxs, tf.cast(tf.fill([inxs.shape[0]], i + 1), tf.float32))

        return out
    
class CategoryEncoding(Layer):
    def __init__(self, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens

    def call(self, input, training=False):
        if input.shape[-1] == 1:
            input = tf.reshape(input, [-1])
    
        out = tf.one_hot(tf.cast(input, tf.int32), self.num_tokens)

        if len(out.shape) == 3:
            out = tf.reduce_max(out, axis=1)

        return out
    
class StringLookup(Layer):
    def __init__(self, vocabulary=None, output_mode='int', num_oov_indices=1, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary = vocabulary
        self.output_mode = output_mode
        self.num_oov_indices = num_oov_indices

    def adapt(self, data):
        self.vocabulary, _, c = tf.unique_with_counts(data)
        inxs = tf.argsort(c, direction='DESCENDING')
        self.vocabulary = tf.gather(self.vocabulary, inxs)

    def vocabulary_size(self):
        if self.vocabulary is None:
            raise ValueError("StringLookup is not adapted")
        
        return len(self.vocabulary) + self.num_oov_indices

    def call(self, input, training=False):
        if self.vocabulary is None:
            raise ValueError("StringLookup is not adapted")

        input = tf.convert_to_tensor(input)
        matches = tf.equal(input[..., tf.newaxis], self.vocabulary)
        has_match = tf.reduce_any(matches, axis=-1)

        oov_cats = tf.strings.to_hash_bucket_fast(input, self.num_oov_indices)
        cats = tf.argmax(tf.cast(matches, tf.int32), axis=-1)
        cats = tf.where(has_match, cats + self.num_oov_indices, oov_cats)

        match self.output_mode:
            case 'int': return cats
            case 'one_hot': 
                return CategoryEncoding(len(self.vocabulary) + self.num_oov_indices)(cats)

class Hashing(Layer):
    def __init__(self, num_bins, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins

    def call(self, input, training=False):
        return tf.strings.to_hash_bucket_fast(input, self.num_bins)

class Embedding(Layer):
    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super().__init__(**kwargs)

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

        self.embedding_mat = tf.Variable(tf.random.normal(shape=(input_dim, output_dim), 
                                                          mean=0.0, stddev=0.05))
        
        self._trainable_variables.append(self.embedding_mat)
        self.output = [output_dim]

    def __call__(self, layer):
        super().__call__(layer)

        self.input  = layer.output if hasattr(layer.output, '__iter__') else [layer.output]
        self.output = list(self.input) + [self.output_dim]

        return self

    def call(self, x, training=False):
        out = tf.gather(self.embedding_mat, tf.cast(x, tf.int32))

        if not self.mask_zero or not self.next_layer.supports_masking:
            return out
        
        return out, tf.math.not_equal(x, 0)

class TextVectorization(Layer):
    def __init__(self, max_tokens=None, output_mode='int', ragged=False, 
                 split='whitespace', output_sequence_length=None, **kwargs):
        
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.output_mode = output_mode
        self.split = split
        self.ragged = ragged
        self.output_sequence_length = output_sequence_length

    def __call__(self, layer):
        super().__call__(layer)

        self.input  = layer.output
        self.output = list(layer.output) + [self.output_sequence_length] 
        
        return self
    
    def _data_prep(self, data):
        data = tf.convert_to_tensor(data)
        lower = tf.strings.lower(data)
        clean = tf.strings.regex_replace(lower, r"[^\w\s]", "")

        match self.split:
            case 'whitespace': return tf.strings.split(clean)
            case 'character' : return tf.strings.unicode_split(clean, 'UTF-8')
        
        raise ValueError(f'split must be whitespace/character not: {self.split}')

    def adapt(self, data):
        if isinstance(data, Dataset):
            temp = tf.constant('', dtype=tf.string)
            for batch in data:
                for item in batch:
                    temp += item

            data = item

        if isinstance(data, tf.data.Dataset):
            data = data.unbatch()
            data = data.reduce(tf.constant('', dtype=tf.string),
                               lambda acc, x: tf.strings.join([acc, x], separator=' '))

        data = self._data_prep(data)
        flatten = tf.reshape(data, [-1])
        self.vocabulary, _, c = tf.unique_with_counts(flatten)
        inxs = tf.argsort(c, direction='DESCENDING')
        self.vocabulary = tf.gather(self.vocabulary, inxs[:self.max_tokens - 2])

        if self.output_mode != 'tf_idf':
            return

        docs = data
        if isinstance(docs, tf.RaggedTensor):
            docs = docs.to_tensor('')  

        d = tf.cast(tf.shape(docs)[0], tf.float32)
        matches = tf.equal(docs[..., tf.newaxis], self.vocabulary)
        appears = tf.reduce_any(matches, axis=1)  
        f = tf.reduce_sum(tf.cast(appears, tf.float32), axis=0)  
        self.idf = tf.math.log(1.0 + d / (f + 1.0))

    def call(self, input, training=False):
        tokens = self._data_prep(input)

        if isinstance(tokens, tf.RaggedTensor):
            tokens = tokens.to_tensor('')    

        matches = tf.equal(tokens[..., tf.newaxis], self.vocabulary)                                       
        has_match = tf.reduce_any(matches, axis=-1) 
        is_padding = tf.equal(tokens, '')

        match self.output_mode:
            case 'int':
                matches = tf.cast(matches, tf.int32)
                inxs = tf.argmax(matches, axis=-1, output_type=tf.int32) + 2
                inxs = tf.where(has_match, inxs, tf.ones_like(inxs))
                inxs = tf.where(is_padding, tf.zeros_like(inxs), inxs)

                if self.output_sequence_length is not None:
                    inxs = inxs[:, :self.output_sequence_length]
                    
                    padding = self.output_sequence_length - tf.shape(inxs)[1]
                    inxs = tf.pad(inxs, [[0, 0], [0, tf.maximum(0, padding)]])

                if self.ragged:
                    inxs = tf.ragged.boolean_mask(inxs, tf.not_equal(inxs, 0))

                return inxs
            
            case 'tf_idf':
                matches = tf.cast(matches, tf.float32)

                tf_counts = tf.reduce_sum(matches, axis=1)    
                tfidf_vocab = tf_counts * self.idf               
            
                is_oov = tf.logical_and(~has_match, ~is_padding)
                oov_count = tf.reduce_sum(tf.cast(is_oov, tf.float32), axis=1, keepdims=True)

                oov_idf = tf.reduce_mean(self.idf)
                oov_tfidf = oov_count * oov_idf

                tfidf = tf.concat([oov_tfidf, tfidf_vocab], axis=1)
                return tfidf
            
            case 'one_hot':
                is_oov = tf.logical_and(~has_match, ~is_padding)
                oov = tf.cast(is_oov, tf.float32)[..., tf.newaxis]
                vocab = tf.cast(matches, tf.float32)

                return tf.concat([oov, vocab], axis=-1)
            
            case 'multi_hot':
                is_oov = tf.logical_and(~has_match, ~is_padding)
                oov = tf.reduce_any(is_oov, axis=1, keepdims=True) 
                oov = tf.cast(oov, tf.float32)

                vocab = tf.reduce_any(matches, axis=1)  
                vocab = tf.cast(vocab, tf.float32)

                return tf.concat([oov, vocab], axis=1)
            
            case 'count':
                is_oov = tf.logical_and(~has_match, ~is_padding)
                oov = tf.cast(is_oov, tf.float32)
                oov = tf.reduce_sum(oov, axis=1, keepdims=True) 

                vocab = tf.cast(matches, tf.float32)
                vocab = tf.reduce_sum(vocab, axis=1)  

                return tf.concat([oov, vocab], axis=1)
            
        raise NotImplemented

# ------------------------ DROPOUT ------------------------

class Dropout(Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.trainable = False

    def __call__(self, layer, *args, **kwds):
        super().__call__(layer)
        self.input  = layer.output
        self.output = layer.output
        return self

    @tf.function(jit_compile=True)
    def _calc(self, X):
        mask = tf.cast(tf.random.uniform(minval=0, maxval=1, shape=X.shape) 
                            > self.rate, tf.float32) / (1 - self.rate)
        return X * mask
    
    def call(self, X, training=True):
        if not self.is_build:
            raise ValueError(f'This instance {self.name} is not builded yet')

        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        
        if not training:
            return X
        
        return self._calc(X)

class AlphaDropout(Dropout):
    def __init__(self, rate, name=None):
        super().__init__(rate, name)
        alpha = 1.67
        scale = 1.05
        self.alpha_p = -scale * alpha

    def call(self, X, training=True):
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        if not training:
            return X

        keep_prob = 1 - self.rate
        self.mask = (tf.random.uniform(*X.shape) < keep_prob)
        X_const = self.mask * X + (~self.mask) * self.alpha_p
        scale = 1.0 / tf.sqrt(keep_prob * (1 + self.rate * self.alpha_p**2))
        bias = -scale * self.rate * self.alpha_p

        return scale * X_const + bias

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
        self._opt1_map = {}
        self._opt2_map = {}

    def get_learning_rate(self, itr):
        return self.learning_rate

    def apply_gradients(self, grads, itr):
        for param, grad in grads:
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)

            self.step(param, grad, itr)
    
    def _clip_delta(self, grad):
        if self.clipvalue is not None:
            return tf.clip_by_value(grad, -self.clipvalue, self.clipvalue)

        if self.clipnorm is not None:
            total_norm = tf.norm(grad)
            scale = self.clipnorm / tf.maximum(total_norm, self.clipnorm)
            return grad * scale

        return grad

    def _get_opt(self, param, mode=1):
        opt_map = self._opt1_map if mode == 1 else self._opt2_map
        opt = opt_map.get(id(param))

        if opt is not None:
            return opt
        
        opt = tf.Variable(tf.zeros_like(param))
        opt_map[id(param)] = opt
        return opt
    
class SGD(OptimizersBase):
    def __init__(self, learning_rate=0.01, clipvalue=None, clipnorm=None, 
                 momentum=0.0, nesterov=False, weight_decay=None, decay=None, c=1):
        super().__init__(learning_rate, clipvalue, clipnorm)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.decay = decay
        self.c = c

    def step(self, param, grad, itr):
        self._clip_delta(grad)
        opt = self._get_opt(param)

        if self.decay is not None:
            self.learning_rate /= 1 + (self.decay / itr) ** self.c

        if self.weight_decay is not None:
            param.assign_sub(self.learning_rate * self.weight_decay * param)

        opt.assign(self.momentum * opt - self.learning_rate * grad)

        if not self.nesterov:
            param.assign_add(opt)
            return

        param.assign_add(self.momentum * opt - self.learning_rate * grad)

class RMSProp(OptimizersBase):
    def __init__(self, learning_rate=0.001, clipvalue=None, clipnorm=None, rho=0.9, epsilon=1e-07):
        super().__init__(learning_rate, clipvalue, clipnorm)
        self.rho = rho
        self.epsilon = epsilon

    def step(self, param, grad, itr):
        self._clip_delta(grad)
        opt = self._get_opt(param)
        opt.assign(self.rho * opt + (1 - self.rho) * tf.square(grad))
        param.assign_sub(self.learning_rate * grad / tf.sqrt(opt + self.epsilon)) 

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

    def step(self, param, grad, itr):
        self._clip_delta(grad)
        opt  = self._get_opt(param)
        opt2 = self._get_opt(param, mode=2)

        opt.assign(self.beta_1 * opt - (1 - self.beta_1) * grad)

        if not self.max:
            opt2.assign(self.beta_2 * opt2 + (1 - self.beta_2) * tf.square(grad))
            s_hat_w = opt2 / (1 - self.beta_2 ** itr)
            denominator_w = tf.sqrt(s_hat_w + self.epsilon)
        else:
            opt2.assign(tf.maximum(self.beta_2 * opt2, tf.abs(grad)))
            denominator_w = opt2 + self.epsilon

        m_hat_w = opt / (1 - self.beta_1 ** itr)
        param.assign_add(self.learning_rate * m_hat_w / denominator_w)

        if self.weight_decay is not None:
            param.assign_sub(self.learning_rate * self.weight_decay * param)

# ------------------- Regularization -------------------

class RegularizationBase(ABC):
    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2
        self.layer = None

    def update(self):
        if self.l2 > 0:
            self.layer.delta_weight.assign_add(2 * self.l2 * self.layer.weights)
            self.layer.delta_bias.assign_add(2 * self.l2 * self.layer.bias)
        
        if self.l1 > 0:
            self.layer.delta_weight.assign_add(self.l1 * tf.sign(self.layer.weights))
            self.layer.delta_bias.assign_add(self.l1 * tf.sign(self.layer.bias))

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
        return params * self.r / tf.norm(params, ord=2)

# ------------------- Additional classes -------------------

class Sequential(Model, Layer):
    def __init__(self, layers=None, name=None):
        self.inputs, self.outputs = None, None
        self.name = name or 'sequential'
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
        super().compile(optimizer, loss, metrics)
        if loss == 'sparse_categorical_crossentropy':
            self.n_classes = self.layers[-1].units

        for i in range(1, len(self.layers)):
            self.layers[i] = self.layers[i](self.layers[i-1])

            if isinstance(self.layers[i], Normalization):
                self.layers[i].adaptive = True

        super().__init__([self.layers[0]], self.layers[-1])
        # super().__init__()

    def __call__(self, input, training=False):
        if isinstance(input, tuple):
            input = input[0]

        x = input

        for layer in self.layers:
            if isinstance(x, tuple) and layer.supports_masking:
                inp, mask = x
                x = layer.call(inp, mask=mask, training=training)
                continue

            x = layer.call(x, training=training)

        return x

    def call(self, input, training=False):
        return self.__call__(input, training)
    
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
    # return joblib.load(filename)

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    layers = []
    for layer_data in data['architecture']:
        layer = layer_data['object']()
        layer.__dict__.update(layer_data['dict'])
        layers.append(layer)

    match data['class']:
        case 'Sequential': model = Sequential(layers)
        case 'Model': model = Model([layers[0]], layers[-1])

    model.compile(optimizer=data['optimizer'], loss=data['loss'], metrics=data['metrics'])
    return model

class VarianceScaling:
    def __init__(self, scale=1.0, mode='fan_in', distribution='normal'):
        if mode not in ("fan_in", "fan_out", "fan_avg"):
            raise ValueError(f'Invalid mode: {mode}')
        
        if distribution not in ("normal", "uniform"):
            raise ValueError(f'Invalid distribution: {distribution}')
        
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

class ExponentialDecay:
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def get_learning_rate(self, itr):
        return self.initial_learning_rate * (self.decay_rate ** (itr / self.decay_steps))

class MCModel(Model):
    def predict(self, X, n_predictions=100):
        check_is_fitted(self)

        if not isinstance(X, tuple):
            X = (X,)


        self._forward(self.inputs, X, training=True)
        res = self.outputs.active_res
        self._active_res_tonone(self.inputs)

        preidctions = tf.Variable(tf.empty_like(res, shape=(n_predictions, *res.shape)))
        preidctions[0].assign(res)

        for i in range(1, n_predictions):
            self._forward(self.inputs, X, training=True)
            preidctions[i].assign(self.outputs.active_res)
            self._active_res_tonone(self.inputs)

        return tf.reduce_mean(preidctions, axis=0)

# ------------------------ Callbacks ------------------------

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
        self.model.optimizer_.learning_rate = self.func(epoch)

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
                self.model.optimizer_.learning_rate *= self.factor
                self._counter = 0
        else:
            self._best_val_loss = val_loss
            self._counter = 0

class EarlyStopping(Callback):
    def __init__(self, patience, monitor='val_loss', restore_best_weights=False, min_delta=0.001):
        self.patience = patience
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.min_delta = min_delta
        self._best_loss = np.inf
        self._best_weights = None
        self._counter = 0
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        new_loss = logs[self.monitor]

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

# ------------------------ Activations ------------------------

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
        alpha_delta = tf.where(raw_res > 0, 0, raw_res) 
        self.alpha_grad = tf.reduce_mean(grad * alpha_delta)

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
        self.beta_grad = tf.reduce_mean(grad * raw_res**2 * s * (1 - s))

    def step(self, lr, *_):
        if not self.trainable:
            return 
        
        self.beta -= lr * self.beta_grad

# ---------------------------- Metrics ----------------------------

class Metric(ABC):
    def __init__(self, name=None):
        super().__init__()
        self.name = name or self.__class__.__name__.lower()
        self._statefull = {}

    def add_weight(self, name, initializer, shape=tuple(), **kwargs):
        match initializer:
            case 'ones':  tensor = tf.Variable(tf.ones(shape, **kwargs))
            case 'zeros': tensor = tf.Variable(tf.zeros(shape, **kwargs))
            case _: tensor = tf.Variable(initializer, shape=shape, **kwargs) 
        
        self._statefull[name] = tensor
        return tensor

    def __call__(self, *args, **kwds):
        self.update_state(*args, **kwds)
        return self
    
    @abstractmethod
    def update_state(self, y_true, y_pred):
        pass

    @abstractmethod
    def result(self):
        pass

    def reset_states(self):
        for variable in self._statefull.values():
            variable.assign(tf.zeros_like(variable))

class MeanSquaredError(Metric):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.mean_sum = self.add_weight('mean_sum', initializer='zeros')
        self.count    = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        error = tf.cast(y_true - y_pred, tf.float32)
        self.mean_sum.assign_add(tf.reduce_sum(tf.square(error)))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))
    
    def result(self):
        return self.mean_sum / self.count

class MeanAbsoluteError(Metric):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.mean_sum = self.add_weight('mean_sum', initializer='zeros')
        self.count    = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        error = tf.cast(y_true - y_pred, tf.float32)
        self.mean_sum.assign_add(tf.reduce_sum(tf.abs(error)))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))
    
    def result(self):
        return self.mean_sum / self.count

class MeanAbsolutePercentageError(Metric):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.mean_sum = self.add_weight('mean_sum', initializer='zeros')
        self.count    = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        error = tf.cast((y_true - y_pred) / y_true, tf.float32) 
        self.mean_sum.assign_add(tf.reduce_sum(tf.abs(error)))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))
    
    def result(self):
        return self.mean_sum / self.count

class RootMeanSquaredError(Metric):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.rmse_sum = self.add_weight('rmse_sum', initializer='zeros')
        self.count    = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        error = tf.cast(y_true - y_pred, tf.float32)
        self.rmse_sum.assign_add(tf.reduce_sum(tf.square(error)))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))
    
    def result(self):
        return tf.sqrt(self.rmse_sum / self.count)
    
class Accuracy(Metric):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.acc_sum = self.add_weight('acc_sum', initializer='zeros')
        self.count    = self.add_weight('count', initializer='zeros')

    @staticmethod
    def one_shot(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        if len(y_true.shape) == 1:
            y_true = tf.reshape(y_true, [-1, 1])

        acc = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        count = tf.cast(tf.size(y_true), tf.float32)

        return acc / count

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        if len(y_true.shape) == 1:
            y_true = tf.reshape(y_true, [-1, 1])

        acc = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        self.acc_sum.assign_add(acc)
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.acc_sum / self.count
    
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

class HuberMetric(Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs) 
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
  
    def update_state(self, y_true, y_pred):
        sample_metrics = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(sample_metrics))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / self.count

class Precision(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.true_positives =  self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        tp = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
        fp = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 1), tf.float32))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        return tf.math.divide_no_nan(self.true_positives, (self.true_positives + self.false_positives))

class Mean(Metric):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, value):
        self.total.assign_add(value)
        self.count.assign_add(1)
    
    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

# ----------------------------- ResidualBlock -----------------------------

class ResidualBlock(Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Dense(n_neurons, activation="relu",
                            kernel_initializer="he_normal")
                            for _ in range(n_layers)]
   
    def call(self, inputs, *args):
        Z = inputs
        for layer in self.hidden:
            Z = layer.call(Z)

        return inputs + Z
    
# ----------------------------- ResidualRegressor -----------------------------

class ResidualRegressor(Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(30, activation="relu",
                                    kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = Dense(output_dim)
    
    def call(self, inp, *args, **kwargs):
        Z = self.hidden1.call(inp)
        for _ in range(1 + 3):
            Z = self.block1.call(Z)
            
        Z = self.block2.call(Z)
        return self.out.call(Z)
    
class SimpleResidualRegressor(Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(100, activation="relu",
                                    kernel_initializer="he_normal")
        self.block2 = Dense(100, activation="relu",
                                    kernel_initializer="he_normal")
        self.out = Dense(output_dim)
    
    def call(self, inp, *args, **kwargs):
        Z = self.hidden1.call(inp)
        Z = self.block2.call(Z)
        return self.out.call(Z)
    