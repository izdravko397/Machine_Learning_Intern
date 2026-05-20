from collections import defaultdict
from io import StringIO
import numpy as np
from sklearn.base import check_array
from sklearn.preprocessing import OneHotEncoder, add_dummy_feature
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
from tqdm import tqdm

sigmoid = lambda z: 1 / (1 + np.exp(-z))

class Input:
    def __init__(self, shape):
        self.is_build = True
        self.shape = shape
        self._output = shape

class Flatten:
    def __init__(self, data_format=None):
        self.is_build = False
        if data_format is not None:
            self.build(data_format)

        self.data_format = data_format

    def build(self, input):
        self.is_build = True
        self.n_features_ = np.cumprod(input)[-1]
        self._output = self.n_features_

    def transform(self, data):
        if not self.is_build:
            raise ValueError('Flatten is not builded yet')
        
        data = np.asarray(data)
        return add_dummy_feature(data.reshape(-1, self.n_features_))

class Dense:
    def __init__(self, units, activation, name=None):
        self.is_build = False
        self.units = units
        self.activation = activation
        self.name = name if name is not None else 'dense'
        self._output = self.units

    def build(self, input):
        self.is_build = True
        self.weights = np.random.randn(self.units, input) * np.sqrt(2 / input)

    def _get_active_res(self, raw_res):
        match self.activation:
            case 'relu': 
                res = raw_res.copy()
                res[res < 0] = 0
                return res
            case 'tanh': return 2 * sigmoid(2 * raw_res) - 1
            case 'logistic': return sigmoid(raw_res)
            case 'softmax': return self._softmax_func(raw_res)

    def _get_active_prim(self, active_res):
        match self.activation:
            case 'relu': return (active_res > 0).astype(int)
            case 'tanh': return 1 - active_res ** 2
            case 'logistic': return active_res * (1 - active_res)

    @staticmethod
    def _softmax_func(z):
        classes_exp = np.exp(z)
        return classes_exp / classes_exp.sum(axis=1, keepdims=True)

    def get_activation(self, data):
        raw = data @ self.weights.T
        return self._get_active_res(raw)
    




class Sequential:
    def __init__(self, layers=None, random_state=None):
        self._names_counts = defaultdict(int)
        self._names = {}
        self.layers = []

        if layers is None:
            layers = []

        for layer in layers:
            layer_name = layer.__class__.__name__.lower()
            self._add_name(layer_name, layer)

        if random_state is not None:
            np.random.seed(random_state)

    def _add_name(self, name, obj):
        self._names_counts[name] += 1
        new_name = name
        if (count := self._names_counts[name]) > 1:
            new_name = name + f'_{count - 1}'
            self._names_counts[new_name]

        self._names[new_name] = obj

        if not obj.is_build and len(self.layers):
            output = self.layers[-1]._output
            if isinstance(obj, Dense) and not isinstance(self.layers[-1], Dense):
                output += 1

            obj.build(output)
        
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

        if isinstance(self.layers[0], Input):
            del self.layers[0]

    def _learning_rate(self):
        match self.optimizer_:
            case 'sgd': return 0.01

    def _get_loss(self, y_true, y_pred):
        match self.loss_:
            case "sparse_categorical_crossentropy" | "categorical_crossentropy":
                return -np.mean((y_true * np.log(y_pred)).sum(axis=1))
            
            case "binary_corssentropy":
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            
            case "mse":
                return ((y_true - y_pred) ** 2).mean()
            
    def get_scores(self, y_true, y_pred):
        scores = []
        for metric in self.metrics_:
            match metric:
                case 'accuracy':
                    score = accuracy_score(y_true, y_pred)
                case 'auc':
                    score = roc_auc_score(y_true, y_pred)
                case 'f1':
                    score = f1_score(y_true, y_pred)
                case 'f1_macro':
                    score = f1_score(y_true, y_pred, average='macro')

            scores.append(score)

        return scores

    def _backpropagation(self, xi, yi, yi_true, lr, val_data=None):
        # forward
        last_output = xi.copy()
        val_last_output = val_data[0].copy()
        layers_active_outputs = []

        search_firts_hid_layer = True
        for layer in self.layers:
            if search_firts_hid_layer and isinstance(layer, Dense):
                last_output = add_dummy_feature(last_output)
                val_last_output = add_dummy_feature(val_last_output)
                search_firts_hid_layer = False

            if isinstance(layer, Flatten):
                search_firts_hid_layer = False
                last_output = layer.transform(last_output)
                if val_data is not None:
                    val_last_output = layer.transform(val_last_output)
            else:
                last_output = layer.get_activation(last_output)
                if val_data is not None:
                    val_last_output = layer.get_activation(val_last_output)

            layers_active_outputs.append(last_output)

        # loss
        loss = self._get_loss(yi, last_output)
        val_loss = self._get_loss(val_data[1], val_last_output) if val_data is not None else None

        # scores
        scores = self.get_scores(yi_true, last_output.argmax(axis=1))
        val_scores = self.get_scores(val_data[2], val_last_output.argmax(axis=1)) if val_data is not None else None

        # backward
        last_b = (last_output - yi) 
        gradients = [(last_b.T @ layers_active_outputs[-2]) / xi.shape[0]]
        for i in range(len(self.layers) -2, len(self.layers) - self.n_dense_layers - 1, -1):
            last_b = (last_b @ self.layers[i + 1].weights) * self.layers[i]._get_active_prim(layers_active_outputs[i])
            prev_layer_pred = layers_active_outputs[i - 1]
            gradients.append((last_b.T @ prev_layer_pred) / xi.shape[0])

        # Update
        gradients.reverse()
        counter = 0
        for layer in self.layers:
            if not isinstance(layer, Dense):
                continue
            
            layer.weights -= lr * gradients[counter]
            counter += 1

        return loss, val_loss, scores, val_scores
    
    def fit(self, X, y, epochs=1, validation_data=None, batch_size=32):
        if self.loss_ is None:
            raise ValueError('This instance is not compiled yet')
        
        self.n_samples_, *self.n_features_in_ = X.shape
        self.n_dense_layers = np.sum([1 for l in self.layers if isinstance(l, Dense)])

        if not self.layers[0].is_build:
            input = self.n_features_in_
            if isinstance(self.layers[0], Dense):
                input += 1

            self.layers[0].build(input)

        y_t = np.asarray(y.copy()).reshape(-1, 1)
        if np.issubdtype(y_t.dtype, np.bool_):
            y_t = y_t.astype(int)

        if self.loss_ == "sparse_categorical_crossentropy":
            onehot = OneHotEncoder(sparse_output=False)
            y_t = onehot.fit_transform(y_t)
            if validation_data is not None:
                if not isinstance(validation_data, list):
                    validation_data = list(validation_data)

                y_val_true = validation_data[1].reshape(-1, 1)
                validation_data[1] = onehot.transform(y_val_true)
                validation_data.append(y_val_true)

        n_batches = self.n_samples_ // batch_size
        loss_curve = []
        loss_val_curve = []
        score_curve = []
        score_val_curve = []
        for epoch in range(epochs):
            mean_loss = []
            mean_val_loss = []
            mean_scores = []
            mean_val_scores = []
            print(f'Epoch {epoch + 1}/{epochs}')

            for i in tqdm(range(n_batches)):
                batch_inxs = np.random.choice(self.n_samples_, batch_size, False)
                xi, yi, yi_true = X[batch_inxs], y_t[batch_inxs], y[batch_inxs]
                lr = self._learning_rate()
                loss, val_loss, scores, val_scores = self._backpropagation(xi, yi, yi_true, lr, validation_data)
                
                mean_loss.append(loss)
                mean_val_loss.append(val_loss)
                mean_scores.append(scores)
                mean_val_scores.append(val_scores)

            mean_loss = np.mean(mean_loss)
            mean_val_loss = np.mean(mean_val_loss)
            mean_scores = np.array(mean_scores).mean(axis=0)
            mean_val_scores = np.array(mean_val_scores).mean(axis=0)

            loss_curve.append(mean_loss)
            loss_val_curve.append(mean_val_loss)
            score_curve.append(mean_scores)
            score_val_curve.append(mean_val_scores)

            train_stats = f'    - loss: {mean_loss}'
            for i, m in enumerate(self.metrics_):
                train_stats += f' - {m}: {mean_scores[i]}'
            print(train_stats)

            val_stats = f'    - val_loss: {mean_val_loss}'
            for i, m in enumerate(self.metrics_):
                val_stats += f' - {'val_' + m}: {mean_val_scores[i]}'
            print(val_stats)

        score_curve = np.array(score_curve)
        score_val_curve = np.array(score_val_curve)
        curves = {'loss': loss_curve, 'val_loss': loss_val_curve}

        for i, m in enumerate(self.metrics_):
            curves[m] = score_curve[:, i]
            curves['val_' + m] = score_val_curve[:, i]

        self.curves_ = curves
        return self

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
            elif isinstance(layer, Flatten):
                n_neurons = layer.n_features_
                param = 0
            else:
                n_neurons = layer.units
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



import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:10_000], y_train_full[:10_000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]


# ser = pd.Series(y_train)
# print(ser.value_counts() / len(ser))

X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

model = Sequential([
    Input([28, 28]),
    Flatten(),
    Dense(30, 'relu'),
    Dense(20, 'relu'),
    Dense(10, 'softmax')
])

# model = Sequential()
# model.add(Input([28, 28]))
# model.add(Flatten())
# model.add(Dense(30, 'relu'))
# model.add(Dense(20, 'relu'))
# model.add(Dense(10, 'softmax'))

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd",
metrics=["accuracy", 'f1_macro'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(model.curves_).plot()
plt.show()