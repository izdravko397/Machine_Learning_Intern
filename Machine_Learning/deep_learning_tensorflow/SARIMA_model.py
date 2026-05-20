from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np


# class SARIMA: 
#     def __init__(self, data, order, seasonal_order=None):
#         if not isinstance(data, pd.Series):
#             data = pd.Series(data)

#         self.data = data

#         self.order = order
#         self.seasonal_order = seasonal_order

#         self.p, self.d, self.q = order
#         self.P, self.D, self.Q, self.s = seasonal_order or (0, 0, 0, 2)

#     def fit(self):
#         data = self.data.copy()

#         self.seasonal_diffs = []
#         for _ in range(self.D):
#             self.seasonal_diffs.append(data)
#             data = data - data.shift(self.s)

#         self.diffs = []
#         for _ in range(self.d):
#             self.diffs.append(data.copy())
#             data = data.diff()

#         self.arma = ARIMA(data, order=(self.p, 0, self.q), seasonal_order=(self.P, 0, self.Q, self.s)).fit()
#         return self

#     def forecast(self, steps=1):
#         pred = self.arma.forecast(steps)
#         restored = pred.copy()

#         for k in reversed(range(self.d)):
#             last_vals = self.diffs[k]
#             last = last_vals.iloc[-1]

#             for i in range(len(restored)):
#                 restored.iloc[i] = last + restored.iloc[i]
#                 last = restored.iloc[i]
        
#         for k in reversed(range(self.D)):
#             last_vals = self.seasonal_diffs[k]
#             last_season = last_vals.iloc[-self.s:]

#             for i in range(len(restored)):
#                 base = last_season.iloc[i] if i < self.s else restored.iloc[i - self.s]
#                 restored.iloc[i] = restored.iloc[i] + base

#         return restored



class SARIMA: 
    def __init__(self, data, order, seasonal_order=None):
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        self.data = data

        self.order = order
        self.seasonal_order = seasonal_order

        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.s = seasonal_order or (0, 0, 0, 2)

    def fit(self):
        data = self.data.copy()

        self.seasonal_diffs = []
        for _ in range(self.D):
            self.seasonal_diffs.append(data)
            data = data - data.shift(self.s)

        self.diffs = []
        for _ in range(self.d):
            self.diffs.append(data.copy())
            data = data.diff()

        data = data.dropna()

        self.arma = ARIMA(data, order=(self.p, 0, self.q)).fit()
        self.z = data.values

        arma_fitted = self.arma.fittedvalues.values

        min_len = min(len(self.z), len(arma_fitted))
        z_cut = self.z[-min_len:]
        errors = z_cut - arma_fitted[-min_len:]
        self.errors = errors

        X, target = [], []
        for t in range(self.P * self.s, len(z_cut)):
            row = []
            for i in range(1, self.P + 1):
                row.append(z_cut[t - i*self.s])

            X.append(row)
            target.append(z_cut[t])

        X, target = np.array(X), np.array(target)
        self.theta_P, *_ = np.linalg.lstsq(X, target)

        X2, target2 = [], []
        for t in range(self.Q * self.s, len(errors)):
            row = []
            for i in range(1, self.Q + 1):
                row.append(errors[t - i*self.s])

            X2.append(row)
            target2.append(errors[t])

        X2 = np.array(X2)
        target2 = np.array(target2)
        self.theta_Q, *_ = np.linalg.lstsq(X2, target2)

        return self

    def forecast(self, steps=1):
        arma_pred = self.arma.forecast(steps)

        z_last   = list(self.z.copy())
        err_last = list(self.errors.copy())

        result = []
        for i in range(steps):
            z_base = arma_pred[i]

            ar_part = 0
            for k in range(1, self.P+1):
                ar_part += self.theta_P[k-1] * z_last[-k*self.s]

            ma_part = 0
            for k in range(1, self.Q+1):
                ma_part += self.theta_Q[k-1] * err_last[-k*self.s]

            z_new = z_base + ar_part + ma_part
            result.append(z_new)
            z_last.append(z_new)
            err_last.append(0)  

        restored = pd.Series(result, arma_pred.index)

        for k in reversed(range(self.d)):
            last_vals = self.diffs[k]
            last = last_vals.iloc[-1]

            for i in range(len(restored)):
                restored.iloc[i] = last + restored.iloc[i]
                last = restored.iloc[i]
        
        for k in reversed(range(self.D)):
            last_vals = self.seasonal_diffs[k]
            last_season = last_vals.iloc[-self.s:]

            for i in range(len(restored)):
                base = last_season.iloc[i] if i < self.s else restored.iloc[i - self.s]
                restored.iloc[i] = restored.iloc[i] + base

        return restored




















# import tensorflow as tf

# class SARIMA:
#     def __init__(self, data, order, seasonal_order=None, epochs=10, learning_rate=0.01):
#         if not isinstance(data, pd.Series):
#             data = pd.Series(data)

#         self.data = data

#         self.order = order
#         self.seasonal_order = seasonal_order

#         self.p, self.d, self.q = order
#         self.P, self.D, self.Q, self.s = seasonal_order or (0, 0, 0, 1)

#         self.epochs = epochs
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate)

#     def _weight_init(self):
#         self.theta_p = tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=[self.p]))
#         self.theta_q = tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=[self.q]))
#         self.theta_P = tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=[self.P]))
#         self.theta_Q = tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=[self.Q]))
        
#         self.params = [self.theta_p, self.theta_q, self.theta_P, self.theta_Q]

#         self.ar_part = tf.Variable(0, dtype=tf.float32)
#         self.ma_part = tf.Variable(0, dtype=tf.float32)


#     def _forward_pass(self, z):
#         if isinstance(z, pd.Series):
#             z = tf.convert_to_tensor(z.values, dtype=tf.float32)
#         else:
#             z = tf.convert_to_tensor(z, dtype=tf.float32)

#         n = z.shape[0]
#         if n is None:
#             n = tf.shape(z)[0].numpy() 

#         z_hat = tf.TensorArray(dtype=tf.float32, size=n, clear_after_read=False)
#         e = tf.TensorArray(dtype=tf.float32, size=n, clear_after_read=False)
#         start = max(self.p, self.P * self.s, self.q, self.Q * self.s)
        
#         for t in tf.range(start):
#             z_hat = z_hat.write(t, z[t])
#             e = e.write(t, 0.0)

#         for t in tf.range(start, n):
#             for i in tf.range(self.p):
#                 self.ar_part.assign_add(self.theta_p[i] * z_hat.read(t - i - 1))

#             for j in tf.range(self.P):
#                 self.ar_part.assign_add(self.theta_P[j] * z_hat.read(t - (j + 1) * self.s))

#             for i in tf.range(self.q):
#                 self.ma_part.assign_add(self.theta_q[i] * e.read(t - i - 1))

#             for j in tf.range(self.Q):
#                 self.ma_part.assign_add(self.theta_Q[j] * e.read(t - (j + 1) * self.s))

#             pred = self.ar_part + self.ma_part
#             z_hat = z_hat.write(t, pred)
#             e = e.write(t, z[t] - pred)

#         z_hat_final = z_hat.stack()
#         e_final = e.stack()

#         self.ar_part.assign(0)
#         self.ma_part.assign(0)

#         return z_hat_final[start:], e_final[start:]

#     @tf.function
#     def _train(self, data):
#         for _ in range(self.epochs):
#             with tf.GradientTape() as tape:
#                 z, error = self._forward_pass(data)
#                 loss = tf.reduce_mean(tf.abs(error))

#             grads = tape.gradient(loss, self.params)
#             self.optimizer.apply_gradients(zip(grads, self.params))

#     def fit(self):
#         data = self.data.copy()

#         self.seasonal_diffs = []
#         for _ in range(self.D):
#             self.seasonal_diffs.append(data)
#             data = data - data.shift(self.s)

#         self.diffs = []
#         for _ in range(self.d):
#             self.diffs.append(data)
#             data = data.diff()

#         self._weight_init()
#         self._train(data.astype(np.float32))
#         return self

#     def forecast(self, steps=1):
#         data = self.data.copy().astype(np.float32)
#         for k in range(self.D):
#             data = data - data.shift(self.s)
        
#         for k in range(self.d):
#             data = data.diff()
        
#         z = tf.convert_to_tensor(data.values, dtype=tf.float32)
#         n = z.shape[0]

#         z_hat = tf.TensorArray(dtype=tf.float32, size=n + steps, clear_after_read=False)
#         e = tf.TensorArray(dtype=tf.float32, size=n + steps, clear_after_read=False)

#         for t in range(n):
#             z_hat = z_hat.write(t, z[t])
#             e = e.write(t, 0.0)

#         for t in range(n, n + steps):
#             ar_part = tf.reduce_sum([self.theta_p[i] * z_hat.read(t - i - 1) for i in range(self.p)])
#             ar_part += tf.reduce_sum([self.theta_P[j] * z_hat.read(t - (j + 1) * self.s) for j in range(self.P)])

#             ma_part = tf.reduce_sum([self.theta_q[i] * e.read(t - i - 1) for i in range(self.q)])
#             ma_part += tf.reduce_sum([self.theta_Q[j] * e.read(t - (j + 1) * self.s) for j in range(self.Q)])

#             pred = ar_part + ma_part

#             z_hat = z_hat.write(t, pred)
#             e = e.write(t, 0.0)  

#         forecast_vals = z_hat.stack()[n:]

#         for k in reversed(range(self.d)):
#             last_val = self.data.iloc[-1]
#             forecast_vals = forecast_vals + last_val

#         for k in reversed(range(self.D)):
#             last_vals = self.data.iloc[-self.s:]
#             forecast_vals = forecast_vals + tf.convert_to_tensor(last_vals.values, dtype=tf.float32)


#         # for k in reversed(range(self.d)):
#         #     last_vals = self.diffs[k].dropna().values
#         #     last_val = last_vals[-1]
#         #     for i in range(len(forecast_vals)):
#         #         forecast_vals = tf.tensor_scatter_nd_add(forecast_vals, [[i]], [last_val])
#         #         last_val = forecast_vals[i]

#         # for k in reversed(range(self.D)):
#         #     last_vals = self.seasonal_diffs[k].dropna().values[-self.s:]
#         #     for i in range(len(forecast_vals)):
#         #         idx = i % self.s
#         #         forecast_vals = tf.tensor_scatter_nd_add(forecast_vals, [[i]], [last_vals[idx]])


#         return forecast_vals.numpy()

