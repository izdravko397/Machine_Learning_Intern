from tf_model import Model, Input, Dense, SGD, Adam, SimpleRNN, Normalization, Mean, Conv2D, Sequential, Flatten, MaxPool2D, Dropout
import tensorflow as tf

# fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
# (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
# X_train, y_train = X_train_full[:10_000], y_train_full[:10_000]
# X_valid, y_valid = X_train_full[-2000:], y_train_full[-2000:]

# X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# inp = Input(X_train.shape[1:])
# flatten = Flatten()(inp)
# hd1 = Dense(300, activation='relu', kernel_initializer='he_normal', name='hd1')(flatten)
# hd2 = Dense(100, activation='relu', kernel_initializer='he_normal', name='hd2')(hd1)
# output = Dense(10, activation='softmax', name='output')(hd2)


# model = Model([inp], output)
# model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

# import time
# s = time.time()
# model.evaluate(X_test, y_test)
# print(time.time() - s)

# -----------------------------------------------------
# mask_t = (y_train == 0) | (y_train == 1)
# mask_v = (y_valid == 0) | (y_valid == 1)

# X_train, y_train = X_train[mask_t], y_train[mask_t]
# X_valid, y_valid = X_valid[mask_v], y_valid[mask_v]


# inp = Input(X_train.shape[1:])
# flatten = Faltten()(inp)
# hd1 = Dense(300, activation='relu', kernel_initializer='he_normal', name='hd1')(flatten)
# hd2 = Dense(100, activation='relu', kernel_initializer='he_normal', name='hd2')(hd1)
# output = Dense(1, activation='sigmoid', name='output')(hd2)


# model = Model([inp], output)
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', 'precision'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# -----------------------------------------------------
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split

# X, y = fetch_california_housing(return_X_y=True)
# X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True) 
# X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, random_state=42, test_size=0.1, shuffle=True) 

# inp = Input(X_train.shape[1:])
# norm = Normalization()
# norm.adapt(X_train)
# norm = norm(inp)

# hd1 = Dense(300, activation='relu', kernel_initializer='he_normal', name='hd1')(norm)
# hd2 = Dense(100, activation='relu', kernel_initializer='he_normal', name='hd2')(hd1)
# output = Dense(1, name='output')(hd2)

# model = Model([inp], output)
# model.compile(optimizer=Adam(), loss='mse', metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

#----------------------------------------------------------

# class ReconstructingRegressor(Model):
#     def __init__(self, output_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden = [Dense(30, activation="relu",
#                         kernel_initializer="he_normal")
#                         for _ in range(5)]
        
#         self.out = Dense(output_dim)
#         self.reconstruction_mean = Mean(name="reconstruction_error")

#     def build(self, batch_input_shape):
#         n_inputs = batch_input_shape[-1]
#         self.reconstruct = Dense(n_inputs)

#     def call(self, inputs, training=False):
#         Z = inputs
#         for layer in self.hidden:
#            Z = layer.call(Z)

#         reconstruction = self.reconstruct.call(Z)
#         recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
#         self.add_loss(0.05 * recon_loss)

#         if training:
#             result = self.reconstruction_mean(recon_loss)
#             self.add_metric(result)

#         return self.out.call(Z)
    
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
    
# model = ReconstructingRegressor(1)
# model.compile(loss='mse', optimizer=Adam(), metrics=['rmse'])
# model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))


# ====================================== Conv2D =====================

# X_train = X_train.reshape(*X_train.shape, 1)
# X_valid = X_valid.reshape(*X_valid.shape, 1)
# X_test = X_test.reshape(*X_test.shape, 1)

# cnn = Sequential([
#     Input(X_train.shape[1:]),

#     Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
#     MaxPool2D(),
#     Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
#     MaxPool2D(),

#     Flatten(),
#     Dense(64, activation='relu', kernel_initializer='he_normal'),
#     Dense(10, activation='softmax')
# ])

# cnn.compile(loss='sparse_categorical_crossentropy', 
#             optimizer='adam', metrics=['accuracy'])

# cnn.fit(X_train, y_train, epochs=10, batch_size=16,
#         validation_data=(X_valid, y_valid))


# ------------------ task 4 --------------------

# from functools import partial
# DefaultConv2D = partial(Conv2D, kernel_size=3, padding="same",
#                                 activation="relu", kernel_initializer="he_normal")

# model = Sequential([
#     Input(X_train.shape[1:]),
#     DefaultConv2D(filters=64, kernel_size=7),
#     MaxPool2D(),

#     DefaultConv2D(filters=128),
#     DefaultConv2D(filters=128),
#     MaxPool2D(),

#     DefaultConv2D(filters=256),
#     DefaultConv2D(filters=256),
#     MaxPool2D(),

#     Flatten(),
#     Dense(units=128, activation="relu", kernel_initializer="he_normal"),
#     Dropout(0.5),

#     Dense(units=64, activation="relu", kernel_initializer="he_normal"),
#     Dropout(0.5),

#     Dense(units=10, activation="softmax")
# ])

# model.compile(loss='sparse_categorical_crossentropy', 
#             optimizer='adam', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=10, batch_size=16,
#         validation_data=(X_valid, y_valid))




import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from tf_model import *


# tf.random.set_seed(42)
# np.random.seed(42)

# ocean_prox = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

# str_lookup_layer = StringLookup()
# str_lookup_layer.adapt(ocean_prox)

# lookup_and_embed = Sequential([
#     str_lookup_layer,
#     Embedding(input_dim=str_lookup_layer.vocabulary_size(), output_dim=2)
# ])


# X_train_num = np.random.rand(10_000, 8)
# X_train_cat = np.random.choice(ocean_prox, size=10_000)
# y_train = np.random.rand(10_000, 1)

# X_valid_num = np.random.rand(2_000, 8)
# X_valid_cat = np.random.choice(ocean_prox, size=2_000)
# y_valid = np.random.rand(2_000, 1)



# num_input = Input(shape=[8], name="num")
# cat_input = Input(shape=[], name="cat")

# cat_embeddings = lookup_and_embed(cat_input)
# encoded_inputs = Concatenate()([num_input, cat_embeddings])
# outputs = Dense(1)(encoded_inputs)

# model = Model(inputs=[num_input, cat_input], outputs=[outputs])
# model.compile(loss="mse", optimizer="sgd", metrics=['rmse'])

# # history = model.fit((X_train_num, X_train_cat), y_train, epochs=5,
# #                     validation_data=((X_valid_num, X_valid_cat), y_valid))

# history = model.fit({'num': X_train_num, 'cat': X_train_cat}, y_train, epochs=5,
#                     validation_data=({'num': X_valid_num, 'cat': X_valid_cat}, y_valid))

# ==============================================================================




# path = Path("Machine_Learning/tensorflow/datasets/CTA_-_Ridership_-_Daily_Boarding_Totals_20260211.csv")
# df = pd.read_csv(path, parse_dates=["service_date"])
# df.columns = ["date", "day_type", "bus", "rail", "total"] # shorter names
# df = df.sort_values("date").set_index("date")
# df = df.drop("total", axis=1) # no need for total, it's just bus + rail
# df = df.drop_duplicates() # remove duplicated months (2011-10 and 2014-07)

# df['bus'] = df['bus'].str.replace(",", "").astype(float)
# df['rail'] = df['rail'].str.replace(",", "").astype(float)


# bus_rail_train = df[['bus', 'rail']][:'2018-12'].astype(np.float32) / 1e6
# bus_rail_test = df[['bus', 'rail']]['2019-01':'2019-12'].astype(np.float32) / 1e6
# seq_length = 7

# time_data = []
# data_len = len(bus_rail_train)

# for i in range(data_len - seq_length):
#     start = i
#     end   = start + seq_length

#     part = bus_rail_train.iloc[start:end].to_numpy()
#     time_data.append(part)

# train_data = tf.convert_to_tensor(time_data)
# y_train = tf.convert_to_tensor(bus_rail_train.iloc[seq_length:].to_numpy())

# time_data = []
# data_len = len(bus_rail_test)

# for i in range(data_len - seq_length):
#     start = i
#     end   = start + seq_length

#     part = bus_rail_test.iloc[start:end].to_numpy()
#     time_data.append(part)

# test_data = tf.convert_to_tensor(time_data)
# y_test = tf.convert_to_tensor(bus_rail_test.iloc[seq_length:].to_numpy())


# multivar_model = Sequential([
#     Input([7, 2]),
#     SimpleRNN(16),
#     Dense(2) 
# ])

# multivar_model.compile(optimizer='adam',
#                      loss='mse',
#                      metrics=['mse'])

# print(train_data.shape, y_train.shape)
# print(test_data.shape, y_test.shape)

# multivar_model.fit(train_data, y_train, validation_data=(test_data, y_test), epochs=40)

























# bus_rail_train = df[['rail']]['2010':'2018-12'].astype(np.float32) / 1e6
# bus_rail_test = df[['rail']]['2019-01':'2019-12'].astype(np.float32) / 1e6
# seq_length = 10

# time_data = []
# data_len = len(bus_rail_train)

# for i in range(data_len - seq_length):
#     start = i
#     end   = start + seq_length

#     part = bus_rail_train.iloc[start:end].to_numpy()
#     time_data.append(part)

# X_train = tf.convert_to_tensor(time_data)
# y_train = tf.convert_to_tensor(bus_rail_train.iloc[seq_length:].to_numpy())

# time_data = []
# data_len = len(bus_rail_test)

# for i in range(data_len - seq_length):
#     start = i
#     end   = start + seq_length

#     part = bus_rail_test.iloc[start:end].to_numpy()
#     time_data.append(part)

# X_test = tf.convert_to_tensor(time_data)
# y_test = tf.convert_to_tensor(bus_rail_test.iloc[seq_length:].to_numpy())



# deep_model = Sequential([
#     Input([seq_length, 1]),
#     SimpleRNN(32, return_sequences=True),
#     SimpleRNN(32, return_sequences=True),
#     SimpleRNN(32),
#     Dense(1)
# ])

# deep_model.compile(optimizer='adam',
#                      loss='mse',
#                      metrics=['mse'])

# deep_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))







# df_mulvar = df[["bus", "rail"]] / 1e6
# df_mulvar["next_day_type"] = df["day_type"].shift(-1)
# df_mulvar = pd.get_dummies(df_mulvar).astype(np.float32)

# mulvar_train = df_mulvar["2016-01":"2018-12"]
# mulvar_valid = df_mulvar["2019-01":"2019-05"]
# mulvar_test = df_mulvar["2019-06":]

# def to_windows(dataset, length):
#     dataset = dataset.window(length, shift=1, drop_remainder=True)
#     return dataset.flat_map(lambda window_ds: window_ds.batch(length))

# seq_length = 56

# def to_seq2seq_dataset(series, seq_length=56, ahead=14, target_col=1,
#                                 batch_size=32, shuffle=False, seed=None):
#     from tf_data import Dataset
                                
#     ds = to_windows(Dataset.from_tensor_slices(series), ahead + 1)
#     ds = to_windows(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, target_col]))
    
#     if shuffle:
#         ds = ds.shuffle(8 * batch_size, seed=seed)
    
#     return ds.batch(batch_size)


# seq2seq_train = to_seq2seq_dataset(mulvar_train, shuffle=True, seed=42)
# seq2seq_valid = to_seq2seq_dataset(mulvar_valid)

# from tf_model import Sequential, SimpleRNN, Dense, Input, EarlyStopping, Adam
# seq2seq_model = Sequential([
#     Input([None, 5]),
#     SimpleRNN(32, return_sequences=True),
#     Dense(14)
# ])

# seq2seq_model.compile(loss='mae', optimizer=Adam(0.01), metrics=['mape'])
# callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
# seq2seq_model.fit(seq2seq_train, epochs=100, validation_data=(seq2seq_valid), callbacks=callbacks)





shakespeare_url = "https://homl.info/shakespeare" # shortcut URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)

with open(filepath) as f:
    shakespeare_text = f.read()

text_vec_layer = tf.keras.layers.TextVectorization(split="character", standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]

encoded -= 2 
n_tokens = text_vec_layer.vocabulary_size() - 2 
dataset_size = len(encoded)

length = 100

def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))

    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)

    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)

stateful_train_set = to_dataset(tf.cast(encoded, tf.float32)[:10_000], length)
stateful_valid_set = to_dataset(tf.cast(encoded, tf.float32)[10_000:13_000], length)

from tf_model import *

model = Sequential([
    Input([None], batch_size=32),
    Embedding(input_dim=n_tokens, output_dim=16),
    GRU(128, return_sequences=True, stateful=True),
    Dense(n_tokens, activation="softmax")
])

class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(stateful_train_set, validation_data=(stateful_valid_set),
epochs=10, callbacks=[ResetStatesCallback()])


