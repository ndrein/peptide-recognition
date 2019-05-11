import pandas as pd
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from embed import embed
from Bio.SubsMat.MatrixInfo import blosum62

# data = next(pd.read_csv('indices.csv', delimiter=',', header=None, chunksize=None)).to_numpy()
data = pd.read_csv('indices.csv', delimiter=',', header=None).to_numpy()
x_train, y_train = data[:, :-1], data[:, -1:]
print(x_train.shape, y_train.shape)
# data = pd.read_csv('embedded.csv', delimiter=',', header=None).to_numpy()
# x_train, y_train = data[:, :-1], data[:, -1:]
# print(x_train.shape, y_train.shape)
# pd.read_csv('embedded.csv', delimiter=',', header=None).to_csv('embedded.csv', header=None, float_format='%.3f', index=False)
# x_train, y_train = np.hsplit(pd.read_csv('train.csv', delimiter=',', header=None).to_numpy(), 2)
# x_train, y_train = x_train[:1000, 0], y_train[:1000, 0]
# x_train, y_train = x_train[:, 0], y_train[:, 0]
# x_train = embed(x_train, y_train)
# x_train, y_train = x_train[..., np.newaxis], y_train[..., np.newaxis]
# print(x_train.shape)
# print(y_train.shape)

# model = Sequential()
# model.add(LSTM(32, input_shape=(None, 1), activation='sigmoid'))
# model.add(Dense(1, activation='softmax'))
#
# model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])
# print(model.summary())
#
# model.fit(x_train, y_train, batch_size=64, epochs=50)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(24, 64, input_length=419),
    tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(24, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=10, callbacks=[tf.keras.callbacks.ModelCheckpoint('model.h5')])
