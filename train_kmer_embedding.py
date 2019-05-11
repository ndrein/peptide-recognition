import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from embed import embed
from Bio.SubsMat.MatrixInfo import blosum62

# data = next(pd.read_csv('indices.csv', delimiter=',', header=None, chunksize=None)).to_numpy()
# data = next(pd.read_csv('kmer_embedded.csv', delimiter=',', header=None, chunksize=20000)).to_numpy()
data = pd.read_csv('kmer_embedded.csv', delimiter=',', header=None).to_numpy()
x_train, y_train = data[:, :-1], data[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.33, random_state=42)
# y_train = y_train[..., np.newaxis]
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

num_kmers = 23 ** 3
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(num_kmers + 1, 64),
#     tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
model = tf.keras.Sequential([
    # tf.keras.layers.Embedding(num_kmers + 1, 4, input_length=417), # Add 1 for 0
    tf.keras.layers.Embedding(num_kmers + 1, 64, input_length=417), # Add 1 for 0
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.compile(loss='mse',
#               optimizer='adam',
#               metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, validation_split=0.3, batch_size=64, epochs=6, callbacks=[tf.keras.callbacks.ModelCheckpoint('model2.h5')])
