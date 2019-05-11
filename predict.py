import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('indices.csv', delimiter=',', header=None).to_numpy()
x_train, y_train = data[:, 0], data[:, 1]

model = tf.keras.models.load_model('model.h5')
print(model.evaluate(x_train, y_train))
