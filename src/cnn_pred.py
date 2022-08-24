from sklearn import datasets
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.activations import relu

import matplotlib.pyplot as plt


def cnn_pred(X_t, X_te, y_t, y_te):
	x_train = X_t.to_numpy().astype(np.float32)
	y_train = y_t.to_numpy().astype(np.float32)
	x_test = X_te.to_numpy().astype(np.float32)
	y_test = y_te.to_numpy().astype(np.float32)
	print(x_train.shape, y_train.shape)
	print(set(y_train))

	y_hot = to_categorical(y_train)
	n_feat = x_train.shape[1]
	n_class = len(set(y_train))
	epo = 100

	model = Sequential()
	model.add(Dense(20, input_dim=n_feat))
	model.add(BatchNormalization())
	model.add(Activation(relu))
	model.add(Dense(n_class))
	model.add(Activation('relu'))

	model.summary()

	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	hist = model.fit(x_train, y_train, epochs=epo, batch_size=5)

	print(model.evaluate(x_train, y_train)[1])
	print(model.evaluate(x_test, y_test)[1])

	epoch = np.arange(1, epo + 1)
	accuracy = hist.history['accuracy']
	loss = hist.history['loss']

	plt.plot(epoch, accuracy, label='Training accuracy')
	return np.argmax(model.predict(x_test), axis=1)
