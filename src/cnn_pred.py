from sklearn import datasets
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.activations import relu

import matplotlib.pyplot as plt

np.random.seed(0)
tf.random.set_seed(0)

raw_wine = datasets.load_wine()

X = raw_wine.data
y = raw_wine.target

print(X.shape, y.shape)
print(set(y))

y_hot = to_categorical(y)

X_tn, X_te, y_tn, y_te = train_test_split(X, y_hot, random_state=0)

n_feat = X_tn.shape[1]
n_class = len(set(y))
epo = 100

model = Sequential()
model.add(Dense(20, input_dim=n_feat))
model.add(BatchNormalization())
model.add(Activation(relu))
model.add(Dense(n_class))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_tn, y_tn, epochs=epo, batch_size=5)

print(model.evaluate(X_tn, y_tn)[1])
print(model.evaluate(X_te, y_te)[1])

epoch = np.arange(1, epo+1)
accuracy = hist.history['accuracy']
loss = hist.history['loss']

plt.plot(epoch, accuracy, label='Training accuracy')
