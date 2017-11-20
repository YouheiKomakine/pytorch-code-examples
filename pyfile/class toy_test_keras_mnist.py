import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

from keras.utils.generic_utils import CustomObjectScope
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from keras.backend.tensorflow_backend import _to_tensor
from keras.backend.common import epsilon

def custom_categorical_crossentropy(target, output, from_logits=False, delta=1e-7):
    if not from_logits:
        output /= tf.reduce_sum(output,
                                axis=len(output.get_shape()) - 1,
                                keep_dims=True)
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output + delta),
                               axis=len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)


class toy_test_keras_mnist(object):
    def __init__(self, data_size=10000, n_hidden_layer = 3, n_hidden_cell=200, 
                 train_size=0.8, test_size=0.2, learning_rate=0.01, keep_prob=0.5, np_seed=None):
        self.data_size = data_size
        self.n_hidden_layer = max(n_hidden_layer, 1)
        self.n_hidden_cell = max(n_hidden_cell, 1)
        self.train_size = train_size
        self.test_size = test_size
        self.lr = learning_rate
        self.keep_prob = keep_prob
        self.np_seed = np_seed
        
    def set_each_dataset(self):
        mnist = fetch_mldata('MNIST original', data_home="./dataset")
        indices = np.random.permutation(len(mnist.data))[:self.data_size]
        self.X = mnist.data[indices]
        self.Y = mnist.target[indices]
        self.Y_onehot = np.eye(10)[self.Y.astype(int)]
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y_onehot, train_size=self.train_size, test_size=self.test_size)
        self.n_in, self.n_out = self.X_train[0].shape[0], self.Y_train[0].shape[0]

    def generate_model(self, activation_function='tanh', loss_function='categorical_crossentropy'):
        with CustomObjectScope({'custom_categorical_crossentropy':custom_categorical_crossentropy}):
            self.model = Sequential()

            self.model.add(Dense(self.n_hidden_cell, input_dim=self.n_in))
            self.model.add(Activation(activation_function))
            self.model.add(Dropout(self.keep_prob))

            for i in range(self.n_hidden_layer - 1):
                self.model.add(Dense(self.n_hidden_cell))
                self.model.add(Activation(activation_function))
                self.model.add(Dropout(self.keep_prob))

            self.model.add(Dense(self.n_out))
            self.model.add(Activation('softmax'))

            self.model.compile(loss = loss_function,
                               optimizer = SGD(lr=self.lr),
                               metrics=['accuracy']
                               )
        
        
    def fit(self, epochs=150, batch_size=200):
        with CustomObjectScope({'custom_categorical_crossentropy':custom_categorical_crossentropy}):
            self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size)
            loss_and_metrics = self.model.evaluate(self.X_test, self.Y_test)
            print(loss_and_metrics)
            
