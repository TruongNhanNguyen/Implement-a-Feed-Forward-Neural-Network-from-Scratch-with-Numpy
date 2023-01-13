import numpy as np
EPS = np.finfo(np.float64).eps

# Loss functions + derivatives


def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true)**2, axis=1).reshape((-1, 1))


def d_mean_squared_error(y_pred, y_true):
    return np.expand_dims((2 / y_pred.shape[1]) * (y_pred - y_true), 1)


def categorical_crossentropy(y_pred, y_true):
    return -np.log(np.sum(y_true * y_pred, axis=1) + EPS)


def d_categorical_crossentropy(y_pred, y_true):
    return np.expand_dims(-y_true / (y_pred + EPS), 1)

def binary_crossentropy(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + EPS) + (1 - y_true) * np.log(1 - y_pred + EPS))

def d_binary_crossentropy(y_pred, y_true):
    return -np.expand_dims((y_true / (y_pred + EPS) - (1 - y_true) / (1 - y_pred + EPS)), 1)

def kullback_leibler_divergence(y_pred, y_true):
    return np.sum(y_true * np.log(y_true / (y_pred + EPS)), axis=1)

def d_kullback_leibler_divergence(y_pred, y_true):
    return -np.expand_dims(y_true / (y_pred + EPS), 1)

def poisson(y_pred, y_true):
    return np.mean((y_pred - y_true * np.log(y_pred + EPS)), axis=1)

def d_poisson(y_pred, y_true):
    return np.expand_dims((y_pred - y_true) / (y_pred + EPS), 1)
