import numpy as np

EPS = np.finfo(np.float64).eps

# Activation functions + derivatives


def identity(x):
    """Output activation function `identity`"""
    return x


def d_identity(x):
    """Derivative of output activation function `identity`."""
    return np.expand_dims(np.identity(x.shape[1]), 0)


def relu(x):
    """Hidden activation function `ReLU`."""
    np.maximum(x, 0, x)
    return x


def d_relu(x):
    """Derivative activation function `ReLU`."""
    return np.where(x > 0, 1, 0)


def softmax(x):
    """Output activation function `softmax`"""
    x = x - x.max(axis=1).reshape((-1, 1))
    exp = np.exp(x)
    s = np.sum(exp, axis=1).reshape((-1, 1))
    return exp / (s+EPS)


def d_softmax(x):
    """Derivative of output activation function `identity`"""
    s = softmax(x)
    return s - np.einsum('ij,ij->i', s, s)[:, np.newaxis]


def sigmoid(x):
    """Hidden activation function `sigmoid`."""
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def d_sigmoid(x):
    """Derivative of hidden activation function `sigmoid`."""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """Hidden activation function `tanh`."""
    return np.tanh(x)


def d_tanh(x):
    """Derivative of hidden activation function `tanh`."""
    t = tanh(x)
    return 1 - t**2


def leaky_relu(x, alpha=0.01):
    """Hidden activation function `leaky ReLU`."""
    return np.where(x > 0, x, x * alpha)


def d_leaky_relu(x, alpha=0.01):
    """Derivative of hidden activation function `leaky ReLU`."""
    return np.where(x > 0, 1, alpha)


def elu(x, alpha=1):
    """Hidden activation function `ELU`."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def d_elu(x, alpha=1):
    """Derivative of hidden activation function `ELU`."""
    return np.where(x > 0, 1, elu(x) + alpha)
