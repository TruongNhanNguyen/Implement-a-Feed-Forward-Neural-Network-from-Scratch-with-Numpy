import numpy as np

# Optimizers


class SGD:
    """Stochastic gradient descent with momentum as an 
    optimization method. But you can adapt other optimizers
    such as `Adam`, `RMSprop`, `Adagrad` to the `Neural Network` class
    by using `.update(old_params, gradient)` method. It returns the
    updated parameters."""

    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, old_params, gradient):
        """Returns the updated parameters. The neural network class will 
        receive an optimizer as a parameter. So, someone who wants to use 
        other optimization methods can create a class with the required 
        interface and pass it to the neural network class when instantiating."""
        if not hasattr(self, 'delta_params'):
            self.delta_params = np.zeros_like(old_params)

        self.delta_params = self.momentum * \
            self.delta_params - self.learning_rate * gradient
        new_params = old_params + self.delta_params

        return new_params


class Adam:
    """Adam optimization algorithm. It is an optimization algorithm that uses moving averages of the parameters."""

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, old_params, gradient):
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)

        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient**2
        m_hat = self.m / (1 - self.beta_1**self.t)
        v_hat = self.v / (1 - self.beta_2**self.t)
        new_params = old_params - self.learning_rate * \
            m_hat / (np.sqrt(v_hat) + self.epsilon)

        return new_params


class RMSprop:
    """RMSprop optimization algorithm. It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters."""

    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon

        self.g_sq = None

    def update(self, old_params, gradient):
        if self.g_sq is None:
            self.g_sq = np.zeros_like(gradient)

        self.g_sq = self.decay_rate * self.g_sq + \
            (1 - self.decay_rate) * gradient**2
        new_params = old_params - self.learning_rate * \
            gradient / (np.sqrt(self.g_sq) + self.epsilon)

        return new_params
