from losses import *
from activations import *
import numpy as np

# Our neural Network class


class NeuralNetwork:
    def __init__(self, layers, hidden_activation, output_activation, loss, d_loss, optimizer, dropout=0, batch_norm=False, l1=0, l2=0):
        """
        # Parameters:
            - layers: a list of integers representing the number of neurons in each layer.
            - hidden_activation: activation function for hidden layers
            - output_activation: activation function for output layer
            - loss: loss function
            - d_loss: derivative of the loss function
            - optimizer: optimizer
            - dropout: dropout rate
            - batch_norm: whether to use batch normalization
            - l1: L1 regularization rate
            - l2: L2 regularization rate
        """

        self.layers = layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.d_loss = d_loss
        self.optimizer = optimizer
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.l1 = l1
        self.l2 = l2

        self.weights = []
        self.biases = []
        self.z = []
        self.a = []
        self.batch_mean = []
        self.batch_var = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]))
            self.biases.append(np.random.randn(1, layers[i+1]))
            if self.batch_norm:
                self.batch_mean.append(np.zeros(layers[i+1]))
                self.batch_var.append(np.zeros(layers[i+1]))

    def forward(self, x):
        """Performs a forward pass on the data"""

        self.z = []
        self.a = [x]

        for i in range(len(self.layers) - 2):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            if self.batch_norm:
                self.batch_mean[i] = np.mean(z,
                                             axis=0)
                self.batch_var[i] = np.var(z, axis=0)
                z = (z - self.batch_mean[i]) / \
                    np.sqrt(self.batch_var[i] + 1e-8)
                if self.dropout:
                    z *= np.random.binomial([np.ones_like(z)],
                                            (1-self.dropout))[0]
                self.z.append(z)
                a = self.hidden_activation(z)
                self.a.append(a)

            z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
            self.z.append(z)
            a = self.output_activation(z)
            self.a.append(a)

            return a

    def backward(self, x, y, d_loss):
        """Performs a backward pass on the data"""

        delta = d_loss(self.a[-1], y)
        gradients = []

        for i in range(len(self.layers) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i].T)
            if self.batch_norm:
                delta = delta * (self.batch_var[i] + 1e-8)**(-1/2)
            if self.hidden_activation == d_relu:
                delta *= (self.z[i] > 0).astype(int)
            elif self.hidden_activation == d_sigmoid:
                delta *= self.a[i] * (1 - self.a[i])
            elif self.hidden_activation == d_tanh:
                delta *= (1 - self.a[i]**2)
            elif self.hidden_activation == d_leaky_relu:

                delta *= np.vectorize(lambda v: 1 if v > 0 else 0.01)(self.z[i]
                                                                      )
            elif self.hidden_activation == d_elu:
                delta *= np.vectorize(lambda v: v if v >
                                      0 else self.alpha*(np.exp(v)-1))(self.z[i])
            gradients.append((np.dot(self.a[i].T, delta) + self.l1*np.sign(
                self.weights[i]) + self.l2*self.weights[i])/x.shape[0])
            self.biases[i] -= self.optimizer.learning_rate * \
                np.mean(delta, axis=0)

        return gradients[::-1]

    def train(self, x, y, x_val=None, y_val=None, epochs=100, batch_size=32, early_stopping=True, verbose=True):
        """Trains the model on the data"""

        if early_stopping and (x_val is None or y_val is None):
            raise ValueError(
                "x_val and y_val must be provided if early stopping is enabled")

        best_params = (np.copy(self.weights), np.copy(self.biases))
        best_val_loss = float("inf")
        counter = 0

        for i in range(epochs):
            for j in range(0, x.shape[0], batch_size):
                x_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]

                y_pred = self.forward(x_batch)
                loss = self.loss(y_pred, y_batch)

                if verbose:
                    print(
                        f"Epoch {i+1}/{epochs}, Batch {j/batch_size+1}/{x.shape[0]/batch_size}, Loss: {np.mean(loss)}")

                gradients = self.backward(x_batch, y_batch, self.d_loss)

                if x_val is not None and y_val is not None:
                    y_val_pred = self.forward(x_val)
                    val_loss = self.loss(y_val_pred, y_val)

                    if np.mean(val_loss) < best_val_loss:
                        best_val_loss = np.mean(val_loss)
                        best_params = (np.copy(self.weights),
                                       np.copy(self.biases))
                        counter = 0
                    else:
                        counter += 1

                    if counter >= 10 and early_stopping:
                        self.weights, self.biases = best_params
                        return

    def predict(self, x):
        """Predicts the output for a given input"""
        return self.forward(x)
