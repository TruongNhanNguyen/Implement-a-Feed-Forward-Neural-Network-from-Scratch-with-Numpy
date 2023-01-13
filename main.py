from losses import *
from activations import *
from optimizers import *
from ff_nn import NeuralNetwork
from utils import to_categorical
import numpy as np

if __name__ == '__main__':
    # Generating dummy data
    np.random.seed(0)
    x = np.random.rand(1000, 100)
    y = np.random.rand(1000, 10)

    x_train, x_val, y_train, y_val = x[:900], x[900:], y[:900], y[900:]

    # Instantiating the NeuralNetwork class
    model = NeuralNetwork([100, 50, 20, 10], relu, softmax, d_softmax, categorical_crossentropy, SGD(
        learning_rate=0.01, momentum=0.9))

    # Training the model with validation set
    model.train(x_train, y_train, x_val, y_val, epochs=50, batch_size=32)

    # Testing the model
    y_pred = model.predict(x_val)
