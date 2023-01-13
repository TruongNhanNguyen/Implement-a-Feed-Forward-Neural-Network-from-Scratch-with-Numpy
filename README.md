# Implement a Feed Forward Neural Network from Scratch with Numpy

This is an implementation of a feed forward neural network (FFNN) from scratch using numpy. The goal of this project is to provide a simple and easy-to-understand implementation of a FFNN that can be used as a starting point for further experimentation and research.

The implementation includes the following features:

- Support for fully connected layers
- Support for different types of activation functions (`sigmoid`, `ReLU`, `tanh`, etc.)
- Support for different types of loss functions (mean squared error, categorical cross-entropy, etc.)
- Support for different types of optimizers (stochastic gradient descent, Adam, etc.)
- Support for batch normalization
- Support for dropout regularization
- Support for `L1` and `L2` regularization

The implementation is fully functional and can be used to train and test a FFNN on a variety of tasks. The code is well-documented and easy to understand, making it a great starting point for anyone interested in understanding the inner workings of a FFNN.

## Usage

To use the implementation, simply import the `NeuralNetwork` class and instantiate it with the desired number of layers, activation functions, loss function, and optimizer. The train method can then be used to `train` the model on a given dataset, and the `predict` method can be used to make predictions on new data.

```python
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
```

## Requirements

This implementation requires `numpy` to be installed. It is recommended to use the latest version of `numpy`.

## Future Work

- Adding more activation functions
- Adding more loss functions
- Adding more optimizers
- Adding more test cases to evaluate the Neural Network implementation

## Contribution

If you're interested in contributing to this project, please feel free to open a pull request or contact me. Any contributions, big or small, are always welcome

## License

This project is licensed under the MIT License - see the [LICENSE](https://chat.openai.com/chat/LICENSE) file for details