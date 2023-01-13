import numpy as np

def to_categorical(labels):
    """Convert class labels in classsification tasks to one-hot
    encoding"""
    n_classes = labels.max() + 1
    return np.eye(n_classes)[labels]
