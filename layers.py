import numpy as np

import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.W = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.b = np.zeros(n_neurons)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dL_dz):
        self.dW = np.outer(self.x, dL_dz)
        self.db = dL_dz
        dx = np.dot(dL_dz, self.W.T)
        return dx

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

def relu(x):
    return np.maximum(0, x)