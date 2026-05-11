import numpy as np

def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

def one_hot(y, num_classes):
    vector = np.zeros(num_classes)
    vector[y] = 1
    return vector

def relu_gradient(x):
    return (x>0).astype(float)