from layers import Dense, relu
from utils import softmax, relu_gradient

class MLP:
    def __init__(self):
        self.fc1 = Dense(784, 128)

        self.fc2 = Dense(128, 10)

    def forward(self, x):
        # z1 = W1 * x + b1
        self.z1 = self.fc1.forward(x)
        self.a1 = relu(self.z1)
        # z2 = W2 * a1 + b2
        self.z2 = self.fc2.forward(self.a1)
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, dL_dz2):
        dL_da1 = self.fc2.backward(dL_dz2)
        dL_dz1 = dL_da1 * relu_gradient(self.z1)
        dx = self.fc1.backward(dL_dz1)
        return dx

    def update(self, learning_rate):
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)