from data import load_training_images, preprocess_data
from model import MLP
from losses import cross_entropy, cross_entropy_gradient
from utils import one_hot
import numpy as np
from sklearn.model_selection import train_test_split
# Load and preprocess data
X, y = load_training_images("data/trainingSet/trainingSet")
X = preprocess_data(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create model
model = MLP()

# Hyperparameters
epochs = 10
learning_rate = 0.01


for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_epoch = X_train[indices]
    y_epoch = y_train[indices]
    total_loss = 0

    for i in range(len(X_epoch)):
        # Get one sample
        x = X_epoch[i]
        y_true = one_hot(y_epoch[i], 10)

        # Forward pass
        y_pred = model.forward(x)

        # Loss
        loss = cross_entropy(y_true, y_pred)
        total_loss += loss
        
        # Output gradient
        dL_dz2 = cross_entropy_gradient(y_true, y_pred)

        # Backprop through whole model
        model.backward(dL_dz2)

        # Update whole model
        model.update(learning_rate)

    avg_loss = total_loss / len(X_epoch)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")


# Test one sample
correct = 0

for i in range(len(X_test)):
    output = model.forward(X_test[i])
    prediction = np.argmax(output)

    if prediction == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")