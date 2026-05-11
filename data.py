import os
import numpy as np
import matplotlib.pyplot as plt


def load_training_images(path):
    images = []
    labels = []
    for label in sorted(os.listdir(path)):
        label_path = os.path.join(path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            img = plt.imread(file_path)
            labels.append(int(label))
            images.append(img)
    return np.array(images), np.array(labels)

def preprocess_data(X):
    X = X.reshape(X.shape[0], -1)
    X = X / 255.0
    print(f"X.shape: {X.shape}")
    return X
    
def visualize_sample(X, y, index):
    plt.imshow(X[index].reshape(28, 28))
    plt.title(f"Label: {y[index]}")
    plt.show()