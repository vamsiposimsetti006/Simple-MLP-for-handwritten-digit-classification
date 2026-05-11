# MNIST MLP From Scratch

This project implements a multi-layer perceptron (MLP) from scratch for handwritten digit classification.

It manually implements:
- Forward propagation
- Backpropagation
- ReLU activation
- Softmax activation
- Cross-entropy loss
- SGD-style gradient descent updates

## Dataset

- **MNIST** database of handwritten digits
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

## Concepts Implemented

- Dense layers
- He initialization
- ReLU
- Softmax
- Cross entropy
- Backpropagation
- Gradient descent

## Architecture

`784 -> 128 -> 10`

## Validation Accuracy

`97%`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```
