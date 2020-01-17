import numpy as np

def sigmoid(X, weight):
    z = np.dot(X, weight)
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]


def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient


def log_likelihood(x, y, weights):
    z = np.dot(x, weights)
    ll = np.sum( y*z - np.log(1 + np.exp(z)) )
    return ll

def gradient_ascent(X, h, y):
    return np.dot(X.T, y - h)


def update_weight_mle(weight, learning_rate, gradient):
    return weight + learning_rate * gradient