import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_gradient(x):
    s = softmax(x)
    return s*(1-s)


def cross_entropy(x):
    return -np.log(x)


def cross_entropy_gradient(x):
    return -1/x


def regularized_cross_entropy(layers, lam, x):
    loss = cross_entropy(x)
    for layer in layers:
        loss += lam * (np.linalg.norm(layer.get_weights()) ** 2)
    return loss


def leakyRelu(x, alpha=0.01):
    return x * alpha if x < 0 else x


def leakyRelu_gradient(x, alpha=0.01):
    return alpha if x < 0 else 1


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_gradient(x):
    s = sigmoid(x)
    return s*(1-s)


def relu(x):
    return np.maximum(0, x)


def relu_gradient(x):
    return np.where(x > 0, 1, 0)


def lr_schedule(learning_rate, iteration):
    if (iteration >= 0) and (iteration <= 10000):
        return learning_rate
    if iteration > 10000:
        return learning_rate * 0.1