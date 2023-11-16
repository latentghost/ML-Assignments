import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def sigmoid_gradient(A):
    dA = A * (1 - A)
    return dA


def tanh(Z):
    A = np.tanh(Z)
    return A


def tanh_gradient(A):
    dA = 1 - np.square(A)
    return dA


def relu(Z):
    A = np.maximum(0, Z)
    return A


def relu_gradient(Z):
    dZ = np.where(Z > 0, 1, 0)
    return dZ


def leaky_relu(Z, alpha=0.01):
    A = np.where(Z > 0, Z, alpha * Z)
    return A


def leaky_relu_gradient(Z, alpha=0.01):
    dZ = np.where(Z > 0, 1, alpha)
    return dZ


def linear(Z):
    A = Z
    return A


def linear_gradient(Z):
    dZ = np.ones_like(Z)
    return dZ


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    return A


def softmax_gradient(A, Y):
    dA = A - Y
    return dA


def zero_init(shape):
    # Initialize weights to all 0s
    weight = np.zeros(shape)
    return weight


def random_init(shape):
    # Initialize weights with random values between -1 and 1
    weight = np.random.uniform(low=-1, high=1, size=shape)
    return weight


def normal_init(shape):
    # Initialize weights with random values from a normal distribution with mean 0 and standard deviation 1
    weight = np.random.normal(0, 1, size=shape)
    return weight