from utils import *


class NeuralNetwork:
    def __init__(self, N, layer_sizes, lr, activation, weight_init, bias_init, epochs, batch_size):
        self.N = N
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.activation = activation
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.epochs = epochs
        self.batch_size = batch_size

        self.weights, self.biases = self.initialize_weights_and_biases()


    def initialize_weights_and_biases(self):
        weights = []
        biases = []

        for i in range(1, self.N):
            input_size = self.layer_sizes[i - 1]
            output_size = self.layer_sizes[i]

            # Initialize weights using the specified weight initialization function
            weight = self.weight_init(input_size, output_size)
            bias = np.zeros((1, output_size))

            weights.append(weight)
            biases.append(bias)

        return weights, biases
    

    def forward_pass(self, X):
        layer_output = X
        layer_outputs = [X]
        
        for i in range(self.N - 1):
            Z = np.dot(layer_output, self.weights[i]) + self.biases[i]
            if i == self.N - 2:
                layer_output = softmax(Z)
            else:
                layer_output = self.activation(Z)
            layer_outputs.append(layer_output)
        
        self.layer_outputs = layer_outputs


    def backward_pass(self, X, Y):
        m = X.shape[0]
        dZ = self.layer_outputs[-1] - Y
        dW = np.dot(self.layer_outputs[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, self.weights[-1].T)
        
        self.weights[-1] -= self.lr * dW
        self.biases[-1] -= self.lr * db
        
        for i in range(self.N - 2, 0, -1):
            dZ = dA_prev * (self.layer_outputs[i] > 0)
            dW = np.dot(self.layer_outputs[i - 1].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA_prev = np.dot(dZ, self.weights[i].T)
            
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db


    def fit(self, X, Y):
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size]
                Y_batch = Y[i:i + self.batch_size]

                self.forward_pass(X_batch)
                self.backward_pass(X_batch, Y_batch)


    def predict(self, X):
        self.forward_pass(X)
        return (self.layer_outputs[-1] > 0.5).astype(int)


    def predict_proba(self, X):
        self.forward_pass(X)
        return self.layer_outputs[-1]


    def score(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.mean(Y_pred == Y)
        return accuracy