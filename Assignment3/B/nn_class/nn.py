from .utils import *

class NeuralNetwork:
    def __init__(self, N, input_size, layer_sizes, weight_init, bias_init):
        self.N = N
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        
        self.weight_init = weight_init
        self.bias_init = bias_init


    def initialize_weights_and_biases(self):
        weights = []
        biases = []

        for i in range(0, self.N):
            if(i==0):
                input_size = self.input_size
            else:
                input_size = self.layer_sizes[i - 1]
            output_size = self.layer_sizes[i]

            weight = self.weight_init(shape=(input_size, output_size))
            bias = self.bias_init(shape=(1, output_size))

            weights.append(weight)
            biases.append(bias)

        return weights, biases
    

    def forward_pass(self, X):
        layer_output = X
        layer_outputs = [X]
        
        for i in range(self.N):
            Z = np.dot(layer_output, self.weights[i]) + self.biases[i]
            if i == self.N - 1:
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
        
        for i in range(self.N - 2, -1, -1):
            dZ = dA_prev * self.grad_act(np.dot(self.layer_outputs[i], self.weights[i]) + self.biases[i])
            dW = np.dot(self.layer_outputs[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA_prev = np.dot(dZ, self.weights[i].T)
            
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db


    def fit(self, X, Y, lr, activation, grad_act, epochs, batch_size=None):
        self.lr = lr

        self.activation = activation
        self.grad_act = grad_act

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_nodes = X.shape[0]

        self.weights, self.biases = self.initialize_weights_and_biases()

        if(self.batch_size is None):
            self.batch_size = self.num_nodes

        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size): 
                X_batch = X[i:min(X.shape[0],i + self.batch_size)]
                Y_batch = Y[i:min(X.shape[0],i + self.batch_size)]

                self.forward_pass(X_batch)
                self.backward_pass(X_batch, Y_batch)


    def predict(self, X):
        self.forward_pass(X)
        return np.argmax(self.layer_outputs[-1], axis=1)


    def predict_proba(self, X):
        self.forward_pass(X)
        return self.layer_outputs[-1]


    def score(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.mean(Y_pred == Y)
        return accuracy
