from .utils import *
np.random.seed(21)


class Convolutional:
    def __init__(self, name, num_filters=3, stride=1, size=3, activation=None, activation_derivative=None):
        self.name = name
        self.filters = np.random.randn(num_filters, 3, 3)
        self.stride = stride
        self.size = size
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.prev = None

    def forward(self, inp):
        ## keep track of last input for later backward propagation
        self.prev = inp

        input_dimension = inp.shape[1]
        output_dimension = int((input_dimension - self.size) / self.stride) + 1

        out = np.zeros((self.filters.shape[0], output_dimension, output_dimension))

        ## for each kernel channel, convolve the input over the kernel
        for f in range(self.filters.shape[0]):
            ## actual convolution operation
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = inp[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    out[f, out_y, out_x] += np.sum(self.filters[f] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        if self.activation is not None:
            out = self.activation(out)
        return out

    def backward(self, din, learn_rate=0.005):
        input_dimension = self.prev.shape[1]

        if self.activation_derivative is not None:
            din = self.activation_derivative(din)

        ## gradient of loss
        dout = np.zeros(self.prev.shape)
        dfilt = np.zeros(self.filters.shape)

        for f in range(self.filters.shape[0]):
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = self.prev[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    dfilt[f] += np.sum(din[f, out_y, out_x] * patch, axis=0)
                    dout[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] += din[f, out_y, out_x] * self.filters[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        ## update the kernel weights and return loss gradient
        self.filters -= learn_rate * dfilt
        return dout

    def get_weights(self):
        return np.reshape(self.filters, -1)


class MaxPooling:
    def __init__(self, name, stride=2, size=2):
        self.name = name
        self.prev = None
        self.stride = stride
        self.size = size

    def forward(self, inp):
        self.prev = inp

        ## output dimensions
        num_channels, h_prev, w_prev = inp.shape
        h = int((h_prev - self.size) / self.stride) + 1
        w = int((w_prev - self.size) / self.stride) + 1

        ## max pooling
        downsampled = np.zeros((num_channels, h, w))

        ## pooling through vertical and horizontal elements
        for i in range(num_channels):
            curr_y = out_y = 0
            while curr_y + self.size <= h_prev:
                curr_x = out_x = 0
                while curr_x + self.size <= w_prev:
                    patch = inp[i, curr_y:curr_y + self.size, curr_x:curr_x + self.size]
                    downsampled[i, out_y, out_x] = np.max(patch) 
                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_y += 1

        return downsampled

    def backward(self, din, learning_rate):
        num_channels, orig_dim, *_ = self.prev.shape 

        dout = np.zeros(self.prev.shape)

        ## compute gradient of loss for pooling layer
        for c in range(num_channels):
            tmp_y = out_y = 0
            while tmp_y + self.size <= orig_dim:
                tmp_x = out_x = 0
                while tmp_x + self.size <= orig_dim:
                    patch = self.prev[c, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)
                    dout[c, tmp_y + x, tmp_x + y] += din[c, out_y, out_x]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        return dout

    def get_weights(self):
        return 0
