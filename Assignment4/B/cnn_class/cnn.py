from .layers import *
from .utils import *
import time



class CNN:
    def __init__(self):
        self.layers = []


    def build_model(self):
        self.layers.append(Convolutional(name='conv1', num_filters=8, stride=2, size=3, activation=relu))
        self.layers.append(MaxPooling(name='pool1', stride=2, size=2))
        self.layers.append(Convolutional(name='conv2', num_filters=2, stride=1, size=3, activation=relu))
        self.layers.append(MaxPooling(name='pool2', stride=2, size=2))


    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp


    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)


    def train(self, train_data, train_labels, val_data, val_labels, num_epochs, learning_rate, validate, regularization, verbose):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}:')

            loss = 0
            tmp_loss = 0
            num_corr = 0
            initial_time = time.time()
            
            for i in range(len(train_data)):
                if i % 100 == 99:
                    accuracy = (num_corr / (i + 1)) * 100
                    loss = tmp_loss / (i + 1)

                    history['loss'].append(loss)
                    history['accuracy'].append(accuracy)

                    if validate:
                        indices = np.random.permutation(val_data.shape[0])

                        val_loss, val_accuracy = self.predict(
                            val_data[indices,:],
                            val_labels[indices],
                            regularization,
                            verbose=0
                        )

                        history['val_loss'].append(val_loss)
                        history['val_accuracy'].append(val_accuracy)

                        if verbose:
                            print(f'[Step {i+1}]: Loss {loss} | Accuracy: {accuracy} | Time: {time.time()-initial_time} seconds | '
                                  f'Validation Loss {val_loss} | Validation Accuracy: {val_accuracy}')
                    elif verbose:
                        print(f'[Step {i+1}]: Loss {loss} | Accuracy: {accuracy} | Time: {time.time() - initial_time} seconds')

                    # restart time
                    initial_time = time.time()

                inp = train_data[i]
                label = train_labels[i]

                tmp_output = self.forward(inp)

                ## compute (regularized) cross-entropy and update loss
                tmp_loss += regularized_cross_entropy(self.layers, regularization, tmp_output[label])

                if np.argmax(tmp_output) == label:
                    num_corr += 1

                gradient = np.zeros(10)
                gradient[label] = -1 / tmp_output[label] + np.sum(
                    [2 * regularization * np.sum(np.absolute(layer.get_weights())) for layer in self.layers])

                # learning_rate = lr_schedule(learning_rate, iteration=i)

                self.backward(gradient, learning_rate)

        if verbose:
            print(f'Train Loss: {history["loss"][-1]}')
            print(f'Train Accuracy: {history["accuracy"][-1]}')


    def predict(self, X, y, regularization, verbose):
        loss, num_correct = 0, 0

        for i in range(len(X)):
            tmp_output = self.forward(X[i])
            loss += regularized_cross_entropy(self.layers, regularization, tmp_output[y[i]])
            prediction = np.argmax(tmp_output)

            if prediction == y[i]:
                num_correct += 1

        test_size = len(X)
        accuracy = (num_correct / test_size) * 100
        loss = loss / test_size

        if verbose:
            print(f'Test Loss: {loss}')
            print(f'Test Accuracy: {accuracy}')
        return loss, accuracy