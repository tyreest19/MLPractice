import numpy as np

class FeedForwardNeuralNetwork:
    # AND Network
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.weights = np.random.rand(3, self.num_layers)
        # print('weights', self.weights)
        # print('shape of weights', np.shape(self.weights))
        #self.learing_rate = learing_rate

    def activation_function(self, x):
        return 1/(1+ np.exp(-x))

    def derivative_activation_function(self, x):
        return x * (1 - x)

    def train(self, iterations, training_set_inputs, training_set_outputs):
        for i in range(iterations):
            prediction = self.test(training_set_inputs)
            error = training_set_outputs - prediction
            # print('error is', error)
            # print('print prediction is', prediction)
            update = np.dot(training_set_inputs.T, error * self.derivative_activation_function(prediction))
            # print('error mumbo jumbo', error * self.derivative_activation_function(prediction))
            # print('updated weights variable result', update)
            print(self.weights)
            self.weights = self.weights + update


    def test(self, test_inputs):
        # print(self.weights)
        test_inputs = np.asarray(test_inputs)
        z = np.dot(test_inputs, self.weights)[0]
        print('test_inputs', test_inputs)
        return self.activation_function(z)

if __name__ == '__main__':
    network = FeedForwardNeuralNetwork(1)
    training_set_inputs = np.asarray([[0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.asarray([[0, 1, 1, 1]]).T
    network.train(5000, training_set_inputs, training_set_outputs)
    print(network.test([[1,0,0]]))
