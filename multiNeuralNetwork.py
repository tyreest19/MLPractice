import numpy as np
import scipy.special

class MultiLayerNeuralNetwork:

    def __init__(self, learing_rate=0.00001):
        self.learing_rate = learing_rate
        self.weights_hidden = np.random.normal(3, 3)
        self.weights_outputs = np.random.normal(3, 3)

    def activation_function(self, x):
        return scipy.special.expit(x)

    def test(self, test_inputs):
        hidden_output = np.dot(self.weights_hidden, test_inputs)
        hidden_output = self.activation_function(hidden_output)
        output = np.dot(self.weights_outputs, hidden_output)
        output = self.activation_function(output)
        return output

if __name__ == '__main__':
    neuralNetwork = MultiLayerNeuralNetwork()
    input_nodes = np.random.rand(3, 1)
    output = neuralNetwork.test(input_nodes)
    print(output)
