import math
import numpy as np
import random

class NeuralNetwork:
    def __init__(self, inp, hidden, output):
        self.input_nodes = inp
        self.hidden_nodes = hidden
        self.output_nodes = output

        self.weights_ih = np.zeros((self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.zeros((self.output_nodes, self.hidden_nodes))
        self.weights_ih = self.randomize(self.weights_ih)
        self.weights_ho = self.randomize(self.weights_ho)

        self.bias_h = np.zeros((self.hidden_nodes, 1))
        self.bias_o = np.zeros((self.output_nodes, 1))
        self.bias_h = self.randomize(self.bias_h)
        self.bias_o = self.randomize(self.bias_o)

        self.learning_rate = 0.1

    def randomize(self, m: np.ndarray):
        with np.nditer(m, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = random.random() * 2 - 1
        return m

    def feedforward(self, inp: np.ndarray):
        # calculating outputs of the hidden layer
        hidden = self.weights_ih @ inp
        hidden += self.bias_h

        # activation function
        def sigmoid(x):
            return 1/(1+math.exp(1)**(-x))

        hidden = sigmoid(hidden)

        output = self.weights_ho @ hidden
        output += self.bias_o
        output = sigmoid(output)

        return output

    def dsigmoid(self, y):
        # sigmoid(x) * (1 - sigmoid(x))
        return y * (1 - y)

    def train(self, inputs: np.ndarray, answers: np.ndarray):
        hidden = self.weights_ih @ inputs
        hidden += self.bias_h

        # activation function
        def sigmoid(x):
            return 1/(1+math.exp(1)**(-x))

        hidden = sigmoid(hidden)

        outputs = self.weights_ho @ hidden
        outputs += self.bias_o
        outputs = sigmoid(outputs)
        output_errors = answers - outputs

        gradients = self.dsigmoid(outputs)
        gradients = output_errors * gradients
        gradients *= self.learning_rate

        hidden_t = hidden.T
        weight_ho_deltas = gradients @ hidden_t

        # adjusting the weights and biases by their deltas
        self.weights_ho += weight_ho_deltas
        self.bias_o += gradients

        # calculating hidden layer errors
        who_t = self.weights_ho.T
        hidden_errors = who_t @ output_errors

        # calculating hidden gradients
        hidden_gradients = self.dsigmoid(hidden)
        hidden_gradients = hidden_gradients * hidden_errors
        hidden_gradients *= self.learning_rate

        # calculating input to hidden deltas
        inputs_t = inputs.T
        weight_ih_deltas = hidden_gradients * inputs_t

        # adjusting the weights and biases by their deltas
        self.weights_ih += weight_ih_deltas
        self.bias_h += hidden_gradients
