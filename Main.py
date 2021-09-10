from NeuralNetwork import NeuralNetwork
import random
import numpy as np


training_data = [
    {"inputs": np.array([[1], [1]]), "answer": np.array([[0]])},
    {"inputs": np.array([[0], [1]]), "answer": np.array([[1]])},
    {"inputs": np.array([[1], [0]]), "answer": np.array([[1]])},
    {"inputs": np.array([[0], [0]]), "answer": np.array([[0]])}
]

nn = NeuralNetwork(2, 2, 1)

# training thr neural network
for i in range(50000):
    data = random.choice(training_data)
    nn.train(data["inputs"], data["answer"])

print(nn.feedforward(np.array([[0], [0]])))
print(nn.feedforward(np.array([[0], [1]])))
print(nn.feedforward(np.array([[1], [0]])))
print(nn.feedforward(np.array([[1], [1]])))

