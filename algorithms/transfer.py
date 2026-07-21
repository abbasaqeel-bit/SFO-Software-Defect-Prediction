import numpy as np


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def position_to_binary(position):
    probabilities = sigmoid(position)
    binary = (probabilities >= 0.5).astype(int)

    if np.sum(binary) == 0:
        random_index = np.random.randint(0, len(binary))
        binary[random_index] = 1

    return binary