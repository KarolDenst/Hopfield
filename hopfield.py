import numpy as np


class Hopfield:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train_hebb(self, patterns, lr=1.0):
        self.weights = np.zeros((self.size, self.size))

        for pattern in patterns:
            self.weights += lr * np.outer(pattern, pattern)

        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def train_oja(self, patterns, lr=0.01):
        self.weights = np.random.uniform(-1, 1, (self.size, self.size)) / np.sqrt(
            self.size
        )
        np.fill_diagonal(self.weights, 0)

        for _ in range(100):
            for pattern in patterns:
                x = pattern
                w = self.weights
                y = w.T @ x
                # dw = np.outer(y, x) - np.outer(y, y) @ w
                # dw = np.outer(y, (x - y @ w))
                dw = np.outer(x, x) - np.outer(y, x)
                self.weights += lr * dw
            self.weights /= np.linalg.norm(self.weights)

    def energy(self, state):
        return -0.5 * (state @ self.weights @ state)

    def update(self, state, num_iterations=100, update_type="sync"):
        state = state.copy()

        for _ in range(num_iterations):
            if update_type == "async":
                for i in np.random.permutation(self.size):
                    activation = np.dot(self.weights[i], state)
                    state[i] = 1 if activation >= 0 else -1
            else:
                activation = np.dot(self.weights, state)
                state = np.where(activation >= 0, 1, -1)

        return state
