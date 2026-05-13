#!/usr/bin/env python3
"""Defines a deep neural network performing multiclass classification."""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep neural network for multiclass classification."""

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize a DeepNeuralNetwork.

        Args:
            nx (int): Number of input features.
            layers (list): Nodes per layer.
            activation (str): 'sig' for sigmoid or 'tanh' for tanh.

        Raises:
            TypeError: If types are invalid.
            ValueError: If values are out of range or activation is
                not 'sig' or 'tanh'.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights["W1"] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                )
            else:
                self.__weights["W" + str(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1])
                    * np.sqrt(2 / layers[i - 1])
                )
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary."""
        return self.__weights

    @property
    def activation(self):
        """Getter for the activation function type."""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: (output, cache).
        """
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            A_prev = self.__cache["A" + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            if i == self.__L:
                t = np.exp(Z)
                self.__cache["A" + str(i)] = t / np.sum(t, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))
                else:
                    self.__cache["A" + str(i)] = np.tanh(Z)
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculate the cross-entropy cost.

        Args:
            Y (numpy.ndarray): One-hot labels with shape (classes, m).
            A (numpy.ndarray): Activated output with shape (classes, m).

        Returns:
            float: The cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): One-hot labels with shape (classes, m).

        Returns:
            tuple: (one-hot prediction, cost).
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A == np.amax(A, axis=0), 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculate one pass of gradient descent.

        Args:
            Y (numpy.ndarray): One-hot labels.
            cache (dict): All intermediary values.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache["A" + str(i - 1)]
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if i > 1:
                W = weights_copy["W" + str(i)]
                if self.__activation == 'sig':
                    dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))
                else:
                    dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)
            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the deep neural network.

        Returns:
            tuple: The evaluation of training data after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph is True:
            import matplotlib.pyplot as plt
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Save the instance to a pickle file."""
        if not isinstance(filename, str):
            return
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object."""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return None
