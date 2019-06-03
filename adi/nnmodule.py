from typing import List

import numpy as np

from adi.utils import OperatorPair


class NNModule:
    def __init__(self, sizes: List[int], activation_operators: OperatorPair):
        self._sizes = sizes
        self._activ = activation_operators
        self._setup()

    def _setup(self):
        self._nlayers = len(self._sizes)
        self._z = [np.zeros((s, 1)) for s in self._sizes]
        self._a = [np.zeros((s, 1)) for s in self._sizes]
        self._init_weights()
        self._init_biases()

    def _init_weights(self):
        """Glorot initialization"""
        self._W = [np.random.randn(next, prev) * np.sqrt(2. / (prev + next))
                   for prev, next in zip(self._sizes[:-1], self._sizes[1:])]

    def _init_biases(self):
        self._b = [np.zeros((s, 1)) for s in self._sizes[1:]]

    def evaluate(self, X: np.array, activation_applied: bool = True) -> np.array:
        self._feed_forward(X)
        return self._a[-1] if activation_applied else self._z[-1]

    def learn(self, X: np.array, Y: np.array):
        raise NotImplemented

    def _feed_forward(self, X: np.array):
        assert X.shape[0] == self._sizes[0]
        assert X.shape[1] >= 1

        self._a[0] = self._z[0] = X
        for l in range(self._nlayers - 1):
            self._z[l + 1] = self._W[l] @ self._a[l] + self._b[l]
            self._a[l + 1] = self._activ.func(self._z[l + 1])

        assert self._a[-1].shape == (self._sizes[-1], X.shape[1])
        assert self._z[-1].shape == (self._sizes[-1], X.shape[1])
