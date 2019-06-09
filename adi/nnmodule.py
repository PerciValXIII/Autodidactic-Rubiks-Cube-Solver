from typing import List, Union

import numpy as np

from adi.utils import OperatorPair, sigmoid_operators, softmax, cross_entropy, ELU_operators, MSE_operators

MSE_COSTS = []
SOFTMAX_COSTS = []

class NNModule:
    def __init__(self, sizes: List[int], activation_operators: OperatorPair, learning_rate: float):
        self._sizes = sizes
        self._activ = activation_operators
        self._learning_rate = learning_rate
        self._setup()

    def _setup(self):
        self._nlayers = len(self._sizes)
        self._z = [np.zeros((s, 1)) for s in self._sizes]
        self._a = [np.zeros((s, 1)) for s in self._sizes]
        self._init_weights()
        self._init_biases()
        self._init_velocities()

        self._vel_factor = 0.9
        self._vel_eps = 1e-5

    def _init_weights(self):
        """Glorot initialization"""
        self._W = [np.random.randn(next, prev) * np.sqrt(2. / (prev + next))
                   for prev, next in zip(self._sizes[:-1], self._sizes[1:])]

    def _init_biases(self):
        self._b = [np.zeros((s, 1)) for s in self._sizes[1:]]

    def _init_velocities(self):
        self._velW = [np.zeros_like(w) for w in self._W]
        self._velb = [np.zeros_like(b) for b in self._b]

    def evaluate(self, X: np.array, activation_applied: bool = True) -> np.array:
        self._feed_forward(X)
        return self._a[-1] if activation_applied else self._z[-1]

    def learn(self, X: np.array, Y: Union[np.array, List[int]], weights: List[float]) -> np.array:
        raise NotImplemented

    def learn_from_delta(self, delta: np.array, rate: float):
        self._propagate_and_return_delta(delta, rate)

    def _feed_forward(self, X: np.array):
        assert X.shape[0] == self._sizes[0]
        if len(X.shape) == 1: X = X[:, None]

        self._a[0] = self._z[0] = X
        for l in range(self._nlayers - 1):
            self._z[l + 1] = self._W[l] @ self._a[l] + self._b[l]
            self._a[l + 1] = self._activ.func(self._z[l + 1])

        assert self._a[-1].shape == (self._sizes[-1], X.shape[1])
        assert self._z[-1].shape == (self._sizes[-1], X.shape[1])

    def _propagate_and_return_delta(self, delta: np.array, rate: float) -> np.array:
        nabla_W = [delta @ self._a[-2].T]
        nabla_b = [np.sum(delta, 1)[:, None]]

        assert nabla_W[0].shape == self._W[-1].shape
        assert nabla_b[0].shape == self._b[-1].shape

        for l in range(2, self._nlayers):
            delta = np.dot(self._W[-l + 1].T, delta) * self._activ.der(self._z[-l])
            nabla_W.insert(0, delta @ self._a[-l - 1].T)
            nabla_b.insert(0, np.sum(delta, 1)[:, None])
        for l in range(self._nlayers - 1):
            self._velW[l] = self._vel_factor * self._velW[l] + (1. - self._vel_factor) * nabla_W[l] * nabla_W[l]
            self._velb[l] = self._vel_factor * self._velb[l] + (1. - self._vel_factor) * nabla_b[l] * nabla_b[l]
            self._W[l] -= rate * nabla_W[l] / (np.sqrt(self._vel_eps + self._velW[l]))
            self._b[l] -= rate * nabla_b[l] / (np.sqrt(self._vel_eps + self._velb[l]))

        return np.dot(self._W[0].T, delta) * self._activ.der(self._z[0])


class SoftmaxCrossEntropyNNModule(NNModule):
    def __init__(self, sizes: List[int], learning_rate: float):
        super().__init__(sizes, sigmoid_operators, learning_rate)

    def _feed_forward(self, X: np.array):
        super()._feed_forward(X)
        self._a[-1] = np.apply_along_axis(softmax, 0, self._z[-1])

    def learn(self, X: np.array, Y: List[int], weights: List[float]) -> np.array:
        batch_size = X.shape[1]
        assert X.shape[0] == self._sizes[0]
        assert len(Y) == len(weights) == batch_size
        assert all((ans in range(self._sizes[-1]) for ans in Y))

        self._feed_forward(X)
        cost = cross_entropy(self._a[-1], Y, weights)
        SOFTMAX_COSTS.append(cost / batch_size)

        delta = self._a[-1]
        delta[Y, range(batch_size)] -= 1 * weights

        return self._propagate_and_return_delta(delta, self._learning_rate / batch_size)


class MSENNModule(NNModule):
    def __init__(self, sizes: List[int], learning_rate: float):
        super().__init__(sizes, ELU_operators, learning_rate)

    def _setup(self):
        super()._setup()
        self._cost = MSE_operators

    def learn(self, X: np.array, Y: np.array, weights: List[float]) -> np.array:
        batch_size = X.shape[1]
        assert X.shape[0] == self._sizes[0]
        assert Y.shape == (self._sizes[-1], batch_size)
        assert len(weights) == batch_size

        self._feed_forward(X)
        cost = self._cost.func(self._a[-1], Y) * weights
        MSE_COSTS.append(np.max(cost))

        delta = weights * self._cost.der(self._a[-1], Y) * self._activ.der(self._z[-1])

        return self._propagate_and_return_delta(delta, self._learning_rate / batch_size)
