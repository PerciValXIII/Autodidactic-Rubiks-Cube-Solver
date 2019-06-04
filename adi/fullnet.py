from typing import List, NamedTuple

import numpy as np

from adi.nnmodule import NNModule, SoftmaxCrossEntropyNNModule, MSENNModule
from adi.utils import ELU_operators

BODY_LEARNING_RATE = 0.01
POLICY_LEARNING_RATE = 0.1
VALUE_LEARNING_RATE = 0.001

POLICY_PROP_FACTOR = 1.
VALUE_PROP_FACTOR = 1.


class ValuePolicyPair(NamedTuple):
    value: float
    policy: List[float]


class FullNet:
    def __init__(self, body_net_sizes: List[int],
                 value_net_sizes: List[int],
                 policy_net_sizes: List[int]):
        assert body_net_sizes[-1] == value_net_sizes[0] == policy_net_sizes[0]
        assert value_net_sizes[-1] == 1

        self._body_net = NNModule(body_net_sizes, ELU_operators, BODY_LEARNING_RATE)
        self._value_net = MSENNModule(value_net_sizes, VALUE_LEARNING_RATE)
        self._policy_net = SoftmaxCrossEntropyNNModule(policy_net_sizes, POLICY_LEARNING_RATE)

    def evaluate(self, X: np.array) -> List[ValuePolicyPair]:
        body_out = self._body_net.evaluate(X)
        values = [value[0] for value in self._value_net.evaluate(body_out).T]
        policies = self._policy_net.evaluate(body_out).T

        return [ValuePolicyPair(v, p) for v, p in zip(values, policies)]

    def learn(self, X: np.array, values: List[float], policies: List[int]):
        body_out = self._body_net.evaluate(X)
        value_delta = self._value_net.learn(body_out, np.array([values]))
        policy_delta = self._policy_net.learn(body_out, policies)

        self._body_net.learn_from_delta(value_delta, BODY_LEARNING_RATE * VALUE_PROP_FACTOR)
        self._body_net.learn_from_delta(policy_delta, BODY_LEARNING_RATE * POLICY_PROP_FACTOR)
