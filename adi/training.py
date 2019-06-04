import numpy as np

from adi.fullnet import FullNet
from adi.sampling import generate_samples
from cube.model import get_children_of


class AutodidacticIterator:
    def __init__(self):
        self._set_hyper()
        self._net = FullNet(self._body_net_sizes,
                            self._value_net_sizes,
                            self._policy_net_sizes)

    def _set_hyper(self):
        self._iteration_rounds = 100

        self._body_net_sizes = [14 * 6, 128, 64]
        self._value_net_sizes = [self._body_net_sizes[-1], 1]
        self._policy_net_sizes = [self._body_net_sizes[-1], 6]

        self._sampling_depth = 8
        self._sampling_iterations = 4

    def train(self):
        for _ in range(self._iteration_rounds):
            X = list(generate_samples(depth=self._sampling_depth,
                                      iterations=self._sampling_iterations))
            best_values, best_policies = [], []
            for x in X:
                values = []
                for child in get_children_of(x):
                    estimated_value = self._net.evaluate(child.one_hot_encode(float).T[:, None])[0].value
                    reward = 1. if child.is_solved() else -1.
                    values.append(estimated_value + reward)
                best_values.append(np.max(values))
                best_policies.append(np.argmax(values))

            self._net.learn(X=np.array([x.one_hot_encode(float) for x in X]).T,
                            values=best_values, policies=best_policies)


if __name__ == '__main__':
    AutodidacticIterator().train()
