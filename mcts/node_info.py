from typing import NamedTuple, List

import numpy as np


class NodeInfo(NamedTuple):
    is_leaf: bool
    N: List[int]  # number of times particular child has been chosen
    W: List[float]  # maximal values for each child
    L: List[float]  # current virtual loss for choice of child
    P: List[float]  # prior probability of particular child choice

    @classmethod
    def create_new(cls, probs: List[float]) -> 'NodeInfo':
        return NodeInfo(is_leaf=True,
                        N=[0 for _ in probs],
                        W=[0. for _ in probs],
                        L=[0. for _ in probs],
                        P=[p for p in probs], )

    def get_best_action(self, exploration_factor: float) -> int:
        U = [exploration_factor * self.P[action] * np.sqrt(np.sum(self.N)) / (1 + self.N[action])
             for action in range(len(self.N))]
        Q = [self.W[action] - self.L[action]
             for action in range(len(self.N))]
        evaluations = [u + v for u, v in zip(U, Q)]
        return np.argmax(evaluations)

    def update_virtual_loss(self, action: int, loss_step: float):
        assert action in range(len(self.L))
        self.L[action] += loss_step

    def update_on_backup(self, action: int, loss_step: float, propagated_value: float):
        self.update_virtual_loss(action, loss_step)
        self.W[action] = max(self.W[action], propagated_value)
        self.N[action] += 1
