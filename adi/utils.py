from typing import NamedTuple, List

import numpy as np


class OperatorPair(NamedTuple):
    func: callable
    der: callable


def sigmoid(x: float) -> float:
    if x < 0: return 1. - 1. / (1. + np.math.exp(x))
    return 1. / (1. + np.math.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1. - sigmoid(x))


sigmoid_operators = OperatorPair(np.vectorize(sigmoid),
                                 np.vectorize(sigmoid_derivative))


def ELU(x: float, alpha: float = 0.1) -> float:
    if x >= 0: return x
    return np.nan_to_num(alpha * (np.exp(x) - 1))


def ELU_derivative(x: float, alpha: float = 0.1) -> float:
    if x >= 0: return 1.
    return ELU(x) + alpha


ELU_operators = OperatorPair(np.vectorize(ELU),
                             np.vectorize(ELU_derivative))


def MSE(x: float, y: float) -> float:
    return .5 * (x - y) ** 2


def MSE_derivative(x: float, y: float) -> float:
    return x - y


MSE_operators = OperatorPair(np.vectorize(MSE),
                             np.vectorize(MSE_derivative))


def softmax(x: np.array):
    exps = np.exp(x - np.max(x))

    return exps / np.sum(exps)


def cross_entropy(guesses: np.array, answers: List[int], weights: List[float]) -> float:
    return -np.sum(np.nan_to_num(weights * np.log(guesses[answers, range(len(answers))])))
