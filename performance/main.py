import pickle

import numpy as np

from cube.model import Cube
from cube.moves import Move
from mcts.solver import Solver


def generate_random_cube(iterations: int = 100) -> Cube:
    c = Cube()
    for _ in range(iterations):
        c.change_by(np.random.choice(list(Move)))
    return c


if __name__ == '__main__':
    np.random.seed(0)
    with open('trained_net500.pkl', 'rb') as input:
        net = pickle.load(input)
    solver = Solver(net)
    cube = generate_random_cube()
    solver.solve(cube)
