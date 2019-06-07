import numpy as np

from adi.training import AutodidacticIterator
from cube.model import Cube
from cube.moves import Move
from mcts.solver import Solver
from performance.persistence import *
from plotting.plotter import Plotter


def generate_random_cube(iterations: int = 10) -> Cube:
    c = Cube()
    for _ in range(iterations):
        c.change_by(np.random.choice(list(Move)))
    return c


if __name__ == '__main__':
    np.random.seed(0)

    ADI = AutodidacticIterator()
    ADI.train()
    save_net(ADI.net, 'trained_net.pkl')
    net = load_net('trained_net.pkl')
    solver = Solver(net)

    plotter = Plotter()

    cube = generate_random_cube()
    plotter.plot_cube(cube)

    sequence = solver.solve(cube)
    for m in sequence:
        cube.change_by(m)
        plotter.plot_cube(cube)
