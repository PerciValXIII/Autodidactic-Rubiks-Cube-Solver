import pickle

from cube.model import Cube
from mcts.solver import Solver
from performance.effectiveness import generate_random_cube
from plotting.plotter import Plotter

if __name__ == '__main__':
    with open('./nets/trained_net500.pkl', 'rb') as input:
        net = pickle.load(input)
    solver = Solver(net)

    cube = generate_random_cube(iterations=6)

    moves = solver.solve(cube)
    sequence = [Cube(cube)] + [Cube(cube.change_by(move))
                               for move in moves]
    # Plotter().save_sequence(sequence, 'demo.gif')
    Plotter().plot_sequence(sequence)
