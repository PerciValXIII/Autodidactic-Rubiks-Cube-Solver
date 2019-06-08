from time import time
from typing import Tuple, List, Iterable

import matplotlib.pyplot as plt
import numpy as np

from adi.fullnet import FullNet
from cube.model import Cube
from cube.moves import Move
from mcts.solver import Solver


def generate_random_cube(iterations: int = 100) -> Cube:
    c = Cube()
    for _ in range(iterations):
        c.change_by(np.random.choice(list(Move)))
    return c


def measure_effectiveness(net: FullNet, scramble_range: int,
                          ncubes: int, time_per_cube: int) -> Tuple[List[int], List[float], List[int]]:
    samples = (generate_random_cube(scramble_range)
               for _ in range(ncubes))
    return try_solve(net, samples, time_per_cube)


def try_solve(net: FullNet, samples: Iterable[Cube],
              time_per_cube: int) -> Tuple[List[int], List[float], List[int]]:
    tree_sizes, times, lengths = [], [], []
    for sample in samples:
        solver = Solver(net)

        start = time()
        ans = solver.solve(sample, time_per_cube)
        end = time()

        if ans is not None:
            tree_sizes.append(len(solver._tree))
            times.append(end - start)
            lengths.append(len(ans))
    return tree_sizes, times, lengths


def plot_stats(ncubes: int, scramble_range: int,
               times: List[float], tree_sizes: List[int], lengths: List[int]):
    fig, axs = plt.subplots(3)
    plt.suptitle(f'Effectiveness on {ncubes} cubes (up to {scramble_range} moves)'
                 f' - {100. * len(tree_sizes) / ncubes:.2f}% of success')

    axs[0].hist(times, 10)
    axs[0].set_title('\n\nTime needed in secs')
    axs[0].set_xlabel('seconds')
    axs[0].set_ylabel('solved cubes')

    axs[1].hist(tree_sizes, 20)
    axs[1].set_title('Monte Carlo Tree size')
    axs[1].set_xlabel('nodes visited before solving')
    axs[1].set_ylabel('solved cubes')

    axs[2].hist(lengths, 10)
    axs[2].set_title('Solution length')
    axs[2].set_xlabel('solution length')
    axs[2].set_ylabel('solved cubes')

    fig.tight_layout()
    plt.show()
