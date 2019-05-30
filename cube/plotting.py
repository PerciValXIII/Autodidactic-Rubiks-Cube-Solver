import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from cube.model import Cube, COLORS_MAPPING


class Plotter:
    def __init__(self):
        self._net = np.ones((6, 8), dtype=int) * -1

    def plot_cube(self, cube: Cube):
        self._mark_faces(cube)
        plt.imshow(self._net,
                   cmap=ListedColormap(['gray'] + list(COLORS_MAPPING.keys())))

        ax = plt.gca()
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=4)

        plt.show()

    def _mark_faces(self, cube: Cube):
        self._net[0, 2:4] = cube.up[:2]
        self._net[1, 2:4] = cube.up[2:]

        self._net[2, 0:2] = cube.left[:2]
        self._net[2, 2:4] = cube.front[:2]
        self._net[2, 4:6] = cube.right[:2]
        self._net[2, 6:8] = cube.back[:2]

        self._net[3, 0:2] = cube.left[2:]
        self._net[3, 2:4] = cube.front[2:]
        self._net[3, 4:6] = cube.right[2:]
        self._net[3, 6:8] = cube.back[2:]

        self._net[4][2:4] = cube.down[:2]
        self._net[5][2:4] = cube.down[2:]
