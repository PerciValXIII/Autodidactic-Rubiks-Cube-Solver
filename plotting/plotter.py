from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import ListedColormap

from cube.model import Cube, COLORS_MAPPING


class Plotter:
    def plot_cube(self, cube: Cube):
        self._set_cube_axes(plt.gca())
        img = self._generate_image(cube)
        plt.show()

    def plot_sequence(self, cubes: List[Cube]):
        fig = plt.figure()
        plots = [[self._generate_image(cube)] for cube in cubes]
        animation = ArtistAnimation(fig, plots, interval=1000, repeat_delay=2000)
        plt.show()

    def save_sequence(self, cubes: List[Cube], filename: str):
        fig = plt.figure()
        plots = [[self._generate_image(cube)] for cube in cubes]
        ArtistAnimation(fig, plots, interval=1000, repeat_delay=2000).save(filename)

    def plot_costs(self, mse_costs: List[float], softmax_costs: List[float]):
        fig, ax1 = plt.subplots()
        ax1.plot(range(len(mse_costs)), mse_costs)
        ax1.set_ylabel('mse cost', color='b')
        ax2 = ax1.twinx()
        ax2.plot(range(len(softmax_costs)), softmax_costs, 'g')
        ax2.set_ylabel('softmax cost', color='g')
        plt.show()

    def _generate_image(self, cube: Cube):
        net = np.ones((6, 8), dtype=int) * -1
        self._mark_faces(cube, net)
        img = plt.imshow(net, cmap=ListedColormap(['gray'] + list(COLORS_MAPPING.keys())))
        self._set_cube_axes(img.axes)
        return img

    def _mark_faces(self, cube: Cube, net: np.array):
        net[0, 2:4] = cube.up[:2]
        net[1, 2:4] = cube.up[2:]
        net[2, 0:2] = cube.left[:2]
        net[2, 2:4] = cube.front[:2]
        net[2, 4:6] = cube.right[:2]
        net[2, 6:8] = cube.back[:2]
        net[3, 0:2] = cube.left[2:]
        net[3, 2:4] = cube.front[2:]
        net[3, 4:6] = cube.right[2:]
        net[3, 6:8] = cube.back[2:]
        net[4][2:4] = cube.down[:2]
        net[5][2:4] = cube.down[2:]

    def _set_cube_axes(self, ax):
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=3)
