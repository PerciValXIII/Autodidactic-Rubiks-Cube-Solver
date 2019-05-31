import copy
from typing import Optional, Iterable

import numpy as np

from cube.moves import Move

COLORS_MAPPING = {'white' : 0, 'blue': 1, 'red': 2,
                  'orange': 3, 'green': 4, 'yellow': 5}


class Cube:
    def __init__(self, cube: Optional['Cube'] = None):
        if cube is None: self.reset()
        else: self._faces = copy.deepcopy(cube._faces)

    def reset(self):
        """Face indexing:   [0][1]
                            [2][3], due to the stationary position"""
        self._faces = {
            'up'   : 4 * [COLORS_MAPPING['white']],
            'front': 4 * [COLORS_MAPPING['red']],
            'right': 4 * [COLORS_MAPPING['blue']],
            'left' : 4 * [COLORS_MAPPING['green']],
            'back' : 4 * [COLORS_MAPPING['orange']],
            'down' : 4 * [COLORS_MAPPING['yellow']]
        }

    def change_by(self, move: Move) -> 'Cube':
        if move < 0: [self.change_by(-move) for _ in range(3)]
        elif move == Move.LEFT: self._move_left()
        elif move == Move.BACK: self._move_back()
        elif move == Move.DOWN: self._move_down()
        else: assert False, 'Invalid move'
        return self

    def is_solved(self) -> bool:
        return all([len(set(face)) == 1
                    for face in self._faces.values()])

    def one_hot_encode(self, dtype: type = bool) -> np.array:
        result = np.zeros((14, 6), dtype=dtype)

        result[0, self.up[2]] = 1
        result[1, self.front[0]] = 1
        result[2, self.up[0]] = 1
        result[3, self.left[0]] = 1
        result[4, self.up[1]] = 1
        result[5, self.back[0]] = 1

        result[6, self.down[0]] = 1
        result[7, self.front[2]] = 1
        result[8, self.down[1]] = 1
        result[9, self.right[2]] = 1
        result[10, self.down[2]] = 1
        result[11, self.left[2]] = 1
        result[12, self.down[3]] = 1
        result[13, self.back[2]] = 1

        return result

    def _move_left(self):
        self._rotate_clockwise('left')
        a, b = self.front[0], self.front[2]
        self.front[0], self.front[2] = self.up[0], self.up[2]
        self.down[0], self.down[2], a, b = a, b, self.down[0], self.down[2]
        self.back[3], self.back[1], a, b = a, b, self.back[3], self.back[1]
        self.up[0], self.up[2] = a, b

    def _move_back(self):
        self._rotate_clockwise('back')
        a, b = self.left[0], self.left[2]
        self.left[0], self.left[2], = self.up[1], self.up[0]
        self.down[2], self.down[3], a, b = a, b, self.down[2], self.down[3]
        self.right[3], self.right[1], a, b = a, b, self.right[3], self.right[1]
        self.up[1], self.up[0] = a, b

    def _move_down(self):
        self._rotate_clockwise('down')
        a, b, = self.front[2], self.front[3]
        self.front[2], self.front[3] = self.left[2], self.left[3]
        self.right[2], self.right[3], a, b = a, b, self.right[2], self.right[3]
        self.back[2], self.back[3], a, b = a, b, self.back[2], self.back[3]
        self.left[2], self.left[3] = a, b

    def _rotate_clockwise(self, face_name: str):
        f = self._faces[face_name]
        self._faces[face_name] = [f[2], f[0], f[3], f[1]]

    @property
    def up(self): return self._faces['up']

    @property
    def front(self): return self._faces['front']

    @property
    def right(self): return self._faces['right']

    @property
    def left(self): return self._faces['left']

    @property
    def back(self): return self._faces['back']

    @property
    def down(self): return self._faces['down']

    def __eq__(self, other):
        return isinstance(other, Cube) and self._faces == other._faces

    def __hash__(self):
        tuples = [tuple([key] + values) for key, values in self._faces.items()]
        return hash(tuple(tuples))


class ImmutableCube(Cube):
    def change_by(self, move: Move) -> Cube:
        return Cube(self).change_by(move)

    @property
    def up(self): return copy.deepcopy(super().up)

    @property
    def front(self): return copy.deepcopy(super().front)

    @property
    def right(self): return copy.deepcopy(super().right)

    @property
    def left(self): return copy.deepcopy(super().left)

    @property
    def back(self): return copy.deepcopy(super().back)

    @property
    def down(self): return copy.deepcopy(super().down)


def get_children_of(cube: Cube) -> Iterable[Cube]:
    imm = ImmutableCube(cube)
    return (imm.change_by(move) for move in list(Move))
