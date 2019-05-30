from cube.moves import Move

COLORS_MAPPING = {'white' : 0, 'blue': 1, 'red': 2,
                  'orange': 3, 'green': 4, 'yellow': 5}


class Cube:
    def __init__(self):
        self._reset_faces()

    def _reset_faces(self):
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

    def change_by(self, move: Move):
        if move < 0: [self.change_by(-move) for _ in range(3)]
        elif move == Move.LEFT: self._move_left()
        elif move == Move.BACK: self._move_back()
        elif move == Move.DOWN: self._move_down()
        else: assert False, 'Invalid move'

    def is_solved(self) -> bool:
        return all([len(set(face)) == 1
                    for face in self._faces.values()])

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
