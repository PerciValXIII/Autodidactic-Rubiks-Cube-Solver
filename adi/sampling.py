from typing import Iterable

import numpy as np

from cube.model import Cube
from cube.moves import Move


def generate_samples(depth: int, iterations: int) -> Iterable[Cube]:
    def get_next_move(last_move):
        while True:
            next_move = np.random.choice(list(Move))
            if next_move != last_move: return next_move

    for _ in range(iterations):
        c, last_move = Cube(), None
        for _ in range(depth):
            last_move = move = get_next_move(last_move)
            yield Cube(c.change_by(move))
