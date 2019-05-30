from enum import Enum


class Move(Enum):
    """We can suppose that th cube's front-right-upper cubelet
    is stationary, so there are only 6 moves available"""
    LEFT = 1
    LEFT_CTR = -1
    DOWN = 2
    DOWN_CTR = -2
    BACK = 3
    BACK_CTR = -3
