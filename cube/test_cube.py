from unittest import TestCase

from cube.model import Cube
from cube.moves import Move


class CubeTest(TestCase):
    def test_is_initially_solved(self):
        self.assertTrue(Cube().is_solved())

    def test_not_solved_after_move(self):
        c = Cube()
        c.change_by(Move.DOWN)
        self.assertFalse(c.is_solved())

    def test_down_move_and_return(self):
        c = Cube()
        c.change_by(Move.DOWN)
        c.change_by(Move.DOWN_CTR)
        self.assertTrue(c.is_solved())

    def test_left_move_and_return(self):
        c = Cube()
        c.change_by(Move.LEFT)
        self.assertFalse(c.is_solved())
        c.change_by(Move.LEFT_CTR)
        self.assertTrue(c.is_solved())

    def test_back_move_and_return(self):
        c = Cube()
        c.change_by(Move.BACK)
        self.assertFalse(c.is_solved())
        c.change_by(Move.BACK_CTR)
        self.assertTrue(c.is_solved())

    def test_scatter_and_return(self):
        c = Cube()
        c.change_by(Move.BACK)
        c.change_by(Move.LEFT_CTR)
        c.change_by(Move.DOWN)
        self.assertFalse(c.is_solved())
        c.change_by(Move.DOWN_CTR)
        c.change_by(Move.LEFT)
        c.change_by(Move.BACK_CTR)
        self.assertTrue(c.is_solved())
