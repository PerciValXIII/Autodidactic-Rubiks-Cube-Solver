from unittest import TestCase

from cube.model import Cube, ImmutableCube, get_children_of
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

    def test_mutability(self):
        mut = Cube()
        changed = mut.change_by(Move.DOWN)

        self.assertFalse(mut.is_solved())
        self.assertListEqual(mut.front, changed.front)

    def test_immutability(self):
        imm = ImmutableCube()
        changed = imm.change_by(Move.DOWN)

        self.assertTrue(imm.is_solved())
        self.assertFalse(changed.is_solved())

    def test_get_children_of(self):
        cube = Cube()
        children = list(get_children_of(cube))

        self.assertEqual(6, len(children))

    def test_hash_cube(self):
        hash_map = dict()
        hash_map[ImmutableCube().change_by(Move.DOWN)] = 1
        hash_map[Cube().change_by(Move.DOWN)] = 2
        hash_map[ImmutableCube().change_by(Move.DOWN)] = 5
        hash_map[Cube().change_by(Move.LEFT)] = 3

        self.assertEqual(2, len(hash_map))
        self.assertEqual(5, hash_map[Cube().change_by(Move.DOWN)])
