"""Tests for the option orderings."""

import unittest

from order_dependency.runner.permutations import place_correct


class PlaceCorrectTests(unittest.TestCase):
    """The answer visits every position while the distractors keep their relative order."""

    def test_place_correct_moves_only_the_answer(self):
        """Position of the answer runs 0..3; the other options stay in canonical order."""
        perms = place_correct(4, answer=2)
        self.assertEqual([p.index(2) for p in perms], [0, 1, 2, 3])
        for p in perms:
            self.assertEqual([x for x in p if x != 2], [0, 1, 3])


if __name__ == "__main__":
    unittest.main()
