import unittest
from unittest.mock import patch
import numpy as np
import random_walks_helpers as rw


class Test_Simulate_Roll(unittest.TestCase):

    @patch("numpy.random.randint")
    def test_when_roll_is_1_or_2_then_step_decreases(self, mock_randint):
        mock_randint.return_value = np.array([1])
        roll, step = rw.simulate_roll(current_step=5)
        self.assertEqual(step, 4)
        self.assertEqual(roll, 1)

    @patch("numpy.random.randint")
    def test_when_roll_is_3_4_or_5_then_step_increases(self, mock_randint):
        mock_randint.return_value = np.array([4])
        roll, step = rw.simulate_roll(current_step=5)
        self.assertEqual(step, 6)

    @patch("numpy.random.randint")
    def test_when_roll_is_6_then_reroll_until_not_6(self, mock_randint):
        mock_randint.side_effect = [
            np.array([6]),
            np.array([6]),
            np.array([3])
        ]
        roll, step = rw.simulate_roll(0)
        self.assertEqual(roll, 3)
        self.assertEqual(step, 1)

    @patch("numpy.random.randint")
    def test_when_step_would_be_negative_and_clamp_is_true_then_step_is_zero(self, mock_randint):
        mock_randint.return_value = np.array([1])
        roll, step = rw.simulate_roll(0, should_clamp=True)
        self.assertEqual(step, 0)

    @patch("numpy.random.randint")
    def test_when_clamp_is_false_then_step_may_be_negative(self, mock_randint):
        mock_randint.return_value = np.array([2])
        roll, step = rw.simulate_roll(0, should_clamp=False)
        self.assertEqual(step, -1)

    @patch("numpy.random.rand")
    @patch("numpy.random.randint")
    def test_when_clumsy_and_random_below_threshold_then_step_resets_to_zero(
        self, mock_randint, mock_rand
    ):
        mock_randint.return_value = np.array([5])
        mock_rand.return_value = 0.004
        roll, step = rw.simulate_roll(3, is_clumsy=True)
        self.assertEqual(step, 0)

    @patch("numpy.random.rand")
    @patch("numpy.random.randint")
    def test_when_clumsy_and_random_above_threshold_then_no_reset(
        self, mock_randint, mock_rand
    ):
        mock_randint.return_value = np.array([3])
        mock_rand.return_value = 0.5
        roll, step = rw.simulate_roll(3, is_clumsy=True)
        self.assertEqual(step, 4)


class Test_Simulate_Walk(unittest.TestCase):

    @patch("random_walks_helpers.simulate_roll")
    def test_walk_contains_initial_positions_before_each_roll(self, mock_roll):
        mock_roll.side_effect = [
            (2, 1),
            (2, 2),
            (2, 3)
        ]
        walk = rw.simulate_walk(3)
        self.assertEqual(walk, [0, 1, 2])

    @patch("random_walks_helpers.simulate_roll")
    def test_walk_has_length_equal_to_num_throws(self, mock_roll):
        mock_roll.return_value = (3, 1)
        walk = rw.simulate_walk(5)
        self.assertEqual(len(walk), 5)


class Test_Simulate_N_Walks(unittest.TestCase):

    @patch("random_walks_helpers.simulate_walk")
    def test_returns_list_of_n_walks(self, mock_simulate):
        mock_simulate.return_value = [0, 1, 2]
        walks = rw.simulate_n_walks(4, 3)
        self.assertEqual(len(walks), 4)
        for w in walks:
            self.assertEqual(w, [0, 1, 2])

    @patch("random_walks_helpers.simulate_walk")
    def test_each_walk_has_correct_length(self, mock_simulate):
        mock_simulate.return_value = [0] * 10
        walks = rw.simulate_n_walks(3, 10)
        for w in walks:
            self.assertEqual(len(w), 10)


if __name__ == "__main__":
    unittest.main()