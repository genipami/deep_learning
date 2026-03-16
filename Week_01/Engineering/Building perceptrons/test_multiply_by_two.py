import unittest
import numpy as np
import multiply_by_two as mbt

class Test_Create_Dataset(unittest.TestCase):
    def test_when_positive_integer_n_is_passed_then_dataset_with_n_elements_is_created(self):
        n = 5
        dataset = mbt.create_dataset(n)
        self.assertEqual(len(dataset), n)

    def test_when_negative_integer_n_is_passed_then_dataset_with_0_elements_is_created(self):
        n = -1
        dataset = mbt.create_dataset(n)
        self.assertEqual(len(dataset), 0)

    def test_when_0_is_passed_then_dataset_with_0_elements_is_created(self):
        n = 0
        dataset = mbt.create_dataset(n)
        self.assertEqual(len(dataset), 0)

    def test_when_5_is_passed_then_dataset_is_a_list_of_tuples_with_number_and_number_times_two(self):
        n = 5
        dataset = mbt.create_dataset(n)
        for i in range(n):
            self.assertEqual(dataset[i][0], i)
            self.assertEqual(dataset[i][1], i*2)

class Test_Initialize_Weights(unittest.TestCase):
    def test_when_called_it_returns_a_float_in_range_x_to_y(self):
        result = mbt.initialize_weights(-1,1)
        self.assertGreaterEqual(result, -1)
        self.assertGreaterEqual(1, result)

class Test_Calculate_Loss(unittest.TestCase):

    def test_when_empty_dataset_then_loss_is_zero(self):
        dataset = []
        model = 1.0
        loss = mbt.calculate_loss(model, dataset)
        self.assertEqual(loss, 0)

    def test_when_dataset_has_exact_matches_then_loss_is_zero(self):
        dataset = [(i, i*2) for i in range(5)]
        model = 2.0
        loss = mbt.calculate_loss(model, dataset)
        self.assertEqual(loss, 0)

    def test_when_model_is_wrong_then_loss_is_positive(self):
        dataset = [(i, i*2) for i in range(3)]
        model = 0.0
        loss = mbt.calculate_loss(model, dataset)
        self.assertGreater(loss, 0)

    def test_loss_is_sum_of_squared_errors(self):
        dataset = [(1, 2)]
        model = 0 
        expected = 4
        loss = mbt.calculate_loss(model, dataset)
        self.assertEqual(loss, expected)


class Test_Train_Model(unittest.TestCase):
    def test_when_called_then_weight_moves_in_direction_of_lower_loss(self):
        dataset = mbt.create_dataset(5)
        initial_weight = np.array([5.0])
        final_weight = np.array(initial_weight, dtype=float)

        mbt.train_model(final_weight, dataset, n_epochs=10)

        self.assertLess(abs(final_weight - 2), abs(initial_weight - 2))

    def test_when_learning_rate_is_used_then_updates_are_smaller(self):
        dataset = mbt.create_dataset(5)
        w1 = np.array([5.0], dtype=float)
        w2 = np.array([5.0], dtype=float)

        mbt.train_model(w1, dataset, 1, use_learing_rate=False)
        mbt.train_model(w2, dataset, 1, use_learing_rate=True)

        self.assertLess(abs(w2 - 5.0), abs(w1 - 5.0))

    def test_when_trained_then_loss_decreases_over_time(self):
        dataset = mbt.create_dataset(5)
        weight = np.array([4.0], dtype=float)

        initial_loss = mbt.calculate_loss(weight, dataset)
        mbt.train_model(weight, dataset, n_epochs=20)
        final_loss = mbt.calculate_loss(weight, dataset)

        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()