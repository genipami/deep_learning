
import unittest
import numpy as np
import copy
import Week_01.Engineering.Building_perceptrons.square_nn as sn


class Test_Create_Dataset(unittest.TestCase):

    def test_when_n_is_zero_then_empty_dataset_is_returned(self):
        dataset = sn.create_dataset(0)
        self.assertEqual(dataset, [])

    def test_when_n_is_positive_then_dataset_contains_pairs_x_and_x_squared(self):
        dataset = sn.create_dataset(5)
        expected = [(i, i * i) for i in range(5)]
        self.assertEqual(dataset, expected)


class Test_Forward(unittest.TestCase):

    def test_forward_with_single_layer_linear_model(self):
        weights = [np.array([[2.0]])]
        biases = [np.array([[3.0]])]
        model = (weights, biases)

        result = sn.forward(model, 4)
        self.assertEqual(result, 11)

    def test_forward_applies_relu_on_hidden_layers(self):
        weights = [
            np.array([[-1.0]]),
            np.array([[1.0]])
        ]
        biases = [
            np.array([[0.0]]),
            np.array([[0.0]])
        ]
        model = (weights, biases)
        result = sn.forward(model, 5)

        self.assertEqual(result, 0)


class Test_Calculate_Loss(unittest.TestCase):
    def test_loss_is_mean_squared_error(self):
        weights = [np.array([[0.0]])]
        biases = [np.array([[0.0]])]
        model = (weights, biases)

        dataset = [(2, 4)]
        loss = sn.calculate_loss(model, dataset)
        self.assertEqual(loss, 16)


class Test_Train_Model(unittest.TestCase):

    def test_training_reduces_the_loss(self):
        weights = [
            np.random.uniform(-0.5, 0.5, size=(2, 1)),
            np.random.uniform(-0.5, 0.5, size=(1, 2))
        ]
        biases = [
            np.random.uniform(-0.5, 0.5, size=(2, 1)),
            np.random.uniform(-0.5, 0.5, size=(1, 1))
        ]
        model = (weights, biases)

        dataset = sn.create_dataset(5)

        initial_loss = sn.calculate_loss(model, dataset)
        sn.train_model(model, dataset, n_epochs=3)
        final_loss = sn.calculate_loss(model, dataset)

        self.assertLess(final_loss, initial_loss)

    def test_learning_rate_makes_updates_smaller(self):
        dataset = sn.create_dataset(3)

        w1 = [
            np.random.uniform(-0.5, 0.5, size=(1, 1)),
        ]
        b1 = [
            np.random.uniform(-0.5, 0.5, size=(1, 1)),
        ]
        w2 = copy.deepcopy(w1)
        b2 = copy.deepcopy(b1)

        model_no_lr = (copy.deepcopy(w1), copy.deepcopy(b1))
        model_lr = (copy.deepcopy(w2), copy.deepcopy(b2))

        sn.train_model(model_no_lr, dataset, 1, use_learing_rate=False)
        sn.train_model(model_lr, dataset, 1, use_learing_rate=True)

        orig_w = np.array(w1)
        
        delta_no_lr = np.abs(np.array(model_no_lr[0]) - orig_w).sum()
        delta_lr = np.abs(np.array(model_lr[0]) - orig_w).sum()

        self.assertLess(delta_lr, delta_no_lr)


if __name__ == "__main__":
    unittest.main()
