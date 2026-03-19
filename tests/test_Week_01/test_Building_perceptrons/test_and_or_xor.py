import unittest
import numpy as np
import Week_01.Engineering.Building_perceptrons.and_or_xor as aox


class Test_Create_Dataset(unittest.TestCase):

    def test_when_called_then_dataset_contains_all_binary_pairs(self):
        dataset = aox.create_dataset(lambda x, y: x or y)
        expected = {
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 1)
        }
        self.assertEqual(set(dataset), expected)

    def test_dataset_uses_provided_function(self):
        dataset = aox.create_dataset(lambda x, y: x and y)
        for (x1, x2, y) in dataset:
            self.assertEqual(y, int(x1 and x2))


class Test_Initialize_Weights(unittest.TestCase):

    def test_when_called_weight_is_in_range(self):
        w = aox.initialize_weights(-1, 1)
        self.assertGreaterEqual(w, -1)
        self.assertLessEqual(w, 1)


class Test_Sigmoid(unittest.TestCase):

    def test_sigmoid_of_zero_is_half(self):
        self.assertAlmostEqual(aox.sigmoid(0), 0.5)

    def test_sigmoid_is_in_range(self):
        for v in [-100, -5, 0, 5, 100]:
            s = aox.sigmoid(v)
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, 1)


class Test_Calculate_Loss(unittest.TestCase):

    def test_when_perfect_prediction_then_loss_is_zero(self):
        model = [10, 10, -15]
        dataset = aox.create_dataset(lambda x, y: x and y)
        loss = aox.calculate_loss(model, dataset)
        self.assertAlmostEqual(loss, 0, delta=0.01)

    def test_loss_increases_when_model_is_wrong(self):
        model = [0, 0, 0]
        dataset = aox.create_dataset(lambda x, y: x and y)
        loss = aox.calculate_loss(model, dataset)
        self.assertGreater(loss, 0)


class Test_Train_Model(unittest.TestCase):

    def test_when_training_then_loss_decreases(self):
        dataset = aox.create_dataset(lambda x, y: x and y)
        model = [0.5, 0.5, 0.5]

        initial_loss = aox.calculate_loss(model, dataset)
        losses = aox.train_model(model, dataset, n_epochs=50, use_learing_rate=True)
        final_loss = aox.calculate_loss(model, dataset)

        self.assertLess(final_loss, initial_loss)
        self.assertEqual(len(losses), 50)

    def test_when_learning_rate_used_then_updates_are_smaller(self):
        dataset = aox.create_dataset(lambda x, y: x or y)
        m1 = [0.5, 0.5, 0.5]
        m2 = [0.5, 0.5, 0.5]

        aox.train_model(m1, dataset, 1, use_learing_rate=False)
        aox.train_model(m2, dataset, 1, use_learing_rate=True)

        original = np.array([0.5, 0.5, 0.5])
        self.assertLess(
            np.linalg.norm(np.array(m2) - original),
            np.linalg.norm(np.array(m1) - original)
        )


class Test_Forward(unittest.TestCase):

    def test_forward_computes_sigmoid_of_weighted_sum(self):
        model = [1, 1, 0] 
        out = aox.forward(model, (1, 1))
        expected = aox.sigmoid(2)
        self.assertAlmostEqual(out, expected)


class Test_Single_Layer_NN(unittest.TestCase):

    def test_returns_model_and_losses(self):
        (model, losses) = aox.create_single_layer_bool_nn(lambda x, y: x or y)
        self.assertEqual(len(model), 3)
        self.assertGreater(len(losses), 0)


class Test_XOR_Forward(unittest.TestCase):

    def test_xor_forward_logic(self):
        or_model   = [10, 10, -5]
        nand_model = [-10, -10, 15]
        and_model  = [10, 10, -15]

        xor_model = (or_model, nand_model, and_model)

        # XOR truth table
        xor_tests = [
            ((0,0), 0),
            ((0,1), 1),
            ((1,0), 1),
            ((1,1), 0)
        ]

        for (inputs, expected) in xor_tests:
            out = aox.xor_nn_forward(xor_model, inputs)
            pred = int(round(out))
            self.assertEqual(pred, expected)


if __name__ == "__main__":
    unittest.main()
