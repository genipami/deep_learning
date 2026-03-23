import unittest
from unittest.mock import patch
from dl_lib.value import Value, trace

class TestInit(unittest.TestCase):
    def test_when_creating_with_data_5_and_without_prev_then_prev_and_op_are_none(self):
        #Arrange
        data_to_pass = 5
        expected_data = 5
        expected_prev = None
        expected_op = None
        
        #Act
        actual = Value(data_to_pass)
        
        #Assert
        self.assertEqual(expected_data, actual.data)
        self.assertEqual(expected_prev, actual._prev)
        self.assertEqual(expected_op, actual._op)

    def test_when_creating_with_data_3_and_with_prev_with_data_4_then_data_is_3_and_prev_has_data_4_and_op_is_None(self):
        #Arrange
        data_to_pass = 3
        prev_to_pass = Value(4)

        expected_data = data_to_pass
        expected_prev = prev_to_pass
        expected_op = None

        #Act
        actual = Value(data_to_pass, prev_to_pass)

        #Assert
        self.assertEqual(expected_data, actual.data)
        self.assertEqual(expected_prev, actual._prev)
        self.assertEqual(expected_op, actual._op)

    def test_when_creating_with_data_3_and_label_k_then_data_is_3_and_label_is_k(self):
        #Arrange
        data_to_pass = 3
        prev_to_pass = Value(4)
        label_to_pass = "k"

        expected_data = data_to_pass
        expected_prev = prev_to_pass
        expected_op = None
        expected_label = label_to_pass

        #Act
        actual = Value(data_to_pass, prev_to_pass, None, label_to_pass)

        #Assert
        self.assertEqual(expected_data, actual.data)
        self.assertEqual(expected_prev, actual._prev)
        self.assertEqual(expected_op, actual._op)
        self.assertEqual(expected_label, actual.label)

    def test_when_creating_a_value_with_data_7_and_no_other_params_then_graident_is_zero(self):
        #Arrange
        data_to_pass = 7
        expected_data = data_to_pass
        expected_gradient = 0

        #Act
        actual = Value(data_to_pass)

        #Assert
        self.assertEqual(expected_gradient, actual.gradient)
        self.assertEqual(expected_data, actual.data)

class TestStr(unittest.TestCase):
    def test_when_passing_value_with_data_5_string_representation_is_as_expected(self):
        #Arrange
        value_to_pass = Value(5)
        expected = "Value(data=5)"
        
        #Act
        actual = str(value_to_pass)
        
        #Assert
        self.assertEqual(expected, actual)

class TestRepr(unittest.TestCase):
    @patch('builtins.print')
    def test_when_printing_value_with_data_5_representation_is_as_expected(self, mocked_print):
        #Arrange
        value_to_pass = Value(5)
        expected = "Value(data=5)"

        #Act
        print(value_to_pass)
        actual = str(mocked_print.call_args.args[0])

        #Assert
        self.assertEqual(expected, actual)

class TestAdd(unittest.TestCase):
    def test_when_adding_value_with_data_2_and_value_with_data_3_then_result_is_value_with_data_5_and_prev_includes_the_values_2_and_3_and_op_is_plus(self):
        #Arrange
        two = Value(2)
        three = Value(3)
        expected = Value(5, {two, three}, "+") 

        #Act
        actual = two + three 
    
        #Assert
        self.assertEqual(expected, actual)

class TestAdd(unittest.TestCase):
    def test_when_multiplying_value_with_data_2_and_value_with_data_3_then_result_is_value_with_data_6_and_prev_includes_the_values_2_and_3_and_op_is_mult(self):
        #Arrange
        two = Value(2)
        three = Value(3)
        expected = Value(6, {two, three}, "*")

        #Act
        actual = two * three 
    
        #Assert
        self.assertEqual(expected.data, actual.data)
        self.assertEqual(expected._prev, actual._prev)

class TestTrace(unittest.TestCase):
    def test_when_calling_on_new_value_then_there_is_no_trace(self):
        #Arrange
        value_to_pass = Value(5)
        expected_nodes = {value_to_pass}
        expected_edges = set()

        #Act
        actual_nodes, actual_edges = trace(value_to_pass)

        # Assert
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_edges, actual_edges)    

    def test_when_calling_on_the_sum_of_2_and_3_values_then_there_is_trace_with_nodes_2_and_3_and_correct_edges(self):
        #Arrange
        two = Value(2)
        three = Value(3)
        value_to_pass = two+three
        
        expected_nodes = {two, three, value_to_pass}
        
        expected_edges = {(two, value_to_pass), (three, value_to_pass)}

        #Act
        actual_nodes, actual_edges = trace(value_to_pass)

        # Assert
        self.assertCountEqual(expected_nodes, actual_nodes)
        self.assertCountEqual(expected_edges, actual_edges)   

    def test_when_doing_multiple_opetarions(self):
        #Arrange
        x = Value(2.0)
        y = Value(-3.0)
        z = Value(10.0)
        x_times_y = x*y
        result = x_times_y+z

        expected_nodes = {x, y, z, x_times_y, result}
        expected_edges = {(x, x_times_y), (y, x_times_y), (z, result), (x_times_y, result)}
        
        #Act
        actual_nodes, actual_edges = trace(result)

        #Assert
        self.assertCountEqual(expected_nodes, actual_nodes)
        self.assertCountEqual(expected_edges, actual_edges)

class TestSetLabel(unittest.TestCase):
    def test_when_setting_label_a_to_value_with_data_5_then_label_is_a_and_data_is_5(self):
        #Arrange
        value = Value(5)
        label_to_set = "a"
        expected_data = 5
        expected_label = label_to_set

        #Act
        value.set_label(label_to_set)
        actual = value

        #Assert
        self.assertEqual(expected_data, actual.data)
        self.assertEqual(expected_label, actual.label)

class TestSetGradient(unittest.TestCase):
    def test_when_setting_gradient_7_to_value_with_data_5_then_gradient_is_7_and_data_is_5(self):
        #Arrange
        value = Value(5)
        gradient_to_set = 7
        expected_data = 5
        expected_gradient = gradient_to_set

        #Act
        value.set_gradient(gradient_to_set)
        actual = value

        #Assert
        self.assertEqual(expected_data, actual.data)
        self.assertEqual(gradient_to_set, actual.gradient)