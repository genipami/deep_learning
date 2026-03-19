import unittest
from unittest.mock import patch, call
from dl_lib.value import Value

class TestInit(unittest.TestCase):
    def test_when_creating_with_data_5_and_without_prev_then_prev_is_none(self):
        #Arrange
        data_to_pass = 5
        expected_data = 5
        expected_prev = None
        
        #Act
        actual = Value(data_to_pass)
        
        #Assert
        self.assertEqual(expected_data, actual.data)
        self.assertEqual(expected_prev, actual._prev)

    def test_when_creating_with_data_3_and_with_prev_with_data_4_then_data_is_3_and_prev_has_data_4(self):
        #Arrange
        data_to_pass = 3
        prev_to_pass = Value(4)

        expected_data = data_to_pass
        expected_prev = prev_to_pass

        #Act
        actual = Value(data_to_pass, prev_to_pass)

        #Assert
        self.assertEqual(expected_data, actual.data)
        self.assertEqual(expected_prev, actual._prev)

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
    def test_when_adding_value_with_data_2_and_value_with_data_3_then_result_is_value_with_data_5_and_prev_includes_the_values_2_and_3(self):
        #Arrange
        two = Value(2)
        three = Value(3)
        expected = Value(5, {two, three})

        #Act
        actual = two + three 
    
        #Assert
        self.assertEqual(expected, actual)

class TestAdd(unittest.TestCase):
    def test_when_multiplying_value_with_data_2_and_value_with_data_3_then_result_is_value_with_data_6_and_prev_includes_the_values_2_and_3(self):
        #Arrange
        two = Value(2)
        three = Value(3)
        expected = Value(6, {two, three})

        #Act
        actual = two * three 
    
        #Assert
        self.assertEqual(expected.data, actual.data)
        self.assertEqual(expected._prev, actual._prev)
    