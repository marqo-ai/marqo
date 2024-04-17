import math
import unittest

import numpy as np

from marqo.core.utils.vector_interpolation import Slerp, Nlerp, Lerp
from marqo.exceptions import InternalError


class TestLerp(unittest.TestCase):
    def setUp(self):
        self.vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]

    def test_equal_weights(self):
        lerp = Lerp()
        weights = [1, 1, 1]  # Equal weights
        expected = [1 / 3, 1 / 3, 1 / 3]
        result = lerp.interpolate(self.vectors, weights)
        self.assertEqual(result, expected)

    def test_no_weights(self):
        lerp = Lerp()
        expected = [1 / 3, 1 / 3, 1 / 3]  # Default to equal weighting if none provided
        result = lerp.interpolate(self.vectors)
        self.assertEqual(result, expected)

    def test_uneven_weights(self):
        lerp = Lerp()
        weights = [3, 1, 1]
        expected = [3 / 5, 1 / 5, 1 / 5]
        result = lerp.interpolate(self.vectors, weights)
        self.assertEqual(result, expected)

    def test_single_vector(self):
        lerp = Lerp()
        result = lerp.interpolate([self.vectors[0]])
        self.assertEqual(result, self.vectors[0])

    def test_empty_vectors(self):
        lerp = Lerp()
        with self.assertRaises(ValueError):  # Accessing index 0 in an empty list
            lerp.interpolate([])


class TestNlerp(unittest.TestCase):
    def setUp(self):
        self.vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        self.weights = [0.333, 0.333, 0.333]  # Equal weighting

    def test_nlerp_unit_length(self):
        nlerp = Nlerp()
        result = nlerp.interpolate(self.vectors, self.weights)
        result_length = math.sqrt(sum(x ** 2 for x in result))
        self.assertAlmostEqual(result_length, 1.0, places=5)

    def test_nlerp_single_vector(self):
        nlerp = Nlerp()
        result = nlerp.interpolate([self.vectors[0]])
        np.testing.assert_array_almost_equal(result, self.vectors[0])

    def test_nlerp_uneven_weights(self):
        nlerp = Nlerp()
        weights = [0.7, 0.15, 0.15]
        result = nlerp.interpolate(self.vectors, weights)
        result_length = math.sqrt(sum(x ** 2 for x in result))
        self.assertAlmostEqual(result_length, 1.0, places=5)


class TestSlerp(unittest.TestCase):
    # TODO - Improve tests so that hierarchical and sequential have different results
    # TODO - Test with nonnomralized vectors
    def setUp(self):
        # Define simple vectors along the axes
        self.vectors = [
            [1, 0, 0],  # Vector along the x-axis
            [0, 1, 0]  # Vector along the y-axis
        ]
        self.weights = [0.5, 0.5]  # Equal weighting for a simple midpoint test

    def test_sequential_interpolation_simple(self):
        slerp = Slerp(Slerp.Method.Sequential)
        result = slerp.interpolate(self.vectors, self.weights)
        expected = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0]  # 45 degrees rotation around z-axis
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_hierarchical_interpolation_simple(self):
        slerp = Slerp(Slerp.Method.Hierarchical)
        result = slerp.interpolate(self.vectors, self.weights)
        expected = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0]  # Same expectation for simple case
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_single_vector(self):
        slerp = Slerp()
        result = slerp.interpolate([self.vectors[0]])
        np.testing.assert_array_almost_equal(result, self.vectors[0])

    def test_empty_vectors(self):
        slerp = Slerp()
        with self.assertRaises(ValueError):
            slerp.interpolate([])

    def test_interpolation_method_mismatch(self):
        slerp = Slerp("non_existing_method")
        with self.assertRaises(InternalError):  # Changed to AttributeError
            slerp.interpolate(self.vectors, self.weights)
