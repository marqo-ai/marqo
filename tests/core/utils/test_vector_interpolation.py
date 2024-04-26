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

    def test_interpolate_sequential_success(self):
        cases = [
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ],
                [1],
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Single vector'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ],
                [2],
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Single vector, weight 2'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ] * 2,
                [1] * 2,
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Same vector * 2'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ] * 2,
                [2] * 2,
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Same vector * 2, weight 2'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ] * 5,
                [1] * 5,
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Same vector * 5'
            ),
            (
                [
                    [1, 0, 0],
                    [0, 1, 0]
                ],
                [0.5, 0.5],
                [math.sqrt(2) / 2, math.sqrt(2) / 2, 0],
                '2 vectors, 45 degrees rotation around z-axis'
            ),
            (
                [
                    [0.24079554, -0.50855556, -0.69860205, 0.4419773],
                    [0.60970949, 0.4784225, 0.61885735, -0.12799152],
                    [0.76591685, -0.59245083, 0.08972328, 0.23307321]
                ],
                [1, 2, 0.5],
                [0.9582755764466467, -0.03248832000519242, 0.1691362691503027, 0.22813450030113333],
                '3 vectors'
            ),
            (
                [
                    [0.24079554, -0.50855556, -0.69860205, 0.4419773],
                    [0.60970949, 0.4784225, 0.61885735, -0.12799152],
                    [0.76591685, -0.59245083, 0.08972328, 0.23307321],
                    [0.53470714, 0.7637857, 0.36148952, -0.0067081]
                ],
                [1, -0.5, 2, 1.5],
                [0.8742225032825055, 0.13177697554091694, 0.46012597200027167, 0.0815715999267052],
                '4 vectors'
            ),
        ]

        slerp = Slerp(Slerp.Method.Sequential)

        for vectors, weights, expected, msg in cases:
            for prenormalized in [True, False]:
                with self.subTest(case=msg, prenormalized=prenormalized):
                    result = slerp.interpolate(vectors, weights, prenormalized=prenormalized)
                    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolate_hierarchical_success(self):
        cases = [
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ],
                [1],
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Single vector'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ],
                [2],
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Single vector, weight 2'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ] * 2,
                [1] * 2,
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Same vector * 2'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ] * 2,
                [2] * 2,
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Same vector * 2, weight 2'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ] * 5,
                [1] * 5,
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Same vector * 5'
            ),
            (
                [
                    [1, 0, 0],
                    [0, 1, 0]
                ],
                [0.5, 0.5],
                [math.sqrt(2) / 2, math.sqrt(2) / 2, 0],
                '2 vectors, 45 degrees rotation around z-axis'
            ),
            (
                [
                    [0.24079554, -0.50855556, -0.69860205, 0.4419773],
                    [0.60970949, 0.4784225, 0.61885735, -0.12799152],
                    [0.76591685, -0.59245083, 0.08972328, 0.23307321]
                ],
                [1, 2, 0.5],
                [0.9582755764466467, -0.03248832000519242, 0.1691362691503027, 0.22813450030113333],
                '3 vectors, odd number'
            ),
            (
                [
                    [0.24079554, -0.50855556, -0.69860205, 0.4419773],
                    [0.60970949, 0.4784225, 0.61885735, -0.12799152],
                    [0.76591685, -0.59245083, 0.08972328, 0.23307321],
                    [0.53470714, 0.7637857, 0.36148952, -0.0067081]
                ],
                [1, -0.5, 2, 1.5],
                [0.8307157370201422, 0.0889292113769262, 0.5487942602028577, 0.028771684936241104],
                '4 vectors, even power of 2'
            ),
            (
                [
                    [0.24079554, -0.50855556, -0.69860205, 0.4419773],
                    [0.60970949, 0.4784225, 0.61885735, -0.12799152],
                    [0.76591685, -0.59245083, 0.08972328, 0.23307321],
                    [0.53470714, 0.7637857, 0.36148952, -0.0067081],
                    [0.06271936, 0.67834342, -0.56283931, -0.46811152],
                    [0.30926992, -0.25873565, -0.40838477, -0.81891994]
                ],
                [1, -0.5, 2, 1.5, 0.25, 0.9],
                [0.8759481090365622, 0.05428666526306154, 0.19410932061327218, -0.4382800871883284],
                '6 vectors, even not power of 2'
            ),
        ]

        slerp = Slerp(Slerp.Method.Hierarchical)

        for vectors, weights, expected, msg in cases:
            for prenormalized in [True, False]:
                with self.subTest(case=msg, prenormalized=prenormalized):
                    result = slerp.interpolate(vectors, weights, prenormalized=prenormalized)
                    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolate_nonUnitNorm_success(self):
        vectors = [
            [0.24079554, -0.50855556, -0.69860205, 0.4419773],
            [0.60970949, 0.4784225, 0.61885735, -0.12799152],
            [0.76591685, -0.59245083, 0.08972328, 0.23307321],
            [0.53470714, 1, 0.36148952, -2.0067081, 0.36148952]
        ]
        weights = [1, -0.5, 2, 1.5]
        expected = []

    def test_interpolate_colinearVectors_success(self):
        pass

    def test_interpolate_zeroVector_failure(self):
        pass

    def test_interpolate_emptyVectors_failure(self):
        for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
            with self.subTest(method=method):
                slerp = Slerp(method)
                with self.assertRaises(ValueError):
                    slerp.interpolate([])

    def test_interpolate_differentVectorLengths_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 1],  # length 2
            [0, 0, 1]
        ]
        weights = [1, 1, 1]
        for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
            with self.subTest(method=method):
                slerp = Slerp(method)
                with self.assertRaises(ValueError):
                    slerp.interpolate(vectors, [1] * 3)

    def test_interpolate_wrongWeightsLength_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        weights = [1] * 4
        for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
            with self.subTest(method=method):
                slerp = Slerp(method)
                with self.assertRaises(ValueError):
                    slerp.interpolate(vectors, weights)

    def test_interpolate_missingWeights_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        weights = [1] * 4
        for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
            with self.subTest(method=method):
                slerp = Slerp(method)
                with self.assertRaises(ValueError):
                    slerp.interpolate(vectors, None)

    def test_wrongInterpolationMethod_failure(self):
        slerp = Slerp("non_existing_method")
        with self.assertRaises(InternalError):  # Changed to AttributeError
            slerp.interpolate(self.vectors, self.weights)
