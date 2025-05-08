import math
import unittest

import numpy as np

from marqo.core.utils.vector_interpolation import Slerp, Nlerp, Lerp, ZeroSumWeightsError, ZeroMagnitudeVectorError
from marqo.exceptions import InternalError
from integ_tests.marqo_test import MarqoTestCase


class TestLerp(unittest.TestCase):

    def test_interpolate_success(self):
        cases = [
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1]
                ],
                [1, 1, 1, 1],
                [1 / 4, 1 / 4, 1 / 2],
                'Equal weights'
            ),
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                [3, 1, 1],
                [3 / 5, 1 / 5, 1 / 5],
                'Different weights'
            ),
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
                    [0, 0, 0],
                    [0, 0, 1]
                ],
                [1, 1, 1],
                [1 / 3, 0, 1 / 3],
                'Zero vector'
            ),
            (
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [1, 1, 1],
                [0, 0, 0],
                'All zero vectors'
            ),
        ]

        # Create instance here to also verify statelessness
        lerp = Lerp()

        for vectors, weights, expected, msg in cases:
            with self.subTest(msg):
                result = lerp.interpolate(vectors, weights)
                np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolate_zeroSumWeights_failure(self):
        """
        Test interpolating weights that sum to zero fails
        """
        cases = [
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1]
                ],
                [0, 1, 2, -3],
                'Zero sum of weights'
            ),
            (
                [
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                ],
                [0, 0, 0],
                'Zero sum of weights -- all zero'
            )
        ]

        lerp = Lerp()

        for vectors, weights, msg in cases:
            with self.subTest(msg):
                with self.assertRaises(ZeroSumWeightsError) as ex:
                    lerp.interpolate(vectors, weights)
                self.assertIn('Sum of weights', str(ex.exception))

    def test_interpolate_emptyVectors_failure(self):
        lerp = Lerp()
        with self.assertRaises(ValueError) as ex:
            lerp.interpolate([], [])
        self.assertIn('empty list of vectors', str(ex.exception))

    def test_interpolate_differentVectorLengths_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 1],  # length 2
            [0, 0, 1]
        ]
        weights = [1, 1, 1]

        lerp = Lerp()
        with self.assertRaises(ValueError) as ex:
            lerp.interpolate(vectors, weights)
        self.assertIn('same length', str(ex.exception))

    def test_interpolate_wrongWeightsLength_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        weights = [1] * 2

        lerp = Lerp()
        with self.assertRaises(ValueError) as ex:
            lerp.interpolate(vectors, weights)
        self.assertIn('must have the same length', str(ex.exception))


class TestNlerp(MarqoTestCase):
    def test_interpolate_success(self):
        cases = [
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1]
                ],
                [1, 1, 1, 1],
                [0.4082482904638631, 0.4082482904638631, 0.8164965809277261],
                'Equal weights'
            ),
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                [3, 1, 1],
                [0.9045340337332909, 0.30151134457776363, 0.30151134457776363],
                'Different weights'
            ),
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
                    [0, 0, 0],
                    [0, 0, 1]
                ],
                [1, 1, 1],
                [0.7071067811865475, 0.0, 0.7071067811865475],
                'Zero vector'
            ),
        ]

        # Create instance here to also verify statelessness
        nlerp = Nlerp()

        for vectors, weights, expected, msg in cases:
            with self.subTest(msg):
                result = nlerp.interpolate(vectors, weights)
                np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolate_zeroMagnitude_failure(self):
        """
        Test interpolating vectors that result in zero magnitude fails
        """
        cases = [
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 1],
                    [-0.5, 0, -0.5]
                ],
                [1, 1, 1, 2],
                'Zero magnitude'
            ),
            (
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [1, 1, 1],
                'Zero magnitude, all zero vectors'
            ),
        ]

        for vectors, weights, msg in cases:
            with self.subTest(msg):
                nlerp = Nlerp()
                with self.assertRaisesStrict(ZeroMagnitudeVectorError) as ex:
                    nlerp.interpolate(vectors, weights)
                self.assertIn('zero magnitude', str(ex.exception))

    def test_interpolate_zeroSumWeights_failure(self):
        """
        Test interpolating weights that sum to zero fails
        """
        cases = [
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1]
                ],
                [0, 1, 2, -3],
                'Zero sum of weights'
            ),
            (
                [
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                ],
                [0, 0, 0],
                'Zero sum of weights -- all zero'
            )
        ]

        lerp = Lerp()

        for vectors, weights, msg in cases:
            with self.subTest(msg):
                with self.assertRaises(ZeroSumWeightsError) as ex:
                    lerp.interpolate(vectors, weights)
                self.assertIn('Sum of weights', str(ex.exception))

    def test_interpolate_emptyVectors_failure(self):
        lerp = Lerp()
        with self.assertRaises(ValueError) as ex:
            lerp.interpolate([], [])
        self.assertIn('empty list of vectors', str(ex.exception))

    def test_interpolate_differentVectorLengths_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 1],  # length 2
            [0, 0, 1]
        ]
        weights = [1, 1, 1]

        lerp = Lerp()
        with self.assertRaises(ValueError) as ex:
            lerp.interpolate(vectors, weights)
        self.assertIn('same length', str(ex.exception))

    def test_interpolate_wrongWeightsLength_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        weights = [1] * 2

        lerp = Lerp()
        with self.assertRaises(ValueError) as ex:
            lerp.interpolate(vectors, weights)
        self.assertIn('must have the same length', str(ex.exception))


class TestSlerp(MarqoTestCase):

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

        # Create instance here to also verify statelessness
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
                'Same vector * 2 - colinear'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ] * 2,
                [2] * 2,
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Same vector * 2 - colinear, weight 2'
            ),
            (
                [
                    [math.sqrt(0.5), math.sqrt(0.5), 0]
                ] * 5,
                [1] * 5,
                [math.sqrt(0.5), math.sqrt(0.5), 0],
                'Same vector * 5 - colinear'
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

        # Create instance here to also verify statelessness
        slerp = Slerp(Slerp.Method.Hierarchical)

        for vectors, weights, expected, msg in cases:
            for prenormalized in [True, False]:
                with self.subTest(case=msg, prenormalized=prenormalized):
                    result = slerp.interpolate(vectors, weights, prenormalized=prenormalized)
                    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolate_nonUnitNorm_success(self):
        vectors = [
            [0.24079554, -0.70855556, -0.69860205, 0.4419773],  # non-normalized
            [0.60970949, 0.4784225, 0.61885735, -0.02799152],  # non-normalized
            [0.76591685, -0.59245083, 0.08972328, 0.23307321],
            [0.53470714, 0.36148952, -2.0067081, 0.16148952]  # non-normalized
        ]
        weights = [1, -0.5, 2, 1.5]
        expected = [0.7917869864963851, -0.15221528929873665, -1.428205025357183, 0.14390162492903472]

        slerp = Slerp(Slerp.Method.Hierarchical)
        result = slerp.interpolate(vectors, weights)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolate_colinearVectors_success(self):
        cases = [
            (
                [
                    [1, 0, 0],
                    [1, 0, 0]
                ],
                [1, 2],
                [1, 0, 0],
                'Unit norm vectors'
            ),
            (
                [
                    [1, 2, 0],
                    [2, 4, 0]
                ],
                [1, 2],
                [5 / 3, 10 / 3, 0],
                'Non-unit norm vectors'
            ),
        ]

        for vectors, weights, expected, msg in cases:
            for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
                slerp = Slerp(method)
                with self.subTest(msg):
                    result = slerp.interpolate(vectors, weights)
                    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolate_zeroWeight_failure(self):
        """
        Test interpolating two consecutive vectors where weights sum to zero fails
        """
        cases = [
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1]
                ],
                [1, -1, 1, 1],
                'Zero sum of weights'
            ),
            (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1]
                ],
                [0, 0, 0, 0],
                'All zero weights'
            )
        ]

        for vectors, weights, msg in cases:
            for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
                with self.subTest(case=msg, method=method):
                    slerp = Slerp(method)
                    with self.assertRaisesStrict(ZeroSumWeightsError) as ex:
                        slerp.interpolate(vectors, weights)
                    self.assertIn('Sum of weights', str(ex.exception))

    def test_interpolate_zeroVector_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 0, 0],  # Zero vector
            [0, 0, 1],
            [1, 1, 1]
        ]
        weights = [1, 1, 1, 1]

        for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
            with self.subTest(method=method):
                slerp = Slerp(method)
                with self.assertRaises(ValueError) as ex:
                    slerp.interpolate(vectors, weights)
                self.assertIn('zero length', str(ex.exception))

    def test_interpolate_emptyVectors_failure(self):
        for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
            with self.subTest(method=method):
                slerp = Slerp(method)
                with self.assertRaises(ValueError) as ex:
                    slerp.interpolate([], [])
                self.assertIn('empty list of vectors', str(ex.exception))

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
                with self.assertRaises(ValueError) as ex:
                    slerp.interpolate(vectors, weights)
                self.assertIn('same length', str(ex.exception))

    def test_interpolate_wrongWeightsLength_failure(self):
        vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        weights = [1] * 2
        for method in [Slerp.Method.Sequential, Slerp.Method.Hierarchical]:
            with self.subTest(method=method):
                slerp = Slerp(method)
                with self.assertRaises(ValueError) as ex:
                    slerp.interpolate(vectors, weights)
                self.assertIn('must have the same length', str(ex.exception))

    def test_interpolate_wrongInterpolationMethod_failure(self):
        slerp = Slerp("non_existing_method")
        with self.assertRaises(InternalError):  # Changed to AttributeError
            slerp.interpolate([[1, 1, 1]], [1])
