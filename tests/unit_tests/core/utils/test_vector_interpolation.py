import unittest
from unittest.mock import patch, Mock
import numpy as np

from marqo.core.utils.vector_interpolation import (
    from_interpolation_method, Lerp, Nlerp, Slerp, 
    AllZeroWeightsError, ZeroMagnitudeVectorError
)
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.exceptions import InternalError


class TestFromInterpolationMethod(unittest.TestCase):
    """Test cases for from_interpolation_method function"""

    def test_from_interpolation_method_string_input(self):
        """Test from_interpolation_method with string input"""
        result = from_interpolation_method("slerp")
        self.assertIsInstance(result, Slerp)
        
        result = from_interpolation_method("nlerp")
        self.assertIsInstance(result, Nlerp)
        
        result = from_interpolation_method("lerp")
        self.assertIsInstance(result, Lerp)

    def test_from_interpolation_method_enum_input(self):
        """Test from_interpolation_method with enum input"""
        result = from_interpolation_method(InterpolationMethod.SLERP)
        self.assertIsInstance(result, Slerp)
        
        result = from_interpolation_method(InterpolationMethod.NLERP)
        self.assertIsInstance(result, Nlerp)
        
        result = from_interpolation_method(InterpolationMethod.LERP)
        self.assertIsInstance(result, Lerp)

    def test_from_interpolation_method_invalid_string(self):
        """Test from_interpolation_method with invalid string"""
        with self.assertRaises(InternalError) as cm:
            from_interpolation_method("invalid")
        self.assertIn('Unknown interpolation method', str(cm.exception))


class TestLerpVectorValidation(unittest.TestCase):
    """Test cases for Lerp vector validation (missing lines)"""

    def test_lerp_vector_length_validation(self):
        """Test Lerp vector length validation"""
        lerp = Lerp()
        
        # Test with vectors of different lengths
        vectors = [[1, 2], [3, 4, 5]]  # Different lengths
        weights = [1, 1]
        
        with self.assertRaises(ValueError) as cm:
            lerp.interpolate(vectors, weights)
        self.assertIn('Vectors must have the same length', str(cm.exception))

    def test_lerp_empty_vectors_check(self):
        """Test Lerp empty vectors check"""
        lerp = Lerp()
        
        # Test with empty vectors list
        vectors = []
        weights = []
        
        with self.assertRaises(ValueError) as cm:
            lerp.interpolate(vectors, weights)
        self.assertIn('Cannot interpolate an empty list of vectors', str(cm.exception))

    def test_lerp_weight_sum_calculation(self):
        """Test Lerp weight sum calculation"""
        lerp = Lerp()
        
        # Test with all zero weights
        vectors = [[1, 2], [3, 4]]
        weights = [0, 0]
        
        with self.assertRaises(AllZeroWeightsError) as cm:
            lerp.interpolate(vectors, weights)
        self.assertIn('All weights are zero', str(cm.exception))


class TestNlerpVectorValidation(unittest.TestCase):
    """Test cases for Nlerp vector validation and edge cases"""

    def test_nlerp_vector_length_validation(self):
        """Test Nlerp vector length validation"""
        nlerp = Nlerp()
        
        # Test with vectors of different lengths  
        vectors = [[1, 2], [3, 4, 5]]  # Different lengths
        weights = [1, 1]
        
        with self.assertRaises(ValueError) as cm:
            nlerp.interpolate(vectors, weights)
        self.assertIn('Vectors must have the same length', str(cm.exception))

    def test_nlerp_zero_magnitude_error(self):
        """Test Nlerp zero magnitude error"""
        nlerp = Nlerp()
        
        # Test with vectors that result in zero magnitude when interpolated
        vectors = [[1, 0], [-1, 0]]  # These will cancel out
        weights = [1, 1]
        
        with self.assertRaises(ZeroMagnitudeVectorError) as cm:
            nlerp.interpolate(vectors, weights)
        self.assertIn('Interpolated vector has zero magnitude', str(cm.exception))


class TestSlerpVectorValidation(unittest.TestCase):
    """Test cases for Slerp vector validation and edge cases"""

    def test_slerp_vector_length_validation(self):
        """Test Slerp vector length validation"""
        slerp = Slerp()
        
        # Test with vectors of different lengths
        vectors = [[1, 2], [3, 4, 5]]  # Different lengths
        weights = [1, 1]
        
        with self.assertRaises(ValueError) as cm:
            slerp.interpolate(vectors, weights)
        self.assertIn('Vectors must have the same length', str(cm.exception))

    def test_slerp_all_zero_weights_early_validation(self):
        """Test Slerp all zero weights early validation"""
        slerp = Slerp()
        
        # Test early validation of all zero weights
        vectors = [[1, 0], [0, 1]]
        weights = [0, 0]  # All zero weights
        
        with self.assertRaises(AllZeroWeightsError) as cm:
            slerp.interpolate(vectors, weights)
        self.assertIn('All weights are zero', str(cm.exception))

    def test_slerp_zero_vector_validation(self):
        """Test Slerp zero vector validation"""
        slerp = Slerp()
        
        # Test with zero vector
        vectors = [[0, 0], [1, 0]]  # First vector is zero
        weights = [1, 1]
        
        with self.assertRaises(ValueError) as cm:
            slerp.interpolate(vectors, weights)
        self.assertIn('One or more vectors had zero length', str(cm.exception))

    def test_slerp_single_vector_case(self):
        """Test Slerp single vector case"""
        slerp = Slerp()
        
        # Test with single vector (should return the original vector)
        vectors = [[3, 4]]  # Single vector
        weights = [2]  # Weight doesn't matter for single vector
        
        result = slerp.interpolate(vectors, weights)
        expected = [3, 4]  # Original vector (not normalized)
        np.testing.assert_array_almost_equal(result, expected)

    def test_slerp_numerical_precision_handling(self):
        """Test Slerp numerical precision handling"""
        slerp = Slerp()
        
        # Test with vectors that might cause numerical precision issues
        vectors = [[1, 0], [1, 1e-10]]  # Nearly parallel vectors
        weights = [1, 1]
        
        # Should not raise an error and return a reasonable result
        result = slerp.interpolate(vectors, weights)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_slerp_weight_normalization(self):
        """Test Slerp weight normalization"""
        slerp = Slerp()
        
        # Test with weights that need normalization
        vectors = [[1, 0], [0, 1]]
        weights = [2, 2]  # Non-normalized weights
        
        result = slerp.interpolate(vectors, weights)
        # Should return interpolated result without error
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_slerp_omega_calculation(self):
        """Test Slerp omega calculation"""
        slerp = Slerp()
        
        # Test with orthogonal vectors (90 degree angle)
        vectors = [[1, 0], [0, 1]]
        weights = [1, 1]
        
        result = slerp.interpolate(vectors, weights)
        # Should return valid interpolated result
        self.assertEqual(len(result), 2)
        # Result should be roughly [0.707, 0.707] for equal weights
        np.testing.assert_array_almost_equal(result, [0.7071067811865476, 0.7071067811865475], decimal=10)


class TestSlerpInternalMethods(unittest.TestCase):
    """Test cases for Slerp internal methods to cover missing lines"""

    def test_slerp_unknown_interpolation_method(self):
        """Test Slerp with unknown interpolation method"""
        slerp = Slerp()
        slerp.method = "unknown_method"  # Set invalid method
        
        vectors = [[1, 0], [0, 1]]
        weights = [1, 1]
        
        with self.assertRaises(InternalError) as cm:
            slerp.interpolate(vectors, weights)
        self.assertIn('Unknown interpolation method', str(cm.exception))

    def test_slerp_sequential_all_zero_weights_in_loop(self):
        """Test Slerp sequential method with all zero weights during processing"""
        slerp = Slerp(method=Slerp.Method.Sequential)
        
        # Create scenario where weights become zero during sequential processing
        vectors = [[1, 0], [0, 1], [1, 1]]
        weights = [0, 0, 1]  # First two weights are zero
        
        with self.assertRaises(AllZeroWeightsError) as cm:
            slerp.interpolate(vectors, weights)
        self.assertIn('All weights are zero', str(cm.exception))

    def test_slerp_hierarchical_all_zero_weights_in_loop(self):
        """Test Slerp hierarchical method with all zero weights during processing"""
        slerp = Slerp(method=Slerp.Method.Hierarchical)
        
        # Create scenario where weights become zero during hierarchical processing
        vectors = [[1, 0], [0, 1], [1, 1], [0, 0]]
        weights = [0, 0, 0, 0]  # All weights are zero
        
        with self.assertRaises(AllZeroWeightsError) as cm:
            slerp.interpolate(vectors, weights)
        self.assertIn('All weights are zero', str(cm.exception))

    def test_slerp_sequential_method_execution(self):
        """Test Slerp sequential method execution"""
        slerp = Slerp(method=Slerp.Method.Sequential)
        
        vectors = [[1, 0], [0, 1], [1, 1]]
        weights = [1, 1, 1]
        
        result = slerp.interpolate(vectors, weights)
        # Ensure result is a list of 2 floats
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_slerp_hierarchical_method_execution(self):
        """Test Slerp hierarchical method execution"""
        slerp = Slerp(method=Slerp.Method.Hierarchical)
        
        vectors = [[1, 0], [0, 1], [1, 1]]
        weights = [1, 1, 1]
        
        result = slerp.interpolate(vectors, weights)
        # Ensure result is a list of 2 floats
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_slerp_hierarchical_odd_number_vectors(self):
        """Test Slerp hierarchical method with odd number of vectors"""
        slerp = Slerp(method=Slerp.Method.Hierarchical)
        
        # Use 5 vectors (odd number) to trigger the odd vector handling
        vectors = [[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1]]
        weights = [1, 1, 1, 1, 1]
        
        result = slerp.interpolate(vectors, weights)
        # Ensure result is a list of 2 floats
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_slerp_invalid_method_raises_internal_error(self):
        """Test Slerp with invalid method enum value raises InternalError"""
        slerp = Slerp()
        # Manually set an invalid method to trigger the else branch
        slerp.method = 999  # Invalid enum value
        
        vectors = [[1, 0], [0, 1]]
        weights = [1, 1]
        
        with self.assertRaises(InternalError) as cm:
            slerp.interpolate(vectors, weights)
        self.assertIn('Unknown interpolation method', str(cm.exception))

    def test_slerp_hierarchical_single_vector_final_return(self):
        """Test Slerp hierarchical method returns final vector when only one remains"""
        slerp = Slerp(method=Slerp.Method.Hierarchical)
        
        # Single vector case to test the final return statement
        vectors = [[2, 3]]
        weights = [1]
        
        result = slerp.interpolate(vectors, weights)
        self.assertEqual(result, [2, 3])


if __name__ == '__main__':
    unittest.main() 