import abc
import math
from enum import Enum
from typing import List

import numpy as np

from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.exceptions import InternalError, InvalidArgumentError


class ZeroSumWeightsError(InvalidArgumentError):
    pass


class ZeroMagnitudeVectorError(InvalidArgumentError):
    pass


class VectorInterpolation(abc.ABC):
    @abc.abstractmethod
    def interpolate(self, vectors: List[List[float]], weights: List[float], prenormalized: bool = False) -> List[float]:
        """
        Interpolates a list of vectors using the given weights.

        Args:
            vectors: A list of vectors to interpolate
            weights: A list of weights to use for the interpolation. If None, the interpolation will be done using
            equal weights for all vectors
            prenormalized: If True, the vectors are assumed to be normalized

        Returns:
            The interpolated vector
        """
        pass


def from_interpolation_method(method: InterpolationMethod):
    if method == InterpolationMethod.SLERP:
        return Slerp()
    elif method == InterpolationMethod.NLERP:
        return Nlerp()
    elif method == InterpolationMethod.LERP:
        return Lerp()
    else:
        raise InternalError(f'Unknown interpolation method: {method}')


class Lerp(VectorInterpolation):
    def interpolate(self, vectors: List[List[float]], weights: List[float], prenormalized: bool = False) -> List[float]:
        """
        Interpolates a list of vectors using the given weights.

        Args:
            vectors: A list of vectors to interpolate
            weights: A list of weights to use for the interpolation. If None, the interpolation will be done using
            equal weights for all vectors
            prenormalized: Ignored for LERP

        Returns:
            The interpolated vector

        Raises:
            ZeroSumWeightsError: If the sum of the weights is zero
        """
        if len(vectors) < 1:
            raise ValueError('Cannot interpolate an empty list of vectors')

        if len(vectors) != len(weights):
            raise ValueError('Vectors and weights must have the same length')

        # Convert inputs to NumPy arrays for faster processing
        np_vectors = np.array(vectors)
        np_weights = np.array(weights)
        
        # Check if all vectors have the same length
        if len(set(len(v) for v in vectors)) != 1:
            raise ValueError('Vectors must have the same length')
        
        # Get sum of absolute values of weights
        weight_sum = np.sum(np.abs(np_weights))

        if weight_sum == 0:
            raise ZeroSumWeightsError(
                'Sum of weights is zero. LERP cannot interpolate vectors with zero sum of weights'
            )

        # Calculate normalized weights (divide all by sum of absolute values)
        normalized_weights = np_weights / weight_sum
        
        # Multiply each vector by its corresponding normalized weight, then sum (weighted average)
        result = np.sum(np_vectors * normalized_weights[:, np.newaxis], axis=0)
        
        # Convert back to list for consistent return type
        return result.tolist()


class Nlerp(Lerp):
    def interpolate(self, vectors: List[List[float]], weights: List[float],
                    prenormalized: bool = False) -> List[float]:
        """
        Interpolates a list of vectors using the given weights.

        Args:
            vectors: A list of vectors to interpolate
            weights: A list of weights to use for the interpolation. If None, the interpolation will be done using
            equal weights for all vectors
            prenormalized: Ignored for LERP

        Returns:
            The interpolated vector

        Raises:
            ZeroSumWeightsError: If the sum of the weights is zero
            ZeroMagnitudeVectorError: If the interpolated vector has zero magnitude
        """
        lerp_result = super().interpolate(vectors, weights)
        
        # Convert to NumPy array for efficient operations
        np_result = np.array(lerp_result)
        
        # Calculate the norm using NumPy's faster norm function
        norm = np.linalg.norm(np_result)

        if norm == 0:
            raise ZeroMagnitudeVectorError(
                'Interpolated vector has zero magnitude. Cannot normalize a vector with zero magnitude'
            )

        # Normalize the vector using NumPy's efficient division
        normalized_result = np_result / norm
        
        return normalized_result.tolist()


class Slerp(VectorInterpolation):
    class Method(Enum):
        Sequential = 0
        Hierarchical = 1

    def __init__(self, method: Method = Method.Hierarchical):
        self.method = method

    def interpolate(self, vectors: List[List[float]], weights: List[float], prenormalized: bool = False) -> List[float]:
        """
        Interpolates a list of vectors using the given weights.

        Args:
            vectors: A list of vectors to interpolate
            weights: A list of weights to use for the interpolation. If None, the interpolation will be done using
            equal weights for all vectors
            prenormalized: If True, the vectors are assumed to be normalized

        Returns:
            The interpolated vector

        Raises:
            ZeroSumWeightsError: If the sum of a consecutive pair of weights is zero
        """
        if len(vectors) < 1:
            raise ValueError('Cannot interpolate an empty list of vectors')

        if len(vectors) != len(weights):
            raise ValueError('Vectors and weights must have the same length')

        # Convert inputs to NumPy arrays for faster processing
        np_vectors = np.array(vectors)
        np_weights = np.array(weights)

        if self.method == self.Method.Sequential:
            return self._interpolate_sequential(np_vectors, np_weights, prenormalized)
        elif self.method == self.Method.Hierarchical:
            return self._interpolate_hierarchical(np_vectors, np_weights, prenormalized)
        else:
            raise InternalError(f'Unknown interpolation method: {self.method}')

    def _slerp(self, v0: np.ndarray, v1: np.ndarray, t: float, prenormalized: bool = False) -> List[float]:
        """Spherical linear interpolation between two vectors."""
        if v0.shape != v1.shape:
            raise ValueError(f'Vectors must have the same length. Got {v0.shape} and {v1.shape}')

        dot = np.dot(v0, v1)

        if not prenormalized:
            norm_v0 = np.linalg.norm(v0)
            norm_v1 = np.linalg.norm(v1)

            # Note we can only detect zero length if we calculate the norm
            if norm_v0 == 0 or norm_v1 == 0:
                raise ValueError('One or more vectors had zero length. '
                                'SLERP cannot interpolate vectors with zero length')

            cos = dot / (norm_v0 * norm_v1)
        else:
            cos = dot

        # Ensure the dot product is within the range [-1, 1]
        cos = np.clip(cos, -1.0, 1.0)

        theta = np.arccos(cos)
        sin_theta = np.sin(theta)
        
        if sin_theta == 0:
            # Co-linear vectors, return linear interpolation
            result = (1 - t) * v0 + t * v1
        else:
            # Use NumPy's vectorized operations for the calculation
            s0 = np.sin((1 - t) * theta) / sin_theta
            s1 = np.sin(t * theta) / sin_theta
            result = s0 * v0 + s1 * v1
            
        return result.tolist()

    def _interpolate_sequential(self, vectors: np.ndarray, weights: np.ndarray, prenormalized: bool = False) -> List[float]:
        """Sequential interpolation of vectors."""
        weights_copy = weights.copy()
        result = vectors[0].copy()
        
        for i in range(1, len(vectors)):
            w0 = weights_copy[i - 1]
            w1 = weights_copy[i]
            weight_sum = np.abs(w0) + np.abs(w1)

            if weight_sum == 0:
                raise ZeroSumWeightsError(
                    f'Sum of weights {w0} and {w1} is zero. SLERP cannot interpolate '
                    'vectors with a sum weight of zero'
                )

            # Interpolate between current result and next vector
            result = self._slerp(result, vectors[i], w1 / weight_sum, prenormalized)    # TODO: Check if abs is necessary here because it's not a sum
            weights_copy[i] = weight_sum / 2
            
        return result

    def _interpolate_hierarchical(self, vectors: np.ndarray, weights: np.ndarray, prenormalized: bool = False) -> List[float]:
        """Hierarchical interpolation of vectors."""
        # Work with copies to avoid modifying the originals
        vecs = vectors.copy()
        wts = weights.copy()
        
        while len(vecs) > 1:
            n = len(vecs)
            result_size = (n + 1) // 2  # Ceiling division for odd numbers
            
            # Pre-allocate arrays for results
            result = np.zeros((result_size, vecs.shape[1]), dtype=vecs.dtype)
            new_weights = np.zeros(result_size, dtype=wts.dtype)
            
            # Process pairs of vectors
            pair_count = n // 2
            
            for i in range(pair_count):
                w0 = wts[2*i]
                w1 = wts[2*i + 1]
                weight_sum = np.abs(w0) + np.abs(w1)
                
                if weight_sum == 0:
                    raise ZeroSumWeightsError(
                        f'Sum of weights {w0} and {w1} is zero. SLERP cannot interpolate '
                        'vectors with a sum weight of zero'
                    )
                
                # Calculate interpolation and store result
                t = w1 / weight_sum         # TODO: Find out if abs is necessary here for weight 1
                result[i] = self._slerp(vecs[2*i], vecs[2*i + 1], t, prenormalized)
                new_weights[i] = weight_sum / 2
            
            # Handle odd number of vectors
            if n % 2 == 1:
                result[-1] = vecs[-1]
                new_weights[-1] = wts[-1]
            
            # Update for next iteration
            vecs = result
            wts = new_weights
        
        return vecs[0].tolist()
