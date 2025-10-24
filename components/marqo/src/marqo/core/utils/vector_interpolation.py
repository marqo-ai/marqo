import abc
import math
from enum import Enum
from typing import List

import numpy as np

from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.exceptions import InternalError, InvalidArgumentError


class AllZeroWeightsError(InvalidArgumentError):
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
    method = method.lower()
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
            AllZeroWeightsError: If all weights are zero
        """
        if len(vectors) < 1:
            raise ValueError('Cannot interpolate an empty list of vectors')

        if len(vectors) != len(weights):
            raise ValueError('Vectors and weights must have the same length')

        # Check if all vectors have the same length BEFORE creating NumPy arrays
        # This avoids the NumPy deprecation warning about ragged arrays
        vector_lengths = [len(v) for v in vectors]
        if len(set(vector_lengths)) != 1:
            raise ValueError('Vectors must have the same length')

        # Now safe to convert to NumPy arrays since all vectors have same length
        np_vectors = np.array(vectors)
        np_weights = np.array(weights)
        
        # Get sum of absolute values of weights
        weight_sum = np.sum(np.abs(np_weights))

        if weight_sum == 0:
            raise AllZeroWeightsError(
                'All weights are zero. LERP cannot interpolate vectors with all zero weights.'
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
            AllZeroWeightsError: If all weights are zero
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
            AllZeroWeightsError: If all weights are zero
        """
        if len(vectors) < 1:
            raise ValueError('Cannot interpolate an empty list of vectors')

        if len(vectors) != len(weights):
            raise ValueError('Vectors and weights must have the same length')

        # Check if all vectors have the same length
        if len(set(len(v) for v in vectors)) != 1:
            raise ValueError('Vectors must have the same length')

        # Early validation: check if all weights are zero
        np_weights = np.array(weights)
        weight_sum = np.sum(np.abs(np_weights))
        if weight_sum == 0:
            raise AllZeroWeightsError('All weights are zero. SLERP cannot interpolate vectors with all zero weights.')

        if self.method == self.Method.Sequential:
            return self._interpolate_sequential(vectors, weights, prenormalized)
        elif self.method == self.Method.Hierarchical:
            return self._interpolate_hierarchical(vectors, weights, prenormalized)
        else:
            raise InternalError(f'Unknown interpolation method: {self.method}')

    def _slerp(self, v0: List[float], v1: List[float], t: float, prenormalized: bool = False) -> List[float]:
        v0, v1 = np.array(v0), np.array(v1)

        if len(v0) != len(v1):
            raise ValueError('Vectors must have the same length. Got {} and {}'.format(len(v0), len(v1)))

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
            return ((1 - t) * v0 + t * v1).tolist()

        slerp_v0 = np.sin((1 - t) * theta) / sin_theta * v0
        slerp_v1 = np.sin(t * theta) / sin_theta * v1

        result = slerp_v0 + slerp_v1
        return result.tolist()

    def _interpolate_sequential(self, vectors: List[List[float]], weights: List[float], prenormalized: bool) -> List[float]:
        weights_copy = weights.copy()
        result = vectors[0]
        for i in range(1, len(vectors)):
            w0 = weights_copy[i - 1]
            w1 = weights_copy[i]
            sum = np.abs(w0) + np.abs(w1)

            if sum == 0:
                raise AllZeroWeightsError('All weights are zero. SLERP cannot interpolate '
                                          'vectors with all zero weights.')

            result = self._slerp(result, vectors[i], w1 / sum, prenormalized)
            weights_copy[i] = sum / 2
        return result

    def _interpolate_hierarchical(self, vectors: List[List[float]], weights: List[float], prenormalized: bool) -> List[float]:
        while len(vectors) > 1:
            result = []
            new_weights = []
            for i in range(0, len(vectors), 2):
                if i + 1 == len(vectors):
                    # if there is an odd number of vectors, just append the last one
                    result.append(vectors[i])
                    new_weights.append(weights[i])
                    continue

                w0 = weights[i]
                w1 = weights[i + 1]
                sum = np.abs(w0) + np.abs(w1)

                if sum == 0:
                    raise AllZeroWeightsError('All weights are zero. SLERP cannot interpolate '
                                              'vectors with all zero weights.')

                result.append(
                    self._slerp(vectors[i], vectors[i + 1], w1 / sum, prenormalized)
                )
                new_weights.append(sum / 2)
            vectors = result
            weights = new_weights

        return vectors[0]