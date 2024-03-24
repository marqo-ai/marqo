import abc
import math
from enum import Enum
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from marqo.exceptions import InternalError


class VectorInterpolation(abc.ABC):
    @abc.abstractmethod
    def interpolate(self, vectors: List[List[float]], weights: List[float] = None) -> List[float]:
        """
        Interpolates a list of vectors using the given weights.

        Args:
            vectors: A list of vectors to interpolate.
            weights: A list of weights to use for the interpolation. If None, the interpolation will be done using
            euqual weights for all vectors.

        Returns:
            The interpolated vector.
        """
        pass


class Lerp(VectorInterpolation):
    def interpolate(self, vectors: List[List[float]], weights: List[float] = None) -> List[float]:
        if not weights:
            weights = [1] * len(vectors)
        weight_sum = sum(weights)
        result = [0] * len(vectors[0])
        for vector, weight in zip(vectors, weights):
            for i, value in enumerate(vector):
                result[i] += (weight / weight_sum) * value
        return result


class Nlerp(VectorInterpolation):
    def interpolate(self, vectors: List[List[float]], weights: List[float] = None) -> List[float]:
        lerp_result = Lerp().interpolate(vectors, weights)
        length = math.sqrt(sum(x ** 2 for x in lerp_result))
        return [x / length for x in lerp_result]


class Slerp(VectorInterpolation):
    class Method(Enum):
        Sequential = 0
        Hierarchical = 1

    def __init__(self, method: Method = Method.Hierarchical):
        self.method = method
        self._slerp_impl = self._slerp

    def _slerp_scipy(self, v1: List[float], v2: List[float], t: float) -> List[float]:
        # Convert vectors to quaternions (assuming vectors are 3D)
        q1 = np.array([0] + v1)
        q2 = np.array([0] + v2)

        # Create a Slerp object and interpolate
        slerp = Slerp([0, 1], R.from_quat([q1, q2]))
        interpolated_quaternion = slerp([t]).as_quat()[0]

        # Return the vector part of the interpolated quaternion
        return interpolated_quaternion[1:].tolist()

    def _slerp(self, v0: List[float], v1: List[float], t: float) -> List[float]:
        dot = sum(a * b for a, b in zip(v0, v1))
        theta = math.acos(dot) * t
        relative_vec = [b - a * dot for a, b in zip(v0, v1)]
        return [math.cos(theta) * a + math.sin(theta) * b for a, b in zip(v0, relative_vec)]

    def _interpolate_sequential(self, vectors: List[List[float]], weights: List[float]) -> List[float]:
        result = vectors[0]
        for i in range(1, len(vectors)):
            result = self._slerp_impl(result, vectors[i], weights[i])
        return result

    def _interpolate_hierarchical(self, vectors: List[List[float]], weights: List[float]) -> List[float]:
        if len(vectors) > 1 and len(vectors) % 2 != 0:
            # interpolate the first pair to get an even number of vectors
            vectors = [self._slerp_impl(vectors[0], vectors[1], weights[1] / (weights[0] + weights[1]))] + vectors[2:]
            weights = [(weights[0] + weights[1]) / 2] + weights[2:]

        while len(vectors) > 1:
            result = []
            new_weights = []
            for i in range(0, len(vectors), 2):
                result.append(
                    self._slerp_impl(vectors[i], vectors[i + 1], weights[i + 1] / (weights[i] + weights[i + 1]))
                )
                new_weights.append((weights[i] + weights[i + 1]) / 2)
            vectors = result
            weights = new_weights

        return vectors[0]

    def interpolate(self, vectors: List[List[float]], weights: List[float] = None) -> List[float]:
        if len(vectors) < 1:
            raise ValueError('Cannot interpolate an empty list of vectors')

        if self.method == self.Method.Sequential:
            return self._interpolate_sequential(vectors, weights)
        elif self.method == self.Method.Hierarchical:
            return self._interpolate_hierarchical(vectors, weights)
        else:
            raise InternalError(f'Unknown interpolation method: {self.method}')
