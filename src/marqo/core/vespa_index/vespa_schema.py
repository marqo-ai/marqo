from abc import ABC, abstractmethod

import semver

from marqo.core.models.marqo_index_request import MarqoIndexRequest, StructuredMarqoIndexRequest, \
    UnstructuredMarqoIndexRequest

from marqo.core.models.marqo_index import *

MINIMUM_SEMI_STRUCTURED_INDEX_VERSION = semver.VersionInfo.parse('2.13.0')

class VespaSchema(ABC):
    """
    An abstract class for classes that generate Vespa schemas.
    """

    _INDEX_NAME_ENCODING_MAP = {
        '_': '_00',
        '-': '_01',
    }

    _DISTANCE_METRIC_MAP = {
        DistanceMetric.Euclidean: 'euclidean',
        DistanceMetric.Angular: 'angular',
        DistanceMetric.DotProduct: 'dotproduct',
        DistanceMetric.PrenormalizedAngular: 'prenormalized-angular',
        DistanceMetric.Geodegrees: 'geodegrees',
        DistanceMetric.Hamming: 'hamming'
    }

    def _get_distance_metric(self, marqo_distance_metric: DistanceMetric) -> str:
        try:
            return self._DISTANCE_METRIC_MAP[marqo_distance_metric]
        except KeyError:
            raise ValueError(f'Unknown Marqo distance metric: {marqo_distance_metric}')

    @abstractmethod
    def generate_schema(self) -> (str, MarqoIndex):
        """
        Generate a Vespa schema.

        Returns:
            A string containing the Vespa schema, and the corresponding MarqoIndex.
        """
        pass

    def _get_vespa_schema_name(self, index_name) -> str:
        """
        Get the name of the Vespa schema.

        Args:
            index_name: The name of the index

        Returns:
            The name of the Vespa schema

        Notes:
            Vespa schema names must conform to [a-zA-Z_][a-zA-Z0-9_]*. This method applies an encoding similar to URL
            encoding (using _ instead of %) to ensure that the schema name is valid. This encoding is a bijection, i.e.
            the encoded string is unique and reversible.

            Index names that are already valid Vespa schema names and do not contain underscores are not encoded.
            Encoded names are prefixed with MARQO_RESERVED_PREFIX for easy identification, even though this prefix is
            not required for uniqueness.
        """
        encoded_name_chars = []
        for char in index_name:
            if char in self._INDEX_NAME_ENCODING_MAP:
                encoded_name_chars.append(self._INDEX_NAME_ENCODING_MAP[char])
            else:
                encoded_name_chars.append(char)

        encoded_name = ''.join(encoded_name_chars)

        if encoded_name == index_name:
            return index_name

        return constants.MARQO_RESERVED_PREFIX + encoded_name


def for_marqo_index_request(marqo_index_request: MarqoIndexRequest):
    """
    Get the VespaSchema implementation for the given MarqoIndexRequest.

    Args:
        marqo_index_request: The MarqoIndexRequest to get the implementation for

    Returns:
        The VespaSchema implementation for the given MarqoIndexRequest
    """
    if isinstance(marqo_index_request, StructuredMarqoIndexRequest):
        from marqo.core.structured_vespa_index.structured_vespa_schema import StructuredVespaSchema
        return StructuredVespaSchema(marqo_index_request)
    elif isinstance(marqo_index_request, UnstructuredMarqoIndexRequest):
        if semver.Version.parse(marqo_index_request.marqo_version) < MINIMUM_SEMI_STRUCTURED_INDEX_VERSION:
            from marqo.core.unstructured_vespa_index.unstructured_vespa_schema import UnstructuredVespaSchema
            return UnstructuredVespaSchema(marqo_index_request)
        else:
            from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
            return SemiStructuredVespaSchema(marqo_index_request)
    else:
        raise ValueError(f"No known implementation for index type {type(marqo_index_request)}")
