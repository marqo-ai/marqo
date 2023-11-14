from abc import ABC, abstractmethod

from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index_request import MarqoIndexRequest, StructuredMarqoIndexRequest, \
    UnstructuredMarqoIndexRequest


class VespaSchema(ABC):
    """
    An abstract class for classes that generate Vespa schemas.
    """

    @abstractmethod
    def generate_schema(self) -> (str, MarqoIndex):
        """
        Generate a Vespa schema.

        Returns:
            A string containing the Vespa schema, and the corresponding MarqoIndex.
        """
        pass


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
        from marqo.core.unstructured_vespa_index.unstructured_vespa_schema import UnstructuredVespaSchema
        return UnstructuredVespaSchema(marqo_index_request)
    else:
        raise ValueError(f"No known implementation for index type {type(marqo_index_request)}")
