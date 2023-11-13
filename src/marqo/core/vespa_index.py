from abc import ABC, abstractmethod
from typing import Dict, Any

from marqo.core.models import MarqoQuery, MarqoIndex
from marqo.core.models.marqo_index import StructuredMarqoIndex, UnstructuredMarqoIndex


class VespaIndex(ABC):
    """
    An abstract class for classes that facilitate data and query transformation to and from a Vespa index.

    Methods in this class do not talk to Vespa directly, but rather transform data and queries to and from a format
    that can be used by a VespaClient.
    """

    @abstractmethod
    def to_vespa_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a MarqoDocument to a Vespa document.

        Args:
            marqo_document: The MarqoDocument to convert

        Returns:
            A dictionary containing the Vespa document
        """
        pass

    @abstractmethod
    def to_marqo_document(self, vespa_document: Dict[str, Any], return_highlights: bool = False) -> Dict[str, Any]:
        """
        Convert a Vespa document to a MarqoDocument.

        This method is not guaranteed to be the inverse of to_vespa_document and this depends on the implementation.

        Args:
            vespa_document: The Vespa document to convert
            return_highlights: Whether to return highlights

        Returns:
            A dictionary containing the MarqoDocument
        """
        pass

    @abstractmethod
    def to_vespa_query(self, marqo_query: MarqoQuery) -> Dict[str, Any]:
        """
        Convert a MarqoQuery to a Vespa query.

        Args:
            marqo_query: The MarqoQuery to convert

        Returns:
            A dictionary containing the Vespa query
        """
        pass


def for_marqo_index(marqo_index: MarqoIndex):
    """
    Get the VespaIndex implementation for the given MarqoIndex.

    Args:
        marqo_index: The MarqoIndex to get the implementation for

    Returns:
        The VespaIndex implementation for the given MarqoIndex
    """
    if isinstance(marqo_index, StructuredMarqoIndex):
        from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
        return StructuredVespaIndex(marqo_index)
    elif isinstance(marqo_index, UnstructuredMarqoIndex):
        from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
        return UnstructuredVespaIndex(marqo_index)
    else:
        raise ValueError(f"No known implementation for index type {type(marqo_index)}")
