from abc import ABC, abstractmethod
from typing import Dict, Any

from marqo.core.models import MarqoQuery, MarqoIndex
from marqo.core.models.marqo_index import IndexType


class VespaIndex(ABC):
    """
    An abstract class for classes that facilitate data and query transformation to and from a Vespa index.

    Methods in this class do not talk to Vespa directly, but rather transform data and queries to and from a format
    that can be used by a VespaClient.
    """
    _HANDLEABLE_INDEX_TYPES = None

    def __init_subclass__(cls, **kwargs):
        if cls._HANDLEABLE_INDEX_TYPES is None:
            raise NotImplementedError("VespaIndex._HANDLEABLE_INDEX_TYPES must be defined")
        super().__init_subclass__()

    @classmethod
    @abstractmethod
    def generate_schema(cls, marqo_index: MarqoIndex) -> str:
        """
        Generate a Vespa schema for the given MarqoIndex.

        Args:
            marqo_index: The MarqoIndex to generate a schema for

        Returns:
            A string containing the Vespa schema
        """
        pass

    @classmethod
    @abstractmethod
    def to_vespa_document(cls, marqo_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        """
        Convert a MarqoDocument to a Vespa document.

        Args:
            marqo_document: The MarqoDocument to convert
            marqo_index: The MarqoIndex the document belongs to

        Returns:
            A dictionary containing the Vespa document
        """
        pass

    @classmethod
    @abstractmethod
    def to_marqo_document(
            cls, vespa_document: Dict[str, Any], marqo_index: MarqoIndex, return_highlights: bool = False
    ) -> Dict[str, Any]:
        """
        Convert a Vespa document to a MarqoDocument.

        This method is not guaranteed to be the inverse of to_vespa_document and this depends on the implementation.

        Args:
            vespa_document: The Vespa document to convert
            marqo_index: The MarqoIndex the document belongs to
            return_highlights: Whether to return highlights

        Returns:
            A dictionary containing the MarqoDocument
        """
        pass

    @classmethod
    @abstractmethod
    def to_vespa_query(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        """
        Convert a MarqoQuery to a Vespa query.

        Args:
            marqo_query: The MarqoQuery to convert
            marqo_index: The MarqoIndex the query belongs to

        Returns:
            A dictionary containing the Vespa query
        """
        pass

    @classmethod
    def _validate_index_type(cls, marqo_index: MarqoIndex) -> None:
        if marqo_index.type != cls._HANDLEABLE_INDEX_TYPES:
            raise ValueError(
                f"Vespa index type must be {cls._HANDLEABLE_INDEX_TYPES}. "
                f"This module cannot handle index type {marqo_index.type.name}."
            )


def for_marqo_index(marqo_index: MarqoIndex):
    """
    Get the VespaIndex implementation for the given MarqoIndex.

    Args:
        marqo_index: The MarqoIndex to get the implementation for

    Returns:
        The VespaIndex implementation for the given MarqoIndex
    """
    if marqo_index.type == IndexType.Structured:
        from marqo.core.structured_vespa_index import StructuredVespaIndex
        return StructuredVespaIndex
    elif marqo_index.type == IndexType.Unstructured:
        from marqo.core.unstructured_vespa_index import UnstructuredVespaIndex
        return UnstructuredVespaIndex
    else:
        raise ValueError(f"No known implementation for index type {marqo_index.type}")
