from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from marqo.core import constants
from marqo.core.models import MarqoQuery, MarqoHybridQuery, MarqoTensorQuery, MarqoLexicalQuery, MarqoIndex
from marqo.core.models.marqo_index import StructuredMarqoIndex, UnstructuredMarqoIndex
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.core.models.marqo_index import *
from marqo.exceptions import InternalError


class VespaIndex(ABC):
    """
    An abstract class for classes that facilitate data and query transformation to and from a Vespa index.

    Methods in this class do not talk to Vespa directly, but rather transform data and queries to and from a format
    that can be used by a VespaClient.
    """

    _VERSION_2_9_0 = semver.VersionInfo.parse("2.9.0")
    _VERSION_2_10_0 = semver.VersionInfo.parse("2.10.0")

    def __init__(self, marqo_index: MarqoIndex):
        self._marqo_index = marqo_index
        self._marqo_index_version = marqo_index.parsed_marqo_version()

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

    @abstractmethod
    def get_vector_count_query(self) -> Dict[str, Any]:
        """
        Get a query that returns the number of vectors in the index.

        Returns:
            A query that aggregates over a single field across all documents in the index. For a non-empty index,
            the following expression over a Vespa client QueryResult query_result will extract the number of vectors:
            ```
            list(query_result.root.children[0].children[0].children[0].fields.values())[0]
            ```
            while for an empty index, `query_result.root.children[0].children is None` will hold true.
        """
        pass

    @abstractmethod
    def to_vespa_partial_document(self, marqo_partial_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a marqo_partial_update_document to a Vespa partial document.

        This method converts a MarqoDocument to a VespaDocument in the partial update format. It should only contain
        the fields that are require to be updated.

        Args:
            marqo_partial_document: The marqo_partial_document to convert

        Returns:
            VespaDocument in dictionary format with keys 'fields' and 'id'
        """
        pass

    @abstractmethod
    def get_vespa_id_field(self) -> str:
        """
        Get the name of the id field in Vespa documents, inside the 'fields' dictionary."""
        pass

    def _convert_score_modifiers_to_tensors(self, score_modifiers: List[ScoreModifier]) -> Dict[
        str, Dict[str, float]]:
        """
        Helper function that converts a list of score modifiers into 2 dictionaries:
        These dictionaries are 'mult' and 'add' weights.
        """
        mult_tensor = {}
        add_tensor = {}
        for modifier in score_modifiers:
            if modifier.type == ScoreModifierType.Multiply:
                mult_tensor[modifier.field] = modifier.weight
            elif modifier.type == ScoreModifierType.Add:
                add_tensor[modifier.field] = modifier.weight
            else:
                raise InternalError(f'Unknown score modifier type {modifier.type}')

        return mult_tensor, add_tensor

    def _get_score_modifiers(self, marqo_query: MarqoQuery) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Returns classic score modifiers (from tensor or lexical queries) as a dictionary of dictionaries.
        Split between 'mult' and 'add' weights.
        """
        if marqo_query.score_modifiers:
            mult_tensor, add_tensor = self._convert_score_modifiers_to_tensors(marqo_query.score_modifiers)

            if self._marqo_index_version < self._HYBRID_SEARCH_MINIMUM_VERSION:
                return {
                    constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_2_9: mult_tensor,
                    constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_2_9: add_tensor
                }
            elif isinstance(marqo_query, MarqoTensorQuery):
                return {
                    constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR: mult_tensor,
                    constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR: add_tensor
                }
            elif isinstance(marqo_query, MarqoLexicalQuery):
                return {
                    constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL: mult_tensor,
                    constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL: add_tensor
                }
            else:
                raise InternalError(f'Unknown query type {type(marqo_query)}')

        return None

    def _get_hybrid_score_modifiers(self, hybrid_query: MarqoHybridQuery) -> \
            Optional[Dict[str, Dict[str, Dict[str, float]]]]:

        """
        Specifically for hybrid queries.
        Returns a dictionary with 2 keys: 'lexical' and 'tensor'.
        Each key points to a dictionary containing the score modifiers for the respective field types.

        Example:
        {
            'lexical': {
                'marqo__mult_weights_lexical': {
                    'field1': 0.5, 'field2': 0.4
                },
                'marqo__add_weights_lexical': {
                    'field3': 23, 'field4': 12
                }
            },
            'tensor': {
                'marqo__mult_weights_tensor': {
                    'field5': 0.5, 'field6': 0.4
                },
                'marqo__add_weights_tensor': {
                    'field7': 23, 'field8': 12
                }
            }
        }
        """

        result = {
            constants.MARQO_SEARCH_METHOD_LEXICAL: None,
            constants.MARQO_SEARCH_METHOD_TENSOR: None
        }

        if hybrid_query.score_modifiers_lexical:
            mult_tensor, add_tensor = self._convert_score_modifiers_to_tensors(hybrid_query.score_modifiers_lexical)
            result[constants.MARQO_SEARCH_METHOD_LEXICAL] = {
                constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL: mult_tensor,
                constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL: add_tensor
            }

        if hybrid_query.score_modifiers_tensor:
            mult_tensor, add_tensor = self._convert_score_modifiers_to_tensors(hybrid_query.score_modifiers_tensor)
            result[constants.MARQO_SEARCH_METHOD_TENSOR] = {
                constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR: mult_tensor,
                constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR: add_tensor
            }

        return result


def for_marqo_index(marqo_index: MarqoIndex) -> VespaIndex:
    """
    Get the VespaIndex implementation for the given MarqoIndex.

    Args:
        marqo_index: The MarqoIndex to get the implementation for
        marqo_version: The version of Marqo that the index was created with

    Returns:
        The VespaIndex implementation for the given MarqoIndex
    """
    if isinstance(marqo_index, SemiStructuredMarqoIndex):
        from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
        return SemiStructuredVespaIndex(marqo_index)
    elif isinstance(marqo_index, StructuredMarqoIndex):
        from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
        return StructuredVespaIndex(marqo_index)
    elif isinstance(marqo_index, UnstructuredMarqoIndex):
        from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
        return UnstructuredVespaIndex(marqo_index)
    else:
        raise ValueError(f"No known implementation for index type {type(marqo_index)}")