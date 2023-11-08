from typing import Dict, Any
import textwrap

from marqo.core.models import MarqoQuery, MarqoIndex
from marqo.core.exceptions import InvalidDataTypeError, InvalidFieldNameError, VespaDocumentParsingError
from marqo.core.vespa_index import VespaIndex
from marqo.core.models.unstructured_document import UnstructuredIndexDocument
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery, ScoreModifierType, \
    ScoreModifier
from marqo.exceptions import InternalError


class UnstructuredVespaIndex(VespaIndex):
    """
    An implementation of VespaIndex for unstructured indexes.
    """

    _EMBEDDINGS_FIELDS_NAME = "marqo_embeddings"
    _HANDLEABLE_INDEX_TYPES = IndexType.Unstructured

    @classmethod
    def generate_schema(cls, marqo_index: MarqoIndex) -> str:
        cls._validate_index_type(marqo_index)

        schema = _generate_unstructured_index_schema(marqo_index)

        return schema

    @classmethod
    def to_vespa_document(cls, marqo_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        cls._validate_index_type(marqo_index)

        return UnstructuredIndexDocument.from_marqo_document(marqo_document).to_vespa_document()

    @classmethod
    def to_marqo_document(cls, vespa_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        cls._validate_index_type(marqo_index)

        return UnstructuredIndexDocument.from_vespa_document(vespa_document).to_marqo_document()

    @classmethod
    def to_vespa_query(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        # Verify attributes to retrieve, if defined
        if marqo_query.attributes_to_retrieve is not None:
            for att in marqo_query.attributes_to_retrieve:
                if att not in marqo_index.field_map:
                    raise InvalidFieldNameError(
                        f'Index {marqo_index.name} has no field {att}. '
                        f'Available fields are {", ".join(marqo_index.field_map.keys())}'
                    )

        # Verify score modifiers, if defined
        if marqo_query.score_modifiers is not None:
            for modifier in marqo_query.score_modifiers:
                if modifier.field not in marqo_index.score_modifier_fields_names:
                    raise InvalidFieldNameError(
                        f'Index {marqo_index.name} has no score modifier field {modifier.field}. '
                        f'Available score modifier fields are {", ".join(marqo_index.score_modifier_fields_names)}'
                    )

        if isinstance(marqo_query, MarqoTensorQuery):
            return cls._to_vespa_tensor_query(marqo_query, marqo_index)
        elif isinstance(marqo_query, MarqoLexicalQuery):
            return cls._to_vespa_lexical_query(marqo_query, marqo_index)
        elif isinstance(marqo_query, MarqoHybridQuery):
            return cls._to_vespa_hybrid_query(marqo_query, marqo_index)
        else:
            raise InternalError(f'Unknown query type {type(marqo_query)}')

    @classmethod
    def _to_vespa_lexical_query(cls, marqo_query: MarqoLexicalQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def _to_vespa_hybrid_query(cls, marqo_query: MarqoHybridQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def _to_vespa_tensor_query(cls, marqo_query: MarqoTensorQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        if marqo_query.searchable_attributes is not None:
            raise InvalidArgumentError("searchable_attributes is not supported for an unstructured index")

        tensor_term = cls.__get_tensor_search_term(marqo_query, marqo_index)

        query = {
            'yql': f'select * from {marqo_query.index_name} where {tensor_term}',
            'model_restrict': marqo_query.index_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': {cls._QUERY_INPUT_EMBEDDING: marqo_query.vector_query},
            'ranking': "embedding_similarity"
        }

        return query

    @classmethod
    def _get_tensor_search_term(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> str:
        field_to_search = cls._EMBEDDINGS_FIELDS_NAME

        return (f"({{targetHits:{marqo_query.limit}, approximate:{str(marqo_query.approximate)}}}"
                f"nearestNeighbor({field_to_search}),({cls._QUERY_INPUT_EMBEDDING}))")




def _generate_unstructured_index_schema(marqo_index: MarqoIndex) -> str:
    """
    Args:
        marqo_index: The MarqoIndex to generate a schema for
    Returns:
        A string containing the Vespa schema
    """
    dimension = str(marqo_index.model.get_dimension())

    return textwrap.dedent(
        f"""
        schema {marqo_index.name} {{
            document {marqo_index.name} {{
                field id type string {{
                    indexing: attribute | summary
                }}

                field strings type array<string>{{
                    indexing: index
                    index: enable-bm25
                }}

                field long_string_fields type map<string, string> {{
                    indexing: summary
                }}

                field short_string_fields type map<string, string> {{
                    indexing: summary
                    struct-field key {{ indexing : attribute
                                       attribute: fast-search
                                       rank: filter }}
                    struct-field value {{ indexing : attribute
                       attribute: fast-search
                      rank: filter }}
                    dictionary: hash
                }}

                field string_arrays type array<string> {{
                    indexing: attribute | summary
                    attribute: fast-search
                    rank: filter
                }}

                field int_fields type map<string, int> {{
                    indexing: summary
                    struct-field key {{ indexing : attribute
                                       attribute: fast-search
                                       rank: filter }}
                    struct-field value {{ indexing : attribute
                                       attribute: fast-search
                                       rank: filter }}
                    dictionary: hash
                }}

                field float_fields type map<string, float> {{
                    indexing: summary
                    struct-field key {{ indexing : attribute
                                       attribute: fast-search
                                       rank: filter }}

                    struct-field value {{ indexing : attribute
                                       attribute: fast-search
                                       rank: filter }}
                    dictionary: hash
                }}


                field score_modifiers type tensor<float>(p{{}}) {{
                    indexing: attribute
                }}

                field marqo_chunks type array<string> {{
                    indexing: attribute | summary
                }}

                field marqo_embeddings type tensor<float>(p{{}}, x[{dimension}]) {{
                    indexing: attribute | index | summary
                    attribute {{
                        distance-metric: prenormalized-angular
                    }}
                    index {{
                        hnsw {{
                            max-links-per-node: {marqo_index.hnsw_config.m}
                            neighbors-to-explore-at-insert: {marqo_index.hnsw_config.ef_construction}
                        }}
                    }}
                }}
            }}

            fieldset default {{
                fields: strings
            }}

            rank-profile embedding_similarity inherits default {{
                inputs {{
                    query(query_embedding) tensor<float>(x[{dimension}])
                }}
                first-phase {{
                    expression: closeness(field, marqo_embeddings)
                }}
                match-features: closest(marqo_embeddings)
            }}

            rank-profile bm25 inherits default {{
                first-phase {{
                expression: bm25(strings)
                }}
            }}

            document-summary all-non-vector-summary {{
                summary documentid type string {{}}
                summary strings type array<string> {{}}
                summary long_string_fields type map<string, string> {{}}
                summary short_string_fields type map<string, string> {{}}
                summary string_arrays type array<string> {{}}
                summary int_fields type map<string, int> {{}}
                summary float_fields type map<string, float> {{}}
                summary marqo_chunks type array<string> {{}}
            }}

            document-summary all-summary {{
                summary documentid type string {{}}
                summary strings type array<string> {{}}
                summary long_string_fields type map<string, string> {{}}
                summary short_string_fields type map<string, string> {{}}
                summary string_arrays type array<string> {{}}
                summary int_fields type map<string, int> {{}}
                summary float_fields type map<string, float> {{}}
                summary marqo_embeddings type tensor<float>(p{{}}, x[{dimension}]) {{}}
            }}
        }}
        """
    )


