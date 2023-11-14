from typing import Dict, Any
import textwrap

from marqo.core.models import MarqoQuery, MarqoIndex
import marqo.core.search.search_filter as search_filter
from marqo.core.exceptions import InvalidDataTypeError, InvalidFieldNameError, VespaDocumentParsingError
from marqo.core.vespa_index import VespaIndex
from marqo.core.models.unstructured_document import UnStructuredIndexDocument
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery, ScoreModifierType, \
    ScoreModifier
from marqo.exceptions import InternalError


class UnStructuredVespaIndex(VespaIndex):
    """
    An implementation of VespaIndex for unstructured indexes.
    """

    _EMBEDDINGS_FIELDS_NAME = "marqo_embeddings"
    _HANDLEABLE_INDEX_TYPES = IndexType.Unstructured

    _QUERY_INPUT_EMBEDDING = 'marqo__query_embedding'

    _RANK_PROFILE_EMBEDDING_SIMILARITY = 'embedding_similarity'

    _SUMMARY_ALL_NON_VECTOR = 'all-non-vector-summary'
    _SUMMARY_ALL_VECTOR = 'all-vector-summary'

    @classmethod
    def generate_schema(cls, marqo_index: MarqoIndex) -> str:
        cls._validate_index_type(marqo_index)

        schema = cls._generate_unstructured_index_schema(marqo_index)

        return schema

    @classmethod
    def to_vespa_document(cls, marqo_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        cls._validate_index_type(marqo_index)

        return UnStructuredIndexDocument.from_marqo_document(marqo_document).to_vespa_document()

    @classmethod
    def to_marqo_document(cls, vespa_document: Dict[str, Any], marqo_index: MarqoIndex, return_highlights: bool = False) \
            -> Dict[str, Any]:
        cls._validate_index_type(marqo_index)

        return UnStructuredIndexDocument.from_vespa_document(vespa_document).\
            to_marqo_document(return_highlights = return_highlights)

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
            raise InvalidArgumentError("searchable_attributes is not supported for an UnStructured Index")

        tensor_term = cls._get_tensor_search_term(marqo_query, marqo_index)
        filter_term = cls._get_filter_term(marqo_query, marqo_index)
        if filter_term:
            filter_term = f' AND {filter_term}'
        else:
            filter_term = ''

        select_attributes = cls._get_select_attributes(marqo_query)
        summary = cls._SUMMARY_ALL_VECTOR if marqo_query.expose_facets else cls._SUMMARY_ALL_NON_VECTOR
        score_modifiers = cls._get_score_modifiers(marqo_query)

        # ranking = cls._RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS if score_modifiers \
        #     else cls._RANK_PROFILE_EMBEDDING_SIMILARITY
        ranking = cls._RANK_PROFILE_EMBEDDING_SIMILARITY

        query_inputs = {
            cls._QUERY_INPUT_EMBEDDING: marqo_query.vector_query
        }

        if score_modifiers:
            query_inputs.update(score_modifiers)

        query = {
            'yql': f'select {select_attributes} from {marqo_query.index_name} where {tensor_term}{filter_term}',
            'model_restrict': marqo_query.index_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': query_inputs,
            'presentation.summary': summary,
            'ranking': ranking
        }
        query = {k: v for k, v in query.items() if v is not None}

        return query

    @classmethod
    def _get_tensor_search_term(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> str:
        field_to_search = cls._EMBEDDINGS_FIELDS_NAME

        return (f"({{targetHits:{marqo_query.limit}, approximate:{str(marqo_query.approximate)}}}"
                f"nearestNeighbor({field_to_search}, {cls._QUERY_INPUT_EMBEDDING}))")

    @classmethod
    def _get_select_attributes(cls, marqo_query: MarqoQuery) -> str:
        if marqo_query.attributes_to_retrieve is not None:
            return ', '.join(marqo_query.attributes_to_retrieve)
        else:
            return '*'

    @classmethod
    def _get_score_modifiers(cls, marqo_query: MarqoQuery) -> \
            Optional[Dict[str, Dict[str, float]]]:
        if marqo_query.score_modifiers:
            mult_tensor = {}
            add_tensor = {}
            for modifier in marqo_query.score_modifiers:
                if modifier.type == ScoreModifierType.Multiply:
                    mult_tensor[modifier.field] = modifier.weight
                elif modifier.type == ScoreModifierType.Add:
                    add_tensor[modifier.field] = modifier.weight
                else:
                    raise InternalError(f'Unknown score modifier type {modifier.type}')

            # Note one of these could be empty, but not both
            return {
                cls._QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS: mult_tensor,
                cls._QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS: add_tensor
            }

        return None

    @classmethod
    def _get_filter_term(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> Optional[str]:
        def escape(s: str) -> str:
            return s.replace('\\', '\\\\').replace('"', '\\"')

        def tree_to_filter_string(node: search_filter.Node) -> str:
            if isinstance(node, search_filter.Operator):
                if isinstance(node, search_filter.And):
                    operator = 'AND'
                elif isinstance(node, search_filter.Or):
                    operator = 'OR'
                else:
                    raise InternalError(f'Unknown operator type {type(node)}')

                return f'({tree_to_filter_string(node.left)} {operator} {tree_to_filter_string(node.right)})'
            elif isinstance(node, search_filter.Modifier):
                if isinstance(node, search_filter.Not):
                    return f'!({tree_to_filter_string(node.modified)})'
                else:
                    raise InternalError(f'Unknown modifier type {type(node)}')
            elif isinstance(node, search_filter.Term):
                # if node.field not in marqo_index.filterable_fields_names:
                #     raise InvalidFieldNameError(
                #         f'Index {marqo_index.name} has no filterable field {node.field}. '
                #         f'Available filterable fields are: {", ".join(marqo_index.filterable_fields_names)}'
                #     )

                #TODO Remove the validation and just use the base case
                if isinstance(node, search_filter.EqualityTerm):
                    #TODO Instead of using contains, use sameElement to replace this one
                    return f'{node.field} contains "{escape(node.value)}"'
                elif isinstance(node, search_filter.RangeTerm):
                    #TODO Use sameElement to replace the range
                    lower = f'{node.field} >= {node.lower}' if node.lower is not None else None
                    upper = f'{node.field} <= {node.upper}' if node.upper is not None else None
                    if lower and upper:
                        return f'({lower} AND {upper})'
                    elif lower:
                        return lower
                    elif upper:
                        return upper
                    else:
                        raise InternalError('RangeTerm has no lower or upper bound')

                raise InternalError(f'Unknown node type {type(node)}')

        if marqo_query.filter is not None:
            return tree_to_filter_string(marqo_query.filter.root)

    @classmethod
    def _generate_unstructured_index_schema(cls, marqo_index: MarqoIndex) -> str:
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
                    field marqo__id type string {{
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
    
                rank-profile {cls._RANK_PROFILE_EMBEDDING_SIMILARITY} inherits default {{
                    inputs {{
                        query({cls._QUERY_INPUT_EMBEDDING}) tensor<float>(x[{dimension}])
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
                    summary marqo__id type string {{}}
                    summary strings type array<string> {{}}
                    summary long_string_fields type map<string, string> {{}}
                    summary short_string_fields type map<string, string> {{}}
                    summary string_arrays type array<string> {{}}
                    summary int_fields type map<string, int> {{}}
                    summary float_fields type map<string, float> {{}}
                    summary marqo_chunks type array<string> {{}}
                }}
    
                document-summary all-summary {{
                    summary marqo__id type string {{}}
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


