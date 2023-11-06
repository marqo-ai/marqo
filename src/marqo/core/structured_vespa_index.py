from typing import Union

import marqo.core.constants as constants
import marqo.core.search.search_filter as search_filter
from marqo.core.exceptions import InvalidDataTypeError, InvalidFieldNameError, VespaDocumentParsingError
from marqo.core.models import MarqoQuery
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery, ScoreModifierType
from marqo.core.vespa_index import VespaIndex
from marqo.exceptions import InternalError
from tests.marqo_test import MarqoTestCase


class StructuredVespaIndex(VespaIndex):
    """
    An implementation of VespaIndex for structured indexes.
    """
    _MARQO_TO_VESPA_TYPE_MAP = {
        FieldType.Text: 'string',
        FieldType.Bool: 'bool',
        FieldType.Int: 'int',
        FieldType.Float: 'float',
        FieldType.ArrayText: 'array<string>',
        FieldType.ArrayInt: 'array<int>',
        FieldType.ArrayFloat: 'array<float>',
        FieldType.ImagePointer: 'string',
        FieldType.MultimodalCombination: 'map<string, float>'
    }

    _MARQO_TO_PYTHON_TYPE_MAP = {
        FieldType.Text: str,
        FieldType.Bool: bool,
        FieldType.Int: int,
        FieldType.Float: [float, int],
        FieldType.ArrayText: list,
        FieldType.ArrayInt: list,
        FieldType.ArrayFloat: list,
        FieldType.ImagePointer: str,
        FieldType.MultimodalCombination: dict
    }

    _DISTANCE_METRIC_MAP = {
        DistanceMetric.Euclidean: 'euclidean',
        DistanceMetric.Angular: 'angular',
        DistanceMetric.DotProduct: 'dotproduct',
        DistanceMetric.PrenormalizedAnguar: 'prenormalized-angular',
        DistanceMetric.Geodegrees: 'geodegrees',
        DistanceMetric.Hamming: 'hamming'
    }

    _FIELD_ID = 'marqo__id'
    _FIELD_SCORE_MODIFIERS = 'marqo__score_modifiers'
    _FIELD_INDEX_PREFIX = 'marqo__lexical_'
    _FIELD_FILTER_PREFIX = 'marqo__filter_'
    _FIELD_CHUNKS_PREFIX = 'marqo__chunks_'
    _FIELD_EMBEDDING_PREFIX = 'marqo__embeddings_'

    _RANK_PROFILE_BM25 = 'bm25'
    _RANK_PROFILE_EMBEDDING_SIMILARITY = 'embedding_similarity'
    _RANK_PROFILE_MODIFIERS = 'modifiers'
    _RANK_PROFILE_BM25_MODIFIERS = 'bm25_modifiers'
    _RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS = 'embedding_similarity_modifiers'

    # Note field names are also used as query inputs, so make sure these reserved names have a marqo__ prefix
    _QUERY_INPUT_EMBEDDING = 'marqo__query_embedding'
    _QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS = 'marqo__mult_weights'
    _QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS = 'marqo__add_weights'

    _SUMMARY_ALL_NON_VECTOR = 'all-non-vector-summary'
    _SUMMARY_ALL_VECTOR = 'all-vector-summary'

    _VESPA_DOC_ID = 'id'
    _VESPA_DOC_FIELDS = 'fields'
    _VESPA_DOC_RELEVANCE = 'relevance'
    _VESPA_DOC_MATCH_FEATURES = 'matchfeatures'
    _VESPA_DOC_FIELDS_TO_IGNORE = {'sddocname'}

    @classmethod
    def generate_schema(cls, marqo_index: MarqoIndex) -> str:
        cls._validate_index_type(marqo_index)

        schema = list()

        schema.append(f'schema {marqo_index.name} {{')
        schema.extend(cls._generate_document_section(marqo_index))
        schema.extend(cls._generate_rank_profiles(marqo_index))
        schema.extend(cls._generate_default_fieldset(marqo_index))
        schema.extend(cls._generate_summaries(marqo_index))
        schema.append('}')

        return '\n'.join(schema)

    @classmethod
    def to_vespa_document(cls, marqo_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        cls._validate_index_type(marqo_index)

        # Ensure index object is caching otherwise this implementation will be computationally expensive
        marqo_index = cls._ensure_cache_enabled(marqo_index)

        vespa_id: Optional[int] = None
        vespa_fields: Dict[str, Any] = dict()

        # ID
        if constants.MARQO_DOC_ID in marqo_document:
            vespa_id = marqo_document[constants.MARQO_DOC_ID]
            vespa_fields[cls._FIELD_ID] = vespa_id

        # Fields
        for marqo_field in marqo_document:
            if marqo_field == constants.MARQO_DOC_TENSORS or marqo_field == constants.MARQO_DOC_ID:
                continue  # process tensor fields later

            marqo_value = marqo_document[marqo_field]
            cls._verify_marqo_field_name(marqo_field, marqo_index)
            cls._verify_marqo_field_type(marqo_field, marqo_value, marqo_index)

            index_field = marqo_index.field_map[marqo_field]

            if index_field.lexical_field_name:
                vespa_fields[index_field.lexical_field_name] = marqo_value
            if index_field.filter_field_name:
                vespa_fields[index_field.filter_field_name] = marqo_value
            if not index_field.lexical_field_name and not index_field.filter_field_name:
                vespa_fields[index_field.name] = marqo_value

        # Tensors
        if constants.MARQO_DOC_TENSORS in marqo_document:
            for marqo_tensor_field in marqo_document[constants.MARQO_DOC_TENSORS]:
                marqo_tensor_value = marqo_document[constants.MARQO_DOC_TENSORS][marqo_tensor_field]

                cls._verify_marqo_tensor_field_name(marqo_tensor_field, marqo_index)
                cls._verify_marqo_tensor_field(marqo_tensor_field, marqo_tensor_value)

                chunks = marqo_tensor_value[constants.MARQO_DOC_CHUNKS]
                embeddings = marqo_tensor_value[constants.MARQO_DOC_EMBEDDINGS]

                index_tensor_field = marqo_index.tensor_field_map[marqo_tensor_field]

                vespa_fields[index_tensor_field.chunk_field_name] = chunks
                vespa_fields[index_tensor_field.embeddings_field_name] = \
                    {f'{i}': embeddings[i] for i in range(len(embeddings))}

        vespa_doc = {
            cls._VESPA_DOC_FIELDS: vespa_fields
        }

        if vespa_id is not None:
            vespa_doc[cls._VESPA_DOC_ID] = vespa_id

        return vespa_doc

    @classmethod
    def to_marqo_document(
            cls, vespa_document: Dict[str, Any], marqo_index: MarqoIndex, return_highlights: bool = False
    ) -> Dict[str, Any]:

        if cls._VESPA_DOC_FIELDS not in vespa_document:
            raise VespaDocumentParsingError(f'Vespa document is missing {cls._VESPA_DOC_FIELDS} field')

        fields = vespa_document[cls._VESPA_DOC_FIELDS]
        marqo_document = dict()
        for field, value in fields.items():
            if field in marqo_index.all_field_map:
                marqo_name = marqo_index.all_field_map[field].name
                if marqo_name in marqo_document:
                    # If getting all fields from Vespa, there may be a lexical and a filter field for one Marqo field
                    # They must have the same value
                    if marqo_document[marqo_name] != value:
                        raise VespaDocumentParsingError(
                            f'Vespa document has different values for Marqo field {marqo_name}: '
                            f'{marqo_document[marqo_name]} and {value}'
                        )
                else:
                    marqo_document[marqo_name] = value
            elif field in marqo_index.tensor_subfield_map:
                tensor_field = marqo_index.tensor_subfield_map[field]

                if constants.MARQO_DOC_TENSORS not in marqo_document:
                    marqo_document[constants.MARQO_DOC_TENSORS] = dict()
                if tensor_field.name not in marqo_document[constants.MARQO_DOC_TENSORS]:
                    marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name] = dict()

                if field == tensor_field.chunk_field_name:
                    marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name][constants.MARQO_DOC_CHUNKS] = value
                elif field == tensor_field.embeddings_field_name:
                    try:
                        marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name][
                            constants.MARQO_DOC_EMBEDDINGS] = list(value['blocks'].values())
                    except (KeyError, AttributeError, TypeError) as e:
                        raise VespaDocumentParsingError(
                            f'Cannot parse embeddings field {field} with value {value}'
                        ) from e

                else:
                    raise VespaDocumentParsingError(f'Unexpected tensor subfield {field}')
            elif field == cls._FIELD_ID:
                marqo_document[constants.MARQO_DOC_ID] = value
            elif field == cls._VESPA_DOC_MATCH_FEATURES:
                continue
            elif field in cls._VESPA_DOC_FIELDS_TO_IGNORE | {cls._FIELD_SCORE_MODIFIERS, cls._VESPA_DOC_MATCH_FEATURES}:
                continue
            else:
                raise VespaDocumentParsingError(f'Unknown field {field} for index {marqo_index.name} in Vespa document')

        # Highlights
        if return_highlights and cls._VESPA_DOC_MATCH_FEATURES in fields:
            marqo_document[constants.MARQO_DOC_HIGHLIGHTS] = cls._extract_highlights(
                fields, marqo_index
            )

        return marqo_document

    @classmethod
    def to_vespa_query(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        # Verify attributes to retrieve, if defined
        if marqo_query.attributes_to_retrieve is not None:
            for att in marqo_query.attributes_to_retrieve:
                if att not in marqo_index.field_map:
                    raise InvalidFieldNameError(
                        f'Index {marqo_index.name} has no field {att}. '
                        f'Available fields are: {", ".join(marqo_index.field_map.keys())}'
                    )

        # Verify score modifiers, if defined
        if marqo_query.score_modifiers is not None:
            for modifier in marqo_query.score_modifiers:
                if modifier.field not in marqo_index.score_modifier_fields_names:
                    raise InvalidFieldNameError(
                        f'Index {marqo_index.name} has no score modifier field {modifier.field}. '
                        f'Available score modifier fields are: {", ".join(marqo_index.score_modifier_fields_names)}'
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
    def _to_vespa_tensor_query(cls, marqo_query: MarqoTensorQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        if marqo_query.searchable_attributes is not None:
            for att in marqo_query.searchable_attributes:
                if att not in marqo_index.tensor_field_map:
                    raise InvalidFieldNameError(
                        f'Index {marqo_index.name} has no tensor field {att}. '
                        f'Available tensor fields are: {", ".join(marqo_index.tensor_field_map.keys())}'
                    )

            fields_to_search = marqo_query.searchable_attributes
        else:
            fields_to_search = marqo_index.tensor_field_map.keys()

        tensor_term = cls._get_tensor_search_term(marqo_query, marqo_index)
        filter_term = cls._get_filter_term(marqo_query, marqo_index)
        if filter_term:
            filter_term = f' AND {filter_term}'
        else:
            filter_term = ''
        select_attributes = cls._get_select_attributes(marqo_query)
        summary = cls._SUMMARY_ALL_VECTOR if marqo_query.expose_facets else cls._SUMMARY_ALL_NON_VECTOR
        score_modifiers = cls._get_score_modifiers(marqo_query, marqo_index)
        ranking = cls._RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS if score_modifiers \
            else cls._RANK_PROFILE_EMBEDDING_SIMILARITY

        query_inputs = {
            cls._QUERY_INPUT_EMBEDDING: marqo_query.vector_query
        }
        query_inputs.update({
            f: 1 for f in fields_to_search
        })
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
    def _to_vespa_lexical_query(cls, marqo_query: MarqoLexicalQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        if marqo_query.searchable_attributes is not None:
            for att in marqo_query.searchable_attributes:
                if att not in marqo_index.lexically_searchable_fields_names:
                    raise InvalidFieldNameError(
                        f'Index {marqo_index.name} has no lexically searchable field {att}. '
                        f'Available lexically searchable fields are: '
                        f'{", ".join(marqo_index.lexically_searchable_fields_names)}'
                    )
            fields_to_search = marqo_query.searchable_attributes
        else:
            fields_to_search = marqo_index.lexical_fields_names

        lexical_term = cls._get_lexical_search_term(marqo_query, marqo_index)
        filter_term = cls._get_filter_term(marqo_query, marqo_index)
        if filter_term:
            filter_term = f' AND {filter_term}'
        else:
            filter_term = ''

        select_attributes = cls._get_select_attributes(marqo_query)
        summary = cls._SUMMARY_ALL_VECTOR if marqo_query.expose_facets else cls._SUMMARY_ALL_NON_VECTOR
        score_modifiers = cls._get_score_modifiers(marqo_query, marqo_index)
        ranking = cls._RANK_PROFILE_BM25_MODIFIERS if score_modifiers \
            else cls._RANK_PROFILE_BM25

        query_inputs = {}
        query_inputs.update({
            f: 1 for f in fields_to_search
        })
        if score_modifiers:
            query_inputs.update(score_modifiers)

        query = {
            'yql': f'select {select_attributes} from {marqo_query.index_name} where {lexical_term}{filter_term}',
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
    def _to_vespa_hybrid_query(cls, marqo_query: MarqoHybridQuery, marqo_index: MarqoIndex) -> Dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def _get_tensor_search_term(cls, marqo_query: MarqoTensorQuery, marqo_index: MarqoIndex) -> str:
        if marqo_query.searchable_attributes is not None:
            fields_to_search = [f for f in marqo_query.searchable_attributes if f in marqo_index.tensor_field_map]
        else:
            fields_to_search = marqo_index.tensor_field_map.keys()

        if marqo_query.ef_search is not None:
            additional_hits = f' hnsw.exploreAdditionalHits:{marqo_query.ef_search - marqo_query.limit}'
        else:
            additional_hits = ''

        terms = []
        for field in fields_to_search:
            tensor_field = marqo_index.tensor_field_map[field]
            embedding_field_name = tensor_field.embeddings_field_name
            terms.append(
                f'({{targetHits:{marqo_query.limit}, approximate:{str(marqo_query.approximate)}{additional_hits}}}'
                f'nearestNeighbor({embedding_field_name}, {cls._QUERY_INPUT_EMBEDDING}))'
            )

        if terms:
            return f'({" OR ".join(terms)})'
        else:
            return ''

    @classmethod
    def _get_filter_term(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> Optional[str]:
        def escape(s: str) -> str:
            return s.replace('\\', '\\\\').replace('"', '\\"')

        def node_to_string(node: search_filter.Node) -> str:
            if isinstance(node, search_filter.Operator):
                return f' {node.raw} '
            elif isinstance(node, search_filter.Not):
                return f'!({node_to_string(node.term)})'
            elif isinstance(node, search_filter.Term):
                if node.field not in marqo_index.filterable_fields_names:
                    raise InvalidFieldNameError(
                        f'Index {marqo_index.name} has no filterable field {node.field}. '
                        f'Available filterable fields are: {", ".join(marqo_index.filterable_fields_names)}'
                    )
                if isinstance(node, search_filter.EqualityTerm):
                    return f'{node.field} contains "{escape(node.value)}"'
                elif isinstance(node, search_filter.RangeTerm):
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

        filter_term = []
        if marqo_query.filter is not None:
            # Traverse filter tree in-order to generate Vespa filter string
            # Node types are: Term, Modifier, Operator where Terms and Modifiers are leaves, but Operators are not
            stack: List[Union[str, search_filter.Node]] = []
            current = marqo_query.filter.root
            while current or stack:
                # Loop invariant: for current, stack: s1, s2, ..., sn, order of processing is
                # ..., current, ..., s1, ..., s2, ..., sn, ... where ... is zero or more elements not yet known
                while current:
                    stack.append(current)
                    if isinstance(current, search_filter.Operator):
                        filter_term.append('(')
                        current = current.left
                    else:
                        current = None

                # Now order of processing is s1, ..., s2, ..., sn, so we will process s1
                top = stack.pop()
                if top == ')':
                    filter_term.append(')')
                else:
                    filter_term.append(node_to_string(top))

                if isinstance(top, search_filter.Operator):
                    current = top.right
                    stack.append(')')

            return ''.join(filter_term)

        return None

    @classmethod
    def _get_select_attributes(cls, marqo_query: MarqoQuery) -> str:
        if marqo_query.attributes_to_retrieve is not None:
            return ', '.join(marqo_query.attributes_to_retrieve)
        else:
            return '*'

    @classmethod
    def _get_score_modifiers(cls, marqo_query: MarqoQuery, marqo_index: MarqoIndex) -> \
            Optional[Dict[str, Dict[str, float]]]:
        if marqo_query.score_modifiers:
            mult_tensor = {}
            add_tensor = {}
            for modifier in marqo_query.score_modifiers:
                if modifier.type == ScoreModifierType.Multiply:
                    mult_tensor[modifier.field] = modifier.weight
                elif modifier.type == ScoreModifierType.Add:
                    add_tensor[modifier.field] = modifier.weight

            # Note one of these could be empty, but not both
            return {
                cls._QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS: mult_tensor,
                cls._QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS: add_tensor
            }

        return None

    @classmethod
    def _get_lexical_search_term(cls, marqo_query: MarqoLexicalQuery, marqo_index: MarqoIndex) -> str:
        if marqo_query.or_phrases:
            or_terms = 'weakAnd(%s)' % ', '.join([
                cls._get_lexical_contains_term(phrase, marqo_query, marqo_index) for phrase in marqo_query.or_phrases
            ])
        else:
            or_terms = ''
        if marqo_query.and_phrases:
            and_terms = ' AND '.join([
                cls._get_lexical_contains_term(phrase, marqo_query, marqo_index) for phrase in marqo_query.and_phrases
            ])
            if or_terms:
                and_terms = f' AND ({and_terms})'
        else:
            and_terms = ''

        return f'{or_terms}{and_terms}'

    @classmethod
    def _get_lexical_contains_term(cls, phrase, query: MarqoQuery, marqo_index: MarqoIndex) -> str:
        if query.searchable_attributes is not None:
            return ' OR '.join([
                f'{marqo_index.field_map[field].lexical_field_name} contains "{phrase}"'
                for field in query.searchable_attributes
            ])
        else:
            return f'default contains "{phrase}"'

    @classmethod
    def _verify_marqo_field_name(cls, field_name: str, marqo_index: MarqoIndex):
        field_map = marqo_index.field_map
        if field_name not in marqo_index.field_map:
            raise InvalidFieldNameError(f'Invalid field name {field_name} for index {marqo_index.name}. '
                                        f'Valid field names are {", ".join(field_map.keys())}')

    @classmethod
    def _verify_marqo_tensor_field_name(cls, field_name: str, marqo_index: MarqoIndex):
        tensor_field_map = marqo_index.tensor_field_map
        if field_name not in marqo_index.field_map:
            raise InvalidFieldNameError(f'Invalid tensor field name {field_name} for index {marqo_index.name}. '
                                        f'Valid tensor field names are {", ".join(tensor_field_map.keys())}')

    @classmethod
    def _verify_marqo_tensor_field(cls, field_name: str, field_value: Dict[str, Any]):
        if not set(field_value.keys()) == {constants.MARQO_DOC_CHUNKS, constants.MARQO_DOC_EMBEDDINGS}:
            raise InternalError(f'Invalid tensor field {field_name}. '
                                f'Expected keys {constants.MARQO_DOC_CHUNKS}, {constants.MARQO_DOC_EMBEDDINGS} '
                                f'but found {", ".join(field_value.keys())}')

    @classmethod
    def _verify_marqo_field_type(cls, field_name: str, value: Any, marqo_index: MarqoIndex):
        marqo_type = marqo_index.field_map[field_name].type
        python_type = cls._get_python_type(marqo_type)
        if (
                isinstance(python_type, list) and not any(isinstance(value, t) for t in python_type) or
                not isinstance(python_type, list) and not isinstance(value, python_type)
        ):
            raise InvalidDataTypeError(f'Invalid value {value} for field {field_name} with Marqo type '
                                       f'{marqo_type.name}. Expected a value of type {python_type}, but found '
                                       f'{type(value)}')

    @classmethod
    def _generate_document_section(cls, marqo_index: MarqoIndex) -> List[str]:
        """
        Generate the document (fields) section of the Vespa schema. Update `marqo_index` with Vespa-level field names.
        """
        document: List[str] = list()

        document.append(f'document {marqo_index.name} {{')

        # ID field
        document.append(f'field {cls._FIELD_ID} type string {{ indexing: summary }}')

        for field in marqo_index.fields:
            field_type = cls._get_vespa_type(field.type)

            if FieldFeature.LexicalSearch in field.features:
                field_name = f'{cls._FIELD_INDEX_PREFIX}{field.name}'
                document.append(f'field {field_name} type {field_type} {{')
                document.append(f'indexing: index | summary')
                document.append('index: enable-bm25')
                document.append('}')

                field.lexical_field_name = field_name

            if FieldFeature.Filter in field.features:
                field_name = f'{cls._FIELD_FILTER_PREFIX}{field.name}'
                document.append(f'field {field_name} type {field_type} {{')
                document.append('indexing: attribute | summary')
                document.append('attribute: fast-search')
                document.append('rank: filter')
                document.append('}')

                field.filter_field_name = field_name

            if FieldFeature.LexicalSearch not in field.features and FieldFeature.Filter not in field.features:
                field_name = field.name
                document.append(f'field {field_name} type {field_type} {{')
                document.append('indexing: summary')
                document.append('}')

        # score modifiers
        if marqo_index.score_modifier_fields_names:
            document.append(f'field {cls._FIELD_SCORE_MODIFIERS} type tensor<float>(p{{}}) {{ indexing: attribute }}')

        # tensor fields
        model_dim = marqo_index.model.get_dimension()
        for field in marqo_index.tensor_fields:
            chunks_field_name = f'{cls._FIELD_CHUNKS_PREFIX}{field.name}'
            embedding_field_name = f'{cls._FIELD_EMBEDDING_PREFIX}{field.name}'
            document.append(f'field {chunks_field_name} type array<string> {{')
            document.append('indexing: attribute | summary')
            document.append('}')
            document.append(f'field {embedding_field_name} type tensor<float>(p{{}}, x[{model_dim}]) {{')
            document.append('indexing: attribute | index | summary')
            document.append(f'attribute {{ distance-metric: {cls._get_distance_metric(marqo_index.distance_metric)} }}')
            document.append('index { hnsw {')
            document.append(f'max-links-per-node: {marqo_index.hnsw_config.m}')
            document.append(f'neighbors-to-explore-at-insert: {marqo_index.hnsw_config.ef_construction}')
            document.append('}}')
            document.append('}')

            field.chunk_field_name = chunks_field_name
            field.embeddings_field_name = embedding_field_name

        document.append('}')

        return document

    @classmethod
    def _generate_summaries(cls, marqo_index: MarqoIndex) -> List[str]:
        summaries: List[str] = list()

        non_vector_summary_fields = []
        vector_summary_fields = []

        non_vector_summary_fields.append(
            f'summary {cls._FIELD_ID} type string {{ }}'
        )

        for field in marqo_index.fields:
            target_field_name = field.name
            field_type = cls._get_vespa_type(field.type)

            if field.type == FieldType.MultimodalCombination:
                # return combination weights only for vector summary
                vector_summary_fields.append(
                    f'summary {target_field_name} type {field_type} {{ }}'
                )
                continue

            if field.filter_field_name:
                # Filter fields are in-memory attributes so use this even if there's a lexical field
                source_field_name = field.filter_field_name
            elif field.lexical_field_name:
                source_field_name = field.lexical_field_name
            else:
                source_field_name = field.name

            non_vector_summary_fields.append(
                f'summary {target_field_name} type {field_type} {{ source: {source_field_name} }}'
            )

        for field in marqo_index.tensor_fields:
            non_vector_summary_fields.append(
                f'summary {field.chunk_field_name} type array<string> {{ }}'
            )
            vector_summary_fields.append(
                f'summary {field.embeddings_field_name} type tensor<float>(p{{}}, '
                f'x[{marqo_index.model.get_dimension()}]) {{ }}'
            )

        summaries.append(f'document-summary {cls._SUMMARY_ALL_NON_VECTOR} {{')
        summaries.extend(non_vector_summary_fields)
        summaries.append('}')
        summaries.append(f'document-summary {cls._SUMMARY_ALL_VECTOR} {{')
        summaries.extend(non_vector_summary_fields)
        summaries.extend(vector_summary_fields)
        summaries.append('}')

        return summaries

    @classmethod
    def _generate_default_fieldset(cls, marqo_index: MarqoIndex) -> List[str]:
        fieldsets: List[str] = list()

        fieldset_fields = marqo_index.lexical_fields_names

        if fieldset_fields:
            fieldsets.append('fieldset default {')
            if fieldset_fields:
                fieldsets.append(f'fields: {", ".join(fieldset_fields)}')
            fieldsets.append('}')

        return fieldsets

    @classmethod
    def _generate_rank_profiles(cls, marqo_index: MarqoIndex) -> List[str]:
        rank_profiles: List[str] = list()

        lexical_fields = marqo_index.lexical_fields_names
        score_modifier_fields = marqo_index.score_modifier_fields_names
        tensor_fields = [field.name for field in marqo_index.tensor_fields]
        model_dim = marqo_index.model.get_dimension()

        bm25_expression = ' + '.join([f'bm25({field})' for field in lexical_fields])
        embedding_similarity_expression = ' + '.join([
            f'if (query({field.name}) > 0, closeness(field, {field.embeddings_field_name}), 0)' for field in
            marqo_index.tensor_fields
        ])
        embedding_match_features_expression = \
            'match-features: ' + \
            ' '.join([f'closest({field.embeddings_field_name})' for field in marqo_index.tensor_fields]) + \
            ' ' + \
            ' '.join([f'distance(field, {field.embeddings_field_name})' for field in marqo_index.tensor_fields])

        if lexical_fields:
            rank_profiles.append(f'rank-profile {cls._RANK_PROFILE_BM25} inherits default {{ first-phase {{')
            rank_profiles.append(f'expression: {bm25_expression}')
            rank_profiles.append('}}')

        if tensor_fields:
            rank_profiles.append(f'rank-profile {cls._RANK_PROFILE_EMBEDDING_SIMILARITY} inherits default {{')

            rank_profiles.append('inputs {')
            rank_profiles.append(f'query({cls._QUERY_INPUT_EMBEDDING}) tensor<float>(x[{model_dim}])')
            for field in tensor_fields:
                rank_profiles.append(f'query({field}): 0')
            rank_profiles.append('}')

            rank_profiles.append('first-phase {')
            rank_profiles.append(f'expression: {embedding_similarity_expression}')
            rank_profiles.append('}')
            rank_profiles.append(embedding_match_features_expression)
            rank_profiles.append('}')

        if score_modifier_fields:
            expression = f'if (count(query({cls._QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS})) == 0, 1, ' \
                         f'reduce(query({cls._QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS}) ' \
                         f'* attribute({cls._FIELD_SCORE_MODIFIERS}), prod)) * score ' \
                         f'+ reduce(query({cls._QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS}) ' \
                         f'* attribute({cls._FIELD_SCORE_MODIFIERS}), sum)'
            rank_profiles.append(f'rank-profile {cls._RANK_PROFILE_MODIFIERS} inherits default {{')
            rank_profiles.append('inputs {')
            rank_profiles.append(f'query({cls._QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS})  tensor<float>(p{{}})')
            rank_profiles.append(f'query({cls._QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS})  tensor<float>(p{{}})')
            rank_profiles.append('}')
            rank_profiles.append('function modify(score) {')
            rank_profiles.append(f'expression: {expression}')
            rank_profiles.append('}}')

            if lexical_fields:
                rank_profiles.append(f'rank-profile {cls._RANK_PROFILE_BM25_MODIFIERS} '
                                     f'inherits {cls._RANK_PROFILE_MODIFIERS} {{ first-phase {{')
                rank_profiles.append(f'expression: modify({bm25_expression})')
                rank_profiles.append('}}')

            if tensor_fields:
                rank_profiles.append(
                    f'rank-profile {cls._RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS} '
                    f'inherits {cls._RANK_PROFILE_MODIFIERS} {{')

                rank_profiles.append('inputs {')
                rank_profiles.append(f'query({cls._QUERY_INPUT_EMBEDDING}) tensor<float>(x[{model_dim}])')
                for field in tensor_fields:
                    rank_profiles.append(f'query({field}): 0')
                rank_profiles.append('}')

                rank_profiles.append('first-phase {')
                rank_profiles.append(f'expression: modify({embedding_similarity_expression})')
                rank_profiles.append('}')
                rank_profiles.append(embedding_match_features_expression)
                rank_profiles.append('}')

        return rank_profiles

    @classmethod
    def _extract_highlights(cls, vespa_document_fields: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        # For each tensor field we will have closest(tensor_field) and distance(tensor_field) in match features
        # If a tensor field hasn't been searched, closest(tensor_field)[cells] will be empty and distance(tensor_field)
        # will be max double
        match_features = vespa_document_fields[cls._VESPA_DOC_MATCH_FEATURES]

        min_distance = None
        closest_tensor_field = None
        for tensor_field in marqo_index.tensor_fields:
            closest_feature = f'closest({tensor_field.embeddings_field_name})'
            if closest_feature in match_features and len(match_features[closest_feature]['cells']) > 0:
                distance_feature = f'distance(field,{tensor_field.embeddings_field_name})'
                if distance_feature not in match_features:
                    raise VespaDocumentParsingError(
                        f'Expected {distance_feature} in match features but it was not found'
                    )
                distance = match_features[distance_feature]
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    closest_tensor_field = tensor_field

        if closest_tensor_field is not None:
            # Get chunk index
            chunk_index_str = next(iter(
                match_features[f'closest({closest_tensor_field.embeddings_field_name})']['cells']
            ))
            try:
                chunk_index = int(chunk_index_str)
            except ValueError as e:
                raise VespaDocumentParsingError(
                    f'Expected integer as chunk index, but found {chunk_index_str}', cause=e
                ) from e

            # Get chunk value
            try:
                chunk = vespa_document_fields[closest_tensor_field.chunk_field_name][chunk_index]
            except (KeyError, TypeError) as e:
                raise VespaDocumentParsingError(
                    f'Cannot extract chunk value from {closest_tensor_field.chunk_field_name}: {str(e)}',
                    cause=e
                ) from e

            return {
                closest_tensor_field.name: chunk
            }

        raise VespaDocumentParsingError('Failed to extract highlights from Vespa document')

    @classmethod
    def _get_vespa_type(cls, marqo_type: FieldType) -> str:
        try:
            return cls._MARQO_TO_VESPA_TYPE_MAP[marqo_type]
        except KeyError:
            raise InternalError(f'Unknown Marqo type: {marqo_type}')

    @classmethod
    def _get_python_type(cls, marqo_type: FieldType) -> type:
        try:
            return cls._MARQO_TO_PYTHON_TYPE_MAP[marqo_type]
        except KeyError:
            raise InternalError(f'Unknown Marqo type: {marqo_type}')

    @classmethod
    def _get_distance_metric(cls, marqo_distance_metric: DistanceMetric) -> str:
        try:
            return cls._DISTANCE_METRIC_MAP[marqo_distance_metric]
        except KeyError:
            raise ValueError(f'Unknown Marqo distance metric: {marqo_distance_metric}')

    @classmethod
    def _validate_index_type(cls, marqo_index: MarqoIndex) -> None:
        if marqo_index.type != IndexType.Structured:
            raise ValueError(f'Vespa index type must be {IndexType.Structured.name}. '
                             f'This module cannot handle index type {marqo_index.type.name}')

    @classmethod
    def _ensure_cache_enabled(cls, marqo_index):
        if not marqo_index.model_enable_cache:
            return marqo_index.copy_with_caching()

        return marqo_index

if __name__ == '__main__':
    marqo_query = MarqoTensorQuery(
        index_name='test',
        limit=10,
        vector_query=[1, 2, 3],
        filter='tags:shirt OR tags:jeans AND NOT tags:blue AND price:[* TO 20]',
    )
    marqo_index = MarqoTestCase.marqo_index(
        name='my_index',
        model=Model(name='ViT-B/32'),
        distance_metric=DistanceMetric.PrenormalizedAnguar,
        type=IndexType.Structured,
        vector_numeric_type=VectorNumericType.Float,
        hnsw_config=HnswConfig(ef_construction=100, m=16),
        fields=[
            Field(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
            Field(name='description', type=FieldType.Text),
            Field(name='tags', type=FieldType.ArrayText, features=[FieldFeature.Filter]),
            Field(name='price', type=FieldType.Float, features=[FieldFeature.Filter, FieldFeature.ScoreModifier]),
        ],
        tensor_fields=[
            TensorField(name='title'),
        ],
    )
    vespa_query = StructuredVespaIndex.to_vespa_query(marqo_query, marqo_index)
    print(vespa_query)
    pass
