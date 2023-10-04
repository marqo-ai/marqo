from typing import Dict, Any, List

from marqo.core.models import MarqoQuery, MarqoIndex
from marqo.core.models.marqo_index import FieldType, FieldFeature, DistanceMetric, VectorNumericType, HnswConfig, Field, \
    TensorField, IndexType
from marqo.core.vespa_index import VespaIndex


class TypedVespaIndex(VespaIndex):
    _TYPE_MAP = {
        FieldType.Text: 'string',
        FieldType.Bool: 'bool',
        FieldType.Int: 'int',
        FieldType.Float: 'float',
        FieldType.ArrayText: 'array<string>',
        FieldType.ArrayBool: 'array<bool>',
        FieldType.ArrayInt: 'array<int>',
        FieldType.ArrayFloat: 'array<float>',
        FieldType.ImagePointer: 'string'
    }

    _DISTANCE_METRIC_MAP = {
        DistanceMetric.PrenormalizedAnguar: 'prenormalized-angular'
    }

    _INDEX_FIELD_PREFIX = 'marqo__lexical_'
    _FILTER_FIELD_PREFIX = 'marqo__filter_'
    _CHUNKS_FIELD_PREFIX = 'marqo__chunks_'
    _EMBEDDING_FIELD_PREFIX = 'marqo__embeddings_'

    _SCORE_MODIFIERS_FIELD = 'marqo__score_modifiers'

    _BM25_RANK_PROFILE = 'bm25'
    _EMBEDDING_SIMILARITY_RANK_PROFILE = 'embedding_similarity'
    _MODIFIERS_RANK_PROFILE = 'modifiers'

    _RANK_INPUT_QUERY_EMBEDDING = 'query_embedding'
    _RANK_INPUT_MULT_WEIGHTS = 'mult_weights'
    _RANK_INPUT_ADD_WEIGHTS = 'add_weights'

    @classmethod
    def generate_schema(cls, marqo_index: MarqoIndex) -> str:
        cls._validate_index_type(marqo_index)

        schema = []

        schema.append(f'schema {marqo_index.name} {{')
        cls._generate_document(marqo_index, schema)
        cls._generate_rank_profiles(marqo_index, schema)
        cls._generate_default_fieldset(marqo_index, schema)
        cls._generate_summaries(marqo_index, schema)
        schema.append('}')

        return '\n'.join(schema)

    @classmethod
    def to_vespa_document(cls, marqo_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        pass

    @classmethod
    def to_marqo_document(cls, vespa_document: Dict[str, Any], marqo_index: MarqoIndex) -> Dict[str, Any]:
        pass

    @classmethod
    def to_vespa_query(cls, query: MarqoQuery) -> Dict[str, Any]:
        pass

    @classmethod
    def _generate_document(cls, marqo_index: MarqoIndex, schema: List[str]) -> None:
        """
        Generate the document (fields) section of the Vespa schema. Update `marqo_index` with Vespa-level field names.
        """
        schema.append(f'document {marqo_index.name} {{')
        for field in marqo_index.fields:
            field_type = cls._get_vespa_type(field.type)

            if FieldFeature.LexicalSearch in field.features:
                field_name = f'{cls._INDEX_FIELD_PREFIX}{field.name}'
                schema.append(f'field {field_name} type {field_type} {{')
                schema.append(f'indexing: index | summary')
                schema.append('index: enable-bm25')
                schema.append('}')

                field.lexical_field_name = field_name

            if FieldFeature.Filter in field.features:
                field_name = f'{cls._FILTER_FIELD_PREFIX}{field.name}'
                schema.append(f'field {field_name} type {field_type} {{')
                schema.append('indexing: attribute | summary')
                schema.append('attribute: fast-search')
                schema.append('rank: filter')
                schema.append('}')

                field.filter_field_name = field_name

            if FieldFeature.LexicalSearch not in field.features and FieldFeature.Filter not in field.features:
                field_name = field.name
                schema.append(f'field {field_name} type {field_type} {{')
                schema.append('indexing: summary')
                schema.append('}')

        # score modifiers
        if marqo_index.score_modifier_fields:
            schema.append(f'field {cls._SCORE_MODIFIERS_FIELD} type tensor<float>(p{{}}) {{ indexing: attribute }}')

        # tensor fields
        model_dim = cls._get_model_dimension(marqo_index.model)
        for field in marqo_index.tensor_fields:
            chunks_field_name = f'{cls._CHUNKS_FIELD_PREFIX}{field.name}'
            embedding_field_name = f'{cls._EMBEDDING_FIELD_PREFIX}{field.name}'
            schema.append(f'field {chunks_field_name} type array<string> {{')
            schema.append('indexing: attribute | summary')
            schema.append('}')
            schema.append(f'field {embedding_field_name} type tensor<float>(p{{}}, x[{model_dim}]) {{')
            schema.append('indexing: attribute | index | summary')
            schema.append(f'attribute {{ distance-metric: {cls._get_distance_metric(marqo_index.distance_metric)} }}')
            schema.append('index { hnsw {')
            schema.append(f'max-links-per-node: {marqo_index.hnsw_config.m}')
            schema.append(f'neighbors-to-explore-at-insert: {marqo_index.hnsw_config.ef_construction}')
            schema.append('}}')
            schema.append('}')

            field.chunk_field_name = chunks_field_name
            field.embeddings_field_name = embedding_field_name

        schema.append('}')

    @classmethod
    def _generate_summaries(cls, marqo_index: MarqoIndex, schema: List[str]) -> None:
        non_vector_summary_fields = []
        vector_summary_fields = []
        for field in marqo_index.fields:
            target_field_name = field.name
            field_type = cls._get_vespa_type(field.type)
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
                f'x[{cls._get_model_dimension(marqo_index.model)}]) {{ }}'
            )

        schema.append('document-summary all-non-vector-summary {')
        schema.extend(non_vector_summary_fields)
        schema.append('}')
        schema.append('document-summary all-vector-summary {')
        schema.extend(non_vector_summary_fields)
        schema.extend(vector_summary_fields)
        schema.append('}')

    @classmethod
    def _generate_default_fieldset(cls, marqo_index: MarqoIndex, schema: List[str]) -> None:
        fieldset_fields = []
        for field in marqo_index.fields:
            if field.lexical_field_name:
                fieldset_fields.append(field.lexical_field_name)

        schema.append('fieldset default {')
        if fieldset_fields:
            schema.append(f'fields: {", ".join(fieldset_fields)}')
        schema.append('}')

    @classmethod
    def _generate_rank_profiles(cls, marqo_index: MarqoIndex, schema: List[str]) -> None:
        lexical_fields = marqo_index.lexical_fields
        score_modifier_fields = marqo_index.score_modifier_fields
        tensor_fields = [field.name for field in marqo_index.tensor_fields]
        model_dim = cls._get_model_dimension(marqo_index.model)

        bm25_expression = ' + '.join([f'bm25({field})' for field in lexical_fields])
        embedding_similarity_expression = ' + '.join([
            f'if (query({field.name}) > 0, closeness(field, {field.embeddings_field_name}), 0)' for field in
            marqo_index.tensor_fields
        ])

        if lexical_fields:
            schema.append(f'rank-profile {cls._BM25_RANK_PROFILE} inherits default {{ first-phase {{')
            schema.append(f'expression: {bm25_expression}')
            schema.append('}}')

        if tensor_fields:
            schema.append(f'rank-profile {cls._EMBEDDING_SIMILARITY_RANK_PROFILE} inherits default {{')
            schema.append('inputs {')
            schema.append(f'query({cls._RANK_INPUT_QUERY_EMBEDDING}) tensor<float>(x[{model_dim}])')
            for field in tensor_fields:
                schema.append(f'query({field}): 1')

            schema.append('}')
            schema.append('first-phase {')
            schema.append(f'expression: {embedding_similarity_expression}')
            schema.append('}}')

        if score_modifier_fields:
            expression = f'if (count(query({cls._RANK_INPUT_MULT_WEIGHTS})) == 0, 1, ' \
                         f'reduce(query({cls._RANK_INPUT_MULT_WEIGHTS}) ' \
                         f'* attribute({cls._SCORE_MODIFIERS_FIELD}), prod)) * score ' \
                         f'+ reduce(query({cls._RANK_INPUT_ADD_WEIGHTS}) ' \
                         f'* attribute({cls._SCORE_MODIFIERS_FIELD}), sum)'
            schema.append(f'rank-profile {cls._MODIFIERS_RANK_PROFILE} inherits default {{')
            schema.append('inputs {')
            schema.append(f'query({cls._RANK_INPUT_MULT_WEIGHTS})  tensor<float>(p{{}})')
            schema.append(f'query({cls._RANK_INPUT_ADD_WEIGHTS})  tensor<float>(p{{}})')
            schema.append('}')
            schema.append('function modify(score) {')
            schema.append(f'expression: {expression}')
            schema.append('}}')

            if lexical_fields:
                schema.append(f'rank-profile {cls._BM25_RANK_PROFILE}_{cls._MODIFIERS_RANK_PROFILE} '
                              f'inherits {cls._MODIFIERS_RANK_PROFILE} {{ first-phase {{')
                schema.append(f'expression: modify({bm25_expression})')
                schema.append('}}')

            if tensor_fields:
                schema.append(f'rank-profile {cls._EMBEDDING_SIMILARITY_RANK_PROFILE}_{cls._MODIFIERS_RANK_PROFILE} '
                              f'inherits {cls._MODIFIERS_RANK_PROFILE} {{ first-phase {{')
                schema.append(f'expression: modify({embedding_similarity_expression})')
                schema.append('}}')

    @classmethod
    def _get_vespa_type(cls, marqo_type: FieldType) -> str:
        try:
            return cls._TYPE_MAP[marqo_type]
        except KeyError:
            raise ValueError(f'Unknown Marqo type: {marqo_type}')

    @classmethod
    def _get_distance_metric(cls, marqo_distance_metric: DistanceMetric) -> str:
        try:
            return cls._DISTANCE_METRIC_MAP[marqo_distance_metric]
        except KeyError:
            raise ValueError(f'Unknown Marqo distance metric: {marqo_distance_metric}')

    @classmethod
    def _get_model_dimension(cls, model: str) -> int:
        return 512
    @classmethod
    def _validate_index_type(cls, marqo_index: MarqoIndex) -> None:
        if marqo_index.type != IndexType.Typed:
            raise ValueError(f'Vespa index type must be {IndexType.Typed.name}. '
                             f'This module cannot handle index type {marqo_index.type.name}')


if __name__ == "__main__":
    marqo_index = MarqoIndex(
        name='my_index', model='clip', distance_metric=DistanceMetric.PrenormalizedAnguar,
        type=IndexType.Typed,
        vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=100, m=16),
        fields=[
            Field(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
            Field(name='description', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
            Field(name='is_active', type=FieldType.Bool, features=[FieldFeature.Filter]),
            Field(name='price', type=FieldType.Float, features=[FieldFeature.Filter, FieldFeature.ScoreModifier]),
            Field(name='category', type=FieldType.Text, features=[FieldFeature.Filter, FieldFeature.LexicalSearch]),
        ],
        tensor_fields=[
            TensorField(name='title'),
            TensorField(name='description')
        ]
    )

    print(TypedVespaIndex.generate_schema(marqo_index))
