from typing import Dict, Any, List

from marqo.core.models import MarqoQuery, MarqoIndex
from marqo.core.models.marqo_index import FieldType, FieldFeature, DistanceMetric
from marqo.core.vespa_index import VespaIndex


class TypesVespaIndex(VespaIndex):
    _type_map = {
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

    _distance_metric_map = {
        DistanceMetric.PrenormalizedAnguar: 'prenormalized-angular'
    }

    _index_field_prefix = '__marqo_lexical_'
    _filter_field_prefix = '__marqo_filter_'
    _chunks_field_prefix = '__marqo_chunks_'
    _embedding_field_prefix = '__marqo_embedding_'

    @classmethod
    def generate_schema(cls, marqo_index: MarqoIndex) -> str:
        pass

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
    def _generate_fields(cls, marqo_index: MarqoIndex, schema: List[str]) -> None:
        """
        Generate the fields section of the Vespa schema. Update `marqo_index` with Vespa-level field names.
        """
        for field in marqo_index.fields:
            if FieldFeature.LexicalSearch in field.features:
                field_name = f'{cls._index_field_prefix}{field.name}'
                schema.append(f'field {field_name} type {cls._get_vespa_type(field.type)} {{')
                schema.append(f'indexing: index | summary')
                schema.append('index: enable-bm25')
                schema.append('}')
                field.lexical_field_name = field_name

            if FieldFeature.Filter in field.features:
                field_name = f'{cls._index_field_prefix}{field.name}'
                schema.append(f'field {field_name} type {cls._get_vespa_type(field.type)} {{')
                schema.append('indexing: attribute | summary')
                schema.append('attribute: fast-search')
                schema.append('rank: filter')
                schema.append('}')

                field.filter_field_name = field_name

        model_dim = cls._get_model_dimension(marqo_index.model)
        for field in marqo_index.tensor_fields:
            chunks_field_name = f'{cls._chunks_field_prefix}{field.name}'
            embedding_field_name = f'{cls._embedding_field_prefix}{field.name}'
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

    @classmethod
    def _get_vespa_type(cls, marqo_type: FieldType) -> str:
        try:
            return cls._type_map[marqo_type]
        except KeyError:
            raise ValueError(f'Unknown Marqo type: {marqo_type}')

    @classmethod
    def _get_distance_metric(cls, marqo_distance_metric: DistanceMetric) -> str:
        try:
            return cls._distance_metric_map[marqo_distance_metric]
        except KeyError:
            raise ValueError(f'Unknown Marqo distance metric: {marqo_distance_metric}')

    @classmethod
    def _get_model_dimension(cls, model: str) -> int:
        pass


if __name__ == '__main__':
    print(TypesVespaIndex._getdistance_metric(DistanceMetric.PrenormalizedAnguar)
