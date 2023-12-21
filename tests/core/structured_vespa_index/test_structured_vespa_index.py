from marqo.core.models.marqo_index import *
from marqo.core.structured_vespa_index import common
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from tests.marqo_test import MarqoTestCase


class TestStructuredVespaIndex(MarqoTestCase):

    def test_to_vespa_document_standardMarqoDoc_successful(self):
        marqo_index = self.structured_marqo_index(
            name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAnguar,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                Field(
                    name='title',
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch],
                    lexical_field_name='lexical_title'
                ),
                Field(name='description', type=FieldType.Text),
                Field(
                    name='category',
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                    lexical_field_name='lexical_category',
                    filter_field_name='filter_category'
                ),
                Field(
                    name='tags',
                    type=FieldType.ArrayText,
                    features=[FieldFeature.Filter],
                    filter_field_name='filter_tags'
                ),
                Field(name='image', type=FieldType.ImagePointer),
                Field(
                    name='is_active',
                    type=FieldType.Bool,
                    features=[FieldFeature.Filter],
                    filter_field_name='filter_is_active'
                ),
                Field(
                    name='price',
                    type=FieldType.Float,
                    features=[FieldFeature.ScoreModifier]
                ),
                Field(
                    name='rank',
                    type=FieldType.Int,
                    features=[FieldFeature.ScoreModifier]
                ),
                Field(
                    name='click_per_day',
                    type=FieldType.ArrayInt,
                    features=[FieldFeature.Filter],
                    filter_field_name='filter_click_per_day'
                ),
                Field(
                    name='last_updated',
                    type=FieldType.ArrayFloat,
                    features=[FieldFeature.Filter],
                    filter_field_name='filter_last_updated'
                )
            ],
            tensor_fields=[
                TensorField(
                    name='title',
                    embeddings_field_name='embeddings_title',
                    chunk_field_name='chunks_title'
                ),
            ]
        )

        marqo_doc = {
            '_id': 'my_id',
            'title': 'my title',
            'description': 'my description',
            'category': 'my category',
            'tags': ['tag1', 'tag2'],
            'image': 'https://my-image.com',
            'is_active': True,
            'price': 100.0,
            'rank': 1,
            'click_per_day': [1, 2, 3],
            'last_updated': [1.0, 2.0, 3.0],
            constants.MARQO_DOC_TENSORS: {
                'title': {
                    constants.MARQO_DOC_CHUNKS: ['my', 'title'],
                    constants.MARQO_DOC_EMBEDDINGS: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                }
            }
        }

        actual_vespa_doc = StructuredVespaIndex(marqo_index).to_vespa_document(marqo_doc)
        expected_vespa_doc = {
            'id': 'my_id',
            'fields': {
                common.FIELD_ID: 'my_id',
                common.FIELD_SCORE_MODIFIERS: {'price': 100.0, 'rank': 1},
                common.FIELD_VECTOR_COUNT: 2,
                marqo_index.field_map['title'].lexical_field_name: 'my title',
                'description': 'my description',
                marqo_index.field_map['category'].lexical_field_name: 'my category',
                marqo_index.field_map['category'].filter_field_name: 'my category',
                marqo_index.field_map['tags'].filter_field_name: ['tag1', 'tag2'],
                'image': 'https://my-image.com',
                marqo_index.field_map['is_active'].filter_field_name: True,
                'price': 100.0,
                'rank': 1,
                marqo_index.field_map['click_per_day'].filter_field_name: [1, 2, 3],
                marqo_index.field_map['last_updated'].filter_field_name: [1.0, 2.0, 3.0],
                marqo_index.tensor_field_map['title'].chunk_field_name: ['my', 'title'],
                marqo_index.tensor_field_map['title'].embeddings_field_name: {'0': [1.0, 2.0, 3.0],
                                                                              '1': [4.0, 5.0, 6.0]}
            }
        }

        self.assertEqual(expected_vespa_doc, actual_vespa_doc)

    def test_to_vespa_document_invalidDataType_fails(self):
        """
        Test that an error is raised when a field has an invalid data type e.g., a string for a float field.
        """
        pass

    def test_to_vespa_document_fieldNotInIndex_fails(self):
        """
        Test that an error is raised when a field is not in the index.
        """
        pass
