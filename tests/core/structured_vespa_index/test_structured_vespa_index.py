from marqo.core.models.marqo_index import *
from marqo.core.structured_vespa_index import StructuredVespaIndex
from tests.marqo_test import MarqoTestCase


class TestStructuredVespaIndex(MarqoTestCase):

    def test_to_vespa_document_standardMarqoDoc_successful(self):
        marqo_index = self.marqo_index(
            name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Structured,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                Field(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                Field(name='description', type=FieldType.Text),
                Field(name='category', type=FieldType.Text, features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                Field(name='tags', type=FieldType.ArrayText, features=[FieldFeature.Filter]),
                Field(name='image', type=FieldType.ImagePointer),
                Field(name='is_active', type=FieldType.Bool, features=[FieldFeature.Filter]),
                Field(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier]),
                Field(name='rank', type=FieldType.Int, features=[FieldFeature.ScoreModifier]),
                Field(name='click_per_day', type=FieldType.ArrayInt, features=[FieldFeature.Filter]),
                Field(name='last_updated', type=FieldType.ArrayFloat, features=[FieldFeature.Filter]),
            ],
            tensor_fields=[
                TensorField(name='title'),
            ]
        )

        # Generate schema to prepare index
        StructuredVespaIndex.generate_schema(marqo_index)

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

        actual_vespa_doc = StructuredVespaIndex.to_vespa_document(marqo_doc, marqo_index)
        expected_vespa_doc = {
            'id': 'my_id',
            'fields': {
                'id': 'my_id',
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
