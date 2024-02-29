import unittest

from marqo.core.models.marqo_index import *
from marqo.core.structured_vespa_index import common
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core import exceptions as core_exceptions
from tests.marqo_test import MarqoTestCase


class TestStructuredVespaIndex(MarqoTestCase):
    def setUp(self) -> None:
        """
        Create a structured Marqo index and a StructuredVespaIndex object for testing.
        """
        self.marqo_index = self.structured_marqo_index(
            name='my_index',
            schema_name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
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

        self.vespa_index = StructuredVespaIndex(self.marqo_index)

    def test_to_vespa_document_standardMarqoDoc_successful(self):
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

        actual_vespa_doc = self.vespa_index.to_vespa_document(marqo_doc)
        expected_vespa_doc = {
            'id': 'my_id',
            'fields': {
                common.FIELD_ID: 'my_id',
                common.FIELD_SCORE_MODIFIERS: {'price': 100.0, 'rank': 1},
                common.FIELD_VECTOR_COUNT: 2,
                self.marqo_index.field_map['title'].lexical_field_name: 'my title',
                'description': 'my description',
                self.marqo_index.field_map['category'].lexical_field_name: 'my category',
                self.marqo_index.field_map['category'].filter_field_name: 'my category',
                self.marqo_index.field_map['tags'].filter_field_name: ['tag1', 'tag2'],
                'image': 'https://my-image.com',
                self.marqo_index.field_map['is_active'].filter_field_name: True,
                'price': 100.0,
                'rank': 1,
                self.marqo_index.field_map['click_per_day'].filter_field_name: [1, 2, 3],
                self.marqo_index.field_map['last_updated'].filter_field_name: [1.0, 2.0, 3.0],
                self.marqo_index.tensor_field_map['title'].chunk_field_name: ['my', 'title'],
                self.marqo_index.tensor_field_map['title'].embeddings_field_name: {'0': [1.0, 2.0, 3.0],
                                                                              '1': [4.0, 5.0, 6.0]}
            }
        }

        self.assertEqual(expected_vespa_doc, actual_vespa_doc)

    def test_to_vespa_document_invalidDataType_fails(self):
        """
        Test that an error is raised when a field has an invalid data type e.g., a string for a float field.
        """
        invalid_data_type_docs = [
            ("int when it should be text", {
                '_id': 'my_id',
                'title': 'my title',
                'description': 12345,   
            }),
            ("array of text when it should be text", {
                '_id': 'my_id',
                'title': 'my title',
                'description': ['my description', 'hello'],   
            }),
            ("float when it should be int", {
                '_id': 'my_id',
                'title': 'my title',
                'description': 123.45,   
            }),
            ("array of int when it should be int", {
                '_id': 'my_id',
                'title': 'my title',
                'description': [123, 456],   
            }),
            ("bool when it should be int", {
                '_id': 'my_id',
                'title': 'my title',
                'description': True,   
            }),
            ("array of bool when it should be int", {
                '_id': 'my_id',
                'title': 'my title',
                'description': [True, False],   
            }),
            ("float when it should be bool", {
                '_id': 'my_id',
                'title': 'my title',
                'description': 123.45,   
            }),
            ("array of float when it should be bool", {
                '_id': 'my_id',
                'title': 'my title',
                'description': [123.45, 678.90],   
            }),
            ("text when it should be array of text", {
                '_id': 'my_id',
                'title': 'my title',
                'tags': 'fail',   
            }),
            ("int when it should be array of int", {
                '_id': 'my_id',
                'title': 'my title',
                'description': 123,   
            }),
            ("bool when it should be array of bool", {
                '_id': 'my_id',
                'title': 'my title',
                'description': True,   
            }),
            ("float when it should be array of float", {
                '_id': 'my_id',
                'title': 'my title',
                'description': 123.45,   
            }),
            ("array of float when it should be array of int", {
                '_id': 'my_id',
                'title': 'my title',
                'description': [123.45, 678.90],   
            }),
            ("array of bool when it should be array of int", {
                '_id': 'my_id',
                'title': 'my title',
                'description': [True, False],   
            }),
        ]
            
        for case_desc, marqo_doc in invalid_data_type_docs:
            with self.subTest(case_desc):
                with self.assertRaises(core_exceptions.InvalidDataTypeError):
                    self.vespa_index.to_vespa_document(marqo_doc)
    
    def test_to_vespa_document_fieldNotInIndex_fails(self):
        """
        Test that an error is raised when a field is not in the index.
        """
        with self.assertRaises(core_exceptions.InvalidFieldNameError):
            self.vespa_index.to_vespa_document(
                {
                    '_id': 'my_id',
                    'title': 'my title',
                    'nonexistent field': 'fail'
                }
            )
        
