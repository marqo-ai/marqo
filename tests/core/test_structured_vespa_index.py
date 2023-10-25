import os
import re

from marqo.core import constants
from marqo.core.models.marqo_index import *
from marqo.core.structured_vespa_index import StructuredVespaIndex
from tests.marqo_test import MarqoTestCase


class TestStructuredVespaIndex(MarqoTestCase):

    def test_generate_schema_standardIndex_successful(self):
        """
        Test an index that has all field types and configurations.
        """
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
                TensorField(name='description')
            ]
        )

        actual_schema = StructuredVespaIndex.generate_schema(marqo_index)
        expected_schema = self._read_schema_from_file('test_schemas/healthy_schema_1.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_noLexicalFields_successful(self):
        marqo_index = MarqoIndex(
            name='my_index', model=Model(name='ViT-B/32'), distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Structured,
            vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                Field(name='title', type=FieldType.Text),
                Field(name='description', type=FieldType.Text),
                Field(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier])
            ],
            tensor_fields=[
                TensorField(name='title'),
                TensorField(name='description')
            ]
        )

        actual_schema = StructuredVespaIndex.generate_schema(marqo_index)
        expected_schema = self._read_schema_from_file('test_schemas/no_lexical_fields.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_noScoreModifierFields_successful(self):
        marqo_index = MarqoIndex(
            name='my_index', model=Model(name='ViT-B/32'), distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Structured,
            vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                Field(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                Field(name='description', type=FieldType.Text),
                Field(name='price', type=FieldType.Float)
            ],
            tensor_fields=[
                TensorField(name='title'),
                TensorField(name='description')
            ]
        )

        actual_schema = StructuredVespaIndex.generate_schema(marqo_index)
        expected_schema = self._read_schema_from_file('test_schemas/no_score_modifiers.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_noTensorFields_successful(self):
        marqo_index = MarqoIndex(
            name='my_index', model=Model(name='ViT-B/32'), distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Structured,
            vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                Field(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                Field(name='description', type=FieldType.Text),
                Field(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier])
            ],
            tensor_fields=[]
        )

        actual_schema = StructuredVespaIndex.generate_schema(marqo_index)
        expected_schema = self._read_schema_from_file('test_schemas/no_tensor_fields.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_dynamicIndex_fails(self):
        marqo_index = MarqoIndex(
            name='my_index', model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Unstructured,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                Field(name='title', type=FieldType.Text)
            ],
            tensor_fields=[
                TensorField(name='title'),
                TensorField(name='description')
            ]
        )

        with self.assertRaises(ValueError):
            StructuredVespaIndex.generate_schema(marqo_index)

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

    def _read_schema_from_file(self, path: str) -> str:
        currentdir = os.path.dirname(os.path.abspath(__file__))
        abspath = os.path.join(currentdir, path)

        with open(abspath, 'r') as f:
            schema = f.read()

        return schema

    def _remove_whitespace_in_schema(self, schema: str) -> str:
        """
        This function removes as much whitespace as possible from a schema without affecting its semantics.
        It is intended to help compare schemas independent of non-consequential syntactical differences such as
        new lines and indentation. Note, however, that not every new line can be removed without breaking the schema.
        """
        chars = re.escape('{}=+-<>():,;[]|')

        # Replace whitespace (including newlines) before or after any of the chars
        pattern = rf"(\s*([{chars}])\s*)"
        schema = re.sub(pattern, r"\2", schema)

        # Replace multiple spaces with a single space
        schema = re.sub(r' +', ' ', schema)

        # Replace leading whitespace and blank lines
        schema = re.sub(r'^\s+', '', schema, flags=re.MULTILINE)

        return schema


def _read_schema_from_file(path: str) -> str:
    currentdir = os.path.dirname(os.path.abspath(__file__))
    abspath = os.path.join(currentdir, path)

    with open(abspath, 'r') as f:
        schema = f.read()

    return schema
