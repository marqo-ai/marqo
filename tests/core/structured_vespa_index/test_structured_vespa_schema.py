import os
import re

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.structured_vespa_index.structured_vespa_schema import StructuredVespaSchema
from tests.marqo_test import MarqoTestCase


class TestStructuredVespaSchema(MarqoTestCase):
    # TODO -- tests aren't verifying the Marqo Index return value. Verify this
    def test_generate_schema_standardIndex_successful(self):
        """
        Test an index that has all field types and configurations.
        """
        marqo_index_request = self.structured_marqo_index_request(
            name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='description', type=FieldType.Text),
                FieldRequest(name='category', type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name='tags', type=FieldType.ArrayText, features=[FieldFeature.Filter]),
                FieldRequest(name='image', type=FieldType.ImagePointer),
                FieldRequest(name='is_active', type=FieldType.Bool, features=[FieldFeature.Filter]),
                FieldRequest(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier]),
                FieldRequest(name='rank', type=FieldType.Int, features=[FieldFeature.ScoreModifier]),
                FieldRequest(name='click_per_day', type=FieldType.ArrayInt, features=[FieldFeature.Filter]),
                FieldRequest(name='last_updated', type=FieldType.ArrayFloat, features=[FieldFeature.Filter]),
            ],
            tensor_fields=['title', 'description']
        )

        actual_schema, _ = StructuredVespaSchema(marqo_index_request).generate_schema()
        expected_schema = self._read_schema_from_file('test_schemas/healthy_schema_1.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_oneTensorField_successful(self):
        """
        Test an index that has all field types and configurations. 1 tensor field.
        Rank profile does NOT use max for embedding similarity.
        """
        marqo_index_request = self.structured_marqo_index_request(
            name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='description', type=FieldType.Text),
                FieldRequest(name='category', type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name='tags', type=FieldType.ArrayText, features=[FieldFeature.Filter]),
                FieldRequest(name='image', type=FieldType.ImagePointer),
                FieldRequest(name='is_active', type=FieldType.Bool, features=[FieldFeature.Filter]),
                FieldRequest(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier]),
                FieldRequest(name='rank', type=FieldType.Int, features=[FieldFeature.ScoreModifier]),
                FieldRequest(name='click_per_day', type=FieldType.ArrayInt, features=[FieldFeature.Filter]),
                FieldRequest(name='last_updated', type=FieldType.ArrayFloat, features=[FieldFeature.Filter]),
            ],
            tensor_fields=['title']
        )

        actual_schema, _ = StructuredVespaSchema(marqo_index_request).generate_schema()
        expected_schema = self._read_schema_from_file('test_schemas/one_tensor_field.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_FourTensorFields_successful(self):
        """
        Test an index that has all field types and configurations. 4 tensor fields.
        Rank profile uses nested max functions for embedding similarity.
        """
        marqo_index_request = self.structured_marqo_index_request(
            name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='description', type=FieldType.Text),
                FieldRequest(name='category', type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name='tags', type=FieldType.ArrayText, features=[FieldFeature.Filter]),
                FieldRequest(name='image', type=FieldType.ImagePointer),
                FieldRequest(name='is_active', type=FieldType.Bool, features=[FieldFeature.Filter]),
                FieldRequest(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier]),
                FieldRequest(name='rank', type=FieldType.Int, features=[FieldFeature.ScoreModifier]),
                FieldRequest(name='click_per_day', type=FieldType.ArrayInt, features=[FieldFeature.Filter]),
                FieldRequest(name='last_updated', type=FieldType.ArrayFloat, features=[FieldFeature.Filter]),
            ],
            tensor_fields=['title', 'description', 'category', 'tags']
        )

        actual_schema, _ = StructuredVespaSchema(marqo_index_request).generate_schema()
        expected_schema = self._read_schema_from_file('test_schemas/four_tensor_fields.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_noLexicalFields_successful(self):
        """
        Test an index that has no lexical fields.
        """
        marqo_index_request = self.structured_marqo_index_request(
            name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text),
                FieldRequest(name='description', type=FieldType.Text),
                FieldRequest(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier])
            ],
            tensor_fields=['title', 'description']
        )

        actual_schema, _ = StructuredVespaSchema(marqo_index_request).generate_schema()
        expected_schema = self._read_schema_from_file('test_schemas/no_lexical_fields.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_noScoreModifierFields_successful(self):
        """
        Test an index that has no score modifier fields.
        """
        marqo_index_request = self.structured_marqo_index_request(
            name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='description', type=FieldType.Text),
                FieldRequest(name='price', type=FieldType.Float)
            ],
            tensor_fields=['title', 'description']
        )

        actual_schema, _ = StructuredVespaSchema(marqo_index_request).generate_schema()
        expected_schema = self._read_schema_from_file('test_schemas/no_score_modifiers.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_noTensorFields_successful(self):
        """
        Test an index that has no tensor fields.
        """
        marqo_index_request = self.structured_marqo_index_request(
            name='my_index',
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAngular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                FieldRequest(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
                FieldRequest(name='description', type=FieldType.Text),
                FieldRequest(name='price', type=FieldType.Float, features=[FieldFeature.ScoreModifier])
            ],
            tensor_fields=[]
        )

        actual_schema, _ = StructuredVespaSchema(marqo_index_request).generate_schema()
        expected_schema = self._read_schema_from_file('test_schemas/no_tensor_fields.sd')

        self.assertEqual(
            self._remove_whitespace_in_schema(expected_schema),
            self._remove_whitespace_in_schema(actual_schema)
        )

    def test_generate_schema_all_distance_metrics(self):
        """A test for the unstructured Vespa schema generation with each of the distance metrics."""
        index_name = "test_structured_schema_distance_metric"

        for distance_metric in DistanceMetric:
            with self.subTest(f"Structured index with distance metric: {distance_metric.value}"):
                marqo_index_request = self.structured_marqo_index_request(
                    name=index_name,
                    model=Model(name='ViT-B/32'),
                    distance_metric=distance_metric,    # Manually set distance metric to each one.
                    vector_numeric_type=VectorNumericType.Float,
                    hnsw_config=HnswConfig(ef_construction=100, m=16),
                    fields=[
                        FieldRequest(name='title', type=FieldType.Text),
                        FieldRequest(name='description', type=FieldType.Text)
                    ],
                    tensor_fields=['title', 'description']
                )

                actual_schema, _ = StructuredVespaSchema(marqo_index_request).generate_schema()
                expected_schema = self._read_schema_from_file(f'test_schemas/structured_distance_metric_{distance_metric.value}.sd')

                self.assertEqual(
                    self._remove_whitespace_in_schema(expected_schema),
                    self._remove_whitespace_in_schema(actual_schema)
                )

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
