import pytest
import json

import pydantic

from marqo.core.models.marqo_index import FieldType, FieldFeature, Field, MarqoIndex
from integ_tests.marqo_test import MarqoTestCase


@pytest.mark.unittest
class TestStructuredMarqoIndex(MarqoTestCase):
    def test_filterable_field_names_pre220(self):
        """
        Test that filterable_field_names returns the correct fields for a structured index when
        index Marqo version <2.2.0.
        """
        versions = [
            '2.0.0',
            '2.0.1',
            '2.1.0',
            '2.1.5'
        ]
        for version in versions:
            with self.subTest(version=version):
                marqo_index = self.structured_marqo_index(
                    name='my_index',
                    schema_name='my_index',
                    fields=[
                        Field(name='title', type=FieldType.Text),
                        Field(name='price', type=FieldType.Float, features=[FieldFeature.Filter],
                              filter_field_name='price_filter'),
                        Field(name='tags', type=FieldType.Text, features=[FieldFeature.Filter],
                              filter_field_name='tags_filter')
                    ],
                    tensor_fields=[],
                    marqo_version=version
                )
                self.assertEqual(
                    marqo_index.filterable_fields_names,
                    {'price', 'tags'}
                )

    def test_filterable_field_names_post220(self):
        """
        Test that filterable_field_names returns the correct fields for a structured index when
        index Marqo version >=2.2.0.
        """
        versions = [
            '2.2.0',
            '2.2.1',
            '2.3.0',
            '2.5.5'
        ]
        for version in versions:
            with self.subTest(version=version):
                marqo_index = self.structured_marqo_index(
                    name='my_index',
                    schema_name='my_index',
                    fields=[
                        Field(name='title', type=FieldType.Text),
                        Field(name='price', type=FieldType.Float, features=[FieldFeature.Filter],
                              filter_field_name='price_filter'),
                        Field(name='tags', type=FieldType.Text, features=[FieldFeature.Filter],
                              filter_field_name='tags_filter')
                    ],
                    tensor_fields=[],
                    marqo_version=version
                )
                self.assertEqual(
                    marqo_index.filterable_fields_names,
                    {'_id', 'price', 'tags'}
                )

    def test_deserialization_with_extra_fields(self):
        """
        Test Pydantic allows deserialization of MarqoIndex with extra fields
        """
        indexes = [
            self.structured_marqo_index(
                name='my_index',
                schema_name='my_index',
                fields=[
                    Field(name='title', type=FieldType.Text)
                ],
                tensor_fields=[],
                marqo_version='2.12.0'
            ),
            self.unstructured_marqo_index(
                name='my_index_2',
                schema_name='my_index_2',
                marqo_version='2.12.0'
            )
        ]

        for index in indexes:
            with self.subTest(index=index):
                index_setting_json = json.loads(index.json())
                index_setting_json["random_field"] = "value"

                try:
                    parsed_index = MarqoIndex.parse_raw(json.dumps(index_setting_json))
                    # assert that the extra field is deserialized
                    self.assertEqual("value", parsed_index.random_field)
                    # assert that the extra field is kept after serialization
                    self.assertTrue("random_field" in parsed_index.json())
                except pydantic.error_wrappers.ValidationError as e:
                    self.fail(f"Pydantic validation failed: {e}")
