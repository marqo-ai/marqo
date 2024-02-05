from marqo.core.models.marqo_index import FieldType, FieldFeature, Field
from tests.marqo_test import MarqoTestCase


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
