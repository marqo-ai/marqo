import uuid
from unittest.mock import patch

import vespa.application as pyvespa

from marqo.core.exceptions import IndexExistsError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import *
from marqo.vespa.models import VespaDocument
from tests.marqo_test import MarqoTestCase


class TestIndexManagement(MarqoTestCase):

    def setUp(self):
        self.index_management = IndexManagement(self.vespa_client)

        # TODO - use Marqo Vespa client instead of pyvespa once get document functionality is implemented
        self.pyvespa_client = pyvespa.Vespa(url="http://localhost", port=8080)

    def test_create_index_settingsSchemaDoesNotExist_successful(self):
        """
        Test that a new index is created successfully when the settings schema does not exist
        """
        index_name = 'a' + str(uuid.uuid4()).replace('-', '')
        settings_schema_name = 'a' + str(uuid.uuid4()).replace('-', '')
        marqo_index = self.marqo_index(
            name=index_name,
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Structured,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
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

        with patch.object(IndexManagement, '_MARQO_SETTINGS_SCHEMA_NAME', settings_schema_name):
            self.index_management.create_index(marqo_index)

            # Inserting a document into the new schema to verify it exists
            self.vespa_client.feed_batch(
                [
                    VespaDocument(
                        id='1',
                        fields={}
                    )
                ],
                schema=index_name
            )

            # Verify settings have been saved
            settings_json = self.pyvespa_client.get_data(
                schema=IndexManagement._MARQO_SETTINGS_SCHEMA_NAME,
                data_id=index_name
            ).json['fields']['settings']

            self.assertEqual(settings_json, marqo_index.json())

    def test_create_index_settingsSchemaExists_successful(self):
        index_name_1 = 'a' + str(uuid.uuid4()).replace('-', '')
        index_name_2 = 'a' + str(uuid.uuid4()).replace('-', '')
        marqo_index = self.marqo_index(
            name=index_name_1,
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Structured,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
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
        self.index_management.create_index(marqo_index)

        # Create with a different name now that we know settings schema exists
        marqo_index.name = index_name_2
        self.index_management.create_index(marqo_index)

        # Inserting a document into the new schema to verify it exists
        self.vespa_client.feed_batch(
            [
                VespaDocument(
                    id='1',
                    fields={}
                )
            ],
            schema=index_name_2
        )

        # Verify settings have been saved
        settings_json = self.pyvespa_client.get_data(
            schema=IndexManagement._MARQO_SETTINGS_SCHEMA_NAME,
            data_id=index_name_2
        ).json['fields']['settings']

        self.assertEqual(settings_json, marqo_index.json())

    def test_create_index_indexExists_fails(self):
        index_name = 'a' + str(uuid.uuid4()).replace('-', '')
        marqo_index = self.marqo_index(
            name=index_name,
            model=Model(name='ViT-B/32'),
            distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Structured,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            fields=[
                Field(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
            ],
            tensor_fields=[]
        )

        self.index_management.create_index(marqo_index)

        with self.assertRaises(IndexExistsError):
            self.index_management.create_index(marqo_index)
