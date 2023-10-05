import uuid

import vespa.application as pyvespa

from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import *
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient
from tests.marqo_test import MarqoTestCase


class TestIndexManagement(MarqoTestCase):

    def setUp(self):
        self.client = VespaClient("http://localhost:19071", "http://localhost:8080", "http://localhost:8080")
        self.index_management = IndexManagement(self.client)

        # TODO - use Marqo Vespa client instead of pyvespa once get document functionality is implemented
        self.pyvespa_client = pyvespa.Vespa(url="http://localhost", port=8080)

    def test_create_index_successful(self):
        index_name = str(uuid.uuid4()).replace('-', '')
        marqo_index = MarqoIndex(
            name=index_name, model=Model(name='ViT-B/32'), distance_metric=DistanceMetric.PrenormalizedAnguar,
            type=IndexType.Typed,
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

        self.index_management.create_index(marqo_index)

        # Inserting a document into the new schema to verify it exists
        self.client.feed_batch(
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
