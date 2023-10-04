import json
import os
import textwrap
import xml.etree.ElementTree as ET

import marqo.logging
import marqo.vespa.vespa_client
from marqo.core.exceptions import MarqoIndexExistsError
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index import IndexType, DistanceMetric, VectorNumericType, HnswConfig, Field, FieldType, \
    FieldFeature, TensorField
from marqo.core.vespa_index import for_marqo_index as vespa_index_factory
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient

logger = marqo.logging.get_logger(__name__)


class IndexManagement:
    _MARQO_SETTINGS_SCHEMA_NAME = 'marqo__settings'
    _MARQO_SETTINGS_SCHEMA = textwrap.dedent(
        '''
        schema %s {
            document %s {
                field index_name type string {
                    indexing: attribute | summary
                }
                field settings type string {
                    indexing: attribute | summary
                }
            }
        }
        ''' % (_MARQO_SETTINGS_SCHEMA_NAME, _MARQO_SETTINGS_SCHEMA_NAME)
    )

    def __init__(self, vespa_client: VespaClient):
        self.vespa_client = vespa_client

    def create_index(self, marqo_index: MarqoIndex):
        app = self.vespa_client.download_application()
        self._ensure_marqo_settings_schema(app)

        if self._index_exists(marqo_index.name):
            raise MarqoIndexExistsError(f"Index {marqo_index.name} already exists")

        vespa_index = vespa_index_factory(marqo_index)

        schema = vespa_index.generate_schema(marqo_index)

        self._add_schema(app, marqo_index.name, schema)
        self.vespa_client.deploy_application(app)
        self._save_index_settings(marqo_index)

    def _index_exists(self, name: str) -> bool:
        return False

    def _ensure_marqo_settings_schema(self, app: str) -> None:
        schema_path = os.path.join(app, 'schemas', f'{self._MARQO_SETTINGS_SCHEMA_NAME}.sd')
        if not os.path.exists(schema_path):
            with open(schema_path, 'w') as f:
                f.write(self._MARQO_SETTINGS_SCHEMA)
            self._add_schema_to_services(app, self._MARQO_SETTINGS_SCHEMA_NAME)

    def _add_schema(self, app: str, name: str, schema: str) -> None:
        schema_path = os.path.join(app, 'schemas', f'{name}.sd')
        if os.path.exists(schema_path):
            logger.warn(f"Schema {name} already exists in application package, overwriting")

        with open(schema_path, 'w') as f:
            f.write(schema)

    def _add_schema_to_services(self, app: str, name: str) -> None:
        services_path = os.path.join(app, 'services.xml')

        tree = ET.parse(services_path)  # Replace 'path_to_file.xml' with the path to your XML file
        root = tree.getroot()

        documents_section = root.find(".//documents")

        new_document = ET.SubElement(documents_section, "document")
        new_document.set("type", name)
        new_document.set("mode", "index")

        tree.write(services_path)

    def _save_index_settings(self, marqo_index: MarqoIndex):
        """
        Create or update index settings in Vespa settings schema.
        """
        self.vespa_client.feed_batch(
            [
                VespaDocument(
                    id=self._MARQO_SETTINGS_SCHEMA_NAME,
                    fields={
                        'index_name': marqo_index.name,
                        'settings': json.dumps(marqo_index.json())
                    }
                )
            ],
            schema=self._MARQO_SETTINGS_SCHEMA_NAME
        )


if __name__ == "__main__":
    marqo_index = MarqoIndex(
        name='my_index_1', model='clip', distance_metric=DistanceMetric.PrenormalizedAnguar,
        type=IndexType.Typed,
        vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=100, m=16),
        fields=[
            Field(name='title', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
            Field(name='description', type=FieldType.Text, features=[FieldFeature.LexicalSearch]),
            Field(name='is_active', type=FieldType.Bool, features=[FieldFeature.Filter]),
            Field(name='price', type=FieldType.Float, features=[FieldFeature.Filter, FieldFeature.ScoreModifier]),
            Field(name='category', type=FieldType.Text, features=[FieldFeature.Filter, FieldFeature.LexicalSearch]),
        ],
        tensor_fields=[
            TensorField(name='title'),
            TensorField(name='description')
        ]
    )

    vespa_client = marqo.vespa.vespa_client.VespaClient(
        config_url='http://localhost:19071',
        query_url='http://localhost:8080',
        document_url='http://localhost:8080',
    )

    index_management = IndexManagement(vespa_client)

    index_management.create_index(marqo_index)
