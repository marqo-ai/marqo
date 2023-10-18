import os
import textwrap
import time
import xml.etree.ElementTree as ET
from typing import List

import marqo.logging
import marqo.vespa.vespa_client
from marqo.core.exceptions import IndexExistsError, IndexNotFoundError
from marqo.core.models import MarqoIndex
from marqo.core.vespa_index import for_marqo_index as vespa_index_factory
from marqo.exceptions import MarqoError, InternalError
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient

logger = marqo.logging.get_logger(__name__)


class IndexManagement:
    _MARQO_SETTINGS_SCHEMA_NAME = 'marqo__settings'
    _MARQO_SETTINGS_SCHEMA_TEMPLATE = textwrap.dedent(
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
        '''
    )
    # Number of retries if settings schema feed fails with 400. This can happen when the settings schema is created as
    # part of the index creation and is not yet available for feeding.
    _MARQO_SETTINGS_RETRIES = 30

    def __init__(self, vespa_client: VespaClient):
        self.vespa_client = vespa_client

    def create_index(self, marqo_index: MarqoIndex) -> None:
        """
        Create a Marqo index.
        Args:
            marqo_index: Marqo index to create

        Raises:
            IndexExistsError: If index already exists
            InvalidVespaApplicationError: If Vespa application is invalid after applying the index
        """
        app = self.vespa_client.download_application()
        settings_schema_created = self._create_marqo_settings_schema(app)

        if not settings_schema_created and self._index_exists(marqo_index.name):
            raise IndexExistsError(f"Index {marqo_index.name} already exists")

        vespa_index = vespa_index_factory(marqo_index)

        schema = vespa_index.generate_schema(marqo_index)

        self._add_schema(app, marqo_index.name, schema)
        self._add_schema_to_services(app, marqo_index.name)
        self.vespa_client.deploy_application(app)
        self._save_index_settings(
            marqo_index,
            retries=self._MARQO_SETTINGS_RETRIES if settings_schema_created else 0
        )

    def delete_index(self, marqo_index: MarqoIndex):
        """
        Delete a Marqo index.

        Args:
            marqo_index: Marqo index to delete
        """
        app = self.vespa_client.download_application()

        if not self._index_exists(marqo_index.name):
            raise IndexNotFoundError(f"Cannot delete index {marqo_index.name} as it does not exist")

        self._remove_schema(app, marqo_index.name)
        self._remove_schema_from_services(app, marqo_index.name)
        self.vespa_client.deploy_application(app)
        self._delete_index_settings(marqo_index)

    def get_all_indexes(self) -> List[MarqoIndex]:
        batch_response = self.vespa_client.get_all_documents(self._MARQO_SETTINGS_SCHEMA_NAME, stream=True)
        if batch_response.continuation:
            # TODO - Verify expected behaviour when streaming. Do we need to expect and handle pagination?
            raise InternalError("Unexpected continuation token received")

        return [
            MarqoIndex.parse_raw(response.fields['settings'])
            for response in batch_response.documents
        ]

    def get_index(self, index_name) -> MarqoIndex:
        try:
            response = self.vespa_client.get_document(index_name, self._MARQO_SETTINGS_SCHEMA_NAME)
        except VespaStatusError as e:
            if e.status_code == 404:
                raise IndexNotFoundError(f"Index {index_name} not found")
            raise e

        return MarqoIndex.parse_raw(response.document.fields['settings'])

    def _index_exists(self, name: str) -> bool:
        """
        Check if an index exists.

        Note: Do not call this method if settings schema does not exist.

        Args:
            name:

        Returns:

        """
        try:
            _ = self.get_index(name)
            return True
        except IndexNotFoundError:
            return False

    def _create_marqo_settings_schema(self, app: str) -> bool:
        """
        Create the Marqo settings schema if it does not exist.
        Args:
            app: Path to Vespa application package
        Returns:
            True if schema was created, False if it already existed
        """
        schema_path = os.path.join(app, 'schemas', f'{self._MARQO_SETTINGS_SCHEMA_NAME}.sd')
        if not os.path.exists(schema_path):
            logger.debug('Marqo settings schema does not exist. Creating it')

            schema = self._MARQO_SETTINGS_SCHEMA_TEMPLATE % (
                self._MARQO_SETTINGS_SCHEMA_NAME,
                self._MARQO_SETTINGS_SCHEMA_NAME
            )
            with open(schema_path, 'w') as f:
                f.write(schema)
            self._add_schema_to_services(app, self._MARQO_SETTINGS_SCHEMA_NAME)

            return True
        return False

    def _add_schema(self, app: str, name: str, schema: str) -> None:
        schema_path = os.path.join(app, 'schemas', f'{name}.sd')
        if os.path.exists(schema_path):
            logger.warn(f"Schema {name} already exists in application package, overwriting")

        with open(schema_path, 'w') as f:
            f.write(schema)

    def _remove_schema(self, app: str, name: str) -> None:
        schema_path = os.path.join(app, 'schemas', f'{name}.sd')
        if not os.path.exists(schema_path):
            logger.warn(f"Schema {name} does not exist in application package, nothing to remove")

        os.remove(schema_path)

    def _add_schema_to_services(self, app: str, name: str) -> None:
        services_path = os.path.join(app, 'services.xml')

        tree = ET.parse(services_path)  # Replace 'path_to_file.xml' with the path to your XML file
        root = tree.getroot()

        documents_section = root.find(".//documents")

        new_document = ET.SubElement(documents_section, "document")
        new_document.set("type", name)
        new_document.set("mode", "index")

        tree.write(services_path)

    def _remove_schema_from_services(self, app: str, name: str) -> None:
        services_path = os.path.join(app, 'services.xml')

        tree = ET.parse(services_path)
        root = tree.getroot()

        documents_section = root.find(".//documents")

        deleted = False
        for document in documents_section.findall("document"):
            if document.get("type") == name:
                documents_section.remove(document)
                deleted = True
                break

        if not deleted:
            logger.warn(f"Schema {name} does not exist in services.xml, nothing to remove")
        else:
            tree.write(services_path)

    def _save_index_settings(self, marqo_index: MarqoIndex, retries=0, attempt=0):
        """
        Create or update index settings in Vespa settings schema.
        """
        # Don't store model properties if model is not custom
        if not marqo_index.model.custom:
            marqo_index.model.properties = None

        try:
            self.vespa_client.feed_document(
                VespaDocument(
                    id=marqo_index.name,
                    fields={
                        'index_name': marqo_index.name,
                        'settings': marqo_index.json()
                    }
                ),
                schema=self._MARQO_SETTINGS_SCHEMA_NAME
            )
        except VespaStatusError as e:
            if e.status_code == 400 and attempt < retries:
                logger.warn(f"Failed to feed index settings for {marqo_index.name}, retrying in 1 second")
                time.sleep(1)
                self._save_index_settings(marqo_index, retries, attempt + 1)
            else:
                raise MarqoError(
                    f"Failed to feed index settings for {marqo_index.name}: {str(e)}", cause=e
                ) from e

    def _delete_index_settings(self, marqo_index: MarqoIndex):
        # Note Vespa delete is 200 even if document doesn't exist
        self.vespa_client.delete_document(marqo_index.name, self._MARQO_SETTINGS_SCHEMA_NAME)


if __name__ == '__main__':
    vespa_client = marqo.vespa.vespa_client.VespaClient(
        config_url='http://localhost:19071',
        document_url='http://localhost:8080',
        query_url='http://localhost:8080',
    )

    index_management = IndexManagement(vespa_client)

    index = index_management.get_index('ef05bf9fd96c48a19d9b219521aa9504')
    # indexes = index_management.get_all_indexes()

    print('hi')
