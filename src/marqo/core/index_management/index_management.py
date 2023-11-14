import os
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

import marqo.logging
import marqo.vespa.vespa_client
from marqo.core.exceptions import IndexExistsError, IndexNotFoundError
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index_request import MarqoIndexRequest
from marqo.core.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.exceptions import InternalError
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

    def __init__(self, vespa_client: VespaClient):
        self.vespa_client = vespa_client

    def create_index(self, marqo_index_request: MarqoIndexRequest) -> None:
        """
        Create a Marqo index.
        Args:
            marqo_index_request: Marqo index to create

        Raises:
            IndexExistsError: If index already exists
            InvalidVespaApplicationError: If Vespa application is invalid after applying the index
        """
        app = self.vespa_client.download_application()
        settings_schema_created = self._create_marqo_settings_schema(app)

        if not settings_schema_created and self.index_exists(marqo_index_request.name):
            raise IndexExistsError(f"Index {marqo_index_request.name} already exists")

        vespa_schema = vespa_schema_factory(marqo_index_request)
        schema, marqo_index = vespa_schema.generate_schema()

        self._add_schema(app, marqo_index.name, schema)
        self._add_schema_to_services(app, marqo_index.name)
        self.vespa_client.deploy_application(app)
        self.vespa_client.wait_for_application_convergence()
        self._save_index_settings(marqo_index)

    def batch_create_indexes(self, marqo_index_requests: List[MarqoIndexRequest]) -> None:
        """
        Create multiple Marqo indexes as a single Vespa deployment.

        This method is intended to facilitate testing and should not be used in production.

        Args:
            marqo_index_requests: List of Marqo indexes to create

        Raises:
            IndexExistsError: If an index already exists
            InvalidVespaApplicationError: If Vespa application is invalid after applying the indexes
        """
        app = self.vespa_client.download_application()
        settings_schema_created = self._create_marqo_settings_schema(app)

        if not settings_schema_created:
            for index in marqo_index_requests:
                if self.index_exists(index.name):
                    raise IndexExistsError(f"Index {index.name} already exists")

        schema_responses = {
            index.name: vespa_schema_factory(index).generate_schema()  # Tuple (schema, MarqoIndex)
            for index in marqo_index_requests
        }

        for name, schema_resp in schema_responses.items():
            self._add_schema(app, name, schema_resp[0])
            self._add_schema_to_services(app, name)

        self.vespa_client.deploy_application(app)

        self.vespa_client.wait_for_application_convergence()

        for _, schema_resp in schema_responses:
            self._save_index_settings(schema_resp[1])

    def delete_index(self, marqo_index: MarqoIndex) -> None:
        """
        Delete a Marqo index.

        Args:
            marqo_index: Marqo index to delete
        """
        self.delete_index_by_name(marqo_index.name)

    def delete_index_by_name(self, index_name: str) -> None:
        """
        Delete a Marqo index by name.

        Args:
            index_name: Name of Marqo index to delete
        """
        app = self.vespa_client.download_application()

        if not self.index_exists(index_name):
            raise IndexNotFoundError(f"Cannot delete index {index_name} as it does not exist")

        self._remove_schema(app, index_name)
        self._remove_schema_from_services(app, index_name)
        self._add_schema_removal_override(app)
        self.vespa_client.deploy_application(app)
        self._delete_index_settings_by_name(index_name)

    def batch_delete_indexes(self, marqo_indexes: List[MarqoIndex]) -> None:
        """
        Delete multiple Marqo indexes as a single Vespa deployment.

        This method is intended to facilitate testing and should not be used in production.
        Args:
            marqo_indexes: List of Marqo indexes to delete
        """
        app = self.vespa_client.download_application()

        for index in marqo_indexes:
            if not self.index_exists(index.name):
                raise IndexNotFoundError(f"Cannot delete index {index.name} as it does not exist")

        for index in marqo_indexes:
            self._remove_schema(app, index.name)
            self._remove_schema_from_services(app, index.name)
        self._add_schema_removal_override(app)
        self.vespa_client.deploy_application(app)
        for index in marqo_indexes:
            self._delete_index_settings(index)

    def get_all_indexes(self) -> List[MarqoIndex]:
        """
        Get all Marqo indexes.

        Returns:
            List of Marqo indexes
        """
        batch_response = self.vespa_client.get_all_documents(self._MARQO_SETTINGS_SCHEMA_NAME, stream=True)
        if batch_response.continuation:
            # TODO - Verify expected behaviour when streaming. Do we need to expect and handle pagination?
            raise InternalError("Unexpected continuation token received")

        return [
            MarqoIndex.parse_raw(response.fields['settings'])
            for response in batch_response.documents
        ]

    def get_index(self, index_name) -> MarqoIndex:
        """
        Get a Marqo index by name.

        Args:
            index_name: Name of Marqo index to get

        Returns:
            Marqo index
        """
        try:
            response = self.vespa_client.get_document(index_name, self._MARQO_SETTINGS_SCHEMA_NAME)
        except VespaStatusError as e:
            if e.status_code == 404:
                raise IndexNotFoundError(f"Index {index_name} not found")
            raise e

        return MarqoIndex.parse_raw(response.document.fields['settings'])

    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.

        Note: Calling this method when settings schema does not exist will cause a VespaStatusError to be raised
        with status code 400.

        Args:
            index_name: Name of index to check

        Returns:
            True if index exists, False otherwise
        """
        try:
            _ = self.get_index(index_name)
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

    def _add_schema_removal_override(self, app: str) -> None:
        validation_overrides_path = os.path.join(app, 'validation-overrides.xml')
        date = datetime.utcnow().strftime('%Y-%m-%d')
        content = textwrap.dedent(
            f'''
            <validation-overrides>
                 <allow until='{date}'>schema-removal</allow>
            </validation-overrides>
            '''
        ).strip()

        with open(validation_overrides_path, 'w') as f:
            f.write(content)

    def _save_index_settings(self, marqo_index: MarqoIndex) -> None:
        """
        Create or update index settings in Vespa settings schema.
        """
        # Don't store model properties if model is not custom
        if not marqo_index.model.custom:
            marqo_index.model.properties = None

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

    def _delete_index_settings(self, marqo_index: MarqoIndex):
        self._delete_index_settings_by_name(marqo_index.name)

    def _delete_index_settings_by_name(self, index_name: str):
        # Note Vespa delete is 200 even if document doesn't exist
        self.vespa_client.delete_document(index_name, self._MARQO_SETTINGS_SCHEMA_NAME)
