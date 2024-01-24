import os
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

import marqo.logging
import marqo.vespa.vespa_client
from marqo.core import constants
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

    def create_settings_schema(self) -> bool:
        """
        Create the Marqo settings schema if it does not exist.

        Returns:
            True if schema was created, False if it already existed
        """
        app = self.vespa_client.download_application()
        settings_schema_created = self._add_marqo_settings_schema(app)

        if settings_schema_created:
            self.vespa_client.deploy_application(app)
            self.vespa_client.wait_for_application_convergence()
            return True

        return False

    def create_index(self, marqo_index_request: MarqoIndexRequest) -> MarqoIndex:
        """
        Create a Marqo index.

        Args:
            marqo_index_request: Marqo index to create

        Returns:
            Created Marqo index

        Raises:
            IndexExistsError: If index already exists
            InvalidVespaApplicationError: If Vespa application is invalid after applying the index
        """
        app = self.vespa_client.download_application()
        settings_schema_created = self._add_marqo_settings_schema(app)

        if not settings_schema_created and self.index_exists(marqo_index_request.name):
            raise IndexExistsError(f"Index {marqo_index_request.name} already exists")

        vespa_schema = vespa_schema_factory(marqo_index_request)
        schema, marqo_index = vespa_schema.generate_schema()

        logger.debug(f'Creating index {str(marqo_index)} with schema:\n{schema}')

        self._add_schema(app, marqo_index.schema_name, schema)
        self._add_schema_to_services(app, marqo_index.schema_name)
        self.vespa_client.deploy_application(app)
        self.vespa_client.wait_for_application_convergence()
        self._save_index_settings(marqo_index)

        return marqo_index

    def batch_create_indexes(self, marqo_index_requests: List[MarqoIndexRequest]) -> List[MarqoIndex]:
        """
        Create multiple Marqo indexes as a single Vespa deployment.

        This method is intended to facilitate testing and should not be used in production.

        Args:
            marqo_index_requests: List of Marqo indexes to create

        Returns:
            List of created Marqo indexes

        Raises:
            IndexExistsError: If an index already exists
            InvalidVespaApplicationError: If Vespa application is invalid after applying the indexes
        """
        app = self.vespa_client.download_application()
        settings_schema_created = self._add_marqo_settings_schema(app)

        if not settings_schema_created:
            for index in marqo_index_requests:
                if self.index_exists(index.name):
                    raise IndexExistsError(f"Index {index.name} already exists")

        schema_responses = [
            vespa_schema_factory(index).generate_schema()  # Tuple (schema, MarqoIndex)
            for index in marqo_index_requests
        ]

        for schema, marqo_index in schema_responses:
            logger.debug(f'Creating index {str(marqo_index)} with schema:\n{schema}')
            self._add_schema(app, marqo_index.schema_name, schema)
            self._add_schema_to_services(app, marqo_index.schema_name)

        self.vespa_client.deploy_application(app)

        self.vespa_client.wait_for_application_convergence()

        for _, marqo_index in schema_responses:
            self._save_index_settings(marqo_index)

        return [schema_resp[1] for schema_resp in schema_responses]

    def delete_index(self, marqo_index: MarqoIndex) -> None:
        """
        Delete a Marqo index.

        This method is idempotent and does not raise an error if the index does not exist.

        Args:
            marqo_index: Marqo index to delete
        """
        app = self.vespa_client.download_application()

        self._remove_schema(app, marqo_index.schema_name)
        self._remove_schema_from_services(app, marqo_index.schema_name)
        self._add_schema_removal_override(app)
        self.vespa_client.deploy_application(app)
        self.vespa_client.wait_for_application_convergence()
        self._delete_index_settings_by_name(marqo_index.name)

    def delete_index_by_name(self, index_name: str) -> None:
        """
        Delete a Marqo index by name.

        Args:
            index_name: Name of Marqo index to delete
        Raises:
            IndexNotFoundError: If index does not exist
        """
        marqo_index = self.get_index(index_name)
        self.delete_index(marqo_index)

    def batch_delete_indexes_by_name(self, index_names: List[str]) -> None:
        marqo_indexes = [self.get_index(index_name) for index_name in index_names]
        self.batch_delete_indexes(marqo_indexes)

    def batch_delete_indexes(self, marqo_indexes: List[MarqoIndex]) -> None:
        """
        Delete multiple Marqo indexes as a single Vespa deployment.

        This method is intended to facilitate testing and should not be used in production.

        This method is idempotent and does not raise an error if an index does not exist.

        Args:
            marqo_indexes: List of Marqo indexes to delete
        """
        app = self.vespa_client.download_application()

        for marqo_index in marqo_indexes:
            self._remove_schema(app, marqo_index.schema_name)
            self._remove_schema_from_services(app, marqo_index.schema_name)
        self._add_schema_removal_override(app)
        self.vespa_client.deploy_application(app)
        for marqo_index in marqo_indexes:
            self._delete_index_settings_by_name(marqo_index.name)

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

    def _add_marqo_config(self, app: str) -> bool:
        """
        Add Marqo configuration to Vespa application package if this application package has not already been
        configured for Marqo.

        An application package is considered configured for Marqo if it contains the Marqo settings schema.
        Args:
            app: Path to Vespa application package
        Returns:
            True if configuration was added, False if it already existed
        """
        added = self._add_marqo_settings_schema(app)
        if added:
            self._add_marqo_query_profile(app)
            return True

        return False

    def _add_marqo_settings_schema(self, app: str) -> bool:
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

    def _add_marqo_query_profile(self, app: str) -> bool:
        """
        Create the Marqo query profile if it does not exist.
        Args:
            app: Path to Vespa application package

        Returns:
            True if query profile was created, False if it already existed
        """
        schema_path = os.path.join(app, 'search/query-profiles', f'{constants.MARQO_VESPA_QUERY_PROFILE}.xml')
        if not os.path.exists(schema_path):
            logger.debug('Marqo query profile does not exist. Creating it')

            query_profile = textwrap.dedent(
                f'''
                <query-profile id="{constants.MARQO_VESPA_QUERY_PROFILE}">
                    <field name="ranking.features.query(model)"/>
                </query-profile>
                '''
            ).strip()
            with open(schema_path, 'w') as f:
                f.write(query_profile)

            return True
        pass

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

        tree = ET.parse(services_path)
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

        # TODO - Verify there is only one documents section (one content cluster)
        # Error out otherwise as we don't know which one to use
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
