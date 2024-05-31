import os
import textwrap
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime
from typing import List
from typing import Optional

import marqo.logging
import marqo.vespa.vespa_client
from marqo import version
from marqo.base_model import ImmutableStrictBaseModel
from marqo.core import constants
from marqo.core.distributed_lock.zookeeper_distributed_lock import ZookeeperDistributedLock
from marqo.core.distributed_lock.zookeeper_distributed_lock import get_deployment_lock
from marqo.core.exceptions import IndexExistsError, IndexNotFoundError
from marqo.core.exceptions import OperationConflictError
from marqo.core.exceptions import ZookeeperLockNotAcquiredError, InternalError
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index_request import MarqoIndexRequest
from marqo.core.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.index_settings import IndexSettings
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient
from marqo.vespa.zookeeper_client import ZookeeperClient

logger = marqo.logging.get_logger(__name__)


class _MarqoConfig(ImmutableStrictBaseModel):
    version: str


class IndexManagement:
    _MARQO_SETTINGS_SCHEMA_NAME = 'marqo__settings'
    _MARQO_CONFIG_DOC_ID = 'marqo__config'
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
    _DEFAULT_QUERY_PROFILE_TEMPLATE = textwrap.dedent(
        '''
        <query-profile id="default">
            <field name="maxHits">1000</field>
            <field name="maxOffset">10000</field>
        </query-profile>
        '''
    )

    def __init__(self, vespa_client: VespaClient, zookeeper_client: Optional[ZookeeperClient] = None,
                 enable_index_operations: bool = False):
        """Instantiate an IndexManagement object.

        Args:
            vespa_client: VespaClient object
            zookeeper_client: ZookeeperClient object
            enable_index_operations: A flag to enable index operations. If set to True,
                the object can create/delete indexes, otherwise, it raises an InternalError during index operations.
        """
        self.vespa_client = vespa_client
        self._zookeeper_client = zookeeper_client
        self._zookeeper_deployment_lock: Optional[ZookeeperDistributedLock] = self._instantiate_deployment_lock()
        self._enable_index_operations = enable_index_operations

    def _instantiate_deployment_lock(self) -> Optional[ZookeeperDistributedLock]:
        """Instantiate a ZookeeperDistributedLock."""
        if self._zookeeper_client is None:
            return None
        else:
            return get_deployment_lock(self._zookeeper_client)

    def bootstrap_vespa(self) -> bool:
        """
        Add Marqo configuration to Vespa application package if an existing Marqo configuration is not detected.

        Returns:
            True if Vespa was configured, False if it was already configured
        """
        app = self.vespa_client.download_application()
        configured = self._marqo_config_exists(app)

        if not configured:
            self._add_marqo_config(app)
            self.vespa_client.deploy_application(app)
            self.vespa_client.wait_for_application_convergence()
            self._save_marqo_version(version.get_version())
            return True

        return False

    def create_index(self, marqo_index_request: MarqoIndexRequest) -> MarqoIndex:
        """
        Create a Marqo index in a thread-safe manner.

        Args:
            marqo_index_request: Marqo index to create

        Returns:
            Created Marqo index

        Raises:
            IndexExistsError: If index already exists
            InvalidVespaApplicationError: If Vespa application is invalid after applying the index
            RuntimeError: If deployment lock is not instantiated
            OperationConflictError: If another index creation/deletion operation is
                in progress and the lock cannot be acquired
        """
        with self._deployment_lock_context_manager():
            app = self.vespa_client.download_application(check_for_application_convergence=True)
            configured = self._marqo_config_exists(app)

            if configured and self.index_exists(marqo_index_request.name):
                raise IndexExistsError(f"Index {marqo_index_request.name} already exists")
            else:
                logger.debug('Marqo config does not exist. Configuring Vespa as part of index creation')
                self._add_marqo_config(app)

            # Populate the prefix fields if they are None
            if marqo_index_request.model.text_query_prefix is None:
                marqo_index_request.model.text_query_prefix = marqo_index_request.model.get_default_text_query_prefix()
            if marqo_index_request.model.text_chunk_prefix is None:
                marqo_index_request.model.text_chunk_prefix = marqo_index_request.model.get_default_text_chunk_prefix()

            vespa_schema = vespa_schema_factory(marqo_index_request)
            schema, marqo_index = vespa_schema.generate_schema()

            logger.debug(f'Creating index {str(marqo_index)} with schema:\n{schema}')

            self._add_schema(app, marqo_index.schema_name, schema)
            self._add_schema_to_services(app, marqo_index.schema_name)
            self.vespa_client.deploy_application(app)
            self.vespa_client.wait_for_application_convergence()
            self._save_index_settings(marqo_index)

            if not configured:
                self._save_marqo_version(version.get_version())

            return marqo_index

    @staticmethod
    def validate_index_settings(index_name: str, settings_dict: dict) -> None:
        """
        Validates index settings using the IndexSettings model.

        Args:
            index_name (str): The name of the index to validate settings for.
            settings_dict (dict): A dictionary of settings to validate.

        Raises:
            ValidationError: If the settings are invalid in the context of the IndexSettings model.
            api_exceptions.InvalidArgError: If several settings are invalid in the context of the Marqo API.
            Check the errors in to_marqo_index_request method of IndexSettings model for more details.

        Returns:
            None: If the validation is successful, nothing is returned,
            else InvalidArgumentError is raised.
        """
        index_settings = IndexSettings(**settings_dict)
        index_settings.to_marqo_index_request(index_name)

    def batch_create_indexes(self, marqo_index_requests: List[MarqoIndexRequest]) -> List[MarqoIndex]:
        """
        Create multiple Marqo indexes as a single Vespa deployment, in a thread-safe manner.

        This method is intended to facilitate testing and should not be used in production.

        Args:
            marqo_index_requests: List of Marqo indexes to create

        Returns:
            List of created Marqo indexes

        Raises:
            IndexExistsError: If an index already exists
            InvalidVespaApplicationError: If Vespa application is invalid after applying the indexes
            RuntimeError: If deployment lock is not instantiated
            OperationConflictError: If another index creation/deletion operation is
                in progress and the lock cannot be acquired
        """
        with self._deployment_lock_context_manager():
            app = self.vespa_client.download_application(check_for_application_convergence=True)
            configured = self._add_marqo_config(app)

            if not configured:
                for index in marqo_index_requests:
                    if self.index_exists(index.name):
                        raise IndexExistsError(f"Index {index.name} already exists")

            # Populate the prefix fields if they are None
            for index in marqo_index_requests:
                if index.model.text_query_prefix is None:
                    index.model.text_query_prefix = index.model.get_default_text_query_prefix()
                if index.model.text_chunk_prefix is None:
                    index.model.text_chunk_prefix = index.model.get_default_text_chunk_prefix()

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
        Delete a Marqo index. To make this operation thread-safe, use delete_index_by_name instead.

        This method is idempotent and does not raise an error if the index does not exist.

        Args:
            marqo_index: Marqo index to delete
        """
        app = self.vespa_client.download_application(check_for_application_convergence=True)

        self._remove_schema(app, marqo_index.schema_name)
        self._remove_schema_from_services(app, marqo_index.schema_name)
        self._add_schema_removal_override(app)
        self.vespa_client.deploy_application(app)
        self.vespa_client.wait_for_application_convergence()
        self._delete_index_settings_by_name(marqo_index.name)

    def delete_index_by_name(self, index_name: str) -> None:
        """
        Delete a Marqo index by name, in a thread-safe manner.

        Args:
            index_name: Name of Marqo index to delete
        Raises:
            IndexNotFoundError: If index does not exist
            RuntimeError: If deployment lock is not instantiated
            OperationConflictError: If another index creation/deletion operation is
                in progress and the lock cannot be acquired
        """
        with self._deployment_lock_context_manager():
            marqo_index = self.get_index(index_name)
            self.delete_index(marqo_index)

    def batch_delete_indexes_by_name(self, index_names: List[str]) -> None:
        """
        Delete multiple Marqo indexes by name, in a thread-safe manner.
        Args:
            index_names:
        Raises:
            IndexNotFoundError: If an index does not exist
            RuntimeError: If deployment lock is not instantiated
            OperationConflictError: If another index creation/deletion operation is
                in progress and the lock cannot be acquired
        """
        with self._deployment_lock_context_manager():
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
        app = self.vespa_client.download_application(check_for_application_convergence=True)

        for marqo_index in marqo_indexes:
            self._remove_schema(app, marqo_index.schema_name)
            self._remove_schema_from_services(app, marqo_index.schema_name)
        self._add_schema_removal_override(app)
        self.vespa_client.deploy_application(app)
        self.vespa_client.wait_for_application_convergence()
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
            MarqoIndex.parse_raw(document.fields['settings'])

            for document in batch_response.documents
            if not document.id.split('::')[-1].startswith(constants.MARQO_RESERVED_PREFIX)
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

    def get_marqo_version(self) -> str:
        """
        Get the Marqo version Vespa is configured for.
        """
        try:
            response = self.vespa_client.get_document(self._MARQO_CONFIG_DOC_ID, self._MARQO_SETTINGS_SCHEMA_NAME)
        except VespaStatusError as e:
            if e.status_code == 404:
                logger.debug('Marqo config document does not exist. Assuming Marqo version 2.0.x')
                return '2.0'
            raise e

        return _MarqoConfig.parse_raw(response.document.fields['settings']).version

    def _marqo_config_exists(self, app) -> bool:
        # For Marqo 2.1+, recording Marqo version is the final stage of configuration, and its absence
        # indicates an incomplete configuration (e.g., interrupted configuration). However, for Marqo 2.0.x,
        # Marqo version is not recorded. Marqo 2.0.x can be detected by an existing Marqo settings schema,
        # but no default query profile
        settings_schema_exists = os.path.exists(os.path.join(app, 'schemas', f'{self._MARQO_SETTINGS_SCHEMA_NAME}.sd'))
        query_profile_exists = os.path.exists(
            os.path.join(app, 'search/query-profiles', 'default.xml')
        )

        if settings_schema_exists and not query_profile_exists:
            logger.debug('Detected existing Marqo 2.0.x configuration')
            return True

        if settings_schema_exists and query_profile_exists:
            try:
                self.vespa_client.get_document(self._MARQO_CONFIG_DOC_ID, self._MARQO_SETTINGS_SCHEMA_NAME)
                return True
            except VespaStatusError as e:
                if e.status_code == 404:
                    logger.debug('Marqo config document does not exist. Detected incomplete Marqo configuration')
                    return False
                raise e

        # Settings schema not found, so Marqo config does not exist
        return False

    def _add_marqo_config(self, app: str) -> bool:
        """
        Add Marqo configuration to Vespa application package.

        Args:
            app: Path to Vespa application package
        Returns:
            True if configuration was added, False if all components already existed
        """
        added_settings_schema = self._add_marqo_settings_schema(app)
        added_query_profile = self._add_default_query_profile(app)

        return added_settings_schema or added_query_profile

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
            os.makedirs(os.path.dirname(schema_path), exist_ok=True)
            with open(schema_path, 'w') as f:
                f.write(schema)
            self._add_schema_to_services(app, self._MARQO_SETTINGS_SCHEMA_NAME)

            return True
        return False

    def _add_default_query_profile(self, app: str) -> bool:
        """
        Create the default query profile if it does not exist.
        Args:
            app: Path to Vespa application package

        Returns:
            True if query profile was created, False if it already existed
        """
        profile_path = os.path.join(app, 'search/query-profiles', 'default.xml')
        if not os.path.exists(profile_path):
            logger.debug('Default query profile does not exist. Creating it')

            query_profile = self._DEFAULT_QUERY_PROFILE_TEMPLATE
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            with open(profile_path, 'w') as f:
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

    def _save_marqo_version(self, version: str) -> None:
        self.vespa_client.feed_document(
            VespaDocument(
                id=self._MARQO_CONFIG_DOC_ID,
                fields={
                    'settings': _MarqoConfig(version=version).json()
                }
            ),
            schema=self._MARQO_SETTINGS_SCHEMA_NAME
        )

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

    @contextmanager
    def _deployment_lock_context_manager(self):
        """A context manager for deployment lock acquisition.

        If the _enable_index_operations flag is set to True, the context manager tries to acquire the deployment lock.
            If the lock is acquired, the context manager yields control to the caller. Or if the lock is not acquired,
            it raises an OperationConflictError.
            If the lock is None, the context manager yields control to the caller without acquiring the lock.
        If the _enable_index_operations flag is set to False, the context manager raises an InternalError during
            index operations.

        Raises:
            OperationConflictError: If another index creation/deletion operation is
                in progress and the lock cannot be acquired
            InternalError: If index_management object is not enabled for index operations
        """
        if self._enable_index_operations:
            if self._zookeeper_deployment_lock is None:
                logger.warning(f"No Zookeeper client provided. "
                               f"Concurrent index operations may result in race conditions. ")
                yield  # No lock, proceed without locking
            else:
                try:
                    with self._zookeeper_deployment_lock:
                        yield
                except ZookeeperLockNotAcquiredError:
                    raise OperationConflictError("Another index creation/deletion operation is in progress. "
                                                 "Your request is rejected. Please try again later")
        else:
            raise InternalError("You index_management object is not enabled for index operations. ")
