from contextlib import contextmanager
from typing import List
from typing import Optional

import marqo.logging
import marqo.vespa.vespa_client
from marqo import version
from marqo.core import constants
from marqo.core.distributed_lock.zookeeper_distributed_lock import get_deployment_lock
from marqo.core.exceptions import IndexExistsError, IndexNotFoundError
from marqo.core.exceptions import OperationConflictError
from marqo.core.exceptions import ZookeeperLockNotAcquiredError, InternalError
from marqo.core.index_management.vespa_application_package import VespaApplicationPackage, MarqoConfig
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index_request import MarqoIndexRequest
from marqo.core.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.tensor_search.models.index_settings import IndexSettings
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.vespa_client import VespaClient
from marqo.vespa.zookeeper_client import ZookeeperClient

logger = marqo.logging.get_logger(__name__)


class IndexManagement:
    _MARQO_SETTINGS_SCHEMA_NAME = 'marqo__settings'
    _MARQO_CONFIG_DOC_ID = 'marqo__config'

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
        # FIXME should we set acquire_timeout?
        self._zookeeper_deployment_lock = get_deployment_lock(zookeeper_client) if zookeeper_client else None
        self._enable_index_operations = enable_index_operations

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

    def bootstrap_vespa(self) -> bool:
        """
        Add Marqo configuration to Vespa application package if an existing Marqo configuration is not detected.

        Returns:
            True if Vespa was bootstrapped, False if it was already up-to-date
        """
        with self._vespa_application_package_deployment() as app:
            marqo_version = version.get_version()
            has_marqo_settings_schema = app.has_schema(self._MARQO_SETTINGS_SCHEMA_NAME)
            marqo_config_doc = self._get_marqo_config() if has_marqo_settings_schema else None

            if app.need_bootstrapping(marqo_version, marqo_config_doc):
                existing_indexes = self._get_existing_indexes() if has_marqo_settings_schema else None
                app.bootstrap(marqo_version, existing_indexes)
                return True
            else:
                app.skip_deployment()
                return False

    def _get_marqo_config(self) -> Optional[MarqoConfig]:
        try:
            response = self.vespa_client.get_document(self._MARQO_CONFIG_DOC_ID, self._MARQO_SETTINGS_SCHEMA_NAME)
        except VespaStatusError as e:
            if e.status_code == 404:
                logger.warn(f'Marqo config document is not found in {self._MARQO_SETTINGS_SCHEMA_NAME}')
                return None
            raise e

        return MarqoConfig.parse_raw(response.document.fields['settings'])

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
        with self._vespa_application_package_deployment() as app:
            return self._create_one_index(app, marqo_index_request)

    @staticmethod
    def _create_one_index(app, marqo_index_request):
        if app.has_index(marqo_index_request.name):
            raise IndexExistsError(f"Index {marqo_index_request.name} already exists")

        # FIXME Ideally, this should be populated when used in inference
        if marqo_index_request.model.text_query_prefix is None:
            marqo_index_request.model.text_query_prefix = marqo_index_request.model.get_default_text_query_prefix()
        if marqo_index_request.model.text_chunk_prefix is None:
            marqo_index_request.model.text_chunk_prefix = marqo_index_request.model.get_default_text_chunk_prefix()

        schema, marqo_index = vespa_schema_factory(marqo_index_request).generate_schema()
        app.add_index_setting_and_schema(marqo_index, schema)
        return marqo_index

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
        with self._vespa_application_package_deployment() as app:
            return [self._create_one_index(app, request) for request in marqo_index_requests]

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
        with self._vespa_application_package_deployment() as app:
            app.delete_index_setting_and_schema(index_name)

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
        with self._vespa_application_package_deployment() as app:
            for index_name in index_names:
                app.delete_index_setting_and_schema(index_name)

    def _get_existing_indexes(self) -> List[MarqoIndex]:
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

    def get_all_indexes(self) -> List[MarqoIndex]:
        """
        Get all Marqo indexes.

        Returns:
            List of Marqo indexes
        """
        return self.vespa_client.get_all_index_settings()

    def get_index(self, index_name) -> MarqoIndex:
        """
        Get a Marqo index by name.

        Args:
            index_name: Name of Marqo index to get

        Returns:
            Marqo index
        """
        index = self.vespa_client.get_index_setting_by_name(index_name)
        if index is None:
            raise IndexNotFoundError(f"Index {index_name} not found")
        return index

    def get_marqo_version(self) -> str:
        """
        This method is only used during upgrade and rollback, it might not be needed anymore
        TODO check if this is still needed
        """
        with self._vespa_application_package_deployment() as app:
            app.skip_deployment()
            return app.get_marqo_config().version

    @contextmanager
    def _vespa_application_package_deployment(self):
        """A context manager that manages a vespa application deployment. .

        If the _enable_index_operations flag is set to True, the context manager tries to acquire the deployment lock.
            If the lock is acquired, the context manager yields the application downloaded from Vespa, and deploys
              the application to Vespa before exits.
            If the lock cannot be acquired before the timeout, it raises an OperationConflictError.
            If the lock is None, the context manager downloads and deploys an application without a lock. Please note
              that this might cause rase condition and is not recommended in production envs
        If the _enable_index_operations flag is set to False, the context manager raises an InternalError during
            index operations.

        Raises:
            OperationConflictError: If another index creation/deletion operation is
                in progress and the lock cannot be acquired
            InternalError: If index_management object is not enabled for index operations
        """
        if not self._enable_index_operations:
            raise InternalError("You index_management object is not enabled for index operations. ")

        if self._zookeeper_deployment_lock is None:
            logger.warning(f"No Zookeeper client provided. "
                           f"Concurrent index operations may result in race conditions. ")
            app = VespaApplicationPackage(self.vespa_client)
            yield app  # No lock, proceed without locking
            # we only deploy if no exception was raised
            app.deploy()
        else:
            try:
                with self._zookeeper_deployment_lock:
                    app = VespaApplicationPackage(self.vespa_client)
                    yield app
                    # we only deploy if no exception was raised
                    app.deploy()
            except ZookeeperLockNotAcquiredError:
                raise OperationConflictError("Another index creation/deletion operation is in progress. "
                                             "Your request is rejected. Please try again later")
