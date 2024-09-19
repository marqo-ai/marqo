from contextlib import contextmanager
from typing import List, Tuple
from typing import Optional

import semver

import marqo.logging
import marqo.vespa.vespa_client
from marqo import version
from marqo.core import constants
from marqo.core.distributed_lock.zookeeper_distributed_lock import get_deployment_lock
from marqo.core.exceptions import IndexNotFoundError, ApplicationNotInitializedError, ApplicationRollbackError
from marqo.core.exceptions import OperationConflictError
from marqo.core.exceptions import ZookeeperLockNotAcquiredError, InternalError
from marqo.core.index_management.vespa_application_package import VespaApplicationPackage, MarqoConfig, \
    VespaApplicationFileStore, ApplicationPackageDeploymentSessionStore
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_index_request import MarqoIndexRequest
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.tensor_search.models.index_settings import IndexSettings
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.vespa_client import VespaClient
from marqo.vespa.zookeeper_client import ZookeeperClient

logger = marqo.logging.get_logger(__name__)


class IndexManagement:
    MINIMUM_VESPA_VERSION_TO_SUPPORT_UPLOAD_BINARY_FILES = semver.VersionInfo.parse('8.382.22')
    _MARQO_SETTINGS_SCHEMA_NAME = 'marqo__settings'
    _MARQO_CONFIG_DOC_ID = 'marqo__config'

    def __init__(self,
                 vespa_client: VespaClient,
                 zookeeper_client: Optional[ZookeeperClient] = None,
                 enable_index_operations: bool = False,
                 deployment_timeout_seconds = 60,
                 convergence_timeout_seconds = 120,
                 deployment_lock_timeout = 0,
                 ):
        """Instantiate an IndexManagement object.

        Args:
            vespa_client: VespaClient object
            zookeeper_client: ZookeeperClient object
            enable_index_operations: A flag to enable index operations. If set to True,
                the object can create/delete indexes, otherwise, it raises an InternalError during index operations.
        """
        self.vespa_client = vespa_client
        self._zookeeper_deployment_lock = get_deployment_lock(zookeeper_client, deployment_lock_timeout) \
            if zookeeper_client else None
        self._enable_index_operations = enable_index_operations
        self._deployment_timeout_seconds = deployment_timeout_seconds
        self._convergence_timeout_seconds = convergence_timeout_seconds

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
        with self._vespa_deployment_lock():
            application = self._get_application_context_manager_for_bootstrapping_and_rollback(check_configured=False)
            with application as app:

                marqo_version = version.get_version()
                has_marqo_settings_schema = app.has_schema(self._MARQO_SETTINGS_SCHEMA_NAME)
                marqo_config_doc = self._get_marqo_config() if has_marqo_settings_schema else None

                if not app.need_bootstrapping(marqo_version, marqo_config_doc):
                    application.gen.send(False)  # tell context manager to skip deployment
                    return False

                existing_indexes = self._get_existing_indexes() if has_marqo_settings_schema else ()
                app.bootstrap(marqo_version, existing_indexes)
                return True

    def rollback_vespa(self) -> None:
        with self._vespa_deployment_lock():
            application = self._get_application_context_manager_for_bootstrapping_and_rollback()
            with application as app:
                try:
                    app.rollback(version.get_version())
                except ApplicationRollbackError as e:
                    logger.error(e.message)
                    application.gen.send(False)  # tell context manager to skip deployment
                    raise e

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
        return self.batch_create_indexes([marqo_index_request])[0]

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
        index_to_create: List[Tuple[str, MarqoIndex]] = []
        for request in marqo_index_requests:
            # set the default prefixes if not provided
            if request.model.text_query_prefix is None:
                request.model.text_query_prefix = request.model.get_default_text_query_prefix()
            if request.model.text_chunk_prefix is None:
                request.model.text_chunk_prefix = request.model.get_default_text_chunk_prefix()

            index_to_create.append(vespa_schema_factory(request).generate_schema())

        with self._vespa_deployment_lock():
            with self._vespa_application_with_deployment_session() as app:
                app.batch_add_index_setting_and_schema(index_to_create)

        return [index for _, index in index_to_create]

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
        self.batch_delete_indexes_by_name([index_name])

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
        with self._vespa_deployment_lock():
            with self._vespa_application_with_deployment_session() as app:
                app.batch_delete_index_setting_and_schema(index_names)

    def update_index(self, marqo_index: SemiStructuredMarqoIndex):
        with self._vespa_deployment_lock():
            with self._vespa_application_with_deployment_session() as app:
                schema = SemiStructuredVespaSchema.generate_vespa_schema(marqo_index)
                app.update_index_setting_and_schema(marqo_index, schema)

    def _get_existing_indexes(self) -> List[MarqoIndex]:
        """
        Get all Marqo indexes storing in _MARQO_SETTINGS_SCHEMA_NAME schema (used prior to Marqo v2.12.0).
        This method is now only used to retrieve the existing indexes for bootstrapping from v2.12.0

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

    def _get_marqo_config(self) -> Optional[MarqoConfig]:
        """
        We store Marqo config in _MARQO_CONFIG_DOC_ID doc prior to Marqo v2.12.0
        This method is now only used to retrieve the existing marqo config for bootstrapping
        """
        try:
            response = self.vespa_client.get_document(self._MARQO_CONFIG_DOC_ID, self._MARQO_SETTINGS_SCHEMA_NAME)
        except VespaStatusError as e:
            if e.status_code == 404:
                logger.warn(f'Marqo config document is not found in {self._MARQO_SETTINGS_SCHEMA_NAME}')
                return None
            raise e

        return MarqoConfig.parse_raw(response.document.fields['settings'])

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
        This method is only used during upgrade and rollback
        TODO check if this is still needed
        """
        application = self._vespa_application_with_deployment_session()
        with application as app:
            marqo_version = app.get_marqo_config().version
            application.gen.send(False)  # do not deploy
            return marqo_version

    @contextmanager
    def _vespa_application(self, check_configured: bool = True):
        """
        A context manager that handles the deployment of a Vespa application package by downloading
        all contents in one session and deploying the updated app in another session.
        This is only used for bootstrapping. A better way is to use _vespa_application_with_deployment_session.
        We need to upload a component jar file during bootstrapping. Due to a bug in Vespa, the better
        approach does not support uploading binary files. Therefore, we still need this method.
        With the protection of the distributed lock context manager, we would not run into race condition, unless
        the connection to zookeeper is lost, and more than one instance holds the lock. This should happen very rarely.
        """
        app_root_path = self.vespa_client.download_application(check_for_application_convergence=True)
        app = VespaApplicationPackage(VespaApplicationFileStore(app_root_path))

        if check_configured and not app.is_configured:
            raise ApplicationNotInitializedError()

        should_deploy = yield app

        if should_deploy is None or should_deploy is True:
            self.vespa_client.deploy_application(app_root_path, timeout=self._deployment_timeout_seconds)
            self.vespa_client.wait_for_application_convergence(timeout=self._convergence_timeout_seconds)

        if should_deploy is not None:
            yield

    @contextmanager
    def _vespa_application_with_deployment_session(self, check_configured: bool = True):
        """
        A context manager that handles the deployment of a Vespa application package in a deployment session
        This is a recommended way to deploy a Vespa application package. It leverages the optimistic locking mechanism
        to avoid race conditions. Changes to the application package that do not touch any binary files should use
        this context manager for deployment.
        """
        content_base_url, prepare_url = self.vespa_client.create_deployment_session()
        store = ApplicationPackageDeploymentSessionStore(content_base_url, self.vespa_client)
        app = VespaApplicationPackage(store)

        if check_configured and not app.is_configured:
            raise ApplicationNotInitializedError()

        should_deploy = yield app

        if should_deploy is None or should_deploy is True:
            prepare_response = self.vespa_client.prepare(prepare_url)
            # TODO handle prepare configChangeActions
            # https://docs.vespa.ai/en/reference/deploy-rest-api-v2.html#prepare-session
            self.vespa_client.activate(prepare_response['activate'])
            self.vespa_client.wait_for_application_convergence(timeout=self._convergence_timeout_seconds)

        if should_deploy is not None:
            yield

    def _get_application_context_manager_for_bootstrapping_and_rollback(self, check_configured: bool = True):
        vespa_version = semver.VersionInfo.parse(self.vespa_client.get_vespa_version())

        if vespa_version < self.MINIMUM_VESPA_VERSION_TO_SUPPORT_UPLOAD_BINARY_FILES:
            return self._vespa_application(check_configured=check_configured)
        else:
            return self._vespa_application_with_deployment_session(check_configured=check_configured)

    @contextmanager
    def _vespa_deployment_lock(self):
        """A context manager that manages an optional distributed lock.

        If the _enable_index_operations flag is set to True, the context manager tries to acquire the deployment lock.
            If the lock is acquired, the context manager yields
            If the lock cannot be acquired before the timeout, it raises an OperationConflictError.
            If the lock is None, the context manager yields without locking
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
            yield  # No lock, proceed without locking
        else:
            try:
                with self._zookeeper_deployment_lock:
                    logger.info(f"Retrieved the distributed lock for index operations. ")
                    yield
            except ZookeeperLockNotAcquiredError:
                raise OperationConflictError("Another index creation/deletion operation is in progress. "
                                             "Your request is rejected. Please try again later")

