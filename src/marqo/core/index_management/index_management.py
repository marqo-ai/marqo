from contextlib import contextmanager
from typing import List, Tuple
from typing import Optional

import semver

import marqo.logging
import marqo.vespa.vespa_client
from marqo import version, marqo_docs
from marqo.core import constants
from marqo.core.distributed_lock.zookeeper_distributed_lock import get_deployment_lock
from marqo.core.exceptions import IndexNotFoundError, ApplicationNotInitializedError
from marqo.core.exceptions import OperationConflictError
from marqo.core.exceptions import ZookeeperLockNotAcquiredError, InternalError
from marqo.core.index_management.vespa_application_package import VespaApplicationPackage, VespaApplicationFileStore, \
    ApplicationPackageDeploymentSessionStore
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_index_request import MarqoIndexRequest
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.vespa_index.vespa_schema import for_marqo_index_request as vespa_schema_factory
from marqo.tensor_search.models.index_settings import IndexSettings
from marqo.vespa.vespa_client import VespaClient
from marqo.vespa.zookeeper_client import ZookeeperClient

logger = marqo.logging.get_logger(__name__)


class IndexManagement:
    _MINIMUM_VESPA_VERSION_TO_SUPPORT_UPLOAD_BINARY_FILES = semver.VersionInfo.parse('8.382.22')
    _MINIMUM_VESPA_VERSION_TO_SUPPORT_FAST_FILE_DISTRIBUTION = semver.VersionInfo.parse('8.396.18')
    _MARQO_SETTINGS_SCHEMA_NAME = 'marqo__settings'
    _MARQO_CONFIG_DOC_ID = 'marqo__config'

    def __init__(self,
                 vespa_client: VespaClient,
                 zookeeper_client: Optional[ZookeeperClient] = None,
                 enable_index_operations: bool = False,
                 deployment_timeout_seconds: int = 60,
                 convergence_timeout_seconds: int = 120,
                 deployment_lock_timeout: int = 0,
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

    @classmethod
    def validate_index_settings(cls, index_name: str, settings_dict: dict) -> None:
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
            vespa_app = self._get_vespa_application(check_configured=False, need_binary_file_support=True)

            to_version = version.get_version()
            from_version = vespa_app.get_marqo_config().version if vespa_app.is_configured else None

            if from_version and semver.VersionInfo.parse(from_version) >= semver.VersionInfo.parse(to_version):
                # skip bootstrapping if already bootstrapped to this version or later
                return False

            # Only retrieving existing index when the vespa app is not configured and the index settings schema exists
            existing_indexes = self._get_existing_indexes() if not vespa_app.is_configured and \
                vespa_app.has_schema(self._MARQO_SETTINGS_SCHEMA_NAME) else None

            vespa_app.bootstrap(to_version, existing_indexes)

            return True

    def rollback_vespa(self) -> None:
        with self._vespa_deployment_lock():
            self._get_vespa_application(need_binary_file_support=True).rollback(version.get_version())

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
            self._get_vespa_application().batch_add_index_setting_and_schema(index_to_create)

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
            self._get_vespa_application().batch_delete_index_setting_and_schema(index_names)

    def update_index(self, marqo_index: SemiStructuredMarqoIndex) -> None:
        """
        Update index settings and schema
        Aars:
            marqo_index: Index to update, only SemiStructuredMarqoIndex is supported
        Raises:
            IndexNotFoundError: If an index does not exist
            InternalError: If the index is not a SemiStructuredMarqoIndex.
            RuntimeError: If deployment lock is not instantiated
            OperationConflictError: If another index creation/deletion operation is
                in progress and the lock cannot be acquired
        """
        if not isinstance(marqo_index, SemiStructuredMarqoIndex):
            # This is just a sanity check, it should not happen since we do not expose this method to end user.
            raise InternalError(f'Index {marqo_index.name} can not be updated.')

        with self._vespa_deployment_lock():
            schema = SemiStructuredVespaSchema.generate_vespa_schema(marqo_index)
            self._get_vespa_application().update_index_setting_and_schema(marqo_index, schema)

    def _get_existing_indexes(self) -> List[MarqoIndex]:
        """
        Get all Marqo indexes storing in _MARQO_SETTINGS_SCHEMA_NAME schema (used prior to Marqo v2.13.0).
        This method is now only used to retrieve the existing indexes for bootstrapping from v2.13.0

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
        This method is only used during legacy upgrade and rollback process. Please note that this will create a
        Vespa deployment session and download the margo_config json from the Vespa config server. If we need to
        retrieve this information more often in the future, consider exposing it from Vespa container.

        Returns:
            The marqo version stored in the vespa application package
        """
        return self._get_vespa_application().get_marqo_config().version

    def _get_vespa_application(self, check_configured: bool = True, need_binary_file_support: bool = False) \
            -> VespaApplicationPackage:
        """
        Retrieve a Vespa application package. Depending on whether we need to handle binary files and the Vespa version,
        it uses different implementation of VespaApplicationStore.

        Args:
            check_configured: if set to True, it checks whether the application package is configured or not.
            need_binary_file_support: indicates whether the support for binary file is needed.

        Returns:
            The VespaApplicationPackage instance we can use to do bootstrapping/rollback and any index operations.
        """
        vespa_version = semver.VersionInfo.parse(self.vespa_client.get_vespa_version())

        if vespa_version < self._MINIMUM_VESPA_VERSION_TO_SUPPORT_UPLOAD_BINARY_FILES:
            # Please note that this warning message will only be logged out for OS users running Marqo on external
            # Vespa servers with version prior to 8.382.22. This will be displayed when Marqo starts up and before
            # each index CUD operation
            logger.warning(f'Your Vespa version {vespa_version} is lower than the minimum recommended Vespa version '
                           f'{self._MINIMUM_VESPA_VERSION_TO_SUPPORT_FAST_FILE_DISTRIBUTION}. This could cause '
                           f'unexpected behavior when bootstrapping Marqo. Please upgrade '
                           f'Vespa to version {self._MINIMUM_VESPA_VERSION_TO_SUPPORT_FAST_FILE_DISTRIBUTION} or '
                           f'later. Please see {marqo_docs.troubleshooting()} for more details.')

        if vespa_version < self._MINIMUM_VESPA_VERSION_TO_SUPPORT_FAST_FILE_DISTRIBUTION:
            # Please note that this warning message will only be logged out for OS users running Marqo on external
            # Vespa servers with version prior to 8.396.18. This will be displayed when Marqo starts up and before
            # each index CUD operation
            logger.warning(f'Your Vespa version {vespa_version} is lower than the minimum recommended Vespa version '
                           f'{self._MINIMUM_VESPA_VERSION_TO_SUPPORT_FAST_FILE_DISTRIBUTION}. You may encounter slower '
                           f'response times when creating a Marqo index or adding documents to unstructured indexes. '
                           f'Please upgrade Vespa to version {self._MINIMUM_VESPA_VERSION_TO_SUPPORT_FAST_FILE_DISTRIBUTION} or '
                           f'later. Please see {marqo_docs.troubleshooting()} for more details.')

        if need_binary_file_support and vespa_version < self._MINIMUM_VESPA_VERSION_TO_SUPPORT_UPLOAD_BINARY_FILES:
            # Binary files are only supported using VespaApplicationFileStore prior to Vespa version 8.382.22
            application_package_store = VespaApplicationFileStore(
                vespa_client=self.vespa_client,
                deploy_timeout=self._deployment_timeout_seconds,
                wait_for_convergence_timeout=self._convergence_timeout_seconds
            )
        else:
            application_package_store = ApplicationPackageDeploymentSessionStore(
                vespa_client=self.vespa_client,
                deploy_timeout=self._deployment_timeout_seconds,
                wait_for_convergence_timeout=self._convergence_timeout_seconds
            )

        application = VespaApplicationPackage(application_package_store)

        if check_configured and not application.is_configured:
            raise ApplicationNotInitializedError()

        return application

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
