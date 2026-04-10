import difflib
import json
import semver
from contextlib import contextmanager
from typing import List, Tuple, Dict, Any, Optional

import marqo.logging
import marqo.vespa.vespa_client
from marqo import version, marqo_docs
from marqo.core import constants
from marqo.core.distributed_lock.zookeeper_distributed_lock import get_deployment_lock
from marqo.core.exceptions import IndexNotFoundError, ApplicationNotInitializedError, UnsupportedFeatureError, \
    InvalidModelPropertiesError, OperationConflictError, ZookeeperLockNotAcquiredError, InternalError
from marqo.core.index_management.vespa_application_package import VespaApplicationPackage, VespaApplicationFileStore, \
    ApplicationPackageDeploymentSessionStore
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_index_request import MarqoIndexRequest
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.typeahead.typeahead_vespa_schema import TypeaheadVespaSchema
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
    _ALLOWED_MODIFIED_SETTINGS = {"modelProperties"}

    def __init__(self,
                 vespa_client: VespaClient,
                 zookeeper_client: Optional[ZookeeperClient] = None,
                 enable_index_operations: bool = False,
                 deployment_timeout_seconds: int = 60,
                 convergence_timeout_seconds: int = 120,
                 deployment_lock_timeout_seconds: float = 5,
                 ):
        """Instantiate an IndexManagement object.

        Args:
            vespa_client: VespaClient object
            zookeeper_client: ZookeeperClient object
            enable_index_operations: A flag to enable index operations. If set to True,
                the object can create/delete indexes, otherwise, it raises an InternalError during index operations.
            deployment_timeout_seconds: Vespa deployment timeout in seconds
            convergence_timeout_seconds: Vespa convergence timeout in seconds
            deployment_lock_timeout_seconds: Vespa deployment lock timeout in seconds
        """
        self.vespa_client = vespa_client
        self._zookeeper_deployment_lock = get_deployment_lock(zookeeper_client, deployment_lock_timeout_seconds) \
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

        # We skip the Vespa convergence check here so that Marqo instance can be bootstrapped even when Vespa is
        # not converged.
        to_version = version.get_version()
        vespa_app_for_version_check = self._get_vespa_application(check_configured=False, need_binary_file_support=True,
                                                                  check_for_application_convergence=False)
        from_version = vespa_app_for_version_check.get_marqo_config().version \
            if vespa_app_for_version_check.is_configured else None

        if from_version and semver.VersionInfo.parse(from_version) >= semver.VersionInfo.parse(to_version):
            # skip bootstrapping if already bootstrapped to this version or later
            return False

        with self._vespa_deployment_lock():
            # Initialise another session based on the latest active Vespa session. The reason we do this again while
            # holding the distributed lock is that the Vespa application might be changed by other operations when
            # we wait for the lock. This time, we error out if the Vespa application is not converged, which reduces
            # the chance of running into race conditions.
            vespa_app = self._get_vespa_application(check_configured=False, need_binary_file_support=True,
                                                    check_for_application_convergence=True)

            # Only retrieving existing index when the vespa app is not configured and the index settings schema exists
            existing_indexes = self._get_existing_indexes() if not vespa_app.is_configured and \
                                                               vespa_app.has_schema(
                                                                   self._MARQO_SETTINGS_SCHEMA_NAME) else None

            vespa_app.bootstrap(to_version, existing_indexes)

            return True

    def rollback_vespa(self) -> None:
        """
        Roll back Vespa application package to the previous version backed up in the current app package.
        """
        with self._vespa_deployment_lock():
            vespa_app = self._get_vespa_application(need_binary_file_support=True)
            vespa_app.rollback(version.get_version())

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
        index_to_create: List[Tuple[str, str, MarqoIndex]] = []

        for request in marqo_index_requests:
            # set the default prefixes if not provided
            if request.model.text_query_prefix is None:
                request.model.text_query_prefix = request.model.get_default_text_query_prefix()
            if request.model.text_chunk_prefix is None:
                request.model.text_chunk_prefix = request.model.get_default_text_chunk_prefix()

            schema, marqo_index = vespa_schema_factory(request).generate_schema()
            logger.debug(f'Creating index {request.name} with schema:\n{schema}')

            typeahead_schema, updated_marqo_index = TypeaheadVespaSchema(marqo_index).generate_schema()
            logger.debug(
                f'Creating typeahead schema for index {request.name} with schema: '
                f'{updated_marqo_index.typeahead_schema_name}'
            )

            index_to_create.append((schema, typeahead_schema, updated_marqo_index))

        with self._vespa_deployment_lock():
            vespa_app = self._get_vespa_application()

            # Deploy schemas and index settings (this will deploy everything together)
            vespa_app.batch_add_index_setting_and_schema(index_to_create)

        return [index for _, _, index in index_to_create]

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

    def update_index_settings_by_settings_dict(
            self, index_name: str, settings_dict: dict,
            force: bool = False, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Update index settings by settings dict. No schema update. Currently only modelProperties can be updated.

        When calling this method, you must consider the scenario distributed Marqo instances. Some Marqo instances
        could still be running the old version so the updated modelProperties must be compatible with the old version.

        This method:
        1. Retrieves the existing index
        2. Validates the updated settings
        3. Behavior based on parameters:
           - dry_run=True: Show changes, never deploy
           - dry_run=False, force=False: Deploy only if validation passes
           - dry_run=False, force=True: Always deploy, despite validation errors

        Args:
            index_name: Name of the index to update
            settings_dict: Settings dict to update the index, currently only modelProperties can be updated.
            force: If True, skip validation and proceed with update. Default is False.
            dry_run: If True, show what would change without applying the update. Default is False.

        Returns:
            Dict with update status:
            {
                "updated": bool,                    # Whether settings were actually deployed
                "error": bool,                      # Whether there was a validation error
                "oldSettings": dict,                # Current settings (for updated fields only)
                "newSettings": dict,                # Proposed new settings (for updated fields only)
                "settingsDiff": str,                # Unified diff between old and new settings
                "reason": str,                      # Explanation of the result
            }

        Raises:
            IndexNotFoundError: If an index does not exist
            OperationConflictError: If deployment lock cannot be acquired
        """
        if not set(settings_dict.keys()).issubset(self._ALLOWED_MODIFIED_SETTINGS):  # pragma: no cover
            # Should not happen since we validate the settings in the API layer
            raise InternalError(f"Only the following settings can be updated: {self._ALLOWED_MODIFIED_SETTINGS}. "
                                f"Provided settings: {list(settings_dict.keys())}")

        with self._vespa_deployment_lock():
            existing_index = self.get_index(index_name)
            updated_version = existing_index.version + 1 if existing_index.version is not None else 1
            updated_index = existing_index.copy(deep=True, update={'version': updated_version})

            # Build old and new settings for comparison
            old_settings = {}
            new_settings = {}

            if "modelProperties" in settings_dict:
                old_settings["modelProperties"] = existing_index.model.properties
                new_settings["modelProperties"] = settings_dict["modelProperties"]

            # Check if settings actually changed
            settings_changed = old_settings != new_settings

            # Generate settings diff
            old_settings_json = json.dumps(old_settings, indent=2, sort_keys=True)
            new_settings_json = json.dumps(new_settings, indent=2, sort_keys=True)
            old_settings_lines = old_settings_json.splitlines(keepends=True)
            new_settings_lines = new_settings_json.splitlines(keepends=True)
            diff_lines = list(difflib.unified_diff(
                old_settings_lines,
                new_settings_lines,
                fromfile='old_settings',
                tofile='new_settings',
                lineterm=''
            ))
            settings_diff = ''.join(diff_lines) if diff_lines else ''

            # Initialize response template
            result = {
                "updated": False,
                "error": False,
                "oldSettings": old_settings,
                "newSettings": new_settings,
                "settingsDiff": settings_diff,
                "reason": "",
            }

            # Scenario 1: No changes needed
            if not settings_changed:
                logger.info(f'Settings for index {index_name} are already up to date')
                result["reason"] = "Settings are already up to date"
                return result

            # Validate unless force=True
            validation_error = None
            try:
                if "modelProperties" in settings_dict:
                    self.validate_updated_model_properties(
                        existing_index.model.properties,
                        settings_dict["modelProperties"]
                    )
            except InvalidModelPropertiesError as e:
                validation_error = e
                result["error"] = True

            # Scenario 2: dry_run=True - never deploy, just return info
            if dry_run:
                logger.info(f'Dry run for index {index_name} - showing settings changes without deploying')
                if validation_error:
                    result["reason"] = f"Dry run - validation would fail: {str(validation_error)}"
                else:
                    result["reason"] = "Dry run - no changes deployed"
                return result

            # Scenario 3: dry_run=False, force=False - block if validation fails
            if validation_error and not force:
                result["reason"] = "Validation failed: " + str(validation_error)
                return result

            # Scenario 4: dry_run=False, force=True OR validation passed - proceed with deployment
            if "modelProperties" in settings_dict:
                updated_index = self._updated_index_with_model_properties(updated_index, settings_dict["modelProperties"])

            logger.debug(f'Updating index {updated_index.name} with settings: {settings_dict}')
            self._get_vespa_application().update_index_setting(updated_index)
            logger.info(f'Successfully updated settings for index {index_name}')

            result["updated"] = True
            if force and validation_error:
                result["reason"] = "Update forced despite validation errors: " + str(validation_error)
            else:
                result["reason"] = "Settings updated successfully"
            from marqo.tensor_search import index_meta_cache
            index_meta_cache.get_index(self, index_name, force_refresh=True)  # Refresh cache
            return result

    def _updated_index_with_model_properties(self, index: MarqoIndex, model_properties: dict) -> MarqoIndex:
        """
        Create a new MarqoIndex object with updated model properties.

        Args:
            index: The index object to update.
            model_properties: A dictionary of model properties to update.

        Returns:
            A new MarqoIndex object with updated model properties.
        """
        index.model.properties = model_properties
        index.model.custom = True  # Mark the model as custom if model properties are updated
        return index

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
        # !!! Please note that we need to acquire the lock before retrieving the index setting so that we know the
        # index setting we get is up-to-date. If another process is updating the index, we will wait until it finishes
        with self._vespa_deployment_lock():
            existing_index = self.get_index(marqo_index.name)
            if not isinstance(existing_index, SemiStructuredMarqoIndex):
                # This is just a sanity check, it should not happen since we do not expose this method to end user.
                raise InternalError(f'Index {marqo_index.name} created by Marqo version {marqo_index.marqo_version} '
                                    f'can not be updated.')

            def is_subset(dict_a, dict_b):
                # check if dict_a is a subset of dict_b
                return all(k in dict_b and dict_b[k] == v for k, v in dict_a.items())

            if (is_subset(marqo_index.tensor_field_map, existing_index.tensor_field_map) and
                    is_subset(marqo_index.field_map, existing_index.field_map) and
                    is_subset(marqo_index.name_to_string_array_field_map, existing_index.name_to_string_array_field_map)):
                logger.debug(f'Another thread has updated the index {marqo_index.name} already.')
                return

            schema = SemiStructuredVespaSchema.generate_vespa_schema(marqo_index)
            logger.debug(f'Updating index {marqo_index.name} with schema:\n{schema}')
            self._get_vespa_application().update_index_setting_and_schema(marqo_index, schema)

    def apply_latest_schema_template(self, index_name: str, force: bool = False, dry_run: bool = False) -> Dict[str, any]:
        """
        Update an index's main schema to the latest template version.

        This method:
        1. Retrieves the existing index
        2. Generates a new schema from the latest template
        3. Compares it with the current deployed schema
        4. If different, prepares the deployment
        5. Checks Vespa's configChangeActions
        6. Behavior based on parameters:
           - dry_run=True: Show diff and actions, never deploy
           - dry_run=False, force=False: Deploy only if no configChangeActions
           - dry_run=False, force=True: Always deploy

        Args:
            index_name: Name of the index to update
            force: If True, proceed with update even if configChangeActions are required
            dry_run: If True, show schema diff and configChangeActions without deploying

        Returns:
            Dict with update status:
            {
                "updated": bool,                # Whether schema was actually deployed
                "schemaChanged": bool,          # Whether generated schema differs from current
                "oldSchema": str,               # Current deployed schema
                "newSchema": str,               # Proposed/generated schema
                "schemaDiff": str,              # Unified diff between old and new
                "reason": str,                  # Explanation of the result
                "configChangeActions": {}       # Vespa configChangeActions if any
            }

        Raises:
            IndexNotFoundError: If index doesn't exist
            InternalError: If index type doesn't support schema updates
            UnsupportedFeatureError: If index was created with Marqo < 2.23.0
            OperationConflictError: If deployment lock cannot be acquired
        """
        with self._vespa_deployment_lock():
            # Get existing index
            existing_index = self.get_index(index_name)

            # Only SemiStructuredMarqoIndex supports schema regeneration
            if not isinstance(existing_index, SemiStructuredMarqoIndex):
                raise InternalError(
                    f'Index {index_name} is type {existing_index.type}, '
                    f'only semi-structured indexes support schema updates'
                )

            # Check minimum version requirement
            if existing_index.parsed_marqo_version() < constants.MARQO_UPDATE_SCHEMA_MINIMUM_VERSION:
                raise UnsupportedFeatureError(
                    f"Schema update is only supported for indexes created with Marqo "
                    f"{str(constants.MARQO_UPDATE_SCHEMA_MINIMUM_VERSION)} or later. "
                    f"This index was created with Marqo {existing_index.marqo_version}. "
                    f"Please recreate the index with a newer version of Marqo to use this feature."
                )

            # Validate that index's marqo_version is not greater than current version
            current_version_parsed = semver.VersionInfo.parse(version.get_version())
            if existing_index.parsed_marqo_version() > current_version_parsed:
                raise InternalError(
                    f"Cannot update schema for index '{index_name}' created with Marqo version "
                    f"{existing_index.marqo_version} using current Marqo version {version.get_version()}. "
                    f"The index was created with a newer version of Marqo than is currently running."
                )

            # Early return if schema is already at current version
            if existing_index.schema_template_version == version.get_version():
                logger.info(f'Index {index_name} schema is already at version {version.get_version()}')
                return {
                    "updated": False,
                    "schemaChanged": False,
                    "reason": f"Schema is already at current Marqo version {version.get_version()}"
                }

            # Generate new schema from current settings using latest template
            new_schema = SemiStructuredVespaSchema.generate_vespa_schema(existing_index)

            # Get current deployed schema
            vespa_app = self._get_vespa_application()
            current_schema = vespa_app.get_schema(existing_index.schema_name)

            # Generate schema diff (always, for all scenarios)
            current_schema_lines = (current_schema or '').splitlines(keepends=True)
            new_schema_lines = new_schema.splitlines(keepends=True)
            diff_lines = list(difflib.unified_diff(
                current_schema_lines,
                new_schema_lines,
                fromfile='old_schema',
                tofile='new_schema',
                lineterm=''
            ))
            schema_diff = ''.join(diff_lines) if diff_lines else 'No changes'

            # Check if schemas are identical (based on diff)
            schemas_identical = len(diff_lines) == 0

            # Initialize response template with common fields
            result = {
                "updated": False,
                "schemaChanged": not schemas_identical,
                "oldSchema": current_schema or '',
                "newSchema": new_schema,
                "schemaDiff": schema_diff,
                "configChangeActions": {},
            }

            # Scenario 1: Schemas are identical - no changes needed
            if schemas_identical:
                logger.info(f'Schema for index {index_name} is already up to date')
                result["reason"] = "Schema is already up to date"
                return result

            # Schema is different - prepare deployment to get configChangeActions
            logger.info(f'Schema for index {index_name} has changes, preparing deployment')
            prepare_response = vespa_app.update_index_setting_and_schema(
                existing_index,
                new_schema,
                prepare_only=True
            )

            # Extract configChangeActions from prepare response
            config_change_actions = prepare_response.get('configChangeActions', {})
            result["configChangeActions"] = config_change_actions

            # Check if there are any required actions
            has_actions = bool(config_change_actions.get('restart') or
                               config_change_actions.get('refeed') or
                               config_change_actions.get('reindex'))

            # Scenario 2: dry_run=True - never deploy, just return info
            if dry_run:
                logger.info(f'Dry run for index {index_name} - showing schema diff without deploying')
                result["reason"] = "Dry run - no changes deployed"
                return result

            # Scenario 3: dry_run=False, force=False - block if actions required
            if has_actions and not force:
                logger.warning(f'Schema update for index {index_name} requires Vespa actions: {config_change_actions}')
                result["reason"] = "Vespa requires manual actions before proceeding. Use force=true to proceed anyway."
                return result

            # Scenario 4: dry_run=False, force=True OR no actions - proceed with deployment
            vespa_app.activate_prepared_deployment(prepare_response)
            logger.info(f'Successfully updated schema for index {index_name}')

            result["updated"] = True
            if has_actions:
                result["reason"] = "Update forced despite required actions"
            else:
                result["reason"] = "Schema updated successfully"

            return result

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

    def _get_vespa_application(self, check_configured: bool = True, need_binary_file_support: bool = False,
                               check_for_application_convergence: bool = True) -> VespaApplicationPackage:
        """
        Retrieve a Vespa application package. Depending on whether we need to handle binary files and the Vespa version,
        it uses different implementation of VespaApplicationStore.

        Args:
            check_configured: if set to True, it checks whether the application package is configured or not.
            need_binary_file_support: indicates whether the support for binary file is needed.
            check_for_application_convergence: whether we check convergence of the Vespa app package. If set to true and
              Vespa is not converged, this process will fail with a VespaError raised.

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
                wait_for_convergence_timeout=self._convergence_timeout_seconds,
                check_for_application_convergence=check_for_application_convergence
            )
        else:
            application_package_store = ApplicationPackageDeploymentSessionStore(
                vespa_client=self.vespa_client,
                deploy_timeout=self._deployment_timeout_seconds,
                wait_for_convergence_timeout=self._convergence_timeout_seconds,
                check_for_application_convergence=check_for_application_convergence
            )

        application = VespaApplicationPackage(application_package_store)

        if check_configured and not application.is_configured:
            raise ApplicationNotInitializedError()

        return application

    @staticmethod
    def validate_updated_model_properties(current_model_properties: dict, updated_model_properties: dict) -> None:
        """
        Validate the updated model properties to ensure compatibility with the current model properties.

        Args:
            current_model_properties: current model properties, as a dict
            updated_model_properties: updated model properties, as a dict

        Returns:
            None

        Raises:
            InvalidModelPropertiesError: If the updated model properties is not compatible with the current model properties.
        """
        must_unchanged_keys = ["dimensions", "type"]

        for key in must_unchanged_keys:
            if current_model_properties.get(key) != updated_model_properties.get(key):
                raise InvalidModelPropertiesError(
                    f"Updating model properties resulting in change of '{key}' is not allowed. "
                    f"Current '{key}': {current_model_properties.get(key)}, "
                    f"updated '{key}': {updated_model_properties.get(key)} "
                )

        current_keys = set(current_model_properties.keys())
        updated_keys = set(updated_model_properties.keys())

        if not current_keys.issubset(updated_keys):
            raise InvalidModelPropertiesError(
                f"The updated model properties must contain all keys in the current model properties for compatibility. "
                f"Current model properties keys: {current_keys}, updated model properties keys: {updated_keys} "
            )

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
                    logger.debug(f"Retrieved the distributed lock for index operations. ")
                    yield
            except ZookeeperLockNotAcquiredError:
                # TODO add a doclink for troubleshooting this issue
                raise OperationConflictError("Your indexes are being updated. Please try again shortly.")
