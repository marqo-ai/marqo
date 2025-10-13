import os
import shutil

from marqo import logging
from marqo.api import exceptions as api_exceptions
from marqo.core.index_management.index_management import IndexManagement
from marqo.upgrades.upgrade import Rollback

logger = logging.get_logger(__name__)


class V2V1V2V0Rollback(Rollback):
    """
    Roll back from Marqo v2.1 to v2.0.

    The rollback comprises two steps:
    1. Delete the default query profile
    2. Remove Marqo version from Marqo settings schema
    """

    def __init__(self, vespa_client, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management
        self.settings_schema = IndexManagement._MARQO_SETTINGS_SCHEMA_NAME
        self.config_id = IndexManagement._MARQO_CONFIG_DOC_ID

    def run(self):
        logger.info("Running rollback v21v20")

        try:
            logger.info("Deleting query profile")
            self._delete_query_profile()

            logger.info("Removing Marqo config")
            self._remove_marqo_version()
        except Exception as e:
            raise Exception('Rollback v21v20 failed') from e

        logger.info("Verifying rollback")
        self._verify_query_profile()
        self._verify_marqo_version()

        logger.info("Rollback v21v20 finished")

    def _delete_query_profile(self):
        app = self.vespa_client.download_application()

        shutil.rmtree(os.path.join(app, 'search'), ignore_errors=True)

        self.vespa_client.deploy_application(app)
        self.vespa_client.wait_for_application_convergence()

    def _remove_marqo_version(self):
        self.vespa_client.delete_document(
            id=self.config_id,
            schema=self.settings_schema
        )

    def _verify_query_profile(self):
        app = self.vespa_client.download_application()
        profile_path_exists = os.path.exists(os.path.join(app, 'search/query-profiles', 'default.xml'))
        if profile_path_exists:
            raise api_exceptions.InternalError(
                "Query profile exists. Rollback has not been applied correctly"
            )

    def _verify_marqo_version(self):
        configured_version = self.index_management.get_marqo_version()
        if configured_version != '2.0':
            raise api_exceptions.InternalError(
                f"Marqo version in config is {configured_version}, expected 2.0. "
                f"Rollback has not been applied correctly"
            )
