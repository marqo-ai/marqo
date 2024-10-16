import os

from marqo import logging, version
from marqo.api import exceptions as api_exceptions
from marqo.core.index_management.index_management import IndexManagement, _MarqoConfig
from marqo.upgrades.upgrade import Upgrade
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient

logger = logging.get_logger(__name__)


class V2V0V2V1(Upgrade):
    """
    Upgrade from Marqo v2.0 to v2.1.

    The migration comprises two steps:
    1. Create the default query profile
    2. Add Marqo version to Marqo settings schema
    """

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management
        self.settings_schema = IndexManagement._MARQO_SETTINGS_SCHEMA_NAME
        self.config_id = IndexManagement._MARQO_CONFIG_DOC_ID
        self.default_query_profile = IndexManagement._DEFAULT_QUERY_PROFILE_TEMPLATE

    def run(self):
        logger.info("Running upgrade v20v21")

        try:
            logger.info("Creating query profile")
            self._create_query_profile()

            logger.info("Adding Marqo config")
            self._add_marqo_version()
        except Exception as e:
            raise Exception('Upgrade v20v21 failed. Partial changes may have been applied. This process is '
                            'idempotent. A successful run is required to bring Marqo into a consistent state') from e

        logger.info("Verifying upgrade")
        self._verify_query_profile()
        self._verify_marqo_version()

        logger.info("Upgrade v20v21 finished")

    def _create_query_profile(self):
        app = self.vespa_client.download_application()

        settings_schema_exists = os.path.exists(os.path.join(app, 'schemas', f'{self.settings_schema}.sd'))
        if not settings_schema_exists:
            raise api_exceptions.BadRequestError(f"Settings schema {self.settings_schema} does not exist. "
                                                 f"Has Marqo been bootstraped?")

        profile_path = os.path.join(app, 'search/query-profiles', 'default.xml')
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)
        with open(profile_path, 'w') as f:
            f.write(self.default_query_profile)

        self.vespa_client.deploy_application(app)
        self.vespa_client.wait_for_application_convergence()

    def _add_marqo_version(self):
        self.vespa_client.feed_document(
            VespaDocument(
                id=self.config_id,
                fields={
                    'settings': _MarqoConfig(version=version.get_version()).json()
                }
            ),
            schema=self.settings_schema
        )

    def _verify_query_profile(self):
        app = self.vespa_client.download_application()
        profile_path_exists = os.path.exists(os.path.join(app, 'search/query-profiles', 'default.xml'))
        if not profile_path_exists:
            raise api_exceptions.InternalError(
                f"Query profile does not exist. "
                f"Upgrade has not been applied correctly"
            )

    def _verify_marqo_version(self):
        configured_version = self.index_management.get_marqo_version()
        if configured_version != version.get_version():
            raise api_exceptions.InternalError(
                f"Marqo version in config is {configured_version}, but Marqo version is {version.get_version()}. "
                f"Upgrade has not been applied correctly"
            )
