from abc import ABC, abstractmethod
from typing import Optional

import semver

from marqo import version
from marqo.api import exceptions as api_exceptions
from marqo.core.index_management.index_management import IndexManagement
from marqo.vespa.vespa_client import VespaClient


class Upgrade(ABC):
    """
    Base class for a Marqo upgrade. Instances of this class represent a Marqo upgrade, and are responsible for
    performing the upgrade from one version to another.

    Upgrades must be idempotent, i.e. running an upgrade on a Marqo with the target version should have no effect.

    Source version must be forwards-compatible with the target state, so that a rolling upgrade can be performed.
    """

    @abstractmethod
    def run(self):
        pass


class UpgradeRunner:
    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def upgrade(self):
        config_version_full = self.index_management.get_marqo_version()
        parsed_version = semver.VersionInfo.parse(config_version_full)
        config_version = f'{parsed_version.major}.{parsed_version.minor}'

        upgrade = self._for_version(config_version)

        if upgrade is None:
            raise api_exceptions.BadRequestError(
                f'Cannot upgrade from Marqo version {config_version} to {version.get_version()}'
            )

        upgrade.run()

    def _for_version(self, from_version) -> Optional[Upgrade]:
        to_version = version.get_version()
        if from_version == "2.0" and to_version == '2.1':
            from marqo.upgrades.v20_v21 import v20v21
            return v20v21(self.vespa_client)

        return None
