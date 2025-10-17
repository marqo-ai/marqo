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


class Rollback(Upgrade, ABC):
    pass


class UpgradeRunner:
    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def upgrade(self):
        config_version_full = self.index_management.get_marqo_version()
        parsed_version = semver.VersionInfo.parse(config_version_full, optional_minor_and_patch=True)
        config_version = f'{parsed_version.major}.{parsed_version.minor}'

        upgrade = self._for_version(config_version)

        if upgrade is None:
            raise api_exceptions.BadRequestError(
                f'Cannot upgrade from Marqo version {config_version} to {version.get_version()}'
            )

        upgrade.run()

    def _for_version(self, from_version) -> Optional[Upgrade]:
        to_version_full = version.get_version()
        parsed_version = semver.VersionInfo.parse(to_version_full)
        to_version = f'{parsed_version.major}.{parsed_version.minor}'

        if from_version == "2.0" and to_version == '2.1':
            from marqo.upgrades.v2_v0_v2_v1 import V2V0V2V1
            return V2V0V2V1(self.vespa_client, self.index_management)

        return None


class RollbackRunner:
    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def rollback(self, from_version: str, to_version: str):
        parsed_from_version = semver.VersionInfo.parse(from_version, optional_minor_and_patch=True)
        parsed_to_version = semver.VersionInfo.parse(to_version, optional_minor_and_patch=True)

        rollback = self._for_versions(
            f'{parsed_from_version.major}.{parsed_from_version.minor}',
            f'{parsed_to_version.major}.{parsed_to_version.minor}'
        )

        if rollback is None:
            raise api_exceptions.BadRequestError(
                f'Cannot roll back from Marqo version {from_version} to {to_version}'
            )

        rollback.run()

    def _for_versions(self, from_version, to_version) -> Optional[Rollback]:
        if from_version == "2.1" and to_version == '2.0':
            from marqo.upgrades.v2_v1_v2_v0_rollback import V2V1V2V0Rollback
            return V2V1V2V0Rollback(self.vespa_client, self.index_management)

        return None
