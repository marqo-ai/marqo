from typing import Optional, Union

from kazoo.handlers.threading import KazooTimeoutError

from marqo.core.inference.device_manager import DeviceManager
from marqo.vespa.zookeeper_client import ZookeeperClient
from marqo.core.document.document import Document
from marqo.core.embed.embed import Embed
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.monitoring.monitoring import Monitoring
from marqo.core.search.recommender import Recommender
from marqo.logging import get_logger
from marqo.tensor_search import enums
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.vespa.vespa_client import VespaClient

logger = get_logger(__name__)


class Config:
    def __init__(
            self,
            vespa_client: VespaClient,
            zookeeper_client: Optional[ZookeeperClient] = None,
            default_device: Optional[str] = None,
            timeout: Optional[int] = None,
            backend: Optional[Union[enums.SearchDb, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        url:
            The url to the S2Search API (ex: http://localhost:9200)
        """
        self.vespa_client = vespa_client
        self.set_is_remote(vespa_client)
        self._zookeeper_client = zookeeper_client
        self._connect_to_zookeeper()

        self.timeout = timeout
        self.backend = backend if backend is not None else enums.SearchDb.vespa
        # TODO [Refactoring device logic] deprecate default_device since it's not used
        self.default_device = default_device if default_device is not None else (
            utils.read_env_vars_and_defaults(EnvVars.MARQO_BEST_AVAILABLE_DEVICE))

        # Initialize Core layer dependencies
        deployment_lock_timeout = utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_INDEX_DEPLOYMENT_LOCK_TIMEOUT)
        self.index_management = IndexManagement(vespa_client, zookeeper_client,
                                                enable_index_operations=True,
                                                deployment_lock_timeout_seconds=deployment_lock_timeout)

        self.monitoring = Monitoring(vespa_client, self.index_management)
        self.document = Document(vespa_client, self.index_management)
        self.recommender = Recommender(vespa_client, self.index_management)
        self.embed = Embed(vespa_client, self.index_management, self.default_device)
        self.device_manager = DeviceManager()

    def set_is_remote(self, vespa_client: VespaClient):
        local_host_markers = ["localhost", "0.0.0.0", "127.0.0.1"]

        if any(
                [
                    marker in url
                    for marker in local_host_markers
                    for url in [vespa_client.config_url, vespa_client.query_url, vespa_client.document_url]
                ]
        ):
            self.cluster_is_remote = False

    def _connect_to_zookeeper(self) -> None:
        """Try to connect to Zookeeper. If it fails, log a warning and continue."""
        if self._zookeeper_client is None:
            pass
        else:
            try:
                self._zookeeper_client.start()
            except KazooTimeoutError as e:
                logger.warning(f"Failed to connect to Zookeeper due to timeout. "
                               f"Marqo will still start but create/delete index operations will not work. "
                               f"Please check your Zookeeper configuration and network settings. "
                               f"You need to restart Marqo to connect to Zookeeper once you have fixed the issue. "
                               f"Original error message: {e}")
                pass

    def stop_and_close_zookeeper_client(self) -> None:
        """Stop and close the Zookeeper client."""
        if self._zookeeper_client is not None:
            self._zookeeper_client.stop()
            self._zookeeper_client.close()