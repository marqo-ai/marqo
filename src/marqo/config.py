from typing import Optional, Union

from marqo.core.document.document import Document
from marqo.core.embed.embed import Embed
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.monitoring.monitoring import Monitoring
from marqo.core.search.recommender import Recommender
from marqo.tensor_search import enums
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.vespa.vespa_client import VespaClient
from kazoo.client import KazooClient
from kazoo.handlers.threading import KazooTimeoutError
from marqo.logging import get_logger

logger = get_logger(__name__)


class Config:
    def __init__(
            self,
            vespa_client: VespaClient,
            zookeeper_client: Optional[KazooClient] = None,
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
        self.zookeeper_client = self._set_zookeeper(zookeeper_client)

        self.timeout = timeout
        self.backend = backend if backend is not None else enums.SearchDb.vespa
        self.default_device = default_device if default_device is not None else (
            utils.read_env_vars_and_defaults(EnvVars.MARQO_BEST_AVAILABLE_DEVICE))

        # Initialize Core layer dependencies
        self.index_management = IndexManagement(vespa_client, zookeeper_client)
        self.monitoring = Monitoring(vespa_client, self.index_management)
        self.document = Document(vespa_client, self.index_management)
        self.recommender = Recommender(vespa_client, self.index_management)
        self.embed = Embed(vespa_client, self.index_management, self.default_device)

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

    def _set_zookeeper(self, zookeeper_client: Optional[KazooClient]) -> Optional[KazooClient]:
        """Connect to Zookeeper and return the client. If connection fails, log a warning and return None.
        Args:
            zookeeper_client: The Zookeeper client. We make it optional to allow for the case where the user does not
                want to connect to Zookeeper.

        Returns:
            Optional[KazooClient]: The Zookeeper client if connection is successful, None otherwise.
        """
        if zookeeper_client is None:
            return None
        else:
            try:
                zookeeper_client.start()
            except KazooTimeoutError as e:
                logger.warning(f"Failed to connect to Zookeeper due to timeout. "
                               f"Marqo will still start but is not protected by Zookeeper. "
                               f"Please check your Zookeeper configuration and network settings. "
                               f"Original error message: {e}")
                return None
            return zookeeper_client
