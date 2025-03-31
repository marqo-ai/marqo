from typing import Optional, Union

from kazoo.handlers.threading import KazooTimeoutError

from marqo.core.document.document import Document
from marqo.core.embed.embed import Embed
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.inference.api import Inference, ModelManager
from marqo.core.monitoring.monitoring import Monitoring
from marqo.core.search.recommender import Recommender
from marqo.logging import get_logger
from marqo.tensor_search import enums
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.vespa.vespa_client import VespaClient
from marqo.vespa.zookeeper_client import ZookeeperClient

logger = get_logger(__name__)


class Config:
    def __init__(
            self,
            vespa_client: VespaClient,
            inference: Inference,
            model_manager: Optional[ModelManager] = None,
            zookeeper_client: Optional[ZookeeperClient] = None,
            timeout: Optional[int] = None,
            backend: Optional[Union[enums.SearchDb, str]] = None,
    ) -> None:
        self.vespa_client = vespa_client
        self.set_is_remote(vespa_client)
        self._zookeeper_client = zookeeper_client
        self._connect_to_zookeeper()

        self.timeout = timeout
        self.backend = backend if backend is not None else enums.SearchDb.vespa

        # Initialize Core layer dependencies
        deployment_lock_timeout = utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_INDEX_DEPLOYMENT_LOCK_TIMEOUT)
        self.index_management = IndexManagement(vespa_client, zookeeper_client,
                                                enable_index_operations=True,
                                                deployment_lock_timeout_seconds=deployment_lock_timeout)

        self.inference = inference
        self.monitoring = Monitoring(vespa_client, self.index_management)
        self.document = Document(vespa_client, self.index_management, self.inference)
        self.recommender = Recommender(vespa_client, self.index_management, self.inference)
        self.embed = Embed(vespa_client, self.index_management, self.inference)

        self.model_manager = model_manager

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
