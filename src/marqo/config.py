from typing import Optional, Union

from marqo.core.index_management.index_management import IndexManagement
from marqo.core.monitoring.monitoring import Monitoring
from marqo.tensor_search import enums
from marqo.vespa.vespa_client import VespaClient


class Config:
    def __init__(
            self,
            vespa_client: VespaClient,
            index_management: IndexManagement,
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
        self.timeout = timeout
        self.backend = backend if backend is not None else enums.SearchDb.vespa

        # Initialize Core layer dependencies
        self.index_management = index_management
        self.monitoring = Monitoring(vespa_client, index_management)

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
