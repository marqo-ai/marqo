from typing import Optional, Union

from marqo.core.document.document import Document
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.monitoring.monitoring import Monitoring
from marqo.core.search.recommender import Recommender
from marqo.tensor_search import enums
from marqo.vespa.vespa_client import VespaClient


class Config:
    def __init__(
            self,
            vespa_client: VespaClient,
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
        self.index_management = IndexManagement(vespa_client)
        self.monitoring = Monitoring(vespa_client, self.index_management)
        self.document = Document(vespa_client, self.index_management)
        self.recommender = Recommender(vespa_client, self.index_management)

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
