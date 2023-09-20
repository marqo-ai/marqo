import os
from typing import Optional, Union
from marqo.tensor_search import enums
from vespa.application import Vespa

class Config:
    def __init__(
        self,
        timeout: Optional[int] = None,
        backend: Optional[Union[enums.SearchDb, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        url:
            The url to the S2Search API (ex: http://localhost:9200)
        """
        self.cluster_is_remote = False
        self.vespa_config_client = Vespa(os.getenv("VESPA_CONFIG_URL", "http://localhost:19071/"))
        self.vespa_query_client = Vespa(os.getenv("VESPA_QUERY_URL", "http://localhost:8080/"))
        self.vespa_feed_client = Vespa(os.getenv("VESPA_FEED_URL", "http://localhost:8080/"))
        self.timeout = timeout
        self.backend = backend if backend is not None else enums.SearchDb.opensearch

    def set_url(self, url):
        pass
