from typing import Optional, Union
from marqo.tensor_search import enums


class Config:
    def __init__(
        self,
        url: str,
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
        self.url = self.set_url(url)
        self.timeout = timeout
        self.backend = backend if backend is not None else enums.SearchDb.opensearch

    def set_url(self, url):
        """Set the URL, and infers whether that url is remote"""
        lowered_url = url.lower()
        local_host_markers = ["localhost", "0.0.0.0", "127.0.0.1"]
        if any([marker in lowered_url for marker in local_host_markers]):
            self.cluster_is_remote = False
        else:
            self.cluster_is_remote = True
        self.url = url
        return self.url
