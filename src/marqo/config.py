from typing import Optional, Union
from marqo.tensor_search import enums


class Config:
    def __init__(
        self,
        url: str,
        timeout: Optional[int] = None,
        indexing_device: Optional[Union[enums.Device, str]] = None,
        search_device: Optional[Union[enums.Device, str]] = None
    ) -> None:
        """
        Parameters
        ----------
        url:
            The url to the S2Search API (ex: http://localhost:9200)
        """
        self.cluster_is_remote = False
        self.cluster_is_s2search = False
        self.url = self.set_url(url)
        self.timeout = timeout
        default_device = enums.Device.cpu

        self.indexing_device = indexing_device if indexing_device is not None else default_device
        self.search_device = search_device if search_device is not None else default_device

    def set_url(self, url):
        """Set the URL, and infers whether that url is remote"""
        lowered_url = url.lower()
        local_host_markers = ["localhost", "0.0.0.0", "127.0.0.1"]
        if any([marker in lowered_url for marker in local_host_markers]):
            self.cluster_is_remote = False
        else:
            self.cluster_is_remote = True
            if "s2search.io" in lowered_url:
                self.cluster_is_s2search = True
        self.url = url
        return self.url
