from typing import Optional, Union
from marqo import enums
import torch
import urllib3
import warnings
import warnings


class Config:
    """
    Client's credentials and configuration parameters
    """

    def __init__(
        self,
        url: str,
        timeout: Optional[int] = None,
        indexing_device: Optional[Union[enums.Devices, str]] = None,
        search_device: Optional[Union[enums.Devices, str]] = None
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

        default_device = enums.Devices.cpu

        self.indexing_device = indexing_device if indexing_device is not None else default_device
        self.search_device = search_device if search_device is not None else default_device
        # suppress warnings until we figure out the dependency issues:
        warnings.filterwarnings("ignore")

    def set_url(self, url):
        """Set the URL, and infers whether that url is remote"""
        lowered_url = url.lower()
        if "localhost" in lowered_url or "0.0.0.0" in lowered_url:
            urllib3.disable_warnings()
            self.cluster_is_remote = False
        else:
            warnings.resetwarnings()
            self.cluster_is_remote = True
            if "s2search.io" in lowered_url:
                self.cluster_is_s2search = True
        self.url = url
        return self.url
