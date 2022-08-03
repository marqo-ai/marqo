import base64
import hashlib
import hmac
import json
import datetime
import nltk
import ssl
from typing import Any, Dict, List, Optional, Union
from marqo.index import Index
from marqo.enums import Devices
from marqo.config import Config
from marqo._httprequests import HttpRequests
from marqo.errors import MarqoError
from marqo.neural_search import neural_search, index_meta_cache
import urllib3
from marqo import utils, enums


class Client:
    """
    A client for the marqo API

    A client instance is needed for every marqo API method to know the location of
    marqo and its permissions.
    """
    def __init__(
        self, url: str = "http://localhost:9200", main_user: str = None, main_password: str = None,
        indexing_device: Optional[Union[enums.Devices, str]] = None,
        search_device: Optional[Union[enums.Devices, str]] = None
    ) -> None:
        """
        Parameters
        ----------
        url:
            The url to the S2Search API (ex: http://localhost:9200)
        """
        self._ensure_nltk_setup()

        self.main_user = main_user
        self.main_password = main_password
        if (main_user is not None) and (main_password is not None):
            self.url = utils.construct_authorized_url(url_base=url, username=main_user, password=main_password)
        else:
            self.url = url
        self.config = Config(self.url, indexing_device=indexing_device, search_device=search_device)
        self.http = HttpRequests(self.config)
        index_meta_cache.populate_cache(config=self.config)
        if not self.config.cluster_is_s2search:
            self._turn_off_auto_create_index()

    def _turn_off_auto_create_index(self):
        """turns off auto creation of indices. To be run in client instantiation"""
        self.http.put(
            path="_cluster/settings",
            body={
                "persistent": {"action.auto_create_index": "false"}
            })


    def create_index(
        self, index_name: str,
        treat_urls_and_pointers_as_images=False, model=None,
        normalize_embeddings=True,
        sentences_per_chunk=2,
        sentence_overlap=0
    ) -> Dict[str, Any]:
        """Create an index.

        Parameters
        ----------
        index_name: str
            name of the index.


        Returns
        -------
        task:
            Name of the index

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        return Index.create(
            config=self.config, index_name=index_name,
            treat_urls_and_pointers_as_images=treat_urls_and_pointers_as_images,
            model=model, normalize_embeddings=normalize_embeddings,
            sentences_per_chunk=sentences_per_chunk, sentence_overlap=sentence_overlap
        )

    def delete_index(self, index_name: str) -> Dict[str, Any]:
        """Deletes an index

        Parameters
        ----------
        index_name:
            UID of the index.

        Returns
        -------
        task:
            Dictionary containing a task to track the informations about the progress of an asynchronous process.
            https://docs.marqo.com/reference/api/tasks.html#get-one-task

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """

        return neural_search.delete_index(config=self.config, index_name=index_name)

    def get_index(self, index_name: str) -> Index:
        """Get the index.
        This index should already exist.

        Parameters
        ----------
        index_name:
            UID of the index.

        Returns
        -------
        index:
            An Index instance containing the information of the fetched index.

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        ix = Index(self.config, index_name)
        # verify it exists
        self.http.get(path=index_name)
        return ix

    def index(self, index_name: str) -> Index:
        """Create a local reference to an index identified by UID, without doing an HTTP call.
        Calling this method doesn't create an index in the marqo instance, but grants access to all the other methods in the Index class.

        Parameters
        ----------
        index_name:
            UID of the index.

        Returns
        -------
        index:
            An Index instance.
        """
        if index_name is not None:
            return Index(self.config, index_name=index_name)
        raise Exception('The index UID should not be None')

    def get_version(self) -> Dict[str, str]:
        """Get version marqo

        Returns
        -------
        version:
            Information about the version of marqo.

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        return self.http.get(self.config.paths.version)

    def version(self) -> Dict[str, str]:
        """Alias for get_version

        Returns
        -------
        version:
            Information about the version of marqo.

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        return self.get_version()

    @staticmethod
    def _base64url_encode(
        data: bytes
    ) -> str:
        return base64.urlsafe_b64encode(data).decode('utf-8').replace('=','')

    def _ensure_nltk_setup(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
