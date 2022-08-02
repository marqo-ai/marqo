import json
import pprint

import requests

from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError
from marqo.neural_search import neural_search, constants, index_meta_cache
import unittest
import copy


class TestDeleteIndex(unittest.TestCase):

    def setUp(self) -> None:
        self.endpoint = 'https://admin:admin@localhost:9200'
        self.generic_header = {"Content-type": "application/json"}
        self.client = Client(url=self.endpoint)
        self.index_name_1 = "my-test-index-owwoowow2"
        self.index_name_2 ="test-index-epic"
        self.config = copy.deepcopy(self.client.config)
        self._delete_indices()

    def tearDown(self) -> None:
        self._delete_indices()

    def _delete_indices(self):
        """Helper to just delete testing indices"""
        for ix_name in [self.index_name_1, self.index_name_2]:
            try:
                self.client.delete_index(ix_name)
            except MarqoApiError as s:
                pass

    def test_delete_clears_cache(self):
        """deletes the index info from cache"""
        assert self.index_name_1 not in index_meta_cache.get_cache()
        neural_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        neural_search.create_vector_index(config=self.config, index_name=self.index_name_2)
        assert self.index_name_1 in index_meta_cache.get_cache()
        assert self.index_name_2 in index_meta_cache.get_cache()
        # make sure only index 1 is deleted:
        neural_search.delete_index(config=self.config, index_name=self.index_name_1)
        assert self.index_name_1 not in index_meta_cache.get_cache()
        assert self.index_name_2 in index_meta_cache.get_cache()

