import json
import pprint
import time
from marqo.errors import IndexNotFoundError, MarqoError
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import unittest
import copy
from tests.marqo_test import MarqoTestCase


class TestGetIndexes(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-create-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_get_indexes(self):
        res = tensor_search.get_indexes(config=self.config)
        pprint.pprint(res)
