import json
import pprint
import time
from marqo.neural_search.enums import NeuralField
from marqo.client import Client
from marqo.errors import IndexNotFoundError, MarqoError
from marqo.neural_search import neural_search, constants, index_meta_cache
import unittest
import copy
from tests.marqo_test import MarqoTestCase


class TestAddDocuments(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        self.config = copy.deepcopy(self.client.config)
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_get_stats_empty(self):
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass
        self.client.create_index(self.index_name_1)
        assert self.client.index(self.index_name_1).get_stats()["numberOfDocuments"] == 0

    def test_get_stats_non_empty(self):
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass
        self.client.index(self.index_name_1).add_documents(
            [{"1": "2"},{"134": "2"},{"14": "62"},]
        )
        assert self.client.index(self.index_name_1).get_stats()["numberOfDocuments"] == 3