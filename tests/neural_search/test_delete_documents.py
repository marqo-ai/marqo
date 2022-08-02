import json
import pprint
from marqo.client import Client
import unittest
import copy
import unittest
from marqo.neural_search import neural_search
from marqo.config import Config
from marqo.errors import MarqoError, MarqoApiError
import requests
from marqo.neural_search import utils
from marqo.neural_search.enums import NeuralField


class TestDeleteDocuments(unittest.TestCase):

    def setUp(self) -> None:
        self.endpoint = 'https://admin:admin@localhost:9200'
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        self.config = Config(url=self.endpoint)
        self._delete_testing_indices()

    def _delete_testing_indices(self):
        for ix in [self.index_name_1, self.index_name_2]:
            try:
                neural_search.delete_index(config=self.config, index_name=ix)
            except MarqoApiError as s:
                pass

    def test_delete_documents(self):
        # first batch:
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"f1": "cat dog sat mat", "Sydney": "Australia contains Sydney"},
                {"Lime": "Tree tee", "Magnificent": "Waterfall out yonder"},
            ], auto_refresh=True)
        count0_res = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[
                {"hooped": "absolutely ridic", "Darling": "A harbour in Sydney", "_id": "455"},
                {"efg": "hheeehehehhe", "_id": "at-at"}
            ], auto_refresh=True)
        count1_res = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        neural_search.delete_documents(config=self.config, index_name=self.index_name_1, doc_ids=["455", "at-at"],
                                       auto_refresh=True)
        count_post_delete = requests.post(
            F"{self.endpoint}/{self.index_name_1}/_count",
            timeout=self.config.timeout,
            verify=False
        ).json()["count"]
        assert count_post_delete == count0_res

