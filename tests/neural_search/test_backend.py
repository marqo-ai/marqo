import pprint

import requests

from marqo.neural_search import enums, backend
from marqo.neural_search import neural_search
import unittest
import copy
from marqo.errors import MarqoError, MarqoApiError
from marqo.client import Client


class TestBackend(unittest.TestCase):

    def setUp(self) -> None:
        self.endpoint = 'https://admin:admin@localhost:9200'
        self.generic_header = {"Content-type": "application/json"}
        self.client = Client(url=self.endpoint)
        self.index_name_1 = "my-test-index-1"
        self.config = copy.deepcopy(self.client.config)
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def test_chunk_properties_arent_deleted(self):
        """TODO - make sure adding new properties doesn't discard old ones"""

    def test_get_index_info(self):
        neural_search.create_vector_index(
            config=self.config, index_name=self.index_name_1
        )
        index_info = backend.get_index_info(
            config=self.config, index_name=self.index_name_1)
        assert index_info.model_name
        assert "__field_name" in index_info.properties[enums.NeuralField.chunks]["properties"]
        assert isinstance(index_info.properties, dict)

    def test_get_index_info_no_index(self):
        r1 = requests.get(
            url=f"{self.endpoint}/{self.index_name_1}",
            verify=False
        )
        assert r1.status_code == 404
        try:
            index_info = backend.get_index_info(
                config=self.config, index_name=self.index_name_1)
        except MarqoApiError as s:
            assert "no such index" in str(s)

    def test_get_cluster_indices(self):
        neural_search.create_vector_index(
            config=self.config, index_name=self.index_name_1)
        cluster_indices = backend.get_cluster_indices(config=self.config)
        assert '.opendistro_security' not in cluster_indices
        assert isinstance(cluster_indices, set)
        assert self.index_name_1 in cluster_indices