import json
import pprint
import requests
from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError
from marqo.neural_search import neural_search, constants, index_meta_cache
import unittest
import copy
import unittest
from marqo.neural_search.models.index_info import IndexInfo
from marqo.neural_search.models import index_info
from marqo.config import Config
import requests
from marqo.neural_search import utils
from marqo.neural_search.enums import NeuralField


class TestGetDocument(unittest.TestCase):

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

    def test_get_document(self):
        neural_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        neural_search.add_documents(config=self.config, index_name=self.index_name_1, docs=[
            {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }], auto_refresh=True)
        assert neural_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123") == {
              "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }

    def test_get_document_non_existent_index(self):
        try:
            a = neural_search.get_document_by_id(
                config=self.config, index_name=self.index_name_1,
                document_id="123")
            raise AssertionError
        except MarqoError as e:
            assert 'no such index' in str(e)

    def test_get_document_empty_str(self):
        try:
            a = neural_search.get_document_by_id(
                config=self.config, index_name=self.index_name_1,
                document_id="")
            print(a)
            raise AssertionError
        except MarqoError as e:
            assert "be empty" in str(e)

    def test_get_document_bad_types(self):
        for ar in [12.2, 1, [], {}, None]:
            try:
                a = neural_search.get_document_by_id(
                    config=self.config, index_name=self.index_name_1,
                    document_id=ar)
                print(a)
                raise AssertionError
            except MarqoError as e:
                assert "must be a str" in str(e)