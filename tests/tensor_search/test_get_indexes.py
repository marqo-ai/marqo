import json
import pprint
import time
from marqo.errors import IndexNotFoundError, MarqoError
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import unittest
import copy
from tests.marqo_test import MarqoTestCase


def _index_is_present(index_name, index_results):
    """True IFF index name is in the results of a get_indexes call"""
    return any([index_name in res.values() for res in index_results['results']])


@unittest.skip
class TestGetIndexes(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-create-index-1"
        self.index_name_2 = "my-test-create-index-2"
        self.all_indexes = [self.index_name_1, self.index_name_2]
        for name in self.all_indexes:
            try:
                tensor_search.delete_index(config=self.config, index_name=name)
            except IndexNotFoundError:
                pass

    def tearDown(self) -> None:
        for name in self.all_indexes:
            try:
                tensor_search.delete_index(config=self.config, index_name=name)
            except IndexNotFoundError:
                pass

    def test_get_indexes(self):
        assert not _index_is_present(self.index_name_1, tensor_search.get_indexes(config=self.config))
        assert not _index_is_present(self.index_name_2, tensor_search.get_indexes(config=self.config))

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        assert _index_is_present(self.index_name_1, tensor_search.get_indexes(config=self.config))
        assert not _index_is_present(self.index_name_2, tensor_search.get_indexes(config=self.config))

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_2)
        assert _index_is_present(self.index_name_1, tensor_search.get_indexes(config=self.config))
        assert _index_is_present(self.index_name_2, tensor_search.get_indexes(config=self.config))

    def test_get_indexes_no_invalid_indexes(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        index_results = tensor_search.get_indexes(config=self.config)
        for ix_name in [res['index_name'] for res in index_results['results']]:
            assert ix_name not in constants.INDEX_NAMES_TO_IGNORE
            for bad_prefix in constants.INDEX_NAME_PREFIXES_TO_IGNORE:
                assert not ix_name.startswith(bad_prefix)
