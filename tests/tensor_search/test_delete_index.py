from marqo.errors import MarqoApiError, IndexNotFoundError
from marqo.tensor_search import tensor_search, index_meta_cache
import copy
from tests.marqo_test import MarqoTestCase


class TestDeleteIndex(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-owwoowow2"
        self.index_name_2 ="test-index-epic"
        self._delete_indices()

    def tearDown(self) -> None:
        self._delete_indices()

    def _delete_indices(self):
        """Helper to just delete testing indices"""
        for ix_name in [self.index_name_1, self.index_name_2]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix_name)
            except IndexNotFoundError as s:
                pass

    def test_delete_clears_cache(self):
        """deletes the index info from cache"""
        assert self.index_name_1 not in index_meta_cache.get_cache()
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_2)
        assert self.index_name_1 in index_meta_cache.get_cache()
        assert self.index_name_2 in index_meta_cache.get_cache()
        # make sure only index 1 is deleted:
        tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        assert self.index_name_1 not in index_meta_cache.get_cache()
        assert self.index_name_2 in index_meta_cache.get_cache()

