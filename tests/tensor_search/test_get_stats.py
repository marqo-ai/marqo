from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.errors import IndexNotFoundError, MarqoError
from marqo.tensor_search import tensor_search, constants, index_meta_cache
from tests.marqo_test import MarqoTestCase


class TestGetStats(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_get_stats_empty(self):
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        index_stats = tensor_search.get_stats(config=self.config, index_name=self.index_name_1)
        assert index_stats["numberOfDocuments"] == 0
        assert len(index_stats["size"]) != 0

    def test_get_stats_non_empty(self):
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=[{"1": "2"},{"134": "2"},{"14": "62"}],
                index_name=self.index_name_1,
                auto_refresh=True, device="cpu"
            )
        )
        index_stats = tensor_search.get_stats(config=self.config, index_name=self.index_name_1)
        assert index_stats["numberOfDocuments"] == 3
        assert len(index_stats["size"]) != 0
