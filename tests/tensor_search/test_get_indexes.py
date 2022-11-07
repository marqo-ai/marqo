from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import tensor_search
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError, MarqoWebError


class TestGetIndexes(MarqoTestCase):
    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-create-index-1"
        self.index_name_2 = "my-test-create-index-2"
        self.indices = self.index_name_1, self.index_name_2
        for index_name in self.indices:
            try:
                tensor_search.delete_index(config=self.config, index_name=index_name)
            except IndexNotFoundError as s:
                pass

    def tearDown(self) -> None:
        for index_name in self.indices:
            try:
                tensor_search.delete_index(config=self.config, index_name=index_name)
            except IndexNotFoundError as s:
                pass

    def test_success(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_2)
        indices = tensor_search.get_all_indexes(config=self.config)
        assert self.index_name_1 in str(indices)
        assert self.index_name_2 in str(indices)
