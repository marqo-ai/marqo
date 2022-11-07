from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import tensor_search
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError, MarqoWebError


class TestIndexSettings(MarqoTestCase):
    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-create-index-1"
        self.indices = [self.index_name_1]

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

    def delete_indices_and_test_deleted(self):
        for index_name in self.indices:
            try:
                tensor_search.delete_index(config=self.config, index_name=index_name)
            except IndexNotFoundError as s:
                pass

    def test_get_index_settings(self):
        self.delete_indices_and_test_deleted()
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        settings = tensor_search.get_index_settings(config=self.config, index_name=self.index_name_1)
        assert settings is not None

    def test_update_index_settings(self):
        self.delete_indices_and_test_deleted()
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.update_index_settings(config=self.config, index_name=self.index_name_1,
                                            settings={"settings": {"index.blocks.write": "true"}})
        settings = tensor_search.get_index_settings(config=self.config, index_name=self.index_name_1)
        assert settings[self.index_name_1]['settings']['index']['blocks']['write'] == "true"

        tensor_search.update_index_settings(config=self.config, index_name=self.index_name_1,
                                            settings={"settings": {"index.blocks.write": "false"}})
        settings = tensor_search.get_index_settings(config=self.config, index_name=self.index_name_1)
        assert settings[self.index_name_1]['settings']['index']['blocks']['write'] == "false"
