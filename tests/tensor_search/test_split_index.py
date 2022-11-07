from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import tensor_search
from marqo.errors import MarqoApiError, MarqoError, IndexNotFoundError, MarqoWebError


class TestSplitIndex(MarqoTestCase):
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

    def delete_indices_and_test_deleted(self):
        for index_name in self.indices:
            try:
                tensor_search.delete_index(config=self.config, index_name=index_name)
            except IndexNotFoundError as s:
                pass
        for index_name in self.indices:
            try:
                tensor_search.search(config=self.config, index_name=index_name, text="some text")
                raise AssertionError
            except IndexNotFoundError as e:
                pass

    def test_success(self):
        self.delete_indices_and_test_deleted()
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.update_index_settings(config=self.config, index_name=self.index_name_1,
                                            settings={"index.blocks.write": True})
        settings = {"settings": {"index.number_of_shards": 10}}
        tensor_search.split_index(config=self.config, index_name=self.index_name_1,
                                  new_index_name=self.index_name_2, settings=settings)
        index2_settings = tensor_search.get_index_settings(config=self.config, index_name=self.index_name_2)
        assert index2_settings[self.index_name_2]["settings"]["index"]["number_of_shards"] == '10'

    def test_too_many_shards(self):
        self.delete_indices_and_test_deleted()
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.update_index_settings(config=self.config, index_name=self.index_name_1,
                                            settings={"index.blocks.write": True})
        settings = {"settings": {"index.number_of_shards": 100000}}
        expected_error = "Failed to parse value [100000] for setting [index.number_of_shards] must be <= 1024"
        try:
            tensor_search.split_index(config=self.config, index_name=self.index_name_1,
                                      new_index_name=self.index_name_2, settings=settings)
            raise AssertionError
        except MarqoWebError as e:
            assert expected_error in e.message

    def test_not_multiple_shards(self):
        self.delete_indices_and_test_deleted()
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.update_index_settings(config=self.config, index_name=self.index_name_1,
                                            settings={"index.blocks.write": True})
        settings = {"settings": {"index.number_of_shards": 7}}
        expected_error = "the number of source shards [5] must be a factor of [7]"
        try:
            tensor_search.split_index(config=self.config, index_name=self.index_name_1,
                                      new_index_name=self.index_name_2, settings=settings)
            raise AssertionError
        except MarqoWebError as e:
            assert expected_error in e.message
