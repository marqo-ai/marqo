import pprint

from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.errors import IndexNotFoundError, MarqoError
from marqo.tensor_search import tensor_search, constants, index_meta_cache
from tests.marqo_test import MarqoTestCase
from marqo._httprequests import HttpRequests


class TestGetStats(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_get_stats_empty(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfDocuments"] == 0

    def test_get_stats_non_empty(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=[{"1": "2"}, {"134": "2"}, {"14": "62"}],
                index_name=self.index_name_1,
                auto_refresh=True, device="cpu"
            )
        )
        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfDocuments"] == 3

    def test_get_stats_number_of_vectors(self):

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings={'index_defaults': {"model": "random/small"}})
        expected_number_of_vectors = 7
        expected_number_of_documents = 5
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=[
                    {"description_1": "test-2", "description_2": "test"},  # 2 vectors
                    {"description_1": "test-2", "description_2": "test", "description_3": "test"},  # 3 vectors
                    {"description_2": "test"},  # 1 vector
                    {"my_multi_modal_field": {
                        "text_1": "test", "text_2": "test"}},  # 1 vector
                    {"non_tensor_field": "test"}  # 0 vectors
                ],
                index_name=self.index_name_1,
                auto_refresh=True, device="cpu",
                non_tensor_fields=["non_tensor_field"],
                mappings={"my_multi_modal_field": {"type": "multimodal_combination", "weights": {
                    "text_1": 0.5, "text_2": 0.8}}}
            )
        )

        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfDocuments"] \
               == expected_number_of_documents
        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfVectors"] \
               == expected_number_of_vectors

    def test_get_stats_number_of_vectors(self):

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings={'index_defaults': {"model": "random/small"}})
        expected_number_of_vectors = 7
        expected_number_of_documents = 6
        res = tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                docs=[
                    {"description_1": "test-2", "description_2": "test"},  # 2 vectors
                    {"description_1": "test-2", "description_2": "test", "description_3": "test"},  # 3 vectors
                    {"description_2": "test"},  # 1 vector
                    {"my_multi_modal_field": {
                        "text_1": "test", "text_2": "test"}},  # 1 vector
                    {"non_tensor_field": "test"},  # 0 vectors
                    {"list_field": ["this", "that"]}, # 0 vectors
                ],
                index_name=self.index_name_1,
                auto_refresh=True, device="cpu",
                non_tensor_fields=["non_tensor_field", "list_field"],
                mappings={"my_multi_modal_field":
                              {"type": "multimodal_combination", "weights": {"text_1": 0.5, "text_2": 0.8}}}
            )
        )

        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfDocuments"] \
               == expected_number_of_documents
        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfVectors"] \
               == expected_number_of_vectors
