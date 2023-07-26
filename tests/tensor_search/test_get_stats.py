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
        testing_list = [
            {
                "expected_number_of_vectors": 2,
                "expected_number_of_documents": 1,
                "add_docs_kwargs": {
                    "docs": [{"description_1": "test-2", "description_2": "test"}]
                }
            },
            {
                "expected_number_of_vectors": 3,
                "expected_number_of_documents": 1,
                "add_docs_kwargs": {
                    "docs": [{"description_1": "test-2", "description_2": "test", "description_3": "test"}]
                }
            },
            {
                "expected_number_of_vectors": 1,
                "expected_number_of_documents": 1,
                "add_docs_kwargs": {
                    "docs": [{"description_2": "test"}]
                }
            },
            {
                "expected_number_of_vectors": 3,
                "expected_number_of_documents": 1,
                "add_docs_kwargs": {
                    "docs": [{"description_1": "test-2", "description_2": "test", "description_3": "test"}],
                    "mappings": {
                        "my_multi_modal_field": {
                            "type": "multimodal_combination",
                            "weights": {"text_1": 0.5, "text_2": 0.8}
                        }
                    }
                }
            },
            {
                "expected_number_of_vectors": 0,
                "expected_number_of_documents": 1,
                "add_docs_kwargs": {
                    "docs": [{"non_tensor_field": "test"}],
                    "non_tensor_fields": ["non_tensor_field"]
                }
            },
            {
                "expected_number_of_vectors": 0,
                "expected_number_of_documents": 1,
                "add_docs_kwargs": {
                    "docs": [{"list_field": ["this", "that"]}],
                    "non_tensor_fields": ["list_field"]
                }
            }
        ]
        for test_case in testing_list:
            try:
                tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            except IndexNotFoundError:
                pass

            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                              index_settings={'index_defaults': {"model": "random/small"}})
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1,
                    auto_refresh=True, device="cpu",
                    **test_case["add_docs_kwargs"],
                )
            )
            assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfDocuments"] \
                   == test_case["expected_number_of_documents"]
            assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfVectors"] \
                   == test_case["expected_number_of_vectors"]

    def test_long_text_splitting_vectors_count(self):

        number_of_words = 55

        test_case = {
                "expected_number_of_vectors": 3,
                "expected_number_of_documents": 1,
                "add_docs_kwargs": {
                    "docs": [{"55_words_field": "test " * number_of_words}],
                }
            }

        index_settings = {
            "index_defaults": {
                "normalize_embeddings": True,
                "model": "random/small",
                "text_preprocessing": {
                    "split_length": 20,
                    "split_overlap": 1,
                    "split_method": "word"
                },
            }
        }

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError:
            pass

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=index_settings)

        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1,
                auto_refresh=True, device="cpu",
                **test_case["add_docs_kwargs"],
            )
        )

        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfDocuments"] \
               == test_case["expected_number_of_documents"]
        assert tensor_search.get_stats(config=self.config, index_name=self.index_name_1)["numberOfVectors"] \
               == test_case["expected_number_of_vectors"]



