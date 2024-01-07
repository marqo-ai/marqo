import unittest

from tests.utils.transition import add_docs_caller
from marqo.api.exceptions import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import TensorField, IndexSettingsField, SearchMethod
from marqo.tensor_search.models.search import SearchContext
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity
from marqo.tensor_search.tensor_search import _create_dummy_query_for_zero_vector_search
from tests.marqo_test import MarqoTestCase
from unittest.mock import patch
from unittest import mock
import numpy as np
import os
import pydantic

@unittest.skip
class TestContextSearch(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.endpoint = self.authorized_url

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treatUrlsAndPointersAsImages: True,
                    IndexSettingsField.normalizeEmbeddings: True
                }
            })
        add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
            {
                "Title": "Horse rider",
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "_id": "1"
            },
            {
                "Title": "unrelated",
                "text_field": "bad result.",
                "_id": "2"
            },
        ], auto_refresh=True)
        
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError:
            pass
        self.device_patcher.stop()

    def test_search(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }
        res = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
        SearchContext(**{"tensor": [{"vector": [1, ] * 512, "weight": 2}, {"vector": [2, ] * 512, "weight": -1}], }))

    def test_search_with_incorrect_tensor_dimension(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }
        try:
            tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=SearchContext(
                **{"tensor": [{"vector": [1, ] * 3, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], }))
            raise AssertionError
        except InvalidArgError as e:
            assert "This causes the error when we do `numpy.mean()` over" in e.message

    def test_search_with_incorrect_query_format(self):
        query = "A rider is riding a horse jumping over the barrier"
        try:
            res = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
            SearchContext(**{"tensor": [{"vector": [1, ] * 512, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], }))
            raise AssertionError
        except InvalidArgError as e:
            assert "This is not supported as the context only works when the query is a dictionary." in e.message

    def test_search_score(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }

        res_1 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query)
        res_2 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
        SearchContext(**{"tensor": [{"vector": [1, ] * 512, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], }))
        res_3 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
        SearchContext(**{"tensor": [{"vector": [1, ] * 512, "weight": -1}, {"vector": [1, ] * 512, "weight": 1}], }))

        assert res_1["hits"][0]["_score"] == res_2["hits"][0]["_score"]
        assert res_1["hits"][0]["_score"] == res_3["hits"][0]["_score"]

    def test_search_vectors(self):
        with patch("numpy.mean", wraps = np.mean) as mock_mean:
            query = {
                "A rider is riding a horse jumping over the barrier": 1,
            }
            res_1 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query)

            weight_1, weight_2, weight_3 = 2.5, 3.4, -1.334
            vector_2 = [-1,] * 512
            vector_3 = [1.3,] * 512
            query = {
                "A rider is riding a horse jumping over the barrier": weight_1,
            }

            res_2 = tensor_search.search(config=self.config, index_name=self.index_name_1, text=query, context=
            SearchContext(**{"tensor": [{"vector": vector_2, "weight": weight_2}, {"vector": vector_3, "weight": weight_3}], }))

            args_list = [args[0] for args in mock_mean.call_args_list]
            vectorised_string = args_list[0][0][0]
            weighted_vectors = args_list[1][0]

            assert np.allclose(vectorised_string * weight_1, weighted_vectors[0], atol=1e-9)
            assert np.allclose(np.array(vector_2) * weight_2, weighted_vectors[1], atol=1e-9)
            assert np.allclose(np.array(vector_3) * weight_3, weighted_vectors[2], atol=1e-9)
    
    def test_search_vectors_from_doc(self):
        # if we get_document with tensor facets and pass those as context for search, we should get that document back
        expected_doc_id = "1"
        retrieved_doc = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id=expected_doc_id, show_vectors=True)
        context_vector_1 = retrieved_doc["_tensor_facets"][0]["_embedding"]
        
        res = tensor_search.search(config=self.config, index_name=self.index_name_1, text={"": 0}, 
                                   context=SearchContext(**{"tensor": [{"vector": context_vector_1, "weight": 1}], }))
        assert res["hits"][0]["_id"] == expected_doc_id


class TestContextBulkSearch(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        self.endpoint = self.authorized_url

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treatUrlsAndPointersAsImages: True,
                    IndexSettingsField.normalizeEmbeddings: True
                }
            })
        add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
            {
                "Title": "Horse rider",
                "text_field": "A rider is riding a horse jumping over the barrier.",
                "_id": "1"
            }], auto_refresh=True)
        
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except:
            pass
        self.device_patcher.stop()

    def test_bulk_search(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }
        res = tensor_search.bulk_search(
            marqo_config=self.config,
            query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=query,
                        context=SearchContext(**{"tensor": [{"vector": [1, ] * 512, "weight": 2}, {"vector": [2, ] * 512, "weight": -1}], })
                    )
                ]
            )
        )

    def test_bulk_search_with_incorrect_tensor_dimension(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }
        try:
            res = tensor_search.bulk_search(
                marqo_config=self.config,
                query=BulkSearchQuery(
                    queries=[
                        BulkSearchQueryEntity(
                            index=self.index_name_1,
                            q=query,
                            context=SearchContext(**{"tensor": [{"vector": [1, ] * 3, "weight": 2}, {"vector": [2, ] * 512, "weight": -1}], })
                        )
                    ]
                )
            )
            raise AssertionError
        except InvalidArgError as e:
            assert "This causes the error when we do `numpy.mean()` over" in e.message

    def test_bulk_search_with_incorrect_query_format(self):
        query = {
            "A rider is riding a horse jumping over the barrier",
        }
        try:
            res = tensor_search.bulk_search(
                marqo_config=self.config,
                query=BulkSearchQuery(
                    queries=[
                        BulkSearchQueryEntity(
                            index=self.index_name_1,
                            q=query,
                            context=SearchContext(**{"tensor": [{"vector": [1, ] * 512, "weight": 2}, {"vector": [2, ] * 512, "weight": -1}], })
                        )
                    ]
                )
            )
        except pydantic.ValidationError as e:
            pass
    
    def test_bulk_search_score(self):
        query = {
            "A rider is riding a horse jumping over the barrier": 1,
        }

        res_1 = tensor_search.bulk_search(
            marqo_config=self.config,
            query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=query
                    )
                ]
            )
        )["result"][0]
        res_2 = tensor_search.bulk_search(
            marqo_config=self.config,
            query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=query,
                        context=SearchContext(**{"tensor": [{"vector": [1, ] * 512, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], })
                    )
                ]
            )
        )["result"][0]
        res_3 = tensor_search.bulk_search(
            marqo_config=self.config,
            query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=query,
                        context=SearchContext(**{"tensor": [{"vector": [5, ] * 512, "weight": 1}, {"vector": [5, ] * 512, "weight": -1}], })
                    )
                ]
            )
        )["result"][0]

        assert res_1["hits"][0]["_score"] == res_2["hits"][0]["_score"]
        assert res_1["hits"][0]["_score"] == res_3["hits"][0]["_score"]

    def test_bulk_search_vectors(self):
        with patch("numpy.mean", wraps = np.mean) as mock_mean:
            query = {
                "A rider is riding a horse jumping over the barrier": 1,
            }
            res_1 = tensor_search.bulk_search(
                marqo_config=self.config,
                query=BulkSearchQuery(
                    queries=[
                        BulkSearchQueryEntity(
                            index=self.index_name_1,
                            q=query
                        )
                    ]
                )
            )["result"][0]

            weight_1, weight_2, weight_3 = 2.5, 3.4, -1.334
            vector_2 = [-1,] * 512
            vector_3 = [1.3,] * 512
            query = {
                "A rider is riding a horse jumping over the barrier": weight_1,
            }

            res_2 = tensor_search.bulk_search(
                marqo_config=self.config,
                query=BulkSearchQuery(
                    queries=[
                        BulkSearchQueryEntity(
                            index=self.index_name_1,
                            q=query,
                            context=SearchContext(**{"tensor": [{"vector": vector_2, "weight": weight_2}, {"vector": vector_3, "weight": weight_3}], })
                        )
                    ]
                )
            )["result"][0]

            args_list = [args[0] for args in mock_mean.call_args_list]
            vectorised_string = args_list[0][0][0]
            weighted_vectors = args_list[1][0]

            assert np.allclose(vectorised_string * weight_1, weighted_vectors[0], atol=1e-9)
            assert np.allclose(np.array(vector_2) * weight_2, weighted_vectors[1], atol=1e-9)
            assert np.allclose(np.array(vector_3) * weight_3, weighted_vectors[2], atol=1e-9)
        
    def test_bulk_search_vectors_from_doc(self):
        # if we get_document with tensor facets and pass those as context for bulk search, we should get that document back
        expected_doc_id = "1"
        retrieved_doc = tensor_search.get_document_by_id(config=self.config, index_name=self.index_name_1, document_id=expected_doc_id, show_vectors=True)
        context_vector_1 = retrieved_doc["_tensor_facets"][0]["_embedding"]
        
        res = tensor_search.search(config=self.config, index_name=self.index_name_1, text={"": 0}, 
                                   context=SearchContext(**{"tensor": [{"vector": context_vector_1, "weight": 1}], }))

        res_2 = tensor_search.bulk_search(
            marqo_config=self.config,
            query=BulkSearchQuery(
                queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q={"": 0},
                        context=SearchContext(**{"tensor": [{"vector": context_vector_1, "weight": 1}], })
                    )
                ]
            )
        )["result"][0]
        
        assert res["hits"][0]["_id"] == expected_doc_id

    def test_context_vector_with_zero_vectors_search(self):
        """This is to test an edge case where the query is a 0-vector, 1 shard, 0 replicas"""
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError:
            pass

        dimension = 32
        index_settings= {
            IndexSettingsField.index_defaults: {
                "treat_urls_and_pointers_as_images": True,
                "model": "random/small"
        },
            'number_of_shards': 1,
            'number_of_replicas': 0,
        }
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=index_settings)

        add_docs_caller(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd"},
                {"abc": "random text", "other field": "Close match hehehe"},
            ], auto_refresh=True)

        mock_create_dummy_query = mock.MagicMock()
        mock_create_dummy_query.side_effect = _create_dummy_query_for_zero_vector_search
        @mock.patch('marqo.tensor_search.tensor_search._create_dummy_query_for_zero_vector_search', mock_create_dummy_query)
        def run():

            res = tensor_search.search(config=self.config, index_name=self.index_name_1, text={"test": 0},
                                       context=SearchContext(
                                           **{"tensor": [{"vector": [0.0] * dimension, "weight": 1}], }))

            mock_create_dummy_query.assert_called_once()
            assert res["hits"] == []
            return True

        assert run()

    def test_context_vector_with_zero_vectors_search_in_bulk(self):
        """ Ensure multimodal vectors generated by bulk search are correct (only 1 query in bulk search)
        """
        docs = [
            {"loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
             "_id": 'realistic_hippo'},

            {"loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png",
             "_id": 'artefact_hippo'}
        ]

        dimension = 32
        image_index_config = {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.model: "random/small",
                IndexSettingsField.treatUrlsAndPointersAsImages: True
            },
            'number_of_shards': 1,
            'number_of_replicas': 0,
        }

        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError:
            pass
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings=image_index_config)

        add_docs_caller(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True
        )

        multi_queries = [
            {
                "artefact": 5.0,
                "photo realistic": -1,
            },
            {
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 0.0,
                "photo realistic": 0.0,
            },
            {
                "random-query": 0.0,
            }
        ]

        mock_create_dummy_query = mock.MagicMock()
        mock_create_dummy_query.side_effect = _create_dummy_query_for_zero_vector_search

        @mock.patch('marqo.tensor_search.tensor_search._create_dummy_query_for_zero_vector_search', mock_create_dummy_query)
        def run():
            res = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=multi_query,
                        limit=5,
                        searchMethod=SearchMethod.TENSOR,
                        context=SearchContext(
                            **{"tensor": [{"vector": [0.0] * dimension, "weight": 1}], })
                    ) for multi_query in multi_queries]
                )
            )
            assert mock_create_dummy_query.call_count == 2
            assert res['result'][1]['hits'] == []
            assert res['result'][2]['hits'] == []
            return True
        assert run()