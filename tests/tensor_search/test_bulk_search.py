import copy
import os
import math
import requests
import random
from unittest import mock
from marqo.s2_inference.s2_inference import vectorise
import unittest
from marqo.tensor_search.enums import TensorField, SearchMethod, EnvVars, IndexSettingsField
from marqo.errors import (
    BackendCommunicationError, IndexNotFoundError, InvalidArgError, IllegalRequestedDocCount, BadRequestError
)
from marqo.tensor_search import api
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity
from marqo.tensor_search import tensor_search, constants, index_meta_cache, utils
from fastapi.exceptions import RequestValidationError
import numpy as np
from tests.marqo_test import MarqoTestCase
from typing import List
import pydantic


def pass_through_vectorise(*arg, **kwargs):
    """Vectorise will behave as usual, but we will be able to see the call list
    via mock
    """
    return vectorise(*arg, **kwargs)


class TestBulkSearch(MarqoTestCase):

    def setUp(self) -> None:
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        self.index_name_3 = "my-test-index-3"
        self._delete_test_indices()

    def _delete_test_indices(self, indices=None):
        if indices is None or not indices:
            ix_to_delete = [self.index_name_1, self.index_name_2, self.index_name_3]
        else:
            ix_to_delete = indices
        for ix_name in ix_to_delete:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix_name)
            except IndexNotFoundError as s:
                pass

    def test_bulk_search_w_extra_parameters__raise_exception(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        ) 
        with self.assertRaises(pydantic.ValidationError):
            api.bulk_search(BulkSearchQuery(queries=[{
                "index": self.index_name_1,
                "q": "title about some doc",
                "parameter-not-expected": 1,
            }]))

    @mock.patch('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': '0'}})
    def test_bulk_search_with_excessive_searchable_attributes(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        ) 
        with self.assertRaises(pydantic.ValidationError):
            api.bulk_search(BulkSearchQuery(queries=[{
                "index": self.index_name_1,
                "q": "title about some doc",
                "parameter-not-expected": 1,
                "searchableAttributes": ["abc"]
            }]), marqo_config=self.config)
    
    @mock.patch('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': '100'}})
    def test_bulk_search_with_max_searchable_attributes_no_searchable_attributes_field(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        ) 
        with self.assertRaises(InvalidArgError):
            api.bulk_search(BulkSearchQuery(queries=[{
                "index": self.index_name_1,
                "q": "title about some doc",
            }]), marqo_config=self.config)

    @mock.patch('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': '1'}})
    def test_bulk_search_with_excessive_searchable_attributes(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        ) 
        with self.assertRaises(InvalidArgError):
            api.bulk_search(BulkSearchQuery(queries=[{
                "index": self.index_name_1,
                "q": "title about some doc",
                "searchableAttributes": ["abc", "other field"]
            }]), marqo_config=self.config)

    @mock.patch('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': None}})
    def test_bulk_search_with_no_max_searchable_attributes(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        ) 
        api.bulk_search(BulkSearchQuery(queries=[{
            "index": self.index_name_1,
            "q": "title about some doc",
            "searchableAttributes": ["abc", "other field"]
        }]), marqo_config=self.config, device="cpu")

    @mock.patch('os.environ', {**os.environ, **{'MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES': None}})
    def test_bulk_search_with_no_max_searchable_attributes_no_searchable_attributes_field(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        ) 
        api.bulk_search(BulkSearchQuery(queries=[{
            "index": self.index_name_1,
            "q": "title about some doc",
        }]), marqo_config=self.config, device="cpu")


    def test_bulk_search_no_queries_return_early(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        ) 
        resp = tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(queries=[]))
        assert resp['result'] == []

    def test_bulk_search_multiple_indexes_and_queries(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        )
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_2, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id2-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id2-second"},
            ], auto_refresh=True
        )
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_3, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id3-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id3-second"},
            ], auto_refresh=True
        )
        response = tensor_search.bulk_search(marqo_config=self.config, query=BulkSearchQuery(
            queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q="hehehe", limit=2),
                BulkSearchQueryEntity(index=self.index_name_3, q={"laughter":  1.0, "match": -1.0})
            ]
        ))
        assert len(response['result']) == 2
        res1 = response['result'][0]
        res2 = response['result'][1]
        
        assert res1["query"] == "hehehe"
        assert res2["query"] == {"laughter":  1.0, "match": -1.0}

        assert all([h["_id"].startswith("id1-") for h in res1["hits"]])
        assert all([h["_id"].startswith("id3-") for h in res2["hits"]])
        assert len(res2["hits"]) == 2



    def test_multimodal_tensor_combination_zero_weight(self):
        documents = [{
                "text_field": "A rider is riding a horse jumping over the barrier.",
                # "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
            },{
                "combo_text_image": {
                    "text_field" : "A rider is riding a horse jumping over the barrier.",
                    "image_field" : "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                },
        }]
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings={
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                    IndexSettingsField.normalize_embeddings: False
                }
            })

        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=documents, auto_refresh=True,
            mappings = {"combo_text_image" : {"type":"multimodal_combination", "weights":{"image_field": 0,"text_field": 1}}}
        )

        res = tensor_search.bulk_search(query=BulkSearchQuery(queries=[
            BulkSearchQueryEntity(index=self.index_name_1, q="", limit=2),
        ]), marqo_config=self.config)

        # Get scores [0], [1]
        assert len(res['result']) == 1
        assert res['result'][0]["hits"][0]["_score"] == res['result'][0]["hits"][1]["_score"]

    def test_bulk_search_works_on_uncached_field(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe efgh ", "other field": "baaadd efgh ",
                 "_id": "5678", "finally": "some field efgh "},
                {"abc": "random text efgh ", "other field": "Close matc efgh h hehehe",
                 "_id": "1234", "finally": "Random text here efgh "},
            ], auto_refresh=True)
        index_meta_cache.empty_cache()
        tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some text")]),
            marqo_config=self.config,
        )
        assert self.index_name_1 in index_meta_cache.get_cache()

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_3)
        tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q="some text"),
                BulkSearchQueryEntity(index=self.index_name_3, q="some text")
            ]),
            marqo_config=self.config,
        )
        assert self.index_name_1 in index_meta_cache.get_cache()
        assert self.index_name_3 in index_meta_cache.get_cache()

    def test_bulk_search_query_boosted(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {
                    "Title": "The Travels of Marco Polo",
                    "Description": "A 13th-century travelogue describing Polo's travels",
                    "_id": "article_590"
                },
                {
                    "Title": "Extravehicular Mobility Unit (EMU)",
                    "Description": "The EMU is a spacesuit that provides environmental protection, "
                                   "mobility, life support, and communications for astronauts",
                    "_id": "article_591"
                }
            ], auto_refresh=True
        )
        q = "What is the best outfit to wear on the moon?"
        resp = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q=q),
                BulkSearchQueryEntity(index=self.index_name_1, q=q, boost={"Title": [5, 1]}),
                BulkSearchQueryEntity(index=self.index_name_1, q=q, boost={"Title": [-5, -1]}),
                BulkSearchQueryEntity(index=self.index_name_1, q=q, boost={"Title": [-5], "Description": [-5, -1]})
            ]),
            marqo_config=self.config,
        )
        assert len(resp['result']) == 4

        score = resp['result'][0]['hits'][0]['_score']
        score_boosted = resp['result'][1]['hits'][0]['_score']
        score_half_negative = resp['result'][2]['hits'][0]['_score']
        score_neg_boosted = resp['result'][3]['hits'][0]['_score']

        self.assertGreater(score_boosted, score)
        self.assertEqual(score, score_half_negative) # 
        self.assertGreater(score, score_neg_boosted)
        self.assertGreater(score_boosted, score_neg_boosted)

    def test_bulk_search_query_invalid_boosted(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {
                    "Title": "The Travels of Marco Polo",
                    "Description": "A 13th-century travelogue describing Polo's travels",
                    "_id": "article_590"
                },
                {
                    "Title": "Extravehicular Mobility Unit (EMU)",
                    "Description": "The EMU is a spacesuit that provides environmental protection, "
                                   "mobility, life support, and communications for astronauts",
                    "_id": "article_591"
                }
            ], auto_refresh=True
        )
        try:
            tensor_search.bulk_search(
                query=BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(index=self.index_name_1, q="moon outfits", searchableAttributes=["Description"], boost={"Title": [5, 1]}),
                ]),
                marqo_config=self.config,
            )
            raise AssertionError("Boost attributes are not in searchable attributes")
        except InvalidArgError:
            pass

    def test_bulk_search_multiple_indexes(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_2, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id2-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id2-second"},
            ], auto_refresh=True)
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_3)

        resp = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q="match"),
                BulkSearchQueryEntity(index=self.index_name_2, q="match"),
                BulkSearchQueryEntity(index=self.index_name_3, q="match")
            ]),
            marqo_config=self.config,
        )

        assert len(resp['result']) == 3
        idx1 = resp['result'][0]
        idx2 = resp['result'][1]
        idx3 = resp['result'][2]
        
        assert all([r["_id"][:4] == "id1-" for r in idx1["hits"]])
        assert all([r["_id"][:4] == "id2-" for r in idx2["hits"]])
        assert len(idx3['hits']) == 0

    @mock.patch("marqo.tensor_search.tensor_search.bulk_msearch")
    def test_bulk_search_multiple_queries_single_msearch_request(self, mock_bulk_msearch):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True
        )
        tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q="one thing"),
                BulkSearchQueryEntity(index=self.index_name_1, q="two things"),
                BulkSearchQueryEntity(index=self.index_name_1, q="many things")
            ]),
            marqo_config=self.config,
        )

        self.assertEqual(mock_bulk_msearch.call_count, 1)

    def test_bulk_search_different_models_separate_vectorise_calls(self):
        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @unittest.mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings={
                "index_defaults": {
                    "normalize_embeddings": False,
                }
            })
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_2)
            tensor_search.bulk_search(
                query=BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(index=self.index_name_1, q="one thing"),
                    BulkSearchQueryEntity(index=self.index_name_2, q="two things"),
                ]), marqo_config=self.config
            )
            mock_vectorise.assert_any_call(
                model_name='hf/all_datasets_v4_MiniLM-L6',
                model_properties={'name': 'flax-sentence-embeddings/all_datasets_v4_MiniLM-L6', 'dimensions': 384, 'tokens': 128, 'type': 'hf', 'notes': ''},
                content=['one thing'],
                device='cpu',
                normalize_embeddings=False,
                image_download_headers=None
            )

            print("ock_vectorise.call_args_listock_vectorise.call_args_list",  mock_vectorise.call_args_list)
            mock_vectorise.assert_any_call(
                model_name='hf/all_datasets_v4_MiniLM-L6',
                model_properties={'name': 'flax-sentence-embeddings/all_datasets_v4_MiniLM-L6', 'dimensions': 384,
                                  'tokens': 128, 'type': 'hf', 'notes': ''},
                content=['two things'],
                device='cpu',
                normalize_embeddings=True,
                image_download_headers=None
            )

            self.assertEqual(mock_vectorise.call_count, 2)
            return True
        assert run()

    @mock.patch("marqo.s2_inference.s2_inference.vectorise")
    def test_bulk_search_different_index_same_vector_models_single_vectorise_calls(self, mock_vectorise):

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise
        @unittest.mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run ():
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_2)
            tensor_search.bulk_search(
                query=BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(index=self.index_name_1, q="one thing"),
                    BulkSearchQueryEntity(index=self.index_name_1, q="two things"),
                ]), marqo_config=self.config
            )
            mock_vectorise.assert_any_call(
                model_name='hf/all_datasets_v4_MiniLM-L6',
                model_properties={'name': 'flax-sentence-embeddings/all_datasets_v4_MiniLM-L6', 'dimensions': 384,
                                  'tokens': 128, 'type': 'hf', 'notes': ''},
                content=['one thing', 'two things'],
                device='cpu',
                normalize_embeddings=True,
                image_download_headers=None
            )
            self.assertEqual(mock_vectorise.call_count, 1)
            return True
        assert run()


    def test_bulk_search_multiple_search_methods(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True)

        resp = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q="match", searchMethod="TENSOR"),
                BulkSearchQueryEntity(index=self.index_name_1, q="random text", searchableAttributes=["abc"], searchMethod="LEXICAL"),
            ]),
            marqo_config=self.config,
        )

        assert len(resp['result']) == 2
        tensor_result = resp['result'][0]
        lexical_result = resp['result'][1]

        assert lexical_result["hits"] != []
        assert lexical_result["hits"][0]["_id"] == "id1-second" # Exact match with Lexical search should be first.

        assert len(tensor_result["hits"]) > 0
    
    def test_bulk_search_highlight_per_search_query(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True)

        resp = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q="match", showHighlights=True),
                BulkSearchQueryEntity(index=self.index_name_1, q="match", showHighlights=False),
            ]),
            marqo_config=self.config,
        )
        assert len(resp['result']) == 2
        idx1 = resp['result'][0]
        idx2 = resp['result'][1]

        for h in idx1["hits"]:
            assert len(h.get("_highlights", [])) > 0

        for h in idx2["hits"]:
            assert "_highlights" not in h.keys()

    @mock.patch("marqo.s2_inference.reranking.rerank.rerank_search_results")
    def test_bulk_search_rerank_per_search_query(self, mock_rerank_search_results):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True)

        resp = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q="match with ranking", searchableAttributes=["abc", "other field"], reRanker='_testing'),
                BulkSearchQueryEntity(index=self.index_name_1, q="match", searchableAttributes=["abc", "other field"]),
            ]),
            marqo_config=self.config,
        )

        self.assertEqual(mock_rerank_search_results.call_count, 1)

        call_args = mock_rerank_search_results.call_args_list
        assert len(call_args) == 1

        call_arg = call_args[0].kwargs
        assert call_arg['query'] == "match with ranking"
        assert call_arg['model_name'] == '_testing'
        assert call_arg['device'] == self.config.search_device
        assert call_arg['num_highlights'] == 1
        
    def test_bulk_search_rerank_invalid(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "id1-first"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "id1-second"},
            ], auto_refresh=True)

        try:
            tensor_search.bulk_search(
                query=BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(index=self.index_name_1, q="match", reRanker='_testing'),
                ]),
                marqo_config=self.config,
            )
            raise AssertionError("Cannot use reranker with no searchableAttributes")
        except InvalidArgError:
            pass


    def test_each_doc_returned_once(self):
        """TODO: make sure each return only has one doc for each ID,
                - esp if matches are found in multiple fields
        """
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe efgh ", "other field": "baaadd efgh ",
                 "_id": "5678", "finally": "some field efgh "},
                {"abc": "shouldn't really match ", "other field": "Nope.....",
                 "_id": "1234", "finally": "Random text here efgh "},
            ], auto_refresh=True)
        search_res = tensor_search._bulk_vector_text_search(
            config=self.config, queries=[BulkSearchQueryEntity(index=self.index_name_1, q=" efgh ", limit=10)]
        )
        assert len(search_res) == 1
        assert len(search_res[0]['hits']) == 2

    def test_bulk_vector_text_search_against_empty_index(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        search_res = tensor_search._bulk_vector_text_search(
            config=self.config, queries=[BulkSearchQueryEntity(index=self.index_name_1, q=" efgh ", limit=10)]
        )
        assert len(search_res) > 0
        assert len(search_res[0]['hits']) == 0


    def test_bulk_vector_text_search_against_non_existent_index(self):
        try:
            tensor_search._bulk_vector_text_search(
                config=self.config, queries=[BulkSearchQueryEntity(index=self.index_name_1, q=" efgh ", limit=10)]
            )
            raise AssertionError
        except IndexNotFoundError:
            pass

    def test_bulk_vector_text_search_long_query_string(self):
        query_text = """The Guardian is a British daily newspaper. It was founded in 1821 as The Manchester Guardian, and changed its name in 1959.[5] Along with its sister papers The Observer and The Guardian Weekly, The Guardian is part of the Guardian Media Group, owned by the Scott Trust.[6] The trust was created in 1936 to "secure the financial and editorial independence of The Guardian in perpetuity and to safeguard the journalistic freedom and liberal values of The Guardian free from commercial or political interference".[7] The trust was converted into a limited company in 2008, with a constitution written so as to maintain for The Guardian the same protections as were built into the structure of the Scott Trust by its creators. Profits are reinvested in journalism rather than distributed to owners or shareholders.[7] It is considered a newspaper of record in the UK.[8][9]
                    The editor-in-chief Katharine Viner succeeded Alan Rusbridger in 2015.[10][11] Since 2018, the paper's main newsprint sections have been published in tabloid format. As of July 2021, its print edition had a daily circulation of 105,134.[4] The newspaper has an online edition, TheGuardian.com, as well as two international websites, Guardian Australia (founded in 2013) and Guardian US (founded in 2011). The paper's readership is generally on the mainstream left of British political opinion,[12][13][14][15] and the term "Guardian reader" is used to imply a stereotype of liberal, left-wing or "politically correct" views.[3] Frequent typographical errors during the age of manual typesetting led Private Eye magazine to dub the paper the "Grauniad" in the 1960s, a nickname still used occasionally by the editors for self-mockery.[16]
                    """
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"_id": "12345", "Desc": "The Guardian is newspaper, read in the UK and other places around the world"},
                {"_id": "abc12334", "Title": "Grandma Jo's family recipe. ",
                 "Steps": "1. Cook meat. 2: Dice Onions. 3: Serve."},
            ], auto_refresh=True)
        search_res = tensor_search._bulk_vector_text_search(
            config=self.config, queries=[BulkSearchQueryEntity(index=self.index_name_1, q=query_text)],

        )
        assert len(search_res) == 1
        assert len(search_res[0]['hits']) == 2

    def test_bulk_vector_text_search_searchable_attributes(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
            ], auto_refresh=True)
        search_res = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", searchableAttributes=["other field"])]),
            marqo_config=self.config,
        )
        assert len(search_res['result']) == 1
        assert search_res['result'][0]["hits"][0]["_id"] == "1234"
        for res in search_res['result'][0]["hits"]:
            assert list(res["_highlights"].keys()) == ["other field"]

    def test_bulk_vector_text_search_searchable_attributes_multiple(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd",
                 "Cool Field 1": "res res res", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
                {"Cool Field 1": "somewhat match", "_id": "9000"}
            ], auto_refresh=True)
        search_res = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(
                index=self.index_name_1, q="Exact match hehehe", searchableAttributes=["other field", "Cool Field 1"]
            )]),
            marqo_config=self.config,
        )
        assert len(search_res['result']) == 1
        ids = [search_res['result'][0]["hits"][0]["_id"], search_res['result'][0]["hits"][1]["_id"]]
        assert "1234" in ids
        assert "9000" in ids
        assert ids[0] != ids[1]
        for res in search_res['result'][0]["hits"]:
            assert "abc" not in res["_highlights"]

    def test_search_format(self):
        """Is the result formatted correctly?"""
        q = "Exact match hehehe"
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd",
                 "Cool Field 1": "res res res", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
                {"Cool Field 1": "somewhat match", "_id": "9000"}
            ], auto_refresh=True)
        search_res = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(
                index=self.index_name_1, q=q, searchableAttributes=["other field", "Cool Field 1"], limit=50
            )]),
            marqo_config=self.config,
        )
        assert len(search_res["result"]) == 1
        print(search_res)
        assert "processingTimeMs" in search_res
        assert search_res["processingTimeMs"] > 0
        assert isinstance(search_res["processingTimeMs"], int)

        assert "query" in search_res['result'][0]
        assert search_res['result'][0]["query"] == q

        assert "limit" in search_res['result'][0]
        assert search_res['result'][0]["limit"] == 50

    def test_result_count_validation(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "Exact match hehehe", "other field": "baaadd",
                 "Cool Field 1": "res res res", "_id": "5678"},
                {"abc": "random text", "other field": "Close match hehehe", "_id": "1234"},
                {"Cool Field 1": "somewhat match", "_id": "9000"}
            ], auto_refresh=True)
        try:
            # too big
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", limit=-1)], ))
            raise AssertionError
        except IllegalRequestedDocCount as e:
            pass
        try:
            # too small
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", limit=1000000)], ))
            raise AssertionError
        except IllegalRequestedDocCount as e:
            pass
        try:
            # should not work with 0
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", limit=0)], ))
            raise AssertionError
        except IllegalRequestedDocCount as e:
            pass
        # should work with 1:
        search_results = tensor_search.bulk_search(
            marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="Exact match hehehe", limit=1)], ))
        assert len(search_results["result"]) >= 1
        assert len(search_results["result"][0]['hits']) >= 1

    def test_highlights_tensor(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234"},
            ], auto_refresh=True)

        results = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some text", showHighlights=True)]), marqo_config=self.config
        )
        assert len(results["result"]) == 1
        tensor_highlights = results["result"][0]
        assert len(tensor_highlights["hits"]) == 2
        for hit in tensor_highlights["hits"]:
            assert "_highlights" in hit

        results = tensor_search.bulk_search(
            query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some text", showHighlights=False)]), marqo_config=self.config
        )
        assert len(results["result"]) == 1
        tensor_no_highlights = results["result"][0]
        assert len(tensor_no_highlights["hits"]) == 2
        for hit in tensor_no_highlights["hits"]:
            assert "_highlights" not in hit

    def test_search_vector_int_field(self):
        """doesn't error out if there is a random int field"""
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_int": 144},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "my_int": 88},
            ], auto_refresh=True)

        results = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="88")],
            config=self.config)
        assert len(results) == 1
        s_res = results[0]
        assert len(s_res["hits"]) > 0

    def test_filtering_list_case_tensor(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"abc": "some text", "_id": "1235",  "my_list": ["tag1", "tag2 some"]}
            ], auto_refresh=True, non_tensor_fields=["my_list"])

        res_exists = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_list:tag1")],
            config=self.config)
        res_not_exists = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_list:tag55")],
            config=self.config)
        res_other = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_string:b")],
            config=self.config)

        # strings in lists are converted into keyword, which aren't filterable on a token basis.
        # Because the list member is "tag2 some" we can only exact match (incl. the space).
        # "tag2" by itself doesn't work, only "(tag2 some)"
        res_should_only_match_keyword_bad = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_list:tag2")],
            config=self.config)
        res_should_only_match_keyword_good = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="", filter="my_list:(tag2 some)")],
            config=self.config)
        assert res_exists[0]["hits"][0]["_id"] == "1235"
        assert res_exists[0]["hits"][0]["_highlights"] == {"abc": "some text"}
        assert len(res_exists[0]["hits"]) == 1

        assert len(res_not_exists[0]["hits"]) == 0

        assert res_other[0]["hits"][0]["_id"] == "5678"
        assert len(res_other[0]["hits"]) == 1

        assert len(res_should_only_match_keyword_bad[0]["hits"]) == 0
        assert len(res_should_only_match_keyword_good[0]["hits"]) == 1

    def test_filtering_list_case_image(self):
        settings = {"index_defaults": {"treat_urls_and_pointers_as_images": True, "model": "ViT-B/32"}}
        tensor_search.create_vector_index(index_name=self.index_name_1, index_settings=settings, config=self.config)
        hippo_img = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"img": hippo_img, "abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"img": hippo_img, "abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"img": hippo_img, "abc": "some text", "_id": "1235", "my_list": ["tag1", "tag2 some"]}
            ], auto_refresh=True, non_tensor_fields=["my_list"])

        response = tensor_search._bulk_vector_text_search(
            queries=[
                BulkSearchQueryEntity(index=self.index_name_1, q=hippo_img, filter="my_list:tag1"),
                BulkSearchQueryEntity(index=self.index_name_1, q=hippo_img, filter="my_list:not_exist"),
                BulkSearchQueryEntity(index=self.index_name_1, q="some", filter="my_list:tag1"),
                BulkSearchQueryEntity(index=self.index_name_1, q="some", filter="my_list:not_exist")
            ],
            config=self.config
        )

        assert len(response) == 4
        res_img_2_img, res_img_2_img_none, res_txt_2_img, res_txt_2_imgs2 = response

        assert len(res_txt_2_imgs2["hits"]) == 0
        assert len(res_img_2_img_none["hits"]) == 0

        assert res_txt_2_img["hits"][0]["_id"] == "1235"
        assert res_txt_2_img["hits"][0]["_highlights"] == {"abc": "some text"}
        assert len(res_txt_2_img["hits"]) == 1

        assert res_img_2_img["hits"][0]["_id"] == "1235"
        assert len(res_img_2_img["hits"]) == 1
        assert res_img_2_img["hits"][0]["_highlights"] == {"img": hippo_img}


    def test_filtering(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1233", "my_bool": True}
            ], auto_refresh=True)

        filter_strings = ["my_string:c", "an_int:2", "my_string:b", "my_int_something:5", "an_int:[5 TO 30]", "an_int:[0 TO 30]", "my_bool:true", "an_int:[0 TO 30] OR my_bool:true", "(an_int:[0 TO 30] and an_int:2) AND abc:(some text)"]
        expected_ids = [[], ["1234"], ["5678"], [],[], ["1234"], ["1233"], ["1233", "1234"], ["1234"]]
        for i in range(len(filter_strings)):
            result = tensor_search._bulk_vector_text_search(
                queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some", filter=filter_strings[i])],
                config=self.config
            )
            assert len(result) == 1
            assert len(result[0]["hits"]) == len(expected_ids[i])
            if len(expected_ids[i]) > 0:
                for j in range(len(result[0]["hits"])):
                    assert result[0]["hits"][j]["_id"] in expected_ids[i]

    def test_filter_spaced_fields(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=[
                {"abc": "some text", "other field": "baaadd", "_id": "5678", "my_string": "b"},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1234", "an_int": 2},
                {"abc": "some text", "other field": "Close match hehehe", "_id": "1233", "my_bool": True},
                {"abc": "some text", "Floaty Field": 0.548, "_id": "344", "my_bool": True},
            ], auto_refresh=True)

        filter_to_ids = {
            "other\ field:baaadd": ["5678"],
            "other\ field:(Close match hehehe)": ['1234', '1233'],
            "(Floaty\ Field:[0 TO 1]) AND (abc:(some text))": ["344"]
        }
        response = tensor_search._bulk_vector_text_search(
            queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some", filter=f)
                for f in filter_to_ids.keys()         
            ],
            config=self.config
        )

        assert len(response) == len(filter_to_ids.keys())
        for i, expected in enumerate(filter_to_ids.values()):
            result = response[i]
            assert len(result["hits"]) == len(expected)
            if len(expected) > 0:
                for j in range(len(result["hits"])):
                    assert result["hits"][j]["_id"] in expected


    def test_set_device(self):
        """calling search with a specified device overrides device defined in config"""
        mock_config = copy.deepcopy(self.config)
        mock_config.search_device = "cpu"
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(queries=[BulkSearchQueryEntity(index=self.index_name_1, q="some")]), device="cuda:123")
            return True

        assert run()
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:123"
        assert mock_config.search_device == "cpu"

    def test_search_other_types_subsearch(self):
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[{
                "an_int": 1,
                "a_float": 1.2,
                "a_bool": True,
                "some_str": "blah"
            }])
        for to_search in [1, 1.2, True, "blah"]:
            results = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(index=self.index_name_1, q=str(to_search)
                 )])
            )
            assert "hits" in results["result"][0]

    def test_search_other_types_top_search(self):
        docs = [{
            "an_int": 1,
            "a_float": 1.2,
            "a_bool": True,
            "some_str": "blah"
        }]
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=docs)
        for field, to_search in docs[0].items():
            results = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(index=self.index_name_1, q=str(to_search), filter=f"{field}:{to_search}")])
            )
            assert "hits" in results["result"][0]
            results = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1, q=str(to_search), searchMethod=SearchMethod.LEXICAL,
                        filter=f"{field}:{to_search}")])
            )
            assert "hits" in results["result"][0]

    def test_attributes_to_retrieve_vector(self):
        docs = {
            "5678": {"abc": "Exact match hehehe", "other field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
            "rgjknrgnj": {"Cool Field 1": "somewhat match", "_id": "rgjknrgnj",
                          "abc": "random text", "other field": "Close match hehehe"},
            "9000": {"Cool Field 1": "somewhat match", "_id": "9000", "other field": "weewowow"}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        results = tensor_search.bulk_search(
            marqo_config=self.config, query=BulkSearchQuery(
                queries=[BulkSearchQueryEntity(
                    index=self.index_name_1,
                    q="Exact match hehehe",
                    attributesToRetrieve=["other field", "Cool Field 1"]
                )], ))

        assert len(results['result']) == 1
        search_res = results['result'][0]
        assert len(search_res["hits"]) == 3
        for res in search_res["hits"]:
            assert docs[res["_id"]]["other field"] == res["other field"]
            assert docs[res["_id"]]["Cool Field 1"] == res["Cool Field 1"]
            assert set(k for k in res.keys() if k not in TensorField.__dict__.values()) == \
                   {"other field", "Cool Field 1", "_id"}

    def test_attributes_to_retrieve_empty(self):
        docs = {
            "5678": {"abc": "Exact match hehehe", "other field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
            "rgjknrgnj": {"Cool Field 1": "somewhat match", "_id": "rgjknrgnj",
                          "abc": "random text", "other field": "Close match hehehe"},
            "9000": {"Cool Field 1": "somewhat match", "_id": "9000", "other field": "weewowow"}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        for method in ("LEXICAL", "TENSOR"):
            search_res = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                queries=[BulkSearchQueryEntity(
                    index=self.index_name_1,
                    q="Exact match hehehe",
                    attributesToRetrieve=[],
                    searchMethod=method
                )]
            ))
            assert len(search_res['result'][0]["hits"]) == 3
            print(search_res)
            for res in search_res['result'][0]["hits"]:
                print("res=", res)
                assert set(k for k in res.keys() if k not in TensorField.__dict__.values()) == {"_id"}
    
    def test_attributes_to_retrieve_empty_index(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        assert 0 == tensor_search.get_stats(config=self.config, index_name=self.index_name_1)['numberOfDocuments']
        for to_retrieve in [[], ["some field name"], ["some field name", "wowowow field"]]:
            for method in ("LEXICAL", "TENSOR"):
                search_res = tensor_search.bulk_search(
                    marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q="Exact match hehehe",
                        attributesToRetrieve=to_retrieve,
                        searchMethod=method
                    )]
                ))
                assert len(search_res['result']) > 0
                search_res = search_res['result'][0]
                
                assert len(search_res["hits"]) == 0
                assert search_res['query'] == "Exact match hehehe"
    #
    def test_attributes_to_retrieve_non_existent(self):
        docs = {
            "5678": {"abc": "Exact a match hehehe", "other field": "baaadd",
                     "Cool Field 1": "res res res", "_id": "5678"},
            "rgjknrgnj": {"Cool Field 1": "somewhata  match", "_id": "rgjknrgnj",
                          "abc": "random a text", "other field": "Close match hehehe"},
            "9000": {"Cool Field 1": "somewhat a match", "_id": "9000", "other field": "weewowow"}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        for to_retrieve in [[], ["non existing field name"], ["other field", "non existing field name"]]:
            for method in ("TENSOR", "LEXICAL"):
                search_res = tensor_search.bulk_search(
                    marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q="Exact match hehehe",
                        attributesToRetrieve=to_retrieve,
                        searchMethod=method
                    )]
                ))
                assert len(search_res['result']) > 0
                search_res = search_res['result'][0]
                assert len(search_res["hits"]) == 3
                for res in search_res["hits"]:
                    assert "non existing field name" not in res
                    assert set(k for k in res.keys()
                               if k not in TensorField.__dict__.values() and k != "_id"
                               ).issubset(to_retrieve)

    def test_attributes_to_retrieve_and_searchable_attribs(self):
        docs = {
            "i_1": {"field_1": "a", "other field": "baaadd",
                    "Cool Field 1": "res res res", "_id": "i_1"},
            "i_2": {"field_1": "a", "_id": "i_2",
                    "field_2": "a", "other field": "Close match hehehe"},
            "i_3": {"field_1": " a ", "_id": "i_3", "field_2": "a",
                    "field_3": "a "}
        }
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1, docs=list(docs.values()), auto_refresh=True)
        for to_retrieve, to_search, expected_ids, expected_fields in [
            (["field_1"], ["field_3"], ["i_3"], ["field_1"]),
            (["field_3"], ["field_1"], ["i_1", "i_2", "i_3"], ["field_3"]),
            (["field_1", "field_2"], ["field_2", "field_3"], ["i_2", "i_3"], ["field_1", "field_2"]),
        ]:
            for method in ("TENSOR", "LEXICAL"):
                search_res = tensor_search.bulk_search(
                    marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q="a",
                        attributesToRetrieve=to_retrieve,
                        searchableAttributes=to_search,
                        searchMethod=method
                    )]
                ))
                assert len(search_res["result"]) > 0
                search_res = search_res["result"][0]

                assert len(search_res["hits"]) == len(expected_ids)
                assert set(expected_ids) == {h['_id'] for h in search_res["hits"]}
                for res in search_res["hits"]:
                    relevant_fields = set(expected_fields).intersection(set(docs[res["_id"]].keys()))
                    assert set(k for k in res.keys()
                               if k not in TensorField.__dict__.values() and k != "_id"
                               ) == relevant_fields

    def test_limit_results(self):
        """"""
        vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"
    
        vocab = requests.get(vocab_source).text.splitlines()
    
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"Title": "a " + (" ".join(random.choices(population=vocab, k=25)))}
                  for _ in range(2000)], auto_refresh=False
        )
        tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)
        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            for max_doc in [0, 1, 2, 5, 10, 100, 1000]:
    
                mock_environ = {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(max_doc)}
    
                @mock.patch("os.environ", mock_environ)
                def run():
                    half_search = tensor_search.bulk_search(
                        marqo_config=self.config, query=BulkSearchQuery(
                        queries=[BulkSearchQueryEntity(
                            index=self.index_name_1,
                            q="a",
                            searchMethod=search_method,
                            limit=max_doc//2
                        )]
                    ))
                    assert len(half_search['result']) > 0 
                    half_search = half_search['result'][0]

                    assert half_search['limit'] == max_doc//2
                    assert len(half_search['hits']) == max_doc//2
                    limit_search = tensor_search.bulk_search(
                        marqo_config=self.config, query=BulkSearchQuery(
                        queries=[BulkSearchQueryEntity(
                            index=self.index_name_1,
                            q="a",
                            searchMethod=search_method,
                            limit=max_doc
                        )]
                    ))
                    assert len(limit_search['result']) > 0 
                    limit_search = limit_search['result'][0]

                    assert limit_search['limit'] == max_doc
                    assert len(limit_search['hits']) == max_doc
                    try:
                        tensor_search.bulk_search(
                            marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q="a",
                                searchMethod=search_method,
                                limit=max_doc+1
                                )]
                        ))
                        raise AssertionError("Should not be able to search with limit > max_docs")
                    except IllegalRequestedDocCount:
                        pass

                    try:
                        tensor_search.bulk_search(
                            marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q="a",
                                searchMethod=search_method,
                                limit=(max_doc+1) * 2
                                )]
                        ))
                        raise AssertionError("Should not be able to search with limit > max_docs")
                    except IllegalRequestedDocCount:
                        pass
                    return True
            assert run()
    
    def test_limit_results_none(self):
        """if env var isn't set or is None"""
        vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"
    
        vocab = requests.get(vocab_source).text.splitlines()
    
        tensor_search.add_documents_orchestrator(
            config=self.config, index_name=self.index_name_1,
            docs=[{"Title": "a " + (" ".join(random.choices(population=vocab, k=25)))}
                  for _ in range(700)], auto_refresh=False, processes=4, batch_size=50
        )
        tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)
    
        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            for mock_environ in [dict(), {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: None},
                                 {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: ''}]:
                @mock.patch("os.environ", mock_environ)
                def run():
                    lim = 500
                    half_search = tensor_search.bulk_search(
                            marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q="a",
                                searchMethod=search_method,
                                limit=lim
                            )]
                        ))
                    assert len(half_search['result']) > 0 
                    half_search = half_search['result'][0]

                    assert half_search['limit'] == lim
                    assert len(half_search['hits']) == lim
                    return True
    
                assert run()

    def test_pagination_single_field(self):
        vocab_source = "https://www.mit.edu/~ecprice/wordlist.10000"
    
        vocab = requests.get(vocab_source).text.splitlines()
        num_docs = 2000
    
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"Title": "a " + (" ".join(random.choices(population=vocab, k=25))),
                    "_id": str(i)
                    }
                  for i in range(num_docs)], auto_refresh=False
        )
        tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)
    
        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            for doc_count in [2000]:
                # Query full results
                full_search_results = tensor_search.bulk_search(
                            marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q="a",
                                searchMethod=search_method,
                                limit=doc_count
                            )]
                        ))
                full_search_results = full_search_results['result'][0]
    
                for page_size in [5, 10, 100, 1000, 2000]:
                    paginated_search_results = {"hits": []}
    
                    for page_num in range(math.ceil(num_docs / page_size)):
                        lim = page_size
                        off = page_num * page_size
                        page_res = tensor_search.bulk_search(
                            marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q="a",
                                searchMethod=search_method,
                                limit=lim,
                                offset=off
                            )]
                        ))
                        page_res = page_res['result'][0]
                        paginated_search_results["hits"].extend(page_res["hits"])
    
                    # Compare paginated to full results (length only for now)
                    assert len(full_search_results["hits"]) == len(paginated_search_results["hits"])
    

    def test_pagination_break_limitations(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        # Negative offset
        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            for lim in [1, 10, 1000]:
                for off in [-1, -10, -1000]:
                    try:
                        tensor_search.bulk_search(
                            marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q=" ",
                                searchMethod=search_method,
                                limit=lim,
                                offset=off
                            )]
                        ))
                        raise AssertionError
                    except IllegalRequestedDocCount:
                        pass
    
        # Negative limit
        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            for lim in [0, -1, -10, -1000]:
                for off in [1, 10, 1000]:
                    try:
                        tensor_search.bulk_search(
                            marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q=" ",
                                searchMethod=search_method,
                                limit=lim,
                                offset=off
                            )]
                        ))
                        raise AssertionError
                    except IllegalRequestedDocCount:
                        pass
    
        # Going over 10,000 for offset + limit
        mock_environ = {EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: "10000"}
        @mock.patch("os.environ", mock_environ)
        def run():
            for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
                try:
                    tensor_search.bulk_search(
                            marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q=" ",
                                searchMethod=search_method,
                                limit=10000,
                                offset=1
                            )]
                        ))
                    raise AssertionError
                except IllegalRequestedDocCount:
                    pass
    
            return True
    
        assert run()

    def test_pagination_multi_field_error(self):
        # Try pagination with 0, 2, and 3 fields
        # To be removed when multi-field pagination is added.
        docs = [
            {
                "field_a": 0,
                "field_b": 0,
                "field_c": 0
            },
            {
                "field_a": 1,
                "field_b": 1,
                "field_c": 1
            }
        ]
    
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=False
        )
    
        tensor_search.refresh_index(config=self.config, index_name=self.index_name_1)
    
        for search_method in (SearchMethod.LEXICAL, SearchMethod.TENSOR):
            try:
                tensor_search.bulk_search(
                    marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=" ",
                        searchMethod=search_method,
                        searchableAttributes=["field_a", "field_b"],
                        offset=1
                    )]
                ))
                raise AssertionError
            except InvalidArgError:
                pass
    
            try:
                tensor_search.bulk_search(
                    marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=" ",
                        searchMethod=search_method,
                        offset=1
                    )]
                ))
                raise AssertionError
            except InvalidArgError:
                pass
    
            try:
                tensor_search.bulk_search(
                    marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=" ",
                        searchMethod=search_method,
                        searchableAttributes=[],
                        offset=1
                    )]
                ))
                raise AssertionError
            except InvalidArgError:
                pass

    def test_image_search_highlights(self):
        """does the URL get returned as the highlight? (it should - because no rerankers are being used)"""
        settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": "ViT-B/32",
            }}
        tensor_search.create_vector_index(
            index_name=self.index_name_1, index_settings=settings, config=self.config
        )
        url_1 = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
        url_2 = "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"
        docs = [
            {"_id": "123",
             "image_field": url_1,
             "text_field": "some words here"
             },
            {"_id": "789",
             "image_field": url_2},
        ]
        tensor_search.add_documents(
            config=self.config, auto_refresh=True, index_name=self.index_name_1, docs=docs
        )
        res = tensor_search.bulk_search(
            marqo_config=self.config, query=BulkSearchQuery(
            queries=[BulkSearchQueryEntity(
                index=self.index_name_1,
                q="some text",
                limit=3,
                searchableAttributes=['image_field'],
            )]
        ))
        res = res['result'][0]

        assert len(res['hits']) == 2
        assert {hit['image_field'] for hit in res['hits']} == {url_2, url_1}
        assert {hit['_highlights']['image_field'] for hit in res['hits']} == {url_2, url_1}

    def test_multi_search(self):
        docs = [
            {"field_a": "Doberman, canines, golden retrievers are humanity's best friends",
             "_id": 'dog_doc'},
            {"field_a": "All things poodles! Poodles are great pets",
             "_id": 'poodle_doc'},
            {"field_a": "Construction and scaffolding equipment",
             "_id": 'irrelevant_doc'}
        ]
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True
        )
        queries_expected_ordering = [
            ({"Dogs": 2.0, "Poodles": -2}, ['dog_doc', 'irrelevant_doc', 'poodle_doc']),
            ("dogs", ['dog_doc', 'poodle_doc', 'irrelevant_doc']),
            ({"dogs": 1}, ['dog_doc', 'poodle_doc', 'irrelevant_doc']),
            ({"Dogs": -2.0, "Poodles": 2}, ['poodle_doc', 'irrelevant_doc', 'dog_doc']),
        ]
        for query, expected_ordering in queries_expected_ordering:
            res = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                queries=[BulkSearchQueryEntity(
                    index=self.index_name_1,
                    q=query,
                    limit=5,
                    searchMethod=SearchMethod.TENSOR,
                )]
            ))
            res = res['result'][0]
            # the poodle doc should be lower ranked than the irrelevant doc
            for hit_position, _ in enumerate(res['hits']):
                assert res['hits'][hit_position]['_id'] == expected_ordering[hit_position]

    def test_multi_search_images(self):
        docs = [
            {"loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
             "_id": 'realistic_hippo'},
            {"loc b": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png",
             "_id": 'artefact_hippo'}
        ]
        image_index_config = {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.model: "ViT-B/16",
                IndexSettingsField.treat_urls_and_pointers_as_images: True
            }
        }
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True
        )
        queries_expected_ordering = [
            ({"Nature photography": 2.0, "Artefact": -2}, ['realistic_hippo', 'artefact_hippo']),
            ({"Nature photography": -1.0, "Artefact": 1.0}, ['artefact_hippo', 'realistic_hippo']),
            ({"Nature photography": -1.5, "Artefact": 1.0, "hippo": 1.0}, ['artefact_hippo', 'realistic_hippo']),
            ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": -1.0,
              "blah": 1.0}, ['realistic_hippo', 'artefact_hippo']),
            ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
              "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0},
             ['artefact_hippo', 'realistic_hippo']),
            ({"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
              "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0,
              "artefact": 1.0, "photo realistic": -1,
              },
             ['artefact_hippo', 'realistic_hippo']),
        ]
        for query, expected_ordering in queries_expected_ordering:
            res = tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                queries=[BulkSearchQueryEntity(
                    index=self.index_name_1,
                    q=query,
                    limit=5,
                    searchMethod=SearchMethod.TENSOR,
                )]
            ))
            res = res['result'][0]
            # the poodle doc should be lower ranked than the irrelevant doc
            for hit_position, _ in enumerate(res['hits']):
                assert res['hits'][hit_position]['_id'] == expected_ordering[hit_position]

    def test_multi_search_images_edge_cases(self):
        docs = [
            {"loc": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
             "_id": 'realistic_hippo'},
            {"field_a": "Some text about a weird forest",
             "_id": 'artefact_hippo'}
        ]
        image_index_config = {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.model: "ViT-B/16",
                IndexSettingsField.treat_urls_and_pointers_as_images: True
            }
        }
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True
        )
        invalid_queries = [[{}], [set()], [{"https://marqo_not_real.com/image_1.png": 3}], [{}, set()], [set(), {"https://marqo_not_real.com/image_1.png": 3}]]
        for qs in invalid_queries:
            try:
                tensor_search.bulk_search(
                    marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=q,
                        limit=5,
                        searchMethod=SearchMethod.TENSOR,
                    ) for q in qs]
                ))
                raise AssertionError(f"Invalid query {qs} did not raise error")
            except (InvalidArgError, BadRequestError) as e:
                pass

    def test_multi_search_images_ok_edge_cases(self):
        docs = [
            {"loc": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
             "_id": 'realistic_hippo'},
            {"field_a": "Some text about a weird forest",
             "_id": 'artefact_hippo'}
        ]
        image_index_config = {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.model: "ViT-B/16",
                IndexSettingsField.treat_urls_and_pointers_as_images: True
            }
        }
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True
        )
        alright_queries = [{"v ": 1.2}, {"d ": 0}, {"vf": -1}]
        for q in alright_queries:
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                queries=[BulkSearchQueryEntity(
                    index=self.index_name_1,
                    q=q,
                    limit=5,
                    searchMethod=SearchMethod.TENSOR,
                )]
            ))

    def test_image_search(self):
        """This test is to ensure image search works as expected
        The code paths for image and search have diverged quite a bit
        """
        hippo_image = (
            'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        )
        doc_dict = {
            'realistic_hippo': {"loc": hippo_image,
             "_id": 'realistic_hippo'},
            'artefact_hippo': {"field_a": "Some text about a weird forest",
             "_id": 'artefact_hippo'}
        }
        docs = list(doc_dict.values())
        image_index_config = {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.model: "ViT-B/16",
                IndexSettingsField.treat_urls_and_pointers_as_images: True
            }
        }
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True
        )
        res = tensor_search.bulk_search(
            marqo_config=self.config, query=BulkSearchQuery(
            queries=[BulkSearchQueryEntity(
                index=self.index_name_1,
                q=hippo_image,
                limit=5,
                searchMethod=SearchMethod.TENSOR,
            )]
        ))
        res = res['result'][0]
        assert len(res['hits']) == 2
        for hit in res['hits']:
            original_doc = doc_dict[hit['_id']]
            assert len(hit['_highlights']) == 1
            highlight_field = list(hit['_highlights'].keys())[0]
            assert highlight_field in original_doc
            assert hit[highlight_field] == original_doc[highlight_field]

    @mock.patch('marqo._httprequests.HttpRequests.get')
    def test_bulk_msearch_invalid_response(self, mock_http_get):
        # mock_http_get.side_effect = KeyError()

        # response["responses"][0]["error"]["root_cause"][0]["reason"]
        mock_http_get.return_value = {"responses": [{"error": {"root_cause": [{"reason": "index.max_result_window"}]}}]}
        try:
            tensor_search.bulk_msearch(self.config, [])
            raise AssertionError("With invalid body, msearch should raise error")
        except IllegalRequestedDocCount:
            pass

        # elif 'parse_exception' in response["responses"][0]["error"]["root_cause"][0]["reason"]: 
        mock_http_get.return_value = {"responses": [{"error": {"root_cause": [{"reason": "parse_exception something else"}]}}]}
        try:
            tensor_search.bulk_msearch(self.config, [])
            raise AssertionError("With invalid body, msearch should raise error")
        except InvalidArgError:
            pass

        mock_http_get.return_value = {"responses": [{"error": {"root_cause": [{"reason": "parse_not_exception"}]}}]}
        try:
            tensor_search.bulk_msearch(self.config, [])
            raise AssertionError("With invalid body, msearch should raise error")
        except BackendCommunicationError:
            pass

        mock_http_get.return_value = {}
        try:
            tensor_search.bulk_msearch(self.config, [])
            raise AssertionError("With invalid body, msearch should raise error")
        except KeyError:
            pass

    def test_gather_documents_from_response_remove_no_chunks(self):
        tensor_search.gather_documents_from_response([
            [{"_id": "idddd", "inner_hits": {"__chunks": {"hits": {"hits": []}}}}]
        ])

    def test_bulk_multi_search_check_vector(self):
        """ Ensure multimodal vectors generated by bulk search are correct (only 1 query in bulk search)
        """
        docs = [
            {
                "loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": 'realistic_hippo'},
            {"loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png",
             "_id": 'artefact_hippo'}
        ]
        image_index_config = {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.model: "ViT-B/16",
                IndexSettingsField.treat_urls_and_pointers_as_images: True
            }
        }
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True
        )
        multi_queries = [
            {
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0,
                "artefact": 5.0, "photo realistic": -1,
            },
            {
                "artefact": 5.0,
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
                "photo realistic": -1,
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0
            },
            {
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 3,
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0,
            },
            {
                "hello": 3, "some thing": -1.0,
            },
        ]
        from marqo.tensor_search.utils import dicts_to_jsonl

        for multi_query in multi_queries:
            mock_dicts_to_jsonl = mock.MagicMock()
            mock_dicts_to_jsonl.side_effect = lambda *x, **y: dicts_to_jsonl(*x, **y)

            @mock.patch('marqo.tensor_search.utils.dicts_to_jsonl', mock_dicts_to_jsonl)
            def run() -> List[float]:
                tensor_search.bulk_search(
                        marqo_config=self.config, query=BulkSearchQuery(
                            queries=[BulkSearchQueryEntity(
                                index=self.index_name_1,
                                q=multi_query,
                                limit=5,
                                searchMethod=SearchMethod.TENSOR,
                            )]
                    )
                )
                get_args, get_kwargs = mock_dicts_to_jsonl.call_args
                search_dicts = get_args[0]
                assert len(search_dicts) == 2
                query_dict = search_dicts[1]

                query_vec = query_dict['query']['nested']['query']['knn'][
                    f"{TensorField.chunks}.{utils.generate_vector_name('loc a')}"]['vector']
                return query_vec

            # manually calculate weights:
            weighted_vectors = []
            for q, weight in multi_query.items():
                vec = vectorise(model_name="ViT-B/16", content=[q, ],
                                image_download_headers=None, normalize_embeddings=True)[0]
                weighted_vectors.append(np.asarray(vec) * weight)

            manually_combined = np.mean(weighted_vectors, axis=0)
            norm = np.linalg.norm(manually_combined, axis=-1, keepdims=True)
            if norm > 0:
                manually_combined /= np.linalg.norm(manually_combined, axis=-1, keepdims=True)
            manually_combined = list(manually_combined)

            combined_query = run()
            assert np.allclose(combined_query, manually_combined, atol=1e-6)

    def test_bulk_multi_search_check_vector_multiple_queries(self):
        """ Ensure multimodal vectors generated by bulk search are correct
        (a few queries in bulk search)
        """
        docs = [
            {
                "loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": 'realistic_hippo'},
            {"loc a": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png",
             "_id": 'artefact_hippo'}
        ]
        image_index_config = {
            IndexSettingsField.index_defaults: {
                IndexSettingsField.model: "ViT-B/16",
                IndexSettingsField.treat_urls_and_pointers_as_images: True
            }
        }
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings=image_index_config)
        tensor_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=docs, auto_refresh=True
        )
        multi_queries = [
            {
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0,
                "artefact": 5.0, "photo realistic": -1,
            },
            {
                "artefact": 5.0,
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 2.0,
                "photo realistic": -1,
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0
            },
            {
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png": 3,
                "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png": -1.0,
            },
            {
                "hello": 3, "some thing": -1.0,
            },
        ]
        from marqo.tensor_search.utils import dicts_to_jsonl

        mock_dicts_to_jsonl = mock.MagicMock()
        mock_dicts_to_jsonl.side_effect = lambda *x, **y: dicts_to_jsonl(*x, **y)

        @mock.patch('marqo.tensor_search.utils.dicts_to_jsonl', mock_dicts_to_jsonl)
        def run() -> List[float]:
            tensor_search.bulk_search(
                marqo_config=self.config, query=BulkSearchQuery(
                    queries=[BulkSearchQueryEntity(
                        index=self.index_name_1,
                        q=mq,
                        limit=5,
                        searchMethod=SearchMethod.TENSOR,
                    )
                    for mq in multi_queries
                    ]
                )
            )
            get_args, get_kwargs = mock_dicts_to_jsonl.call_args
            search_dicts = get_args[0]
            assert len(search_dicts) == 2 * len(multi_queries)

            query_dicts = [elem for i, elem in enumerate(search_dicts) if i % 2 == 1]

            query_vecs = [query_dict['query']['nested']['query']['knn'][
                f"{TensorField.chunks}.{utils.generate_vector_name('loc a')}"]['vector']
                           for query_dict in query_dicts]
            return query_vecs
        combined_queries = run()
        assert len(combined_queries) == len(multi_queries)
        for i, multi_query in enumerate(multi_queries):

            # manually calculate weights:
            weighted_vectors = []
            for q, weight in multi_query.items():
                vec = vectorise(model_name="ViT-B/16", content=[q, ],
                                image_download_headers=None, normalize_embeddings=True)[0]
                weighted_vectors.append(np.asarray(vec) * weight)

            manually_combined = np.mean(weighted_vectors, axis=0)
            norm = np.linalg.norm(manually_combined, axis=-1, keepdims=True)
            if norm > 0:
                manually_combined /= np.linalg.norm(manually_combined, axis=-1, keepdims=True)
            manually_combined = list(manually_combined)

            assert np.allclose(combined_queries[i], manually_combined, atol=1e-6)
