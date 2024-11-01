import os
import uuid
from unittest import mock

import marqo.core.exceptions as core_exceptions
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.core.structured_vespa_index import common
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.core.models.add_docs_params import AddDocsParams
from tests.marqo_test import MarqoTestCase
from marqo import exceptions as base_exceptions
import unittest
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.tensor_search.models.api_models import ScoreModifierLists
from marqo.tensor_search.models.search import SearchContext
from marqo.tensor_search import api
import numpy as np
from marqo.tensor_search.models.api_models import CustomVectorQuery
from tests.tensor_search.backwards_compat.resources import results_2_9


class TestSearchRegression(MarqoTestCase):
    """
    Tests for search result and score regression
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # STRUCTURED indexes
        structured_text_index_score_modifiers = cls.structured_marqo_index_request(
            model=Model(name="sentence-transformers/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_2", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="text_field_3", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch, FieldFeature.Filter]),
                FieldRequest(name="add_field_1", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="add_field_2", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="mult_field_1", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
                FieldRequest(name="mult_field_2", type=FieldType.Float,
                             features=[FieldFeature.ScoreModifier]),
            ],
            tensor_fields=["text_field_1", "text_field_2", "text_field_3"]
        )

        unstructured_text_index = cls.unstructured_marqo_index_request(
            model=Model(name="sentence-transformers/all-MiniLM-L6-v2")
        )

        cls.indexes = cls.create_indexes([
            structured_text_index_score_modifiers,
            unstructured_text_index
        ])

        # Assign to objects so they can be used in tests
        cls.structured_text_index_score_modifiers = cls.indexes[0]
        cls.unstructured_text_index = cls.indexes[1]

    def setUp(self) -> None:
        super().setUp()

        self.docs_list = [
            # similar semantics to dogs
            {"_id": "doc1", "text_field_1": "dogs"},
            {"_id": "doc2", "text_field_1": "puppies"},
            {"_id": "doc3", "text_field_1": "canines", "add_field_1": 2.0, "mult_field_1": 3.0},
            {"_id": "doc4", "text_field_1": "huskies"},
            {"_id": "doc5", "text_field_1": "four-legged animals"},

            # shares lexical token with dogs
            {"_id": "doc6", "text_field_1": "hot dogs"},
            {"_id": "doc7", "text_field_1": "dogs is a word"},
            {"_id": "doc8", "text_field_1": "something something dogs", "add_field_1": 1.0, "mult_field_1": 2.0},
            {"_id": "doc9", "text_field_1": "dogs random words"},
            {"_id": "doc10", "text_field_1": "dogs dogs dogs"},

            {"_id": "doc11", "text_field_2": "dogs but wrong field"},
            {"_id": "doc12", "text_field_2": "puppies puppies", "add_field_1": -1.0, "mult_field_1": 0.5},
            {"_id": "doc13", "text_field_2": "canines canines"},
        ]

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_search_result_scores_match_2_9(self):
        """
        Tests that both lexical and tensor search results and scores match those
        of Marqo 2.9.0
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_text_index]:
            with self.subTest(index=type(index)):

                docs_with_same_bm25_score = [("doc8", "doc9")]

                # Select results
                if isinstance(index, (StructuredMarqoIndex, SemiStructuredMarqoIndex)):
                    expected_results = results_2_9.search_results_structured
                elif isinstance(index, UnstructuredMarqoIndex):
                    expected_results = results_2_9.search_results_unstructured
                    docs_with_same_bm25_score.append(("doc7", "doc11"))

                # Add documents
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list,
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] if \
                            isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                for search_method in [SearchMethod.TENSOR, SearchMethod.LEXICAL]:
                    with self.subTest(search_method=search_method):
                        # Search
                        search_res = tensor_search.search(
                            config=self.config,
                            index_name=index.name,
                            text="dogs",
                            search_method=search_method,
                            result_count=10
                        )


                        self.assertEqual(len(search_res["hits"]), len(expected_results[search_method]))
                        for i in range(len(search_res["hits"])):
                            # Docs with same bm25 score are interchangeable in order
                            same_score_group = ()
                            for group in docs_with_same_bm25_score:
                                if search_res["hits"][i]["_id"] in group:
                                    same_score_group = group

                            if (search_res["hits"][i]["_id"] in same_score_group) and (search_method == SearchMethod.LEXICAL):
                                self.assertIn(expected_results[search_method][i]["_id"], same_score_group)
                            else:
                                self.assertEqual(search_res["hits"][i]["_id"], expected_results[search_method][i]["_id"])
                            self.assertTrue(np.allclose(search_res["hits"][i]["_score"], expected_results[search_method][i]["_score"], atol=1e-6),
                                            msg=f'Score of Hit {i} do not match in {search_method} search on {index.type} index, '
                                                f'expected: {expected_results[search_method][i]["_score"]}, '
                                                f'actual: {search_res["hits"][i]["_score"]}')

    def test_document_vectors_match_2_9(self):
        """
        Tests that document vectors match those
        of Marqo 2.9.0
        """

        for index in [self.structured_text_index_score_modifiers, self.unstructured_text_index]:
            with self.subTest(index=index.name):
                # Add documents
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list,
                        tensor_fields=["text_field_1", "text_field_2", "text_field_3"] if \
                            isinstance(index, UnstructuredMarqoIndex) else None
                    )
                )

                # Get document
                fetched_doc = tensor_search.get_document_by_id(self.config, index.name,
                                                               "doc10", show_vectors=True)
                self.assertTrue(np.allclose(fetched_doc["_tensor_facets"][0]["_embedding"],
                                 results_2_9.doc_10_embedding, atol=1e-6))