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
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from marqo import exceptions as base_exceptions
import unittest
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.tensor_search.models.api_models import ScoreModifierLists
from marqo.tensor_search.models.search import SearchContext
from marqo.tensor_search import api
import numpy as np
from marqo.tensor_search.models.api_models import CustomVectorQuery


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

        cls.indexes = cls.create_indexes([
            structured_text_index_score_modifiers,
        ])

        # Assign to objects so they can be used in tests
        cls.structured_text_index_score_modifiers = cls.indexes[0]

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

        self.results_2_9 = dict()

        self.results_2_9["LEXICAL"] = [
            {'_id': 'doc10',
               'text_field_1': 'dogs dogs dogs',
               '_score': 1.1185285961247438,
               '_highlights': []},
          {'_id': 'doc1',
           'text_field_1': 'dogs',
           '_score': 0.9876369518973803,
           '_highlights': []},
          {'_id': 'doc6',
           'text_field_1': 'hot dogs',
           '_score': 0.7968916178399463,
           '_highlights': []},
          {'_id': 'doc9',
           'text_field_1': 'dogs random words',
           '_score': 0.6678983703478686,
           '_highlights': []},
          {'_id': 'doc8',
           'text_field_1': 'something something dogs',
           'add_field_1': 1.0,
           'mult_field_1': 2.0,
           '_score': 0.6678983703478686,
           '_highlights': []},
          {'_id': 'doc11',
           'text_field_2': 'dogs but wrong field',
           '_score': 0.6369665418754974,
           '_highlights': []},
          {'_id': 'doc7',
           'text_field_1': 'dogs is a word',
           '_score': 0.574847513797856,
           '_highlights': []}
        ]

        self.results_2_9["TENSOR"] = [{'_id': 'doc1',
           'text_field_1': 'dogs',
           '_highlights': [{'text_field_1': 'dogs'}],
           '_score': 1.0},
          {'_id': 'doc10',
           'text_field_1': 'dogs dogs dogs',
           '_highlights': [{'text_field_1': 'dogs dogs dogs'}],
           '_score': 0.901753906564094},
          {'_id': 'doc3',
           'text_field_1': 'canines',
           'add_field_1': 2.0,
           'mult_field_1': 3.0,
           '_highlights': [{'text_field_1': 'canines'}],
           '_score': 0.886741153647037},
          {'_id': 'doc2',
           'text_field_1': 'puppies',
           '_highlights': [{'text_field_1': 'puppies'}],
           '_score': 0.8183335751020673},
          {'_id': 'doc13',
           'text_field_2': 'canines canines',
           '_highlights': [{'text_field_2': 'canines canines'}],
           '_score': 0.817154577803434},
          {'_id': 'doc6',
           'text_field_1': 'hot dogs',
           '_highlights': [{'text_field_1': 'hot dogs'}],
           '_score': 0.7853536191636338},
          {'_id': 'doc12',
           'add_field_1': -1.0,
           'mult_field_1': 0.5,
           'text_field_2': 'puppies puppies',
           '_highlights': [{'text_field_2': 'puppies puppies'}],
           '_score': 0.7440953692828544},
          {'_id': 'doc7',
           'text_field_1': 'dogs is a word',
           '_highlights': [{'text_field_1': 'dogs is a word'}],
           '_score': 0.7431715440181559},
          {'_id': 'doc11',
           'text_field_2': 'dogs but wrong field',
           '_highlights': [{'text_field_2': 'dogs but wrong field'}],
           '_score': 0.7098773257203361},
          {'_id': 'doc9',
           'text_field_1': 'dogs random words',
           '_highlights': [{'text_field_1': 'dogs random words'}],
           '_score': 0.7008764748105084}
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

        for index in [self.structured_text_index_score_modifiers]:
            with self.subTest(index=index.name):
                # Add documents
                tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=self.docs_list
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

                        self.assertEqual(len(search_res["hits"]), len(self.results_2_9[search_method]))
                        for i in range(len(search_res["hits"])):
                            self.assertEqual(search_res["hits"][i]["_id"], self.results_2_9[search_method][i]["_id"])
                            self.assertEqual(search_res["hits"][i]["_score"], self.results_2_9[search_method][i]["_score"])