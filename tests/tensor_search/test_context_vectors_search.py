import functools
import os
import pprint
import unittest
import uuid
from unittest import mock

from marqo.core.exceptions import IndexNotFoundError
from marqo.exceptions import InvalidArgumentError
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from unittest.mock import patch
import os
import pprint
from tests.utils.transition import *
from marqo.tensor_search.models.search import SearchContext
from marqo.api.exceptions import InvalidArgError

class TestContextVectors(MarqoTestCase):

    structured_index_with_random_model = "structured_index_with_random_model"
    unstructured_index_with_random_model = "unstructured_index_with_random_model"
    
    # The index in this test is created with 'hf/all-MiniLM-L6-v2' with 384 dimensions
    # Don't use random model for this test suite as we need to guarantee the same query generate the same embeddings
    DIMENSION = 384

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_index_with_random_model_request = cls.structured_marqo_index_request(
            name=cls.structured_index_with_random_model,
            model=Model(name="hf/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(name='text_field_1', type=FieldType.Text),
                FieldRequest(name='image_field_1', type=FieldType.ImagePointer),
            ],
            tensor_fields=["text_field_1", "image_field_1"]
        )
        unstructured_index_with_random_model_request = cls.unstructured_marqo_index_request(
            name=cls.unstructured_index_with_random_model,
            model=Model(name="hf/all-MiniLM-L6-v2"),
        )

        # List of indexes to loop through per test. Test itself should extract index name.
        cls.indexes = cls.create_indexes([
            structured_index_with_random_model_request,
            unstructured_index_with_random_model_request,
        ])

    def setUp(self) -> None:
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_search(self):
        for index_name in [self.structured_index_with_random_model, self.unstructured_index_with_random_model]:
            with self.subTest(msg=index_name):
                query = {
                    "A rider is riding a horse jumping over the barrier": 1,
                }
                res = tensor_search.search(config=self.config, index_name=index_name, text=query,
                                           context=SearchContext(
                                               **{"tensor": [{"vector": [1, ] * self.DIMENSION, "weight": 2},
                                                             {"vector": [2, ] * self.DIMENSION, "weight": -1}]}))

    def test_search_with_incorrect_tensor_dimension(self):
        for index_name in [self.structured_index_with_random_model, self.unstructured_index_with_random_model]:
            with self.subTest(msg=index_name):
                query = {
                    "A rider is riding a horse jumping over the barrier": 1,
                }
                with self.assertRaises(InvalidArgError) as e:
                    tensor_search.search(config=self.config, index_name=index_name, text=query, context=SearchContext(
                        **{"tensor": [{"vector": [1, ] * 3, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}], }))
                self.assertIn("does not match the expected dimension", str(e.exception.message))

    def test_search_with_incorrect_query_format(self):
        for index_name in [self.structured_index_with_random_model, self.unstructured_index_with_random_model]:
            with self.subTest(msg=index_name):
                query = "A rider is riding a horse jumping over the barrier"
                with self.assertRaises(InvalidArgError) as e:
                    res = tensor_search.search(config=self.config, index_name=index_name, text=query, context=
                    SearchContext(
                        **{"tensor": [{"vector": [1, ] * 512, "weight": 0}, {"vector": [2, ] * 512, "weight": 0}]}))
                self.assertIn("This is not supported as the context only works when the query is a dictionary.",
                              str(e.exception.message))

    def test_search_score(self):
        """Test to ensure that the score is the same for the same query with different context vectors combinations."""
        for index_name in [self.structured_index_with_random_model, self.unstructured_index_with_random_model]:
            tensor_fields = ["text_field_1"] if index_name == self.unstructured_index_with_random_model else None
            tensor_search.add_documents(config=self.config, add_docs_params=
                                        AddDocsParams(index_name=index_name,
                                                      docs=[{"text_field_1": "A rider", "_id": "1"}],
                                                      tensor_fields=tensor_fields
                                                      )
                                        )
            with self.subTest(msg=index_name):
                query = {
                    "A rider is riding a horse jumping over the barrier": 1,
                }

                res_1 = tensor_search.search(config=self.config, index_name=index_name, text=query)
                res_2 = tensor_search.search(config=self.config, index_name=index_name, text=query, context=
                SearchContext(**{"tensor": [{"vector": [1, ] * self.DIMENSION, "weight": 0}, {"vector": [2, ] * self.DIMENSION, "weight": 0}], }))
                res_3 = tensor_search.search(config=self.config, index_name=index_name, text=query, context=
                SearchContext(**{"tensor": [{"vector": [1, ] * self.DIMENSION, "weight": -1}, {"vector": [1, ] * self.DIMENSION, "weight": 1}], }))

                self.assertEqual(res_1["hits"][0]["_score"], res_2["hits"][0]["_score"])
                self.assertEqual(res_1["hits"][0]["_score"], res_3["hits"][0]["_score"])

    def test_context_vector_with_none_query(self):
        """Test to ensure that the context vector can be used without a query."""
        for index_name in [self.structured_index_with_random_model, self.unstructured_index_with_random_model]:
            with self.subTest(msg=index_name):
                res = tensor_search.search(text=None, config=self.config, index_name=index_name, context=SearchContext(
                    **{"tensor": [{"vector": [1, ] * self.DIMENSION, "weight": 1},
                                  {"vector": [2, ] * self.DIMENSION, "weight": 2}]}))

    def test_context_vector_raise_error_if_query_and_context_are_none(self):
        """Test to ensure that a proper error is raised if both query and context is None"""
        for index_name in [self.structured_index_with_random_model, self.unstructured_index_with_random_model]:
            with self.subTest(msg=index_name):
                with self.assertRaises(InvalidArgError) as e:
                    res = tensor_search.search(text=None, config=self.config, index_name=index_name, context=None)
                self.assertIn("One of Query(q) or context is required for tensor search",
                              str(e.exception.message))