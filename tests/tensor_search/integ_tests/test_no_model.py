import os
from unittest.mock import patch

import numpy as np

from marqo.api.exceptions import InvalidArgError
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.search import SearchContext
from tests.marqo_test import MarqoTestCase
from tests.utils.transition import *


class TestNoModel(MarqoTestCase):

    structured_index_with_no_model = "structured_index_with_no_model"
    unstructured_index_with_no_model = "unstructured_index_with_no_model"
    DIMENSION = 16

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_index_with_no_model_request = cls.structured_marqo_index_request(
            name=cls.structured_index_with_no_model,
            model=Model(name="no_model", properties={"dimensions": cls.DIMENSION, "type": "no_model"}, custom=True),
            fields=[
                FieldRequest(name='text_field_1', type=FieldType.Text),
                FieldRequest(name='image_field_1', type=FieldType.ImagePointer),
                FieldRequest(name="custom_field_1", type=FieldType.CustomVector)
            ],
            tensor_fields=["text_field_1", "image_field_1", "custom_field_1"]
        )
        unstructured_index_with_no_model_request = cls.unstructured_marqo_index_request(
            name=cls.unstructured_index_with_no_model,
            model=Model(name="no_model", properties={"dimensions": cls.DIMENSION, "type": "no_model"}, custom=True),
        )

        # List of indexes to loop through per test. Test itself should extract index name.
        cls.indexes = cls.create_indexes([
            structured_index_with_no_model_request,
            unstructured_index_with_no_model_request,
        ])

    def setUp(self) -> None:
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_create_index_with_invalid_model_properties(self):
        """Test to ensure an invalid model properties will block index creation."""
        try:
            self.index_management.delete_index_by_name("test_create_index_with_invalid_model_name_or_properties")
        except IndexNotFoundError:
            pass

        # Test that creating an index with an invalid model name or properties raises an error
        test_cases = [
            ("no_model", {"dimensions": 0, "type": "no_model"}, "invalid dimensions value, can't be 0"),
            ("no_model", {"dimensions": -123, "type": "no_model"}, "invalid dimensions value, can't be negative"),
            ("no_model", {"dimensions": 213.213, "type": "no_model"}, "invalid dimensions value, can't be float"),
            ("no_model", {"dimensions": "512", "type": "no_model"}, "invalid dimensions value, can't be string"),
            ("no_model", {"dimensions": 234, "type": "No_model"}, "invalid model type, should be 'no_model'"),
            ("no_model", None, "no model properties provided"),
            ("no_model", {"type": "no_model"}, "dimension not provided"),
            ("my_model", {"dimensions": 512, "type": "no_model"}, "invalid model name"),
            ("no_model", {"dimensions": 512, "type": "open_clip"}, "invalid model properties type")
        ]

        for name, model_properties, msg in test_cases:
            for index_type in ["structured", "unstructured"]:
                with self.subTest(name=name, model_properties=model_properties, msg=f"{index_type} - msg"):
                    with self.assertRaises(InvalidArgumentError) as e:
                        if index_type == "structured":
                            self.index_management.create_index(
                                self.structured_marqo_index_request(
                                    name="test_create_index_with_invalid_model_name_or_properties",
                                    model=Model(name=name, properties=model_properties, custom=True),
                                    fields=[FieldRequest(name="text_field_1", type=FieldType.Text)],
                                    tensor_fields=["text_field_1"]
                                )
                            )
                        else:
                            self.index_management.create_index(
                                self.unstructured_marqo_index_request(
                                    name="test_create_index_with_invalid_model_name_or_properties",
                                    model=Model(name=name, properties=model_properties, custom=True)
                                )
                        )
                    self.assertIn("Invalid model properties", str(e.exception))

    def test_no_model_in_add_documents_error(self):
        """Test to ensure that adding documents to an index with no model raises an error for the
        specific documents."""
        documents = [
            {
                "_id": "1",
                "text_field_1": "text",
            },
            {
                "_id": "2",
                "custom_field_1":
                    {
                        "content": "test custom field content",
                        "vector": [1.0 for _ in range(self.DIMENSION)]
                    }
            }
        ]

        for index_name in [self.structured_index_with_no_model, self.unstructured_index_with_no_model]:
            with (self.subTest(index_name=index_name)):
                tensor_fields = ["text_field_1", "custom_field_1"] if \
                    index_name == self.unstructured_index_with_no_model else None
                mappings = {"custom_field_1": {"type": "custom_vector"}} if \
                    index_name == self.unstructured_index_with_no_model else None
                r = tensor_search.add_documents(config=self.config,
                                                add_docs_params=AddDocsParams(index_name=index_name,
                                                                          docs=documents,
                                                                          tensor_fields=tensor_fields,
                                                                          mappings=mappings))
                self.assertEqual(r["errors"], True)
                self.assertIn("Cannot vectorise anything with 'no_model'", r["items"][0]["error"])
                self.assertEqual(400, r["items"][0]["status"])
                self.assertEqual("invalid_argument", r["items"][0]["code"])
                self.assertEqual("1", r["items"][0]["_id"])

                self.assertEqual("2", r["items"][1]["_id"])
                self.assertEqual(200, r["items"][1]["status"])
                self.assertEqual(1, self.monitoring.get_index_stats_by_name(index_name).number_of_documents)
                self.assertEqual(1, self.monitoring.get_index_stats_by_name(index_name).number_of_vectors)

    def test_no_model_raise_error_if_query_in_search(self):
        """Test to ensure that providing a query to vectorise will raise an error."""
        for index_name in [self.structured_index_with_no_model, self.unstructured_index_with_no_model]:
            with (self.subTest(index_name=index_name)):
                with self.assertRaises(InvalidArgError) as e:
                    r = tensor_search.search(config=self.config, index_name=index_name, text="test")
                self.assertIn("'no_model' cannot vectorise your content.", str(e.exception))

    def test_no_model_work_with_context_vectors_in_search(self):
        """Test to ensure that context vectors work with no_model by setting query as None"""

        custom_vector = [0.655 for _ in range(self.DIMENSION)]

        docs = [
            {
                "_id": "1",
                "custom_field_1":
                    {
                        "content": "test custom field content_1",
                        "vector": np.random.rand(self.DIMENSION).tolist()
                    }
            },
            {
                "_id": "2",
                "custom_field_1":
                    {
                        "content": "test custom field content_2",
                        "vector": custom_vector
                    }
            }
        ]

        for index_name in [self.structured_index_with_no_model, self.unstructured_index_with_no_model]:
            with (self.subTest(index_name=index_name)):
                tensor_fields = ["custom_field_1"] if \
                    index_name == self.unstructured_index_with_no_model else None

                mappings = {"custom_field_1": {"type": "custom_vector"}} if \
                    index_name == self.unstructured_index_with_no_model else None
                add_docs_params = AddDocsParams(index_name=index_name,
                                                docs=docs,
                                                tensor_fields=tensor_fields,
                                                mappings=mappings)
                _ = tensor_search.add_documents(config=self.config,
                                                add_docs_params=add_docs_params)

                r = tensor_search.search(config=self.config, index_name=index_name, text=None,
                                         context=SearchContext(**{"tensor": [{"vector": custom_vector,
                                                                              "weight": 1}], }))
                self.assertEqual(2, len(r["hits"]))
                self.assertEqual("2", r["hits"][0]["_id"])
                self.assertAlmostEqual(1, r["hits"][0]["_score"], places=1)

                self.assertEqual("1", r["hits"][1]["_id"])
                self.assertTrue(r["hits"][1]["_score"], r["hits"][0]["_score"])

    def test_no_model_work_with_custom_vectors_in_search(self):
        """Test to ensure that context vectors work with no_model by setting query as None"""
        for index_name in [self.structured_index_with_no_model, self.unstructured_index_with_no_model]:
            with (self.subTest(index_name=index_name)):
                r = tensor_search.search(config=self.config, index_name=index_name, text=None,
                                         context=SearchContext(**{"tensor": [{"vector": [1, ] * self.DIMENSION,
                                                                              "weight": -1},
                                                                             {"vector": [1, ] * self.DIMENSION,
                                                                              "weight": 1}], }))




    def test_no_model_and_context_vectors_dimension(self):
        """Test to ensure no_model still raises error if context vector dimension is incorrect."""
        for index_name in [self.structured_index_with_no_model, self.unstructured_index_with_no_model]:
            with (self.subTest(index_name=index_name)):
                with self.assertRaises(InvalidArgError) as e:
                    r = tensor_search.search(config=self.config, index_name=index_name, text=None,
                                             context=SearchContext(**{"tensor": [{"vector": [1, ] * (self.DIMENSION + 1),
                                                                                  "weight": -1},
                                                                                 {"vector": [1, ] * (self.DIMENSION + 1),
                                                                                  "weight": 1}], }))
                self.assertIn("does not match the expected dimension", str(e.exception.message))
