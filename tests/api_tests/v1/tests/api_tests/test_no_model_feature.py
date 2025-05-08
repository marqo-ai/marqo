import copy
import uuid

import numpy as np
from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestNoModelFeature(MarqoTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)

        cls.unstructured_no_model_index_name = "unstructured_no_model" + str(uuid.uuid4()).replace('-', '')
        cls.structured_no_model_index_name = "structured_no_model" + str(uuid.uuid4()).replace('-', '')
        cls.DIMENSION = 128

        cls.create_indexes([
            {
                "indexName": cls.unstructured_no_model_index_name,
                "type": "unstructured",
                "model": "no_model",
                "modelProperties": {
                    "dimensions": cls.DIMENSION,
                    "type": "no_model"
                }
            },
            {
                "indexName": cls.structured_no_model_index_name,
                "type": "structured",
                "model": "no_model",
                "modelProperties": {
                    "dimensions": cls.DIMENSION,
                    "type": "no_model"
                },
                "allFields": [
                    {"name": "text_field_1", "type": "text"},
                    {"name": "image_field_1", "type": "image_pointer"},
                    {"name": "custom_field_1", "type": "custom_vector"}
                ],
                "tensorFields": ["text_field_1", "image_field_1", "custom_field_1"]
            }
        ])

        cls.indexes_to_delete = [cls.unstructured_no_model_index_name, cls.structured_no_model_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    @staticmethod
    def strip_marqo_fields(doc, strip_id=True):
        """Strips Marqo fields from a returned doc to get the original doc"""
        copied = copy.deepcopy(doc)

        strip_fields = ["_highlights", "_score"]
        if strip_id:
            strip_fields += ["_id"]

        for to_strip in strip_fields:
            del copied[to_strip]

        return copied

    def test_create_index_with_incorrect_model_properties(self):
        """Test to ensure that an error is raised when the model properties are incorrect"""
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

        for model, model_properties, msg in test_cases:
            for index_type in ["structured", "unstructured"]:
                with self.subTest(msg=f"{index_type} - msg"):
                    with self.assertRaises(MarqoWebError) as e:
                        if index_type == "structured":
                            self.client.create_index("test_create_invalid_no_model", type=index_type,
                                                     model=model, model_properties=model_properties,
                                                     all_fields=[{"name": "text_field_1", "type": "text"}],
                                                     tensor_fields=[]
                                                     )

                        else:
                            self.client.create_index("test_create_invalid_no_model", type=index_type,
                                                     model=model, model_properties=model_properties
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

        for index_name in [self.structured_no_model_index_name, self.unstructured_no_model_index_name]:
            with (self.subTest(index_name=index_name)):
                tensor_fields = ["text_field_1", "custom_field_1"] if \
                    index_name == self.unstructured_no_model_index_name else None
                mappings = {"custom_field_1": {"type": "custom_vector"}} if \
                    index_name == self.unstructured_no_model_index_name else None
                r = self.client.index(index_name).add_documents(
                    documents=documents,
                    tensor_fields=tensor_fields,
                    mappings=mappings)

                self.assertEqual(r["errors"], True)
                self.assertIn("Cannot vectorise anything with 'no_model'", r["items"][0]["error"])
                self.assertEqual(400, r["items"][0]["status"])
                self.assertEqual("inference_error", r["items"][0]["code"])
                self.assertEqual("1", r["items"][0]["_id"])

                self.assertEqual("2", r["items"][1]["_id"])
                self.assertEqual(200, r["items"][1]["status"])
                self.assertEqual(1, self.client.index(index_name).get_stats()["numberOfDocuments"])
                self.assertEqual(1, self.client.index(index_name).get_stats()["numberOfVectors"])

    def test_no_model_raise_error_if_query_in_search(self):
        """Test to ensure that providing a query to vectorise will raise an error."""
        for index_name in [self.structured_no_model_index_name, self.unstructured_no_model_index_name]:
            with (self.subTest(index_name=index_name)):
                with self.assertRaises(MarqoWebError) as e:
                    r = self.client.index(index_name).search(q="test")
                self.assertIn("Cannot vectorise anything with 'no_model'", str(e.exception))

    def test_no_model_work_with_context_vectors_in_search(self):
        """Test to ensure that context vectors work with no_model by setting query as None"""
        # We need to normalise the vectors as the default distance metric is prenormalized-angular
        custom_vector = np.random.randn(self.DIMENSION)
        custom_vector = (np.array(custom_vector) / np.linalg.norm(custom_vector)).tolist()

        random_vector = np.random.randn(self.DIMENSION)
        random_vector = (random_vector / np.linalg.norm(random_vector)).tolist()

        docs = [
            {
                "_id": "1",
                "custom_field_1":
                    {
                        "content": "test custom field content_1",
                        "vector": random_vector
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

        for index_name in [self.structured_no_model_index_name, self.unstructured_no_model_index_name]:
            with (self.subTest(index_name=index_name)):
                tensor_fields = ["text_field_1", "custom_field_1"] if \
                    index_name == self.unstructured_no_model_index_name else None
                mappings = {"custom_field_1": {"type": "custom_vector"}} if \
                    index_name == self.unstructured_no_model_index_name else None

                r = self.client.index(index_name).add_documents(
                    documents=docs,
                    tensor_fields=tensor_fields,
                    mappings=mappings
                )
                r = self.client.index(index_name).search(q=None,
                                                         context={"tensor": [{"vector": custom_vector,
                                                                              "weight": 1}], })

                self.assertEqual(2, len(r["hits"]))
                self.assertEqual("2", r["hits"][0]["_id"])
                self.assertAlmostEqual(1, r["hits"][0]["_score"], places=1)

                self.assertEqual("1", r["hits"][1]["_id"])
                self.assertTrue(r["hits"][1]["_score"], r["hits"][0]["_score"])

    def test_no_model_work_with_custom_vectors_in_search(self):
        """Test to ensure that context vectors work with no_model by setting query as None"""
        for index_name in [self.structured_no_model_index_name, self.unstructured_no_model_index_name]:
            with (self.subTest(index_name=index_name)):
                r = self.client.index(index_name).search(q=None,
                                                         context={"tensor": [{"vector": [1, ] * self.DIMENSION,
                                                                              "weight": -1},
                                                                             {"vector": [1, ] * self.DIMENSION,
                                                                              "weight": 1}], })

    def test_no_model_and_context_vectors_dimension(self):
        """Test to ensure no_model still raises error if context vector dimension is incorrect."""
        for index_name in [self.structured_no_model_index_name, self.unstructured_no_model_index_name]:
            with (self.subTest(index_name=index_name)):
                with self.assertRaises(MarqoWebError) as e:
                    r = self.client.index(index_name).search(q=None,
                                                             context={
                                                                 "tensor": [{"vector": [1, ] * (self.DIMENSION + 1),
                                                                             "weight": -1},
                                                                            {"vector": [1, ] * (self.DIMENSION + 1),
                                                                             "weight": 1}], })
                self.assertIn("does not match the expected dimension", str(e.exception.message))
