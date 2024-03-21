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


class TestNoModel(MarqoTestCase):

    structured_index_with_no_model = "structured_index_with_no_model"
    unstructured_index_with_no_model = "unstructured_index_with_no_model"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_index_with_no_model_request = cls.structured_marqo_index_request(
            name=cls.structured_index_with_no_model,
            model=Model(name="no_model", properties={"dimensions": 16, "type": "no_model"}, custom=True),
            fields=[
                FieldRequest(name='text_field_1', type=FieldType.Text),
                FieldRequest(name='image_field_1', type=FieldType.ImagePointer),
                FieldRequest(name="custom_field_1", type=FieldType.CustomVector)
            ],
            tensor_fields=["text_field_1", "image_field_1", "custom_field_1"]
        )
        unstructured_index_with_no_model_request = cls.unstructured_marqo_index_request(
            name=cls.unstructured_index_with_no_model,
            model=Model(name="no_model", properties={"dimensions": 16, "type": "no_model"}, custom=True),
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
            ("my_model", {"dimensions": 512, "type": "no_model"}, "invalid model name")
        ]

        for name, model_properties, msg in test_cases:
            for index_type in ["structured", "unstructured"]:
                with self.subTest(name=name, model_properties=model_properties, msg=f"{index_type} - msg"):
                    with self.assertRaises(InvalidArgumentError, ) as e:
                        if index_type == "structured":
                            self.index_management.create_index(
                                self.structured_marqo_index_request(
                                    name="test_create_index_with_invalid_model_name_or_properties",
                                    model=Model(name=name, properties=model_properties),
                                    fields=[FieldRequest(name="text_field_1", type=FieldType.Text)],
                                    tensor_fields=["text_field_1"]
                                )
                            )
                        else:
                            self.index_management.create_index(
                                self.unstructured_marqo_index_request(
                                    name="test_create_index_with_invalid_model_name_or_properties",
                                    model=Model(name=name, properties=model_properties)
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
                        "vector": [1.0 for _ in range(16)]
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
                self.assertIn("'no_model' cannot vectorise your content.", r["items"][0]["error"])
                self.assertEqual(400, r["items"][0]["status"])
                self.assertEqual("invalid_argument", r["items"][0]["code"])
                self.assertEqual("1", r["items"][0]["_id"])

                self.assertEqual("2", r["items"][1]["_id"])
                self.assertEqual(200, r["items"][1]["status"])
                self.assertEqual(1, self.monitoring.get_index_stats_by_name(index_name).number_of_documents)
                self.assertEqual(1, self.monitoring.get_index_stats_by_name(index_name).number_of_vectors)