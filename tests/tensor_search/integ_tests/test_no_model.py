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


class TestGetDocuments(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_index_with_no_model_request = cls.structured_marqo_index_request(
            model=Model(name="no_model", properties={"dimensions": 123}),
            fields=[
                FieldRequest(name='text_field_1', type=FieldType.Text),
                FieldRequest(name='image_field_1', type=FieldType.ImagePointer),
            ],
            tensor_fields=["text_field_1", "image_field_1"]
        )
        unstructured_index_with_no_model_request = cls.unstructured_marqo_index_request(
            model=Model(name="no_model", properties={"dimensions": 456}),
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

    def test_create_index_with_invalid_dimensions(self):
        """Test to ensure an invalid dimension value will block the index creation."""
        try:
            self.index_management.delete_index_by_name("test_create_index_with_invalid_model_name_or_properties")
        except IndexNotFoundError:
            pass

        # Test that creating an index with an invalid model name or properties raises an error
        test_cases = [
            ("no_model", {"dimensions": 0, "type": "no_model"}, "invalid dimensions value, can't be 0"),
            ("no_model", {"dimensions": -123, "type": "no_model"}, "invalid dimensions value, can't be negative"),
            ("no_model", {"dimensions": 213.213, "type": "no_model"}, "invalid dimensions value, can't be float"),
        ]

        for name, model_properties, msg in test_cases:
            for index_type in ["structured", "unstructured"]:
                with self.subTest(name=name, model_properties=model_properties, msg=f"{index_type} - msg"):
                    with self.assertRaises(InvalidArgumentError) as e:
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
                    self.assertIn("The given model properties does not contain a valid 'dimensions' value.",
                                  str(e.exception))



