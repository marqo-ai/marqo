import os
import unittest
from unittest import mock

from marqo.core.models.marqo_index import FieldType
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import add_docs
from marqo.core.models.marqo_index import *
from marqo import exceptions as base_exceptions
from marqo.core import exceptions as core_exceptions


class TestAddDocumentsUtils(MarqoTestCase):
    def test_determine_document_dict_field_type(self):
        """
        Only tests for custom_vector or None
        """
        mixed_mappings = {
            "my_custom_vector": {
                "type": "custom_vector"
            },
            "my_bad_type": {
                "type": "DOESNT EXIST IN enums.MappingsObjectType"
            }
        }

        assert add_docs.determine_document_dict_field_type(
            field_name="my_custom_vector",
            field_content={"vector": [1, 2, 3]},
            mappings=mixed_mappings
        ) == FieldType.CustomVector

        with self.assertRaises(base_exceptions.InternalError):
            add_docs.determine_document_dict_field_type(
                field_name="my_bad_type",   # exists in mappings, but not in enums.MappingsObjectType
                field_content={"vector": [1, 2, 3]},
                mappings=mixed_mappings
            )

        with self.assertRaises(base_exceptions.InternalError):
            add_docs.determine_document_dict_field_type(
                field_name="BAD NAME, NOT IN MAPPINGS.",
                field_content={"vector": [1, 2, 3]},
                mappings=mixed_mappings
            )

        assert add_docs.determine_document_dict_field_type(
            field_name="Any name. Doesn't matter.",
            field_content="normal text",
            mappings=mixed_mappings
        ) == None
