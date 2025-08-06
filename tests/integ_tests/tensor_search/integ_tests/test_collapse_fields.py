import os
from unittest import mock

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import CollapseField, SemiStructuredMarqoIndex
from marqo.tensor_search import tensor_search
from tests.integ_tests.marqo_test import MarqoTestCase


class TestCollapseFields(MarqoTestCase):
    """Integration tests for collapse fields functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.unstructured_marqo_index_request(
            collapse_fields=[CollapseField(name="parent_id", minGroups=100)]
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
        ])

        cls.default_text_index = cls.indexes[0]

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_index_should_contain_collapse_field_settings(self):
        index = self.index_management.get_index(self.default_text_index.name)
        self.assertIsInstance(index, SemiStructuredMarqoIndex)
        self.assertIsNotNone(index.collapse_fields)
        self.assertEqual(index.collapse_fields[0].name, "parent_id")
        self.assertEqual(index.collapse_fields[0].min_groups, 100)

    def test_add_documents_mixed_batch_with_collapse_field_errors(self):
        """Test that valid documents succeed while invalid ones fail in same batch"""
        docs = [
            {"_id": "valid1", "title": "Valid document 1", "parent_id": "group_1"},
            {"_id": "invalid1", "title": "Invalid document - missing field"},
            {"_id": "valid2", "title": "Valid document 2", "parent_id": "group_2"},
            {"_id": "invalid2", "title": "Invalid document - wrong type", "parent_id": 456}
        ]

        res = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=[]
            )
        )

        # Check results
        successful_items = [item for item in res.items if item.status == 200]
        failed_items = [item for item in res.items if item.status != 200]

        self.assertEqual(2, len(successful_items), "Expected 2 successful documents")
        self.assertEqual(2, len(failed_items), "Expected 2 failed documents")

        # Verify successful document IDs
        successful_ids = {item.id for item in successful_items}
        self.assertEqual({"valid1", "valid2"}, successful_ids)

        # Verify failed documents
        self.assertIn("Document missing required field 'parent_id'", failed_items[0].message)
        self.assertIn("Field 'parent_id' must be of type string", failed_items[1].message)

        # TODO see if we can get the two valid docs back from Vespa with correct parent_id
        valid_docs = tensor_search.get_documents_by_ids(config=self.config, index_name=self.default_text_index.name,
                                                        document_ids=successful_ids)

        self.assertFalse(valid_docs.errors)
        self.assertEqual(2, len(valid_docs.results))
        for doc in valid_docs.results:
            expected_parent_id = "group_1" if doc["_id"] == "valid1" else "group_2"
            self.assertEqual(expected_parent_id, doc["parent_id"])
