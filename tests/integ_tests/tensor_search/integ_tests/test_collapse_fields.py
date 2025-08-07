import os
from unittest import mock

from marqo.api.exceptions import InvalidArgError
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

        # Verify failed documents
        failed_items = [item for item in res.items if item.status != 200]
        self.assertEqual(2, len(failed_items), "Expected 2 failed documents")
        self.assertIn("Document missing required field 'parent_id'", failed_items[0].message)
        self.assertIn("Field 'parent_id' must be of type string", failed_items[1].message)

        # Verify successful documents
        successful_items = [item for item in res.items if item.status == 200]
        self.assertEqual(2, len(successful_items), "Expected 2 successful documents")

        successful_ids = {item.id for item in successful_items}
        self.assertEqual({"valid1", "valid2"}, successful_ids)

        # Verify we can retrieve the parent_id back
        valid_docs = tensor_search.get_documents_by_ids(config=self.config, index_name=self.default_text_index.name,
                                                        document_ids=successful_ids)

        self.assertFalse(valid_docs.errors)
        self.assertEqual(2, len(valid_docs.results))
        for doc in valid_docs.results:
            expected_parent_id = "group_1" if doc["_id"] == "valid1" else "group_2"
            self.assertEqual(expected_parent_id, doc["parent_id"])

    def test_partial_update_of_collapse_field_does_not_work(self):
        docs = [
            {"_id": "valid1", "title": "Valid document 1", "parent_id": "group_1"},
        ]

        res = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=[]
            )
        )

        self.assertFalse(res.errors)
        self.assertEqual(1, len(res.items))

        update_res = self.config.document.partial_update_documents_by_index_name(
            index_name=self.default_text_index.name,
            partial_documents=[{"_id": "valid1", "parent_id": "group_2"}])

        self.assertTrue(update_res.errors)
        self.assertEqual(400, update_res.items[0].status)

        # TODO please note that this is not working due to a side effect that partial update treats all string fields
        #  as lexical fields. Ideally, partial updates should treat collapse differently to avoid confusing error msg.
        self.assertIn("parent_id of type str does not exist in the original document. "
                      "Marqo does not support adding new lexical fields in partial updates", update_res.items[0].error)

        doc = tensor_search.get_document_by_id(config=self.config, index_name=self.default_text_index.name,
                                               document_id="valid1")

        self.assertEqual(doc["parent_id"], "group_1")

    def test_search_with_invalid_collapse_field_raises_error(self):
        """Test that search with invalid collapse field name raises error"""
        with self.assertRaises(InvalidArgError) as cm:
            tensor_search.search(
                config=self.config,
                index_name=self.default_text_index.name,
                text="test query",
                search_method="HYBRID",
                collapse_field_name="non_existent_field"
            )
        
        self.assertIn("Field 'non_existent_field' is not configured as collapseFields for this index", 
                      str(cm.exception))

    def test_search_with_valid_collapse_field_succeeds(self):
        """Test that search with valid collapse field name succeeds"""
        # Add some test documents
        docs = [
            {"_id": "doc1", "title": "Test document 1", "parent_id": "group_1"},
            {"_id": "doc2", "title": "Test document 2", "parent_id": "group_2"}
        ]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index.name,
                docs=docs,
                tensor_fields=["title"]
            )
        )
        
        # This should not raise an exception
        result = tensor_search.search(
            config=self.config,
            index_name=self.default_text_index.name,
            text="test",
            search_method="HYBRID",
            collapse_field_name="parent_id"
        )
        
        # Verify the search executed successfully
        self.assertIn("hits", result)
        self.assertIsInstance(result["hits"], list)
        # TODO test search collapse on parent_id
