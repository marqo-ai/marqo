from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.models.search import (SearchContextTensor, SearchContextDocuments,
                                               SearchContextDocumentsParameters, SearchContext)
from unittest import TestCase
from pydantic.v1 import ValidationError
from marqo.api.exceptions import InvalidArgError


class TestSearchQueryModel(TestCase):
    """
    Tests for search query and context models were in test_validation.py under integ_tests.
    Moving them to unit tests.
    """

    def test_search_context_only_documents_succeeds(self):
        """Test that SearchQuery with only documents context succeeds"""
        search_query = SearchQuery(
            context=SearchContext(
                documents=SearchContextDocuments(
                    ids={"doc1": 1.0, "doc2": 0.5}
                )
            )
        )
        self.assertIsNotNone(search_query.context)
        self.assertIsNotNone(search_query.context.documents)
        self.assertIsNone(search_query.context.tensor)
        self.assertEqual(search_query.context.documents.ids, {"doc1": 1.0, "doc2": 0.5})

    def test_search_context_documents_with_int_weights_succeeds(self):
        """Test that SearchQuery with documents context using integer weights succeeds."""
        search_query = SearchQuery(
            context=SearchContext(
                documents=SearchContextDocuments(
                    ids={"doc1": 1, "doc2": 2}
                )
            )
        )
        self.assertIsNotNone(search_query.context)
        self.assertIsNotNone(search_query.context.documents)
        self.assertEqual(search_query.context.documents.ids, {"doc1": 1, "doc2": 2})

    def test_search_context_only_tensor_succeeds(self):
        """Test that SearchQuery with only tensor context succeeds"""
        search_query = SearchQuery(
            context=SearchContext(
                tensor=[
                    SearchContextTensor(vector=[0.1, 0.2, 0.3], weight=1.0),
                    SearchContextTensor(vector=[0.4, 0.5, 0.6], weight=0.5)
                ]
            )
        )
        self.assertIsNotNone(search_query.context)
        self.assertIsNotNone(search_query.context.tensor)
        self.assertIsNone(search_query.context.documents)
        self.assertEqual(len(search_query.context.tensor), 2)

    def test_search_context_documents_empty_ids_fails(self):
        """Test that SearchQuery with empty ids in documents context fails"""
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                context=SearchContext(
                    documents=SearchContextDocuments(
                        ids={}
                    )
                )
            )
        self.assertIn('must be present and a non-empty dict', str(cm.exception))

    def test_search_context_documents_no_ids_fails(self):
        """Test that SearchQuery with no ids in documents context fails"""
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                context=SearchContext(
                    documents=SearchContextDocuments(
                        ids=None
                    )
                )
            )
        self.assertIn('must be present and a non-empty dict', str(cm.exception))