import unittest
from unittest.mock import Mock, patch, MagicMock
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.search import VectorisedJobPointer, JHash
from marqo.tensor_search.models.api_models import CustomVectorQuery
from marqo.api import exceptions as api_exceptions
from marqo.core import exceptions as core_exceptions
from marqo.core.models.marqo_index import IndexType, StructuredMarqoIndex, UnstructuredMarqoIndex
from marqo.tensor_search.enums import SearchMethod
from marqo.vespa.exceptions import VespaStatusError


class TestTensorSearchFunctions(unittest.TestCase):
    """Test cases for tensor_search.py functions to cover missing lines"""

    def setUp(self):
        self.mock_config = Mock()
        self.mock_index = Mock()
        self.mock_index.type = IndexType.Structured

    def test_search_lexical_ef_search_parameter_fails(self):
        """Test that search function raises InvalidArgError when efSearch parameter is used with lexical search method."""
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            tensor_search.search(
                config=self.mock_config,
                index_name="test_index",
                text="test query",
                search_method=SearchMethod.LEXICAL,
                ef_search=100  # Invalid for lexical search
            )
        self.assertIn("efSearch", str(cm.exception))

    def test_search_lexical_approximate_parameter_fails(self):
        """Test that search function raises InvalidArgError when approximate parameter is used with lexical search method."""
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            tensor_search.search(
                config=self.mock_config,
                index_name="test_index",
                text="test query",
                search_method=SearchMethod.LEXICAL,
                approximate=False  # Invalid for lexical search
            )
        self.assertIn("approximate", str(cm.exception))

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    def test_construct_vector_input_batches_with_none_query(self, mock_get_index):
        """Test that construct_vector_input_batches returns empty QueryContentCollector when given None query."""
        result = tensor_search.construct_vector_input_batches(None)
        
        # Should return empty query collector
        self.assertEqual(len(result.queries), 0)

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    def test_construct_vector_input_batches_with_invalid_query_type(self, mock_get_index):
        """Test that construct_vector_input_batches raises ValueError when given an invalid query type like integer."""
        with self.assertRaises(ValueError) as cm:
            tensor_search.construct_vector_input_batches(123)  # Invalid type
        self.assertIn("Incorrect type for query", str(cm.exception))

    @patch('marqo.core.vespa_index.vespa_index.for_marqo_index')
    def test_gather_documents_from_response_with_group_facet_id(self, mock_factory):
        """Test that gather_documents_from_response skips group:facet: IDs and returns empty hits list."""
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        
        # Mock QueryResult with a group:facet: document
        mock_response = Mock()
        mock_doc = Mock()
        mock_doc.id = "group:facet:test"
        mock_response.hits = [mock_doc]
        
        mock_vespa_index = Mock()
        mock_factory.return_value = mock_vespa_index
        
        result = tensor_search.gather_documents_from_response(
            response=mock_response,
            marqo_index=mock_index,
            highlights=False
        )
        
        # Should skip the group:facet: document
        self.assertEqual(len(result['hits']), 0)
        # Verify to_marqo_document was never called since we skip group:facet docs
        mock_vespa_index.to_marqo_document.assert_not_called()

    def test_select_attributes_with_flattened_map_fields(self):
        """Test that select_attributes includes flattened map fields when the prefix matches an attribute to retrieve."""
        marqo_doc = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "Test Document",
            "metadata.author": "John Doe",  # Flattened map field
            "metadata.year": "2023",       # Flattened map field
            "description": "Test Description",
            "extra_field": "Should be filtered out"
        }
        
        attributes_to_retrieve_set = {"_id", "_score", "title", "metadata"}
        
        result = tensor_search.select_attributes(marqo_doc, attributes_to_retrieve_set)
        
        # Should include flattened map fields with "metadata" prefix
        expected = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "Test Document",
            "metadata.author": "John Doe",
            "metadata.year": "2023"
        }
        self.assertEqual(result, expected)

    def test_select_attributes_without_flattened_fields(self):
        """Test that select_attributes returns only exact attribute matches when no flattened fields are present."""
        marqo_doc = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "Test Document",
            "description": "Test Description",
            "extra_field": "Should be filtered out"
        }
        
        attributes_to_retrieve_set = {"_id", "_score", "title"}
        
        result = tensor_search.select_attributes(marqo_doc, attributes_to_retrieve_set)
        
        # Should only include exact matches
        expected = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "Test Document"
        }
        self.assertEqual(result, expected)

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    def test_get_content_vector_not_found_error(self, mock_get_index):
        """Test that get_content_vector raises RuntimeError when content is not found in any job."""
        possible_jobs = []  # Empty list
        job_to_vectors = {}
        content = "test_content"
        
        with self.assertRaises(RuntimeError) as cm:
            tensor_search.get_content_vector(possible_jobs, job_to_vectors, content)
        self.assertIn("could not find corresponding vector for content", str(cm.exception))

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    def test_get_content_vector_found_in_job(self, mock_get_index):
        """Test that get_content_vector returns the correct vector when content is found in a job."""
        
        job_hash = JHash(123)
        possible_jobs = [VectorisedJobPointer(job_hash=job_hash, start_idx=0, end_idx=1)]
        job_to_vectors = {job_hash: {"test_content": [0.1, 0.2, 0.3]}}
        content = "test_content"
        
        result = tensor_search.get_content_vector(possible_jobs, job_to_vectors, content)
        
        self.assertEqual(result, [0.1, 0.2, 0.3])


if __name__ == '__main__':
    unittest.main() 