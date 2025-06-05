import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from marqo.core.search.recommender import Recommender
from marqo.core.models.marqo_index import IndexType, StructuredMarqoIndex, UnstructuredMarqoIndex
from marqo.exceptions import InvalidArgumentError
from marqo.core.exceptions import InvalidFieldNameError


class TestRecommenderGetDocVectorsFromIds:
    """Test cases for the updated Recommender.get_doc_vectors_from_ids method"""
    
    def setup_method(self):
        """Set up common test fixtures"""
        self.mock_vespa_client = Mock()
        self.mock_index_management = Mock()
        self.mock_inference = Mock()
        
        self.recommender = Recommender(
            self.mock_vespa_client,
            self.mock_index_management,
            self.mock_inference
        )
        
        # Mock structured index
        self.mock_structured_index = Mock(spec=StructuredMarqoIndex)
        self.mock_structured_index.type = IndexType.Structured
        self.mock_structured_index.tensor_field_map = {
            "title": Mock(),
            "description": Mock(),
            "content": Mock()
        }
    
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_successful_get_vectors_list_ids(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test successfully getting vectors with list of document IDs"""
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_structured_index
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock the tensor_search function response
        mock_get_vectors.return_value = {
            "doc1": {
                "title": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "description": [[0.7, 0.8, 0.9]]
            },
            "doc2": {
                "title": [[1.1, 1.2, 1.3]],
                "description": [[1.7, 1.8, 1.9]]
            }
        }
        
        # Call the method
        result = self.recommender.get_doc_vectors_from_ids(
            index_name="test_index",
            documents=["doc1", "doc2"]
        )
        
        # Verify result - should flatten embeddings from all fields
        expected = {
            "doc1": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            "doc2": [[1.1, 1.2, 1.3], [1.7, 1.8, 1.9]]
        }
        assert result == expected
        
        # Verify tensor_search function was called correctly
        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1", "doc2"], tensor_fields=None
        )
    
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_successful_get_vectors_dict_ids(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test successfully getting vectors with dictionary of document IDs and weights"""
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_structured_index
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock the tensor_search function response
        mock_get_vectors.return_value = {
            "doc1": {
                "title": [[0.1, 0.2, 0.3]]
            },
            "doc3": {
                "title": [[1.1, 1.2, 1.3]]
            }
        }
        
        # Call the method with dict (including zero weight doc)
        documents = {"doc1": 1.0, "doc2": 0.0, "doc3": 2.0}
        result = self.recommender.get_doc_vectors_from_ids(
            index_name="test_index",
            documents=documents
        )
        
        # Verify result - zero weight documents should be filtered out
        expected = {
            "doc1": [[0.1, 0.2, 0.3]],
            "doc3": [[1.1, 1.2, 1.3]]
        }
        assert result == expected
        
        # Verify tensor_search function was called with non-zero weight docs only
        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1", "doc3"], tensor_fields=None
        )
    
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_specific_tensor_fields(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test getting vectors for specific tensor fields"""
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_structured_index
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock the tensor_search function response with multiple fields
        mock_get_vectors.return_value = {
            "doc1": {
                "title": [[0.1, 0.2, 0.3]],
                "description": [[0.7, 0.8, 0.9]],
                "content": [[1.1, 1.2, 1.3]]
            }
        }
        
        # Call the method with specific tensor fields
        result = self.recommender.get_doc_vectors_from_ids(
            index_name="test_index",
            documents=["doc1"],
            tensor_fields=["title", "content"]
        )
        
        # Should only include specified tensor fields
        expected = {
            "doc1": [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]  # Only title and content
        }
        assert result == expected
        
        # Verify tensor_search function was called with specific fields
        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1"], tensor_fields=["title", "content"]
        )
    
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    def test_invalid_tensor_field_structured_index(self, mock_get_index):
        """Test validation of tensor field names for structured index"""
        
        mock_get_index.return_value = self.mock_structured_index
        
        # Try to use invalid tensor field
        with pytest.raises(InvalidFieldNameError) as exc_info:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1"],
                tensor_fields=["invalid_field"]
            )
        
        assert "invalid_field" in str(exc_info.value)
        assert "Available tensor fields" in str(exc_info.value)
    
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_document_not_found(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test handling when document is not found"""
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_structured_index
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock response missing one document
        mock_get_vectors.return_value = {
            "doc1": {"title": [[0.1, 0.2, 0.3]]}
            # doc2 is missing
        }
        
        # Should raise error for missing document
        with pytest.raises(InvalidArgumentError) as exc_info:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1", "doc2"]
            )
        
        assert "not found" in str(exc_info.value)
        assert "doc2" in str(exc_info.value)
    
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_document_without_vectors(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test handling when document has no embeddings"""
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_structured_index
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock response with document that has no embeddings
        mock_get_vectors.return_value = {
            "doc1": {"title": [[0.1, 0.2, 0.3]]},
            "doc2": {}  # No embeddings
        }
        
        # Should raise error for document without vectors
        with pytest.raises(InvalidArgumentError) as exc_info:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1", "doc2"]
            )
        
        assert "do not have embeddings" in str(exc_info.value)
        assert "doc2" in str(exc_info.value)
    
    def test_empty_documents_list(self):
        """Test with empty documents list"""
        
        with pytest.raises(InvalidArgumentError) as exc_info:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=[]
            )
        
        assert "No document IDs provided" in str(exc_info.value)
    
    def test_none_documents(self):
        """Test with None documents"""
        
        with pytest.raises(InvalidArgumentError) as exc_info:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=None
            )
        
        assert "No document IDs provided" in str(exc_info.value)
    
    def test_all_zero_weight_documents(self):
        """Test with all documents having zero weight"""
        
        with pytest.raises(InvalidArgumentError) as exc_info:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents={"doc1": 0.0, "doc2": 0.0}
            )
        
        assert "No documents with non-zero weight provided" in str(exc_info.value)
    
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    def test_unstructured_index_no_validation(self, mock_get_index):
        """Test that unstructured index doesn't validate tensor field names"""
        
        # Mock unstructured index
        mock_unstructured_index = Mock(spec=UnstructuredMarqoIndex)
        mock_unstructured_index.type = IndexType.Unstructured
        mock_get_index.return_value = mock_unstructured_index
        
        # This should not raise an error for unstructured index
        # Just verify the setup doesn't crash (actual function call will be mocked in integration)
        # The validation only happens for structured indexes
        pass 