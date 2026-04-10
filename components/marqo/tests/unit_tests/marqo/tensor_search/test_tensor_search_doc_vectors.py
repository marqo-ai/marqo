import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from marqo.config import Config
from marqo.core.models.marqo_index import IndexType, StructuredMarqoIndex, UnstructuredMarqoIndex, TensorField, Field, FieldType
from marqo.tensor_search.tensor_search import get_embedding_field_names, get_doc_vectors_per_tensor_field_by_ids
from marqo.core import constants
from marqo.vespa.models.get_document_response import GetBatchResponse, GetBatchDocumentResponse
from marqo.vespa.models import VespaDocument
from marqo.core.structured_vespa_index import common as structured_common
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.exceptions import InternalError
from marqo.core import exceptions as core_exceptions

class TestGetEmbeddingFieldNames(unittest.TestCase):
    """Test cases for get_embedding_field_names function"""
    
    def test_structured_index_all_tensor_fields(self):
        """Test getting all embedding field names for structured index"""
        # Create mock tensor fields
        tensor_fields = [
            TensorField(name="title", chunk_field_name="chunks_title", embeddings_field_name="emb_title"),
            TensorField(name="description", chunk_field_name="chunks_desc", embeddings_field_name="emb_desc")
        ]
        
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        mock_index.tensor_fields = tensor_fields
        
        result = get_embedding_field_names(mock_index)
        
        assert result == (["title", "description"], ["emb_title", "emb_desc"])
    
    def test_structured_index_specific_tensor_fields(self):
        """Test getting specific embedding field names for structured index"""
        tensor_fields = [
            TensorField(name="title", chunk_field_name="chunks_title", embeddings_field_name="emb_title"),
            TensorField(name="description", chunk_field_name="chunks_desc", embeddings_field_name="emb_desc"),
            TensorField(name="content", chunk_field_name="chunks_content", embeddings_field_name="emb_content")
        ]
        
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        mock_index.tensor_fields = tensor_fields
        
        result = get_embedding_field_names(mock_index, tensor_field_names=["title", "content"])
        
        assert result == (["title", "content"], ["emb_title", "emb_content"])
    
    def test_semistructured_index_all_tensor_fields(self):
        """Test getting all embedding field names for semi-structured index"""
        tensor_fields = [
            TensorField(name="title", chunk_field_name="chunks_title", embeddings_field_name="emb_title"),
            TensorField(name="description", chunk_field_name="chunks_desc", embeddings_field_name="emb_desc")
        ]
        
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.SemiStructured
        mock_index.tensor_fields = tensor_fields
        
        result = get_embedding_field_names(mock_index)
        
        assert result == (["title", "description"], ["emb_title", "emb_desc"])
    
    def test_legacy_unstructured_index_fails(self):
        """Test getting embedding field names for unstructured index - should raise error"""
        mock_index = Mock(spec=UnstructuredMarqoIndex)
        mock_index.type = IndexType.Unstructured
        mock_index.name = "test_unstructured_index"  # Add name for error message
        
        with pytest.raises(Exception) as exc_info:
            get_embedding_field_names(mock_index)
        
        assert "Attempting to retrieve only embeddings for unstructured index" in str(exc_info.value)
    
    def test_structured_index_no_tensor_fields(self):
        """Test structured index with no tensor fields"""
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        mock_index.tensor_fields = []
        
        result = get_embedding_field_names(mock_index)
        
        assert result == ([], [])
    
    def test_structured_index_missing_tensor_fields_attribute_fails(self):
        """Test structured index without tensor_fields attribute - should raise error"""
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        mock_index.name = "test_index"  # Add name attribute for error message
        # Don't set tensor_fields attribute
        del mock_index.tensor_fields
        
        # Should raise an error now instead of returning empty list
        with pytest.raises(Exception) as exc_info:
            get_embedding_field_names(mock_index)
        
        assert "has no tensor fields" in str(exc_info.value)

    def test_get_embedding_field_names_tensor_field_not_found_fails(self):
        """Test that requesting non-existent tensor field raises InvalidArgumentError"""

        mock_tensor_field = Mock()
        mock_tensor_field.name = "existing_field"

        mock_index = Mock()
        mock_index.type = IndexType.Structured
        mock_index.name = "test_index"
        mock_index.tensor_fields = [mock_tensor_field]

        with self.assertRaises(core_exceptions.InvalidArgumentError) as cm:
            get_embedding_field_names(mock_index, ["non_existent_field"])
        self.assertIn("Tensor field 'non_existent_field' not found in index", str(cm.exception))


class TestGetDocVectorsPerTensorFieldByIds(unittest.TestCase):
    """Test cases for get_doc_vectors_per_tensor_field_by_ids function"""

    def setUp(self):
        """Set up common test fixtures"""
        self.mock_config = Mock(spec=Config)
        self.mock_vespa_client = Mock()
        self.mock_config.vespa_client = self.mock_vespa_client
        self.mock_config.index_management = Mock()  # Add the missing index_management attribute
        
        # Mock index
        self.mock_index = Mock(spec=StructuredMarqoIndex)
        self.mock_index.type = IndexType.Structured
        self.mock_index.schema_name = "test_schema"
        
        # Mock tensor fields
        self.tensor_fields = [
            TensorField(name="title", chunk_field_name="chunks_title", embeddings_field_name="emb_title"),
            TensorField(name="description", chunk_field_name="chunks_desc", embeddings_field_name="emb_desc")
        ]
        self.mock_index.tensor_fields = self.tensor_fields
    
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_get_doc_vectors_structured_index_succeeds(self, mock_vespa_factory, mock_get_index, mock_metrics):
        """Test successfully getting embeddings from structured index"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock Vespa response with proper fields structure
        mock_doc_response = Mock()
        mock_doc_response.status = 200
        mock_doc_response.document.fields = {
            "marqo__id": "doc1",
            "emb_title": {
                "blocks": {
                    "0": [0.1, 0.2, 0.3],
                    "1": [0.4, 0.5, 0.6]
                }
            },
            "emb_desc": {
                "blocks": {
                    "0": [0.7, 0.8, 0.9]
                }
            }
        }
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            ["doc1"],
            tensor_fields=["title", "description"]
        )
        
        # Verify result
        expected = {
            "doc1": {
                "title": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "description": [[0.7, 0.8, 0.9]]
            }
        }
        assert result == expected
        
        # Verify get_batch was called with correct fields
        expected_fields = [structured_common.FIELD_ID, "emb_title", "emb_desc"]
        self.mock_vespa_client.get_batch.assert_called_once_with(
            ["doc1"], "test_schema", fields=expected_fields
        )
    
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_vespa_document_not_found_fails(self, mock_vespa_factory, mock_get_index, mock_metrics):
        """Test handling of document not found (404)"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock Vespa response with 404 - need to mock document.fields even for 404
        mock_doc_response = Mock()
        mock_doc_response.status = 404
        mock_doc_response.message = "Document not found"
        mock_doc_response.document.fields = {"marqo__id": "doc1"}  # Still need doc_id field
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function and expect exception
        with pytest.raises(Exception) as exc_info:
            get_doc_vectors_per_tensor_field_by_ids(
                self.mock_config,
                "test_index",
                ["doc1"]
            )
        
        # Should contain error message about failed retrieval
        assert "Failed to retrieve document" in str(exc_info.value)
    
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_legacy_unstructured_index_fails(self, mock_vespa_factory, mock_get_index, mock_metrics):
        """Test that unstructured index raises error since function is only for structured/semi-structured"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        # Create unstructured index mock
        mock_unstructured_index = Mock(spec=UnstructuredMarqoIndex)
        mock_unstructured_index.type = IndexType.Unstructured
        mock_unstructured_index.schema_name = "test_schema"
        mock_unstructured_index.name = "test_unstructured_index"  # Add name for error message
        
        mock_get_index.return_value = mock_unstructured_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Call the function and expect it to raise an error
        with pytest.raises(Exception) as exc_info:
            get_doc_vectors_per_tensor_field_by_ids(
                self.mock_config, 
                "test_index", 
                ["doc1"]
            )
        
        # Verify error message
        assert "Attempting to retrieve only embeddings for unstructured index" in str(exc_info.value)
    
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_document_without_vectors(self, mock_vespa_factory, mock_get_index, mock_metrics):
        """Test document that has no vector data"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        mock_get_index.return_value = self.mock_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock Vespa response with document that has no embedding fields
        mock_doc_response = Mock()
        mock_doc_response.status = 200
        mock_doc_response.document.fields = {
            "marqo__id": "doc1"
            # Missing emb_title and emb_desc fields
        }
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config,
            "test_index",
            ["doc1"]
        )
        
        # Should return empty embeddings for missing fields
        expected = {
            "doc1": {
                "title": [],
                "description": []
            }
        }
        assert result == expected
    
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_empty_document_ids(self, mock_vespa_factory, mock_get_index, mock_metrics):
        """Test handling of empty document IDs list"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock empty batch response
        mock_batch_response = Mock()
        mock_batch_response.responses = []
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            [],  # Empty document IDs
            tensor_fields=["title"]
        )
        
        # Verify empty result
        assert result == {}
        
        # Verify get_batch was called with empty document list
        expected_fields = [structured_common.FIELD_ID, "emb_title"]
        self.mock_vespa_client.get_batch.assert_called_once_with(
            [], "test_schema", fields=expected_fields
        )

    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.tensor_search._get_latest_index')
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_uses_cache_not_get_latest_index(self, mock_vespa_factory, mock_get_index, mock_get_latest_index, mock_metrics):
        """Test that get_doc_vectors_per_tensor_field_by_ids uses
        index_meta_cache.get_index instead of _get_latest_index for efficiency"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock Vespa response
        mock_doc_response = Mock()
        mock_doc_response.status = 200
        mock_doc_response.document.fields = {
            "marqo__id": "doc1",
            "emb_title": {
                "blocks": {
                    "0": [0.1, 0.2, 0.3]
                }
            }
        }
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            ["doc1"],
            tensor_fields=["title"]
        )
        
        # Verify that index_meta_cache.get_index was called
        mock_get_index.assert_called_once_with(index_management=self.mock_config.index_management, index_name="test_index")
        
        # Verify that _get_latest_index was NOT called
        mock_get_latest_index.assert_not_called()
        
        # Verify result is correct
        expected = {
            "doc1": {
                "title": [[0.1, 0.2, 0.3]]
            }
        }
        assert result == expected

    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_no_call_to_marqo_document_conversion(self, mock_vespa_factory, mock_get_index, mock_metrics):
        """Test that get_doc_vectors_per_tensor_field_by_ids
        does not call vespa_index.to_marqo_document for efficiency"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock Vespa response
        mock_doc_response = Mock()
        mock_doc_response.status = 200
        mock_doc_response.document.fields = {
            "marqo__id": "doc1",
            "emb_title": {
                "blocks": {
                    "0": [0.1, 0.2, 0.3]
                }
            },
            "emb_desc": {
                "blocks": {
                    "0": [0.7, 0.8, 0.9]
                }
            }
        }
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            ["doc1"],
            tensor_fields=["title", "description"]
        )
        
        # Verify that vespa_index_factory was called (this is still needed to create the vespa_index)
        mock_vespa_factory.assert_called_once_with(self.mock_index)
        
        # Verify that to_marqo_document was NOT called on the vespa_index
        # This is the key efficiency optimization - we skip the expensive Pydantic parsing
        mock_vespa_index.to_marqo_document.assert_not_called()
        
        # Verify result is still correct (processed directly from raw Vespa response)
        expected = {
            "doc1": {
                "title": [[0.1, 0.2, 0.3]],
                "description": [[0.7, 0.8, 0.9]]
            }
        }
        assert result == expected 

    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_get_vectors_from_multiple_documents_multiple_tensor_fields(self, mock_vespa_factory, mock_get_index, mock_metrics):
        """Test that loop indexes are handled correctly with 3 responses and 3 tensor fields each
        Tests that looping through result docs and fields is handled correctly"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        # Create mock index with 3 tensor fields
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        mock_index.schema_name = "test_schema"
        
        tensor_fields = [
            TensorField(name="field1", chunk_field_name="chunks_field1", embeddings_field_name="emb_field1"),
            TensorField(name="field2", chunk_field_name="chunks_field2", embeddings_field_name="emb_field2"),
            TensorField(name="field3", chunk_field_name="chunks_field3", embeddings_field_name="emb_field3")
        ]
        mock_index.tensor_fields = tensor_fields
        
        # Mock dependencies
        mock_get_index.return_value = mock_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock 3 Vespa responses with different embedding patterns to verify correct indexing
        mock_responses = []
        for i in range(3):
            doc_id = f"doc{i+1}"
            mock_response = Mock()
            mock_response.status = 200
            mock_response.document.fields = {
                "marqo__id": doc_id,
                "emb_field1": {
                    "blocks": {
                        "0": [0.1 + i, 0.2 + i, 0.3 + i],  # Different values per document
                        "1": [0.4 + i, 0.5 + i, 0.6 + i]
                    }
                },
                "emb_field2": {
                    "blocks": {
                        "0": [1.1 + i, 1.2 + i, 1.3 + i],
                    }
                },
                "emb_field3": {
                    "blocks": {
                        "0": [2.1 + i, 2.2 + i, 2.3 + i],
                        "1": [2.4 + i, 2.5 + i, 2.6 + i],
                        "2": [2.7 + i, 2.8 + i, 2.9 + i]
                    }
                }
            }
            mock_responses.append(mock_response)
        
        mock_batch_response = Mock()
        mock_batch_response.responses = mock_responses
        self.mock_config.vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function with 3 document IDs
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            ["doc1", "doc2", "doc3"],
            tensor_fields=["field1", "field2", "field3"]
        )
        
        # Verify that each document has the correct embeddings for each field
        expected = {
            "doc1": {
                "field1": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "field2": [[1.1, 1.2, 1.3]],
                "field3": [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6], [2.7, 2.8, 2.9]]
            },
            "doc2": {
                "field1": [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
                "field2": [[2.1, 2.2, 2.3]],
                "field3": [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6], [3.7, 3.8, 3.9]]
            },
            "doc3": {
                "field1": [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]],
                "field2": [[3.1, 3.2, 3.3]],
                "field3": [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6], [4.7, 4.8, 4.9]]
            }
        }
        assert result == expected
        
        # Verify get_batch was called with correct fields
        expected_fields = [structured_common.FIELD_ID, "emb_field1", "emb_field2", "emb_field3"]
        self.mock_config.vespa_client.get_batch.assert_called_once_with(
            ["doc1", "doc2", "doc3"], "test_schema", fields=expected_fields
        )