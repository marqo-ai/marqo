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
        
        assert result == ["emb_title", "emb_desc"]
    
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
        
        assert result == ["emb_title", "emb_content"]
    
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
        
        assert result == ["emb_title", "emb_desc"]
    
    def test_unstructured_index(self):
        """Test getting embedding field names for unstructured index"""
        mock_index = Mock(spec=UnstructuredMarqoIndex)
        mock_index.type = IndexType.Unstructured
        
        result = get_embedding_field_names(mock_index)
        
        assert result == [unstructured_common.VESPA_DOC_EMBEDDINGS]
    
    def test_structured_index_no_tensor_fields(self):
        """Test structured index with no tensor fields"""
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        mock_index.tensor_fields = []
        
        result = get_embedding_field_names(mock_index)
        
        assert result == []
    
    def test_structured_index_missing_tensor_fields_attribute(self):
        """Test structured index without tensor_fields attribute"""
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        # Don't set tensor_fields attribute
        del mock_index.tensor_fields
        
        result = get_embedding_field_names(mock_index)
        
        assert result == []


class TestGetDocVectorsPerTensorFieldByIds(unittest.TestCase):
    """Test cases for get_doc_vectors_per_tensor_field_by_ids function"""
    
    def setup_method(self, method):
        """Set up common test fixtures"""
        self.mock_config = Mock(spec=Config)
        self.mock_vespa_client = Mock()
        self.mock_config.vespa_client = self.mock_vespa_client
        
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
    @patch('marqo.tensor_search.tensor_search._get_latest_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_successful_get_embeddings_structured_index(self, mock_vespa_factory, mock_get_index, mock_metrics):
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
        
        # Mock Vespa response
        mock_doc_response = Mock()
        mock_doc_response.status = 200
        mock_doc_response.document.dict.return_value = {
            structured_common.FIELD_ID: "doc1",
            constants.MARQO_DOC_TENSORS: {
                "title": {
                    constants.MARQO_DOC_EMBEDDINGS: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
                },
                "description": {
                    constants.MARQO_DOC_EMBEDDINGS: [[0.7, 0.8, 0.9]]
                }
            }
        }
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Mock vespa index conversion
        mock_vespa_index.to_marqo_document.return_value = {
            '_id': 'doc1',
            constants.MARQO_DOC_TENSORS: {
                "title": {
                    constants.MARQO_DOC_EMBEDDINGS: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
                },
                "description": {
                    constants.MARQO_DOC_EMBEDDINGS: [[0.7, 0.8, 0.9]]
                }
            }
        }
        
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
    @patch('marqo.tensor_search.tensor_search._get_latest_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_document_not_found(self, mock_vespa_factory, mock_get_index, mock_metrics):
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
        
        # Mock Vespa response with 404
        mock_doc_response = Mock()
        mock_doc_response.status = 404
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            ["doc1"]
        )
        
        # Should return empty result for non-200 status
        assert result == {}
    
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.tensor_search._get_latest_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    def test_unstructured_index(self, mock_vespa_factory, mock_get_index, mock_metrics):
        """Test with unstructured index"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        # Create unstructured index mock
        mock_unstructured_index = Mock(spec=UnstructuredMarqoIndex)
        mock_unstructured_index.type = IndexType.Unstructured
        mock_unstructured_index.schema_name = "test_schema"
        
        mock_get_index.return_value = mock_unstructured_index
        mock_vespa_index = Mock()
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Mock Vespa response
        mock_doc_response = Mock()
        mock_doc_response.status = 200
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Mock vespa index conversion
        mock_vespa_index.to_marqo_document.return_value = {
            '_id': 'doc1',
            constants.MARQO_DOC_TENSORS: {
                "field1": {
                    constants.MARQO_DOC_EMBEDDINGS: [[0.1, 0.2, 0.3]]
                }
            }
        }
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            ["doc1"]
        )
        
        # Verify result
        expected = {
            "doc1": {
                "field1": [[0.1, 0.2, 0.3]]
            }
        }
        assert result == expected
        
        # Verify get_batch was called with unstructured embeddings field
        expected_fields = [structured_common.FIELD_ID, unstructured_common.VESPA_DOC_EMBEDDINGS]
        self.mock_vespa_client.get_batch.assert_called_once_with(
            ["doc1"], "test_schema", fields=expected_fields
        )
    
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.tensor_search._get_latest_index')
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
        
        # Mock Vespa response
        mock_doc_response = Mock()
        mock_doc_response.status = 200
        
        mock_batch_response = Mock()
        mock_batch_response.responses = [mock_doc_response]
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Mock vespa index conversion - document without tensors
        mock_vespa_index.to_marqo_document.return_value = {
            '_id': 'doc1'
            # No MARQO_DOC_TENSORS field
        }
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            ["doc1"]
        )
        
        # Should return empty embeddings for document without vectors
        expected = {"doc1": {}}
        assert result == expected
    
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    @patch('marqo.tensor_search.tensor_search._get_latest_index')
    def test_empty_document_ids(self, mock_get_index, mock_metrics):
        """Test with empty document IDs list"""
        
        # Mock RequestMetricsStore
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.time.return_value.__enter__ = Mock()
        mock_metrics_instance.time.return_value.__exit__ = Mock()
        
        mock_get_index.return_value = self.mock_index
        
        # Mock empty batch response
        mock_batch_response = Mock()
        mock_batch_response.responses = []
        self.mock_vespa_client.get_batch.return_value = mock_batch_response
        
        # Call the function
        result = get_doc_vectors_per_tensor_field_by_ids(
            self.mock_config, 
            "test_index", 
            []
        )
        
        # Should return empty result
        assert result == {} 