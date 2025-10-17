import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from marqo.core.search.recommender import Recommender
from marqo.core.models.marqo_index import IndexType, StructuredMarqoIndex, UnstructuredMarqoIndex
from marqo.exceptions import InvalidArgumentError
from marqo.core.exceptions import InvalidFieldNameError
from marqo.vespa.vespa_client import VespaClient
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.inference.api import Inference
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.utils.vector_interpolation import AllZeroWeightsError
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor
from marqo.api.exceptions import InvalidDocumentIdError


class TestRecommenderGetDocVectorsFromIds(unittest.TestCase):
    """Test cases for the updated Recommender.get_doc_vectors_from_ids method"""
    
    def setUp(self):
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
        self.assertEqual(result, expected)
        
        # Verify tensor_search function was called correctly
        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1", "doc2"], tensor_fields=None, allow_missing_documents=False
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
        self.assertEqual(result, expected)
        
        # Verify tensor_search function was called with non-zero weight docs only
        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1", "doc3"], tensor_fields=None, allow_missing_documents=False
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
        self.assertEqual(result, expected)
        
        # Verify tensor_search function was called with specific fields
        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1"], tensor_fields=["title", "content"], allow_missing_documents=False
        )
    
    @patch('marqo.tensor_search.index_meta_cache.get_index')
    def test_invalid_tensor_field_structured_index(self, mock_get_index):
        """Test validation of tensor field names for structured index"""
        
        mock_get_index.return_value = self.mock_structured_index
        
        # Try to use invalid tensor field
        with self.assertRaises(InvalidFieldNameError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1"],
                tensor_fields=["invalid_field"]
            )
        
        self.assertIn("invalid_field", str(cm.exception))
        self.assertIn("Available tensor fields", str(cm.exception))
    
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
        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1", "doc2"]
            )
        
        self.assertIn("not found", str(cm.exception))
        self.assertIn("doc2", str(cm.exception))
    
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
        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1", "doc2"]
            )
        
        self.assertIn("do not have embeddings", str(cm.exception))
        self.assertIn("doc2", str(cm.exception))
    
    def test_empty_documents_list(self):
        """Test with empty documents list"""
        
        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=[]
            )
        
        self.assertIn("No document IDs provided", str(cm.exception))

    def test_non_string_ids_fails(self):
        """Test that document id validation catches non string IDs and errors out"""
        with self.assertRaises(InvalidDocumentIdError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=[123, 456]
            )

        self.assertIn("Document _id must be a string type", str(cm.exception))
    
    def test_none_documents(self):
        """Test with None documents"""
        
        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=None
            )
        
        self.assertIn("No document IDs provided", str(cm.exception))
    
    def test_all_zero_weight_documents(self):
        """Test with all documents having zero weight"""
        
        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents={"doc1": 0.0, "doc2": 0.0}
            )
        
        self.assertIn("No documents with non-zero weight provided", str(cm.exception))

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.config.Config')
    @patch('marqo.tensor_search.tensor_search.search')
    def test_recommend_with_interpolation_method_none(self, mock_search, mock_config_class, mock_get_index):
        """Test recommend method when interpolation_method is None"""

        # Mock dependencies
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.name = "test_index"
        mock_index.normalize_embeddings = True
        mock_index.type = IndexType.Structured
        mock_get_index.return_value = mock_index
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_search.return_value = {"hits": []}
        
        # Mock get_doc_vectors_from_ids to return some vectors
        with patch.object(self.recommender, 'get_doc_vectors_from_ids') as mock_get_vectors:
            mock_get_vectors.return_value = {
                "doc1": [[0.1, 0.2, 0.3]]
            }
            
            # Mock get_default_interpolation_method
            with patch.object(self.recommender, 'get_default_interpolation_method') as mock_get_default:
                mock_get_default.return_value = InterpolationMethod.SLERP
                
                # Mock from_interpolation_method to capture which method is used
                with patch('marqo.core.search.recommender.from_interpolation_method') as mock_from_interp:
                    mock_interpolation = Mock()
                    mock_interpolation.interpolate.return_value = [0.1, 0.2, 0.3]
                    mock_from_interp.return_value = mock_interpolation
                    
                    # Call recommend with interpolation_method=None
                    result = self.recommender.recommend(
                        index_name="test_index",
                        documents=["doc1"],
                        interpolation_method=None  # This triggers interpolation method selection
                    )
                    
                    # This confirms that get_default_interpolation_method is being called
                    mock_get_default.assert_called_once_with(mock_index, ["doc1"])
                    
                    # Assert that the final interpolation method called would be SLERP
                    mock_from_interp.assert_called_once_with(InterpolationMethod.SLERP)

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.config.Config')
    @patch('marqo.tensor_search.tensor_search.search')
    def test_recommend_with_dict_documents_filtering(self, mock_search, mock_config_class, mock_get_index):
        """Test recommend method with dict documents for filtering"""
        
        # Mock dependencies
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.name = "test_index"
        mock_index.normalize_embeddings = True
        mock_get_index.return_value = mock_index
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_search.return_value = {"hits": []}
        
        # Mock get_doc_vectors_from_ids to return some vectors
        with patch.object(self.recommender, 'get_doc_vectors_from_ids') as mock_get_vectors:
            mock_get_vectors.return_value = {
                "doc1": [[0.1, 0.2, 0.3]],
                "doc2": [[0.4, 0.5, 0.6]]
            }
            
            # Mock get_exclusion_filter to verify it's called with all document IDs
            with patch.object(self.recommender, 'get_exclusion_filter') as mock_get_filter:
                mock_get_filter.return_value = "filtered"
                
                # Call recommend with dict documents
                documents = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.0}  # doc3 has zero weight
                result = self.recommender.recommend(
                    index_name="test_index",
                    documents=documents,
                    exclude_input_documents=True
                )
                
                # Verify get_exclusion_filter was called with ALL document IDs
                # including zero-weight documents for proper filtering
                mock_get_filter.assert_called_once()
                args = mock_get_filter.call_args[0]
                all_document_ids = args[1]  # Second argument should be all_document_ids
                self.assertEqual(set(all_document_ids), {"doc1", "doc2", "doc3"})

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.config.Config')
    @patch('marqo.tensor_search.tensor_search.search')
    def test_recommend_slerp_all_zero_weights_error(self, mock_search, mock_config_class, mock_get_index):
        """Test recommend method SLERP error handling"""
        
        # Mock dependencies
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.name = "test_index"
        mock_index.normalize_embeddings = True
        mock_get_index.return_value = mock_index
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock get_doc_vectors_from_ids to return some vectors
        with patch.object(self.recommender, 'get_doc_vectors_from_ids') as mock_get_vectors:
            mock_get_vectors.return_value = {
                "doc1": [[0.1, 0.2, 0.3]]
            }
            
            # Mock vector interpolation to raise AllZeroWeightsError
            with patch('marqo.core.utils.vector_interpolation.from_interpolation_method') as mock_from_interp:
                mock_interpolation = Mock()
                mock_interpolation.interpolate.side_effect = AllZeroWeightsError("All weights are zero")
                mock_from_interp.return_value = mock_interpolation
                
                # Test SLERP error handling
                with self.assertRaises(InvalidArgumentError) as cm:
                    self.recommender.recommend(
                        index_name="test_index",
                        documents={"doc1": 0.0},  # Zero weight to trigger error
                        interpolation_method=InterpolationMethod.SLERP
                    )
                
                # Verify generic error message (same for all interpolation methods)
                self.assertIn("Cannot interpolate vectors with all zero weights", str(cm.exception))

    def test_get_default_interpolation_method_normalize_embeddings_with_context(self):
        """Test get_default_interpolation_method with normalize_embeddings=True and context docs"""
        
        # Mock index with normalize_embeddings=True
        mock_index = Mock()
        mock_index.normalize_embeddings = True
        
        # Test with context documents
        result = self.recommender.get_default_interpolation_method(mock_index, ["doc1"])
        
        # Should return SLERP for normalized embeddings with context docs
        self.assertEqual(result, InterpolationMethod.SLERP)

    def test_get_default_interpolation_method_normalize_embeddings_no_context(self):
        """Test get_default_interpolation_method with normalize_embeddings=True and no context docs"""
        
        # Mock index with normalize_embeddings=True
        mock_index = Mock()
        mock_index.normalize_embeddings = True
        
        # Test with no context documents
        result = self.recommender.get_default_interpolation_method(mock_index, None)
        
        # Should return NLERP for normalized embeddings without context docs
        self.assertEqual(result, InterpolationMethod.NLERP)

    def test_get_default_interpolation_method_no_normalize_embeddings(self):
        """Test get_default_interpolation_method with normalize_embeddings=False"""
        
        # Mock index with normalize_embeddings=False
        mock_index = Mock()
        mock_index.normalize_embeddings = False
        
        # Test with any context documents
        result = self.recommender.get_default_interpolation_method(mock_index, ["doc1"])
        
        # Should return LERP for non-normalized embeddings
        self.assertEqual(result, InterpolationMethod.LERP)

    # Error scenario tests
    def test_get_doc_vectors_from_ids_with_invalid_document_ids_fails(self):
        """Test get_doc_vectors_from_ids with invalid document IDs"""
        
        # Test with invalid document ID format (should be caught by validation elsewhere)
        with self.assertRaises(Exception):  # Specific exception depends on validation layer
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=[""]  # Empty string ID
            )

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    def test_get_doc_vectors_from_ids_with_nonexistent_index_fails(self, mock_get_index):
        """Test get_doc_vectors_from_ids with nonexistent index"""
        
        # Mock index not found
        from marqo.core.exceptions import IndexNotFoundError
        mock_get_index.side_effect = IndexNotFoundError("Index not found")
        
        with self.assertRaises(IndexNotFoundError):
            self.recommender.get_doc_vectors_from_ids(
                index_name="nonexistent_index",
                documents=["doc1"]
            )

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    def test_get_doc_vectors_from_ids_with_invalid_tensor_fields_fails(self, mock_get_index):
        """Test get_doc_vectors_from_ids with invalid tensor fields for structured index"""
        
        # Mock structured index with specific tensor fields
        mock_structured_index = Mock(spec=StructuredMarqoIndex)
        mock_structured_index.type = IndexType.Structured
        mock_structured_index.tensor_field_map = {"valid_field": "some_config"}
        mock_get_index.return_value = mock_structured_index
        
        with self.assertRaises(InvalidFieldNameError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1"],
                tensor_fields=["invalid_field"]
            )
        self.assertIn("Tensor field \"invalid_field\" not found", str(cm.exception))

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_get_doc_vectors_from_ids_with_missing_documents_fails(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test get_doc_vectors_from_ids when some documents are not found"""
        
        # Mock dependencies
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        mock_get_index.return_value = mock_index
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock get_doc_vectors_per_tensor_field_by_ids to return only some documents
        mock_get_vectors.return_value = {
            "doc1": {"field1": [[0.1, 0.2]]}
            # doc2 is missing
        }
        
        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1", "doc2"]  # doc2 will be missing
            )
        self.assertIn("The following document IDs were not found: doc2", str(cm.exception))

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_get_doc_vectors_from_ids_with_documents_without_embeddings_fails(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test get_doc_vectors_from_ids when documents have no embeddings"""
        
        # Mock dependencies
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured
        mock_get_index.return_value = mock_index
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock get_doc_vectors_per_tensor_field_by_ids to return documents with no embeddings
        mock_get_vectors.return_value = {
            "doc1": {"field1": []},  # No embeddings
            "doc2": {"field1": [[0.1, 0.2]]}  # Has embeddings
        }
        
        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1", "doc2"]
            )
        self.assertIn("The following documents do not have embeddings: doc1", str(cm.exception))

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.config.Config')
    @patch('marqo.tensor_search.tensor_search.search')
    def test_recommend_with_lerp_all_zero_weights_error(self, mock_search, mock_config_class, mock_get_index):
        """Test recommend method LERP all zero weights error handling"""
        
        # Mock dependencies
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.name = "test_index"
        mock_index.normalize_embeddings = False
        mock_index.type = IndexType.Structured
        mock_get_index.return_value = mock_index
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock get_doc_vectors_from_ids to return some vectors
        with patch.object(self.recommender, 'get_doc_vectors_from_ids') as mock_get_vectors:
            mock_get_vectors.return_value = {
                "doc1": [[0.1, 0.2, 0.3]]
            }
            
            # Mock vector interpolation to raise AllZeroWeightsError
            with patch('marqo.core.utils.vector_interpolation.from_interpolation_method') as mock_from_interp:
                mock_interpolation = Mock()
                mock_interpolation.interpolate.side_effect = AllZeroWeightsError("All weights are zero")
                mock_from_interp.return_value = mock_interpolation
                
                # Test LERP/NLERP error handling (non-SLERP case)
                with self.assertRaises(InvalidArgumentError) as cm:
                    self.recommender.recommend(
                        index_name="test_index",
                        documents={"doc1": 0.0},  # Zero weight to trigger error
                        interpolation_method=InterpolationMethod.LERP
                    )
                
                # Verify generic error message (same for all interpolation methods)
                self.assertIn("Cannot interpolate vectors with all zero weights", str(cm.exception))

    def test_get_exclusion_filter_for_structured_index(self):
        """Test get_exclusion_filter for structured index format"""
        
        # Mock structured index
        mock_index = Mock()
        mock_index.type = IndexType.Structured
        
        # Test structured index filter format
        result = self.recommender.get_exclusion_filter(
            mock_index, 
            ["doc1", "doc2"], 
            None
        )
        expected = "NOT _id IN (doc1, doc2)"
        self.assertEqual(result, expected)

    def test_get_exclusion_filter_for_unstructured_index(self):
        """Test get_exclusion_filter for unstructured index format"""
        
        # Mock unstructured index
        mock_index = Mock()
        mock_index.type = IndexType.Unstructured
        
        # Test unstructured index filter format
        result = self.recommender.get_exclusion_filter(
            mock_index, 
            ["doc1", "doc2"], 
            None
        )
        expected = "NOT (_id:(doc1) OR _id:(doc2))"
        self.assertEqual(result, expected)

    def test_get_exclusion_filter_with_user_filter(self):
        """Test get_exclusion_filter combined with user filter"""
        
        # Mock structured index
        mock_index = Mock()
        mock_index.type = IndexType.Structured
        
        # Test with user filter
        result = self.recommender.get_exclusion_filter(
            mock_index, 
            ["doc1"], 
            "category:books"
        )
        expected = "(category:books) AND NOT _id IN (doc1)"
        self.assertEqual(result, expected)

    def test_get_exclusion_filter_with_empty_user_filter(self):
        """Test get_exclusion_filter with empty user filter"""
        
        # Mock structured index
        mock_index = Mock()
        mock_index.type = IndexType.Structured
        
        # Test with empty user filter
        result = self.recommender.get_exclusion_filter(
            mock_index, 
            ["doc1"], 
            "   "  # Empty/whitespace filter
        )
        expected = "NOT _id IN (doc1)"
        self.assertEqual(result, expected)

    def test_duplicate_document_ids_in_list_fails(self):
        """Test that duplicate document IDs in a list raise InvalidArgumentError
        for get_doc_vectors_from_ids"""

        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1", "doc2", "doc1", "doc3", "doc2"]
            )

        error_message = str(cm.exception)
        self.assertIn("Duplicate document IDs found", error_message)
        # Should mention both duplicate IDs
        self.assertIn("doc1", error_message)
        self.assertIn("doc2", error_message)

    def test_single_duplicate_document_id_in_list_fails(self):
        """Test that a single duplicate document ID in a list raises InvalidArgumentError"""

        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.get_doc_vectors_from_ids(
                index_name="test_index",
                documents=["doc1", "doc2", "doc3", "doc1"]
            )

        error_message = str(cm.exception)
        self.assertIn("Duplicate document IDs found", error_message)
        self.assertIn("doc1", error_message)

    def test_recommend_with_duplicate_document_ids_fails(self):
        """Test that recommend method also catches duplicate document IDs"""
        
        with self.assertRaises(InvalidArgumentError) as cm:
            self.recommender.recommend(
                index_name="test_index",
                documents=["doc1", "doc2", "doc1"]
            )
        
        error_message = str(cm.exception)
        self.assertIn("Duplicate document IDs found", error_message)
        self.assertIn("doc1", error_message)

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_get_doc_vectors_allow_missing_documents_true(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test that allowMissingDocuments=True allows missing documents and only returns existing ones"""
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_structured_index
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock response with only some documents (simulating missing documents ignored)
        mock_get_vectors.return_value = {
            "doc1": {"title": [[0.1, 0.2, 0.3]]},
            "doc3": {"title": [[0.7, 0.8, 0.9]]}
            # doc2 and doc4 are missing but should be ignored due to allowMissingDocuments=True
        }
        
        # Should succeed with allowMissingDocuments=True
        result = self.recommender.get_doc_vectors_from_ids(
            index_name="test_index",
            documents=["doc1", "doc2", "doc3", "doc4"],
            allow_missing_documents=True
        )
        
        # Should only return vectors for existing documents
        expected = {
            "doc1": [[0.1, 0.2, 0.3]],
            "doc3": [[0.7, 0.8, 0.9]]
        }
        self.assertEqual(result, expected)
        
        # Verify tensor_search function was called with allowMissingDocuments=True
        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1", "doc2", "doc3", "doc4"], tensor_fields=None, allow_missing_documents=True
        )

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_get_doc_vectors_allow_missing_embeddings_true(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test that allowMissingEmbeddings=True allows documents with missing embeddings"""

        # Mock dependencies
        mock_get_index.return_value = self.mock_structured_index
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # Mock response where one document has no embeddings for requested field
        mock_get_vectors.return_value = {
            "doc1": {"title": [[0.1, 0.2, 0.3]]},
            "doc2": {}  # No embeddings for the requested field
        }

        r = self.recommender.get_doc_vectors_from_ids(
            index_name="test_index",
            documents=["doc1", "doc2"],
            tensor_fields=["title"],
            allow_missing_embeddings=True
        )

        self.assertEqual(
            {"doc1": [[0.1, 0.2, 0.3]]}, r
        )

        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1", "doc2"], tensor_fields=["title"],
            allow_missing_documents=False
        )

    @patch('marqo.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.get_doc_vectors_per_tensor_field_by_ids')
    @patch('marqo.config.Config')
    def test_get_doc_vectors_allow_both_missing_parameters_true(self, mock_config_class, mock_get_vectors, mock_get_index):
        """Test that both allowMissingDocuments=True and allowMissingEmbeddings=True work together"""
        
        # Mock dependencies
        mock_get_index.return_value = self.mock_structured_index
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock response where some documents are missing and others lack embeddings
        mock_get_vectors.return_value = {
            "doc1": {"title": [[0.1, 0.2, 0.3]]},
            "doc4": {},
            # doc2 missing entirely, doc3 exists but has no title embeddings
        }
        
        # Should succeed with both parameters set to True
        result = self.recommender.get_doc_vectors_from_ids(
            index_name="test_index",
            documents=["doc1", "doc2", "doc3", "doc4"],
            tensor_fields=["title"],
            allow_missing_documents=True,
            allow_missing_embeddings=True
        )
        
        # Should only return vectors for documents that exist and have embeddings
        expected = {
            "doc1": [[0.1, 0.2, 0.3]],
        }
        self.assertEqual(result, expected)
        
        # Verify tensor_search function was called with both parameters=True
        mock_get_vectors.assert_called_once_with(
            mock_config, "test_index", ["doc1", "doc2", "doc3", "doc4"], tensor_fields=["title"], allow_missing_documents=True
        )