import unittest
from unittest.mock import patch, MagicMock

from marqo.core.search.hybrid_search import HybridSearch
from marqo.core.models.marqo_index import StructuredMarqoIndex
from marqo.config import Config


class TestHybridSearchApproximateThreshold(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_config = MagicMock(spec=Config)
        self.mock_marqo_index = MagicMock(spec=StructuredMarqoIndex)
        self.mock_marqo_index.schema_name = "test_schema"
        
        # Create the HybridSearch instance
        self.hybrid_search = HybridSearch()
        
    @patch('marqo.core.search.hybrid_search.get_lexical_score_modifiers')
    @patch('marqo.core.search.hybrid_search.get_tensor_score_modifiers')
    @patch('marqo.core.search.hybrid_search.ScoringFunctionEnsemble._get_score_modifiers')
    @patch('marqo.core.search.hybrid_search._lexical_search')
    @patch('marqo.core.search.hybrid_search._tensor_search')
    def test_search_passes_approximate_threshold(
        self, 
        mock_tensor_search, 
        mock_lexical_search, 
        mock_get_score_modifiers,
        mock_get_tensor_score_modifiers, 
        mock_get_lexical_score_modifiers
    ):
        """Test that approximate_threshold is passed to _tensor_search"""
        # Set up return values
        mock_tensor_search.return_value = {"hits": [], "query": "test query"}
        mock_lexical_search.return_value = {"hits": [], "query": "test query"}
        mock_get_score_modifiers.return_value = []
        mock_get_tensor_score_modifiers.return_value = []
        mock_get_lexical_score_modifiers.return_value = []
        
        # Call search with approximate_threshold
        self.hybrid_search.search(
            config=self.mock_config,
            marqo_index=self.mock_marqo_index,
            query="test query",
            approximate=True,
            approximate_threshold=0.75
        )
        
        # Check that _tensor_search was called with approximate_threshold
        mock_tensor_search.assert_called_once()
        call_kwargs = mock_tensor_search.call_args[1]
        self.assertEqual(call_kwargs.get('approximate_threshold'), 0.75)
    
    @patch('marqo.core.search.hybrid_search.get_lexical_score_modifiers')
    @patch('marqo.core.search.hybrid_search.get_tensor_score_modifiers')
    @patch('marqo.core.search.hybrid_search.ScoringFunctionEnsemble._get_score_modifiers')
    @patch('marqo.core.search.hybrid_search._lexical_search')
    @patch('marqo.core.search.hybrid_search._tensor_search')
    def test_search_without_approximate_threshold(
        self, 
        mock_tensor_search, 
        mock_lexical_search, 
        mock_get_score_modifiers,
        mock_get_tensor_score_modifiers, 
        mock_get_lexical_score_modifiers
    ):
        """Test that search works without approximate_threshold"""
        # Set up return values
        mock_tensor_search.return_value = {"hits": [], "query": "test query"}
        mock_lexical_search.return_value = {"hits": [], "query": "test query"}
        mock_get_score_modifiers.return_value = []
        mock_get_tensor_score_modifiers.return_value = []
        mock_get_lexical_score_modifiers.return_value = []
        
        # Call search without approximate_threshold
        self.hybrid_search.search(
            config=self.mock_config,
            marqo_index=self.mock_marqo_index,
            query="test query",
            approximate=True
        )
        
        # Check that _tensor_search was called without approximate_threshold
        mock_tensor_search.assert_called_once()
        call_kwargs = mock_tensor_search.call_args[1]
        self.assertIsNone(call_kwargs.get('approximate_threshold'))


if __name__ == "__main__":
    unittest.main() 