import unittest
from unittest.mock import patch, MagicMock

from marqo.config import Config


class TestApproximateThreshold(unittest.TestCase):
    def setUp(self):
        # Mock VespaClient and Config
        self.vespa_client_mock = MagicMock()
        self.inference_mock = MagicMock()
        self.config = Config(self.vespa_client_mock, self.inference_mock)

    @patch('marqo.tensor_search.tensor_search._vector_text_search')
    def test_vector_text_search_with_approximate_threshold(
        self, mock_vector_text_search
    ):
        """Test that approximate_threshold parameter is passed to 
        _vector_text_search"""
        # Mock the return value
        mock_vector_text_search.return_value = {
            "hits": [], "query": "test query"
        }
        
        # Import and call the function directly
        from marqo.tensor_search.tensor_search import _vector_text_search
        
        _vector_text_search(
            config=self.config,
            index_name="test_index",
            query="test query",
            approximate_threshold=0.75
        )
        
        # Check that the function was called with approximate_threshold
        mock_vector_text_search.assert_called_once()
        call_kwargs = mock_vector_text_search.call_args[1]
        self.assertEqual(call_kwargs.get('approximate_threshold'), 0.75)

    @patch('marqo.tensor_search.tensor_search._vector_text_search')
    def test_vector_text_search_without_approximate_threshold(
        self, mock_vector_text_search
    ):
        """Test that _vector_text_search works without approximate_threshold"""
        # Mock the return value
        mock_vector_text_search.return_value = {
            "hits": [], "query": "test query"
        }
        
        # Import and call the function directly
        from marqo.tensor_search.tensor_search import _vector_text_search
        
        _vector_text_search(
            config=self.config,
            index_name="test_index",
            query="test query"
        )
        
        # Check that the function was called without approximate_threshold
        mock_vector_text_search.assert_called_once()
        call_kwargs = mock_vector_text_search.call_args[1]
        self.assertIsNone(call_kwargs.get('approximate_threshold'))


if __name__ == "__main__":
    unittest.main() 