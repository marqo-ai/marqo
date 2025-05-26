import unittest
from unittest.mock import patch, MagicMock

from marqo.tensor_search import api


class TestSearchEndpointApproximateThreshold(unittest.TestCase):
    def setUp(self):
        # Create a mock for tensor_search.search
        self.search_patcher = patch('marqo.tensor_search.api.tensor_search.search')
        self.mock_search = self.search_patcher.start()
        self.mock_search.return_value = {"hits": [], "query": "test query"}
        
        # Create a mock for the dependency injection
        self.config_patcher = patch('marqo.tensor_search.api.get_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_config = MagicMock()
        self.mock_get_config.return_value = self.mock_config
        
        # Create a mock for the API validation
        self.validation_patcher = patch('marqo.tensor_search.api.api_validation.validate_device')
        self.mock_validate_device = self.validation_patcher.start()
        self.mock_validate_device.return_value = "cpu"
        
        # Create a mock for the parse_request_object function
        self.parse_patcher = patch('marqo.tensor_search.api.parse_request_object')
        self.mock_parse = self.parse_patcher.start()
        
    def tearDown(self):
        # Stop all patchers
        self.search_patcher.stop()
        self.config_patcher.stop()
        self.validation_patcher.stop()
        self.parse_patcher.stop()
        
    def test_search_endpoint_with_approximate_threshold(self):
        """Test that the search endpoint passes approximateThreshold to tensor_search.search"""
        # Create a mock SearchQuery with approximateThreshold
        mock_search_query = MagicMock()
        mock_search_query.q = "test query"
        mock_search_query.searchMethod = "tensor"
        mock_search_query.limit = 10
        mock_search_query.offset = 0
        mock_search_query.rerankDepth = None
        mock_search_query.efSearch = None
        mock_search_query.approximate = True
        mock_search_query.approximateThreshold = 0.8
        mock_search_query.reRanker = None
        mock_search_query.filter = None
        mock_search_query.attributesToRetrieve = None
        mock_search_query.boost = None
        mock_search_query.mediaDownloadHeaders = None
        mock_search_query.context = None
        mock_search_query.scoreModifiers = None
        mock_search_query.modelAuth = None
        mock_search_query.showHighlights = True
        mock_search_query.searchableAttributes = None
        mock_search_query.textQueryPrefix = None
        
        # Set up the mock parse_request_object to return our mock SearchQuery
        self.mock_parse.return_value = mock_search_query
        
        # Call the search endpoint
        api.search(
            index_name="test_index",
            search_query_dict={"q": "test query", "approximateThreshold": 0.8},
            device="cpu"
        )
        
        # Check that tensor_search.search was called with the correct parameters
        self.mock_search.assert_called_once()
        call_kwargs = self.mock_search.call_args[1]
        self.assertEqual(call_kwargs.get('approximate_threshold'), 0.8)
        
    def test_search_endpoint_without_approximate_threshold(self):
        """Test that the search endpoint works without approximateThreshold"""
        # Create a mock SearchQuery without approximateThreshold
        mock_search_query = MagicMock()
        mock_search_query.q = "test query"
        mock_search_query.searchMethod = "tensor"
        mock_search_query.limit = 10
        mock_search_query.offset = 0
        mock_search_query.rerankDepth = None
        mock_search_query.efSearch = None
        mock_search_query.approximate = True
        mock_search_query.approximateThreshold = None
        mock_search_query.reRanker = None
        mock_search_query.filter = None
        mock_search_query.attributesToRetrieve = None
        mock_search_query.boost = None
        mock_search_query.mediaDownloadHeaders = None
        mock_search_query.context = None
        mock_search_query.scoreModifiers = None
        mock_search_query.modelAuth = None
        mock_search_query.showHighlights = True
        mock_search_query.searchableAttributes = None
        mock_search_query.textQueryPrefix = None
        
        # Set up the mock parse_request_object to return our mock SearchQuery
        self.mock_parse.return_value = mock_search_query
        
        # Call the search endpoint
        api.search(
            index_name="test_index",
            search_query_dict={"q": "test query"},
            device="cpu"
        )
        
        # Check that tensor_search.search was called with the correct parameters
        self.mock_search.assert_called_once()
        call_kwargs = self.mock_search.call_args[1]
        self.assertIsNone(call_kwargs.get('approximate_threshold'))


if __name__ == "__main__":
    unittest.main() 