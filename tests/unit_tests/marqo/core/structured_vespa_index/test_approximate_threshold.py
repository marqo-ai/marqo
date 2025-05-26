import unittest
from unittest.mock import patch, MagicMock

from marqo.core.models.marqo_index import StructuredMarqoIndex, Model, TextPreProcessing
from marqo.core.models.marqo_index import DistanceMetric, VectorNumericType, HnswConfig
from marqo.core.models.marqo_query import MarqoTensorQuery
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex


class TestStructuredVespaIndexApproximateThreshold(unittest.TestCase):
    def setUp(self):
        # Create a mock MarqoIndex
        self.mock_marqo_index = MagicMock(spec=StructuredMarqoIndex)
        self.mock_marqo_index.name = "test_index"
        self.mock_marqo_index.schema_name = "test_schema"

        # Create the StructuredVespaIndex instance
        self.vespa_index = StructuredVespaIndex(self.mock_marqo_index)
        
        # Mock the vespa_client
        self.vespa_index._vespa_client = MagicMock()

    def test_tensor_search_passes_approximate_threshold(self):
        """Test that _tensor_search includes the approximate_threshold in the query parameters"""
        # Create a tensor query with approximate_threshold
        query = MarqoTensorQuery(
            index_name="test_index",
            vector_query=[0.1, 0.2, 0.3],
            limit=10,
            approximate=True,
            approximate_threshold=0.6
        )
        
        # Mock the response from Vespa
        mock_response = MagicMock()
        mock_response.hits = []
        self.vespa_index._vespa_client.query.return_value = mock_response
        
        # Call _tensor_search
        self.vespa_index._tensor_search(query)
        
        # Check that the vespa_client.query was called with approximate_threshold
        self.vespa_index._vespa_client.query.assert_called_once()
        call_args = self.vespa_index._vespa_client.query.call_args[0][0]
        
        # Verify approximate_threshold parameter was passed
        self.assertEqual(call_args.get('ranking.matching.approximateThreshold'), 0.6)

    def test_tensor_search_without_approximate_threshold(self):
        """Test that _tensor_search works correctly without approximate_threshold"""
        # Create a tensor query without approximate_threshold
        query = MarqoTensorQuery(
            index_name="test_index",
            vector_query=[0.1, 0.2, 0.3],
            limit=10,
            approximate=True
        )
        
        # Mock the response from Vespa
        mock_response = MagicMock()
        mock_response.hits = []
        self.vespa_index._vespa_client.query.return_value = mock_response
        
        # Call _tensor_search
        self.vespa_index._tensor_search(query)
        
        # Check that the vespa_client.query was called
        self.vespa_index._vespa_client.query.assert_called_once()
        call_args = self.vespa_index._vespa_client.query.call_args[0][0]
        
        # Verify that ranking.matching.approximateThreshold is None
        self.assertIsNone(call_args.get('ranking.matching.approximateThreshold'))

    def test_multiple_field_search_with_approximate_threshold(self):
        """Test that multiple field search correctly passes approximate_threshold"""
        # Create a tensor query with approximate_threshold
        query = MarqoTensorQuery(
            index_name="test_index",
            vector_query=[0.1, 0.2, 0.3],
            limit=10,
            approximate=True,
            approximate_threshold=0.75
        )
        
        # Setup for multiple fields
        self.mock_marqo_index.tensor_fields = ['field1', 'field2']
        
        # Mock the response from Vespa
        mock_response = MagicMock()
        mock_response.hits = []
        self.vespa_index._vespa_client.query.return_value = mock_response
        
        # Call _multiple_field_search
        with patch.object(self.vespa_index, '_tensor_search') as mock_tensor_search:
            mock_tensor_search.return_value = []
            self.vespa_index._multiple_field_search(query, ['field1', 'field2'])
            
            # Verify that _tensor_search was called with the approximate_threshold
            mock_tensor_search.assert_called()
            call_args = mock_tensor_search.call_args[0][0]
            self.assertEqual(call_args.approximate_threshold, 0.75) 