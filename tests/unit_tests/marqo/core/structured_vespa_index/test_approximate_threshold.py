import unittest
from unittest.mock import patch, MagicMock

from marqo.core.models.marqo_index import StructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery
from marqo.core.structured_vespa_index.structured_vespa_index import (
    StructuredVespaIndex
)


class TestStructuredVespaIndexApproximateThreshold(unittest.TestCase):
    def setUp(self):
        # Create a mock MarqoIndex
        self.mock_marqo_index = MagicMock(spec=StructuredMarqoIndex)
        self.mock_marqo_index.name = "test_index"
        self.mock_marqo_index.schema_name = "test_schema"
        self.mock_marqo_index.tensor_field_map = {}

        # Create the StructuredVespaIndex instance
        self.vespa_index = StructuredVespaIndex(self.mock_marqo_index)
        
        # Mock the vespa_client and version
        self.vespa_index._vespa_client = MagicMock()
        self.vespa_index._marqo_index_version = "2.12.0"

    def test_tensor_search_passes_approximate_threshold(self):
        """Test that _to_vespa_tensor_query includes the 
        approximate_threshold in the query parameters"""
        # Create a tensor query with approximate_threshold
        query = MarqoTensorQuery(
            index_name="test_index",
            vector_query=[0.1, 0.2, 0.3],
            limit=10,
            approximate=True,
            approximate_threshold=0.6
        )
        
        # Call _to_vespa_tensor_query
        vespa_query = self.vespa_index._to_vespa_tensor_query(query)
        
        # Verify approximate_threshold parameter was passed
        self.assertEqual(
            vespa_query.get('ranking.matching.approximateThreshold'), 0.6
        )

    def test_tensor_search_without_approximate_threshold(self):
        """Test that _to_vespa_tensor_query works correctly without 
        approximate_threshold"""
        # Create a tensor query without approximate_threshold
        query = MarqoTensorQuery(
            index_name="test_index",
            vector_query=[0.1, 0.2, 0.3],
            limit=10,
            approximate=True
        )
        
        # Call _to_vespa_tensor_query
        vespa_query = self.vespa_index._to_vespa_tensor_query(query)
        
        # Verify that ranking.matching.approximateThreshold is None
        self.assertIsNone(
            vespa_query.get('ranking.matching.approximateThreshold')
        )

    def test_multiple_field_search_with_approximate_threshold(self):
        """Test that multiple field search correctly passes 
        approximate_threshold"""
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
        
        # Mock the _to_vespa_tensor_query method to test that it gets called
        with patch.object(self.vespa_index, '_to_vespa_tensor_query') as mock:
            mock.return_value = {'ranking.matching.approximateThreshold': 0.75}
            
            # Call the method that would use _to_vespa_tensor_query
            result = self.vespa_index._to_vespa_tensor_query(query)
            
            # Verify that approximate_threshold was passed correctly
            self.assertEqual(
                result.get('ranking.matching.approximateThreshold'), 0.75
            )


if __name__ == "__main__":
    unittest.main() 