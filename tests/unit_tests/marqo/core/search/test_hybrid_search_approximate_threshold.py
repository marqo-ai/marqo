import unittest
from unittest.mock import patch, MagicMock

from marqo.core.search.hybrid_search import HybridSearch
from marqo.core.models.marqo_index import StructuredMarqoIndex
from marqo.config import Config
from semver import VersionInfo


class TestHybridSearchApproximateThreshold(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_config = MagicMock(spec=Config)
        self.mock_marqo_index = MagicMock(spec=StructuredMarqoIndex)
        self.mock_marqo_index.schema_name = "test_schema"
        self.mock_marqo_index.name = "test_index"
        
        # Mock the model attribute
        self.mock_model = MagicMock()
        self.mock_model.get_text_query_prefix.return_value = None
        self.mock_marqo_index.model = self.mock_model
        
        # Mock the version method to return a version that supports 
        # hybrid search
        self.mock_marqo_index.parsed_marqo_version.return_value = (
            VersionInfo.parse("2.12.0")
        )
        
        # Create the HybridSearch instance
        self.hybrid_search = HybridSearch()
        
    def test_approximate_threshold_parameter_exists(self):
        """Test that HybridSearch.search method accepts 
        approximate_threshold parameter"""
        # This test verifies that the method signature includes 
        # approximate_threshold
        import inspect
        
        # Get the search method signature
        search_method = getattr(self.hybrid_search, 'search')
        sig = inspect.signature(search_method)
        
        # Check that approximate_threshold parameter exists
        self.assertIn('approximate_threshold', sig.parameters)
        
        # Check that it has the correct default value
        param = sig.parameters['approximate_threshold']
        self.assertIsNone(param.default)
        
        # Check that it's properly typed as Optional[float] or 
        # Union[float, NoneType]
        annotation_str = str(param.annotation)
        self.assertIn('float', annotation_str)
        expected_types = [
            'typing.Optional[float]', 
            'typing.Union[float, NoneType]'
        ]
        self.assertTrue(
            annotation_str in expected_types,
            f"Expected Optional[float] or Union[float, NoneType], "
            f"got {annotation_str}"
        )
    
    @patch('marqo.core.search.hybrid_search.MarqoHybridQuery')
    def test_approximate_threshold_passed_to_marqo_query(
        self, mock_marqo_hybrid_query
    ):
        """Test that approximate_threshold is passed to MarqoHybridQuery 
        constructor"""
        # Mock vespa_client on config
        mock_vespa_client = MagicMock()
        self.mock_config.vespa_client = mock_vespa_client
        
        # Mock all the complex dependencies to avoid running full search logic
        with patch(
            'marqo.core.search.hybrid_search.RequestMetricsStore.for_request'
        ) as mock_metrics, \
        patch(
            'marqo.core.search.hybrid_search.run_vectorise_pipeline'
        ) as mock_vectorise, \
        patch(
            'marqo.core.search.hybrid_search.utils.parse_lexical_query'
        ) as mock_parse, \
        patch(
            'marqo.core.search.hybrid_search.vespa_index_factory'
        ) as mock_factory, \
        patch(
            'marqo.core.search.hybrid_search.gather_documents_from_response'
        ) as mock_gather:
            
            # Setup mocks
            mock_metrics_instance = MagicMock()
            mock_metrics.return_value = mock_metrics_instance
            mock_metrics_instance.time.return_value.__enter__ = MagicMock()
            mock_metrics_instance.time.return_value.__exit__ = MagicMock()
            mock_metrics_instance.start.return_value = None
            mock_metrics_instance.stop.return_value = 0.0
            
            mock_vectorise.return_value = {0: [0.1, 0.2, 0.3]}
            mock_parse.return_value = ([], ["test"])
            
            mock_vespa_index = MagicMock()
            mock_factory.return_value = mock_vespa_index
            mock_vespa_index.to_vespa_query.return_value = {}
            
            mock_vespa_response = MagicMock()
            mock_vespa_response.root.coverage.coverage = 100
            mock_vespa_response.root.coverage.degraded = None
            mock_vespa_client.query.return_value = mock_vespa_response
            
            mock_gather.return_value = {"hits": []}
            
            # Call search with approximate_threshold
            try:
                self.hybrid_search.search(
                    config=self.mock_config,
                    marqo_index=self.mock_marqo_index,
                    query="test query",
                    approximate=True,
                    approximate_threshold=0.75
                )
            except Exception:
                # We expect this to fail due to complex dependencies,
                # but we want to check that MarqoHybridQuery was called 
                # correctly
                pass
            
            # Check that MarqoHybridQuery was called with approximate_threshold
            mock_marqo_hybrid_query.assert_called_once()
            call_kwargs = mock_marqo_hybrid_query.call_args[1]
            self.assertEqual(call_kwargs.get('approximate_threshold'), 0.75)


if __name__ == "__main__":
    unittest.main() 