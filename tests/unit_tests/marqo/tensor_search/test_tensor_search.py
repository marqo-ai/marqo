import unittest
from unittest.mock import Mock, patch

from marqo import version

from marqo.config import Config
from marqo.core.models.marqo_index import MarqoIndex, IndexType, Model
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import CustomVectorQuery
from marqo.vespa.models import QueryResult
from marqo.vespa.models.query_result import Child, Root, Coverage


class TestTensorSearch(unittest.TestCase):
    """Test basic search functionality for lexical, tensor, and hybrid search methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.config.vespa_client = Mock()
        self.config.index_management = Mock()
        self.config.inference = Mock()

        # Mock index
        self.mock_index = Mock(spec=MarqoIndex)
        self.mock_index.name = "test-index"
        self.mock_index.type = IndexType.SemiStructured
        self.mock_index.schema_name = "test_schema"
        self.mock_index.normalize_embeddings = True
        self.mock_index.model = Mock(spec=Model)
        self.mock_index.model.name = "test-model"
        self.mock_index.model.get_text_query_prefix.return_value = ""
        self.mock_index.model.get_dimension.return_value = 512
        self.mock_index.model.get_properties.return_value = {}
        self.mock_index.parsed_marqo_version.return_value = version.__version__

        # Setup mock Vespa response
        self.mock_hit = Child(
            id="doc1",
            relevance=0.95,
            fields={"field1": "value1", "field2": "value2"}
        )

        # Setup proper QueryResult mock
        mock_coverage = Coverage(coverage=100, degraded=None, documents=1, full=True, nodes=1, results=1, resultsFull=1)
        mock_root = Root(relevance=0, coverage=mock_coverage)
        mock_root.children = [self.mock_hit]

        self.mock_query_result = QueryResult(root=mock_root)

    def _setup_common_mocks(self, mock_metrics, mock_vespa_factory, mock_get_index):
        """Helper method to set up common mocks used across multiple tests."""
        # Setup index mock
        mock_get_index.return_value = self.mock_index
        
        # Setup vespa index mock
        mock_vespa_index = Mock()
        mock_vespa_index.to_vespa_query.return_value = {"query": "test_query"}
        mock_vespa_index.to_marqo_document.return_value = {
            "_id": "doc1",
            "field1": "value1",
            "field2": "value2"
        }
        mock_vespa_factory.return_value = mock_vespa_index
        
        # Setup metrics mock
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.start.return_value = None
        mock_metrics_instance.stop.return_value = 100.0
        mock_metrics_instance.time.return_value.__enter__ = Mock(return_value=None)
        mock_metrics_instance.time.return_value.__exit__ = Mock(return_value=None)
        
        # Setup Vespa response
        self.config.vespa_client.query.return_value = self.mock_query_result
        
        return mock_vespa_index

    def _setup_lexical_mocks(self, mock_parse_query):
        """Helper method to set up lexical search specific mocks."""
        mock_parse_query.return_value = (["test"], ["query"])

    def _setup_tensor_mocks(self, mock_vectorise):
        """Helper method to set up tensor search specific mocks."""
        mock_vectorise.return_value = {0: [0.1, 0.2, 0.3]}

    def _assert_basic_search_response(self, result, expected_query, expected_limit=10, expected_offset=0):
        """Helper method to assert basic search response structure."""
        self.assertEqual(result['query'], expected_query)
        self.assertEqual(result['limit'], expected_limit)
        self.assertEqual(result['offset'], expected_offset)
        self.assertIn('hits', result)
        self.assertIn('processingTimeMs', result)
        self.assertEqual(len(result['hits']), 1)
        self.assertEqual(result['hits'][0]['_id'], 'doc1')
        self.assertEqual(result['hits'][0]['_score'], 0.95)

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    @patch('marqo.tensor_search.tensor_search.utils.parse_lexical_query')
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    def test_search_lexical_method(self, mock_metrics, mock_parse_query, mock_vespa_factory, mock_get_index):
        """Test search with lexical method returns expected results."""
        # Setup
        mock_vespa_index = self._setup_common_mocks(mock_metrics, mock_vespa_factory, mock_get_index)
        self._setup_lexical_mocks(mock_parse_query)
        mock_vespa_index.to_vespa_query.return_value = {"query": "test"}

        # Execute
        result = tensor_search.search(
            config=self.config,
            index_name="test-index",
            text="test query",
            result_count=10,
            search_method=SearchMethod.LEXICAL
        )

        # Verify vespa_client.query was called with correct parameters
        self.config.vespa_client.query.assert_called_once_with(query="test")
        
        # Verify search results
        self._assert_basic_search_response(result, 'test query')

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    @patch('marqo.tensor_search.tensor_search.run_vectorise_pipeline')
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    def test_search_tensor_method(self, mock_metrics, mock_vectorise, mock_vespa_factory, mock_get_index):
        """Test search with tensor method returns expected results."""
        # Setup
        mock_vespa_index = self._setup_common_mocks(mock_metrics, mock_vespa_factory, mock_get_index)
        self._setup_tensor_mocks(mock_vectorise)
        mock_vespa_index.to_vespa_query.return_value = {"query": "vector_query"}

        # Execute
        result = tensor_search.search(
            config=self.config,
            index_name="test-index",
            text="test query",
            result_count=10,
            search_method=SearchMethod.TENSOR
        )

        # Verify vespa_client.query was called with correct parameters
        self.config.vespa_client.query.assert_called_once_with(query="vector_query")
        
        # Verify search results
        self._assert_basic_search_response(result, 'test query')

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    @patch('marqo.core.search.hybrid_search.HybridSearch')
    def test_search_hybrid_method(self, mock_hybrid_search_class, mock_get_index):
        """Test search with hybrid method returns expected results."""
        # Setup
        mock_get_index.return_value = self.mock_index

        # Mock hybrid search instance
        mock_hybrid_instance = Mock()
        mock_hybrid_search_class.return_value = mock_hybrid_instance

        # Mock the search result
        mock_hybrid_instance.search.return_value = {
            'hits': [{'_id': 'doc1', '_score': 0.95, 'field1': 'value1'}]
        }

        # Execute
        result = tensor_search.search(
            config=self.config,
            index_name="test-index",
            text="test query",
            result_count=10,
            search_method=SearchMethod.HYBRID
        )

        # Verify
        mock_hybrid_instance.search.assert_called_once()
        self.assertEqual(result['query'], 'test query')
        self.assertEqual(result['limit'], 10)
        self.assertEqual(result['offset'], 0)
        self.assertIn('hits', result)
        self.assertIn('processingTimeMs', result)

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    @patch('marqo.tensor_search.tensor_search.run_vectorise_pipeline')
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    def test_search_with_base64_query_omitted_in_response(self, mock_metrics, mock_vectorise, mock_vespa_factory, mock_get_index):
        """Test that search with base64 content in query returns sanitized query in response."""
        # Setup
        mock_vespa_index = self._setup_common_mocks(mock_metrics, mock_vespa_factory, mock_get_index)
        self._setup_tensor_mocks(mock_vectorise)
        mock_vespa_index.to_vespa_query.return_value = {"query": "vector_query"}

        test_cases = [
            {
                "name": "base64_image_string",
                "query": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ...",
                "expected": "data:image/[omitted]"
            },
            {
                "name": "dict_with_base64_key",
                "query": {
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34...": 0.8,
                    "regular_field": 0.2
                },
                "expected": {
                    "data:image/[omitted]": 0.8,
                    "regular_field": 0.2
                }
            }
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                # Execute search with test case query
                result = tensor_search.search(
                    config=self.config,
                    index_name="test-index",
                    text=case["query"],
                    result_count=10,
                    search_method=SearchMethod.TENSOR
                )

                # Verify search results including sanitized query
                self._assert_basic_search_response(result, case["expected"])


if __name__ == '__main__':
    unittest.main()
