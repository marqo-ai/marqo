import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from marqo.config import Config
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.core.models.marqo_index import MarqoIndex, IndexType, Model
from marqo.vespa.models import QueryResult
from marqo.vespa.models.query_result import Child, Root, Coverage


class MockChild(dict):
    """Mock Child object that behaves like a dict"""

    def __init__(self, id, relevance, fields):
        super().__init__()
        self.id = id
        self.relevance = relevance
        self.fields = fields
        self.update({"id": id, "relevance": relevance, "fields": fields})


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
        self.mock_index.parsed_marqo_version.return_value = "2.17.0"

        # Setup mock Vespa response
        self.mock_hit = MockChild(
            id="doc1",
            relevance=0.95,
            fields={"field1": "value1", "field2": "value2"}
        )

        # Setup proper QueryResult mock
        mock_root = Mock(spec=Root)
        mock_coverage = Mock(spec=Coverage)
        mock_coverage.coverage = 100
        mock_coverage.degraded = None
        mock_root.coverage = mock_coverage
        mock_root.children = [self.mock_hit]

        self.mock_query_result = Mock(spec=QueryResult)
        self.mock_query_result.root = mock_root
        self.mock_query_result.hits = [self.mock_hit]

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    @patch('marqo.tensor_search.tensor_search.utils.parse_lexical_query')
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    def test_search_lexical_method(self, mock_metrics, mock_parse_query, mock_vespa_factory, mock_get_index):
        """Test search with lexical method returns expected results."""
        # Setup
        mock_get_index.return_value = self.mock_index
        mock_parse_query.return_value = (["test"], ["query"])

        # Mock vespa index
        mock_vespa_index = Mock()
        mock_vespa_index.to_vespa_query.return_value = {"query": "test"}
        mock_vespa_index.to_marqo_document.return_value = {
            "_id": "doc1",
            "field1": "value1",
            "field2": "value2"
        }
        mock_vespa_factory.return_value = mock_vespa_index

        # Mock metrics
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.start.return_value = None
        mock_metrics_instance.stop.return_value = 100.0
        mock_metrics_instance.time.return_value.__enter__ = Mock(return_value=None)
        mock_metrics_instance.time.return_value.__exit__ = Mock(return_value=None)

        # Mock Vespa response
        self.config.vespa_client.query.return_value = self.mock_query_result

        # Execute
        result = tensor_search.search(
            config=self.config,
            index_name="test-index",
            text="test query",
            result_count=10,
            search_method=SearchMethod.LEXICAL
        )

        # Verify
        self.assertEqual(result['query'], 'test query')
        self.assertEqual(result['limit'], 10)
        self.assertEqual(result['offset'], 0)
        self.assertIn('hits', result)
        self.assertIn('processingTimeMs', result)
        self.assertEqual(len(result['hits']), 1)
        self.assertEqual(result['hits'][0]['_id'], 'doc1')
        self.assertEqual(result['hits'][0]['_score'], 0.95)

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    @patch('marqo.tensor_search.tensor_search.vespa_index_factory')
    @patch('marqo.tensor_search.tensor_search.run_vectorise_pipeline')
    @patch('marqo.tensor_search.tensor_search.RequestMetricsStore')
    def test_search_tensor_method(self, mock_metrics, mock_vectorise, mock_vespa_factory, mock_get_index):
        """Test search with tensor method returns expected results."""
        # Setup
        mock_get_index.return_value = self.mock_index
        mock_vectorise.return_value = {0: [0.1, 0.2, 0.3]}

        # Mock vespa index
        mock_vespa_index = Mock()
        mock_vespa_index.to_vespa_query.return_value = {"query": "vector_query"}
        mock_vespa_index.to_marqo_document.return_value = {
            "_id": "doc1",
            "field1": "value1",
            "field2": "value2"
        }
        mock_vespa_factory.return_value = mock_vespa_index

        # Mock metrics
        mock_metrics_instance = Mock()
        mock_metrics.for_request.return_value = mock_metrics_instance
        mock_metrics_instance.start.return_value = None
        mock_metrics_instance.stop.return_value = 100.0
        mock_metrics_instance.time.return_value.__enter__ = Mock(return_value=None)
        mock_metrics_instance.time.return_value.__exit__ = Mock(return_value=None)

        # Mock Vespa response
        self.config.vespa_client.query.return_value = self.mock_query_result

        # Execute
        result = tensor_search.search(
            config=self.config,
            index_name="test-index",
            text="test query",
            result_count=10,
            search_method=SearchMethod.TENSOR
        )

        # Verify
        self.assertEqual(result['query'], 'test query')
        self.assertEqual(result['limit'], 10)
        self.assertEqual(result['offset'], 0)
        self.assertIn('hits', result)
        self.assertIn('processingTimeMs', result)
        self.assertEqual(len(result['hits']), 1)
        self.assertEqual(result['hits'][0]['_id'], 'doc1')
        self.assertEqual(result['hits'][0]['_score'], 0.95)

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


if __name__ == '__main__':
    unittest.main()
