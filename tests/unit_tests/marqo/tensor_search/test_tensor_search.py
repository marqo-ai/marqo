import unittest
from unittest.mock import Mock, patch

from marqo import version
from marqo.api import exceptions as api_exceptions
from marqo.config import Config
from marqo.core.models.marqo_index import MarqoIndex, IndexType, Model, StructuredMarqoIndex
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.search import VectorisedJobPointer, JHash
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

        # Verify vespa_client.query was called with correct parameters
        self.config.vespa_client.query.assert_called_once_with(query="test")
        
        # Verify search results
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

        # Verify vespa_client.query was called with correct parameters
        self.config.vespa_client.query.assert_called_once_with(query="vector_query")
        
        # Verify search results
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


class TestTensorSearchFunctions(unittest.TestCase):
    """Test cases for tensor_search.py functions to cover specific functionality"""

    def setUp(self):
        self.mock_config = Mock()
        self.mock_index = Mock()
        self.mock_index.type = IndexType.Structured

    def test_search_lexical_ef_search_parameter_fails(self):
        """Test that search function raises InvalidArgError when efSearch parameter is used with lexical search method."""
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            tensor_search.search(
                config=self.mock_config,
                index_name="test_index",
                text="test query",
                search_method=SearchMethod.LEXICAL,
                ef_search=100  # Invalid for lexical search
            )
        self.assertIn("efSearch", str(cm.exception))

    def test_search_lexical_approximate_parameter_fails(self):
        """Test that search function raises InvalidArgError when approximate parameter is used with lexical search method."""
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            tensor_search.search(
                config=self.mock_config,
                index_name="test_index",
                text="test query",
                search_method=SearchMethod.LEXICAL,
                approximate=False  # Invalid for lexical search
            )
        self.assertIn("approximate", str(cm.exception))

    def test_construct_vector_input_batches_with_none_query(self):
        """Test that construct_vector_input_batches returns empty QueryContentCollector when given None query."""
        result = tensor_search.construct_vector_input_batches(None)

        # Should return empty query collector
        self.assertEqual(len(result.queries), 0)

    def test_construct_vector_input_batches_with_invalid_query_type(self):
        """Test that construct_vector_input_batches raises ValueError when given an invalid query type like integer."""
        with self.assertRaises(ValueError) as cm:
            tensor_search.construct_vector_input_batches(123)  # Invalid type
        self.assertIn("Incorrect type for query", str(cm.exception))

    @patch('marqo.core.vespa_index.vespa_index.for_marqo_index')
    def test_gather_documents_from_response_with_group_facet_id(self, mock_factory):
        """Test that gather_documents_from_response skips group:facet: IDs and returns empty hits list."""
        mock_index = Mock(spec=StructuredMarqoIndex)
        mock_index.type = IndexType.Structured

        # Mock QueryResult with a group:facet: document
        mock_response = Mock()
        mock_doc = Mock()
        mock_doc.id = "group:facet:test"
        mock_response.hits = [mock_doc]

        mock_vespa_index = Mock()
        mock_factory.return_value = mock_vespa_index

        result = tensor_search.gather_documents_from_response(
            response=mock_response,
            marqo_index=mock_index,
            highlights=False
        )

        # Should skip the group:facet: document
        self.assertEqual(len(result['hits']), 0)
        # Verify to_marqo_document was never called since we skip group:facet docs
        mock_vespa_index.to_marqo_document.assert_not_called()

    def test_select_attributes_with_flattened_map_fields(self):
        """Test that select_attributes includes flattened map fields when the prefix matches an attribute to retrieve."""
        marqo_doc = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "Test Document",
            "metadata.author": "John Doe",  # Flattened map field
            "metadata.year": "2023",  # Flattened map field
            "description": "Test Description",
            "extra_field": "Should be filtered out"
        }

        attributes_to_retrieve_set = {"_id", "_score", "title", "metadata"}

        result = tensor_search.select_attributes(marqo_doc, attributes_to_retrieve_set)

        # Should include flattened map fields with "metadata" prefix
        expected = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "Test Document",
            "metadata.author": "John Doe",
            "metadata.year": "2023"
        }
        self.assertEqual(result, expected)

    def test_select_attributes_without_flattened_fields(self):
        """Test that select_attributes returns only exact attribute matches when no flattened fields are present."""
        marqo_doc = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "Test Document",
            "description": "Test Description",
            "extra_field": "Should be filtered out"
        }

        attributes_to_retrieve_set = {"_id", "_score", "title"}

        result = tensor_search.select_attributes(marqo_doc, attributes_to_retrieve_set)

        # Should only include exact matches
        expected = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "Test Document"
        }
        self.assertEqual(result, expected)

    def test_get_content_vector_not_found_error(self):
        """Test that get_content_vector raises RuntimeError when content is not found in any job."""
        possible_jobs = []  # Empty list
        job_to_vectors = {}
        content = "test_content"

        with self.assertRaises(RuntimeError) as cm:
            tensor_search.get_content_vector(possible_jobs, job_to_vectors, content)
        self.assertIn("could not find corresponding vector for content", str(cm.exception))

    def test_get_content_vector_found_in_job(self):
        """Test that get_content_vector returns the correct vector when content is found in a job."""

        job_hash = JHash(123)
        possible_jobs = [VectorisedJobPointer(job_hash=job_hash, start_idx=0, end_idx=1)]
        job_to_vectors = {job_hash: {"test_content": [0.1, 0.2, 0.3]}}
        content = "test_content"

        result = tensor_search.get_content_vector(possible_jobs, job_to_vectors, content)

        self.assertEqual(result, [0.1, 0.2, 0.3])


if __name__ == '__main__':
    unittest.main()
