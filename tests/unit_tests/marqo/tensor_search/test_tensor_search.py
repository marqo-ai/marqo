import unittest
from unittest.mock import Mock, patch

from marqo import version
from marqo.api import exceptions as api_exceptions
from marqo.config import Config
from marqo.core.models.marqo_index import MarqoIndex, IndexType, Model, StructuredMarqoIndex, SemiStructuredMarqoIndex
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.search import VectorisedJobPointer, JHash
from marqo.vespa.models import QueryResult
from marqo.vespa.models.query_result import Child, Root, Coverage
from marqo.core import exceptions as core_exceptions
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor


class TestTensorSearch(unittest.TestCase):
    """Test core search functionality and utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.config.vespa_client = Mock()
        self.config.index_management = Mock()
        self.config.inference = Mock()

        # Mock index - use proper SemiStructuredMarqoIndex
        self.mock_index = Mock(spec=SemiStructuredMarqoIndex)
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

    def test_construct_vector_input_batches_with_none_query(self):
        """Test construct_vector_input_batches with None query returns empty collector"""
        result = tensor_search.construct_vector_input_batches(None)
        self.assertEqual(len(result.queries), 0)

    def test_construct_vector_input_batches_with_invalid_query_type(self):
        """Test construct_vector_input_batches with invalid query type raises ValueError"""
        with self.assertRaises(ValueError):
            tensor_search.construct_vector_input_batches(123)

    def test_select_attributes_with_flattened_map_fields(self):
        """Test select_attributes includes flattened map fields with matching prefixes"""
        marqo_doc = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "test title",
            "metadata.category": "science",
            "metadata.tags": ["tag1", "tag2"],
            "other_field": "value",
            "unrelated.field": "unrelated"
        }
        
        attributes_to_retrieve = {"_id", "_score", "title", "metadata"}
        
        result = tensor_search.select_attributes(marqo_doc, attributes_to_retrieve)
        
        expected = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "test title",
            "metadata.category": "science",
            "metadata.tags": ["tag1", "tag2"]
        }
        
        self.assertEqual(result, expected)

    def test_select_attributes_without_flattened_fields(self):
        """Test select_attributes with only exact field matches"""
        marqo_doc = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "test title",
            "description": "test description",
            "other_field": "value"
        }
        
        attributes_to_retrieve = {"_id", "_score", "title"}
        
        result = tensor_search.select_attributes(marqo_doc, attributes_to_retrieve)
        
        expected = {
            "_id": "doc1",
            "_score": 0.95,
            "title": "test title"
        }
        
        self.assertEqual(result, expected)

    def test_get_content_vector_not_found_error(self):
        """Test get_content_vector raises RuntimeError when content not found"""
        possible_jobs = []
        job_to_vectors = {}
        content = "test content"
        
        with self.assertRaises(RuntimeError) as cm:
            tensor_search.get_content_vector(possible_jobs, job_to_vectors, content)
        
        self.assertIn("could not find corresponding vector for content", str(cm.exception))

    def test_get_content_vector_found_in_job(self):
        """Test get_content_vector successfully finds and returns vector"""
        # Use proper JHash type for job_hash (should be an integer)
        job_hash = JHash(123)
        job_pointer = VectorisedJobPointer(job_hash=job_hash, start_idx=0, end_idx=1)
        possible_jobs = [job_pointer]
        job_to_vectors = {job_hash: {"test content": [0.1, 0.2, 0.3]}}
        content = "test content"
        
        result = tensor_search.get_content_vector(possible_jobs, job_to_vectors, content)
        
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_get_query_vectors_string_query_with_context_fails(self):
        """Test that using context with a string query raises InvalidArgumentError"""
        # Create a proper MarqoIndex mock that will pass pydantic validation
        mock_index = Mock(spec=SemiStructuredMarqoIndex)
        mock_index.model = Mock()
        mock_index.model.get_dimension.return_value = 512
        mock_index.name = "test-index"
        mock_index.type = IndexType.SemiStructured
        # Add the dict() method that pydantic expects
        mock_index.dict.return_value = {
            'name': 'test-index',
            'type': 'semi_structured'
        }
        
        # Create a query with string q and context (this should fail)
        context = SearchContext(tensor=[SearchContextTensor(vector=[0.1, 0.2, 0.3], weight=1.0)])
        query = BulkSearchQueryEntity(
            q="test string query",  # String query
            context=context,        # With context - this combination should fail
            index=mock_index,
            searchMethod=SearchMethod.TENSOR,
            limit=10,
            offset=0,
            showHighlights=False
        )
        
        # Mock the required parameters for get_query_vectors_from_jobs
        queries = [query]
        qidx_to_job = {0: []}  # Empty job pointers for string query
        job_to_vectors = {}    # Empty job vectors
        jobs = {}              # Empty jobs
        
        with self.assertRaises(core_exceptions.InvalidArgumentError) as cm:
            tensor_search.get_query_vectors_from_jobs(
                queries, qidx_to_job, job_to_vectors, Mock(), jobs
            )
        
        # Verify the specific error message from line 966
        self.assertIn("Cannot use 'context' for a search with a string 'q'", str(cm.exception))
        self.assertIn("test string query", str(cm.exception))
        self.assertIn("provide a dictionary or a CustomVectorQuery object", str(cm.exception))


class TestTensorSearchValidation(unittest.TestCase):
    """Test validation and error handling for tensor search operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.config.index_management = Mock()
        self.mock_index = Mock(spec=SemiStructuredMarqoIndex)
        self.mock_index.type = IndexType.SemiStructured
        self.mock_index.parsed_marqo_version.return_value = version.__version__

    def test_search_result_count_validation_negative_fails(self):
        """Test that negative result_count raises IllegalRequestedDocCount"""
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=-1
            )
        self.assertIn("result_count must be an integer greater than 0", str(cm.exception))

    def test_search_result_count_validation_zero_fails(self):
        """Test that zero result_count raises IllegalRequestedDocCount"""
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=0
            )
        self.assertIn("result_count must be an integer greater than 0", str(cm.exception))

    def test_search_result_count_validation_non_integer_fails(self):
        """Test that non-integer result_count raises IllegalRequestedDocCount"""
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=5.5
            )
        self.assertIn("result_count must be an integer greater than 0", str(cm.exception))

    def test_search_offset_validation_negative_fails(self):
        """Test that negative offset raises IllegalRequestedDocCount"""
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=5,
                offset=-1
            )
        self.assertIn("search result offset cannot be less than 0", str(cm.exception))

    @patch.dict('os.environ', {'MARQO_MAX_RETRIEVABLE_DOCS': '10'})
    def test_search_max_docs_limit_validation_fails(self):
        """Test that exceeding max retrievable docs raises IllegalRequestedDocCount"""
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=8,
                offset=5  # 8 + 5 = 13 > 10
            )
        self.assertIn("search result limit + offset must be less than or equal", str(cm.exception))

    @patch.dict('os.environ', {'MARQO_MAX_SEARCH_LIMIT': '5'})
    def test_search_max_search_limit_validation_fails(self):
        """Test that exceeding max search limit raises IllegalRequestedDocCount"""
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=10  # > 5
            )
        self.assertIn("search result limit must be less than or equal to the MARQO_MAX_SEARCH_LIMIT", str(cm.exception))

    @patch.dict('os.environ', {'MARQO_MAX_SEARCH_OFFSET': '100'})
    def test_search_max_search_offset_validation_fails(self):
        """Test that exceeding max search offset raises IllegalRequestedDocCount"""
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=5,
                offset=150  # > 100
            )
        self.assertIn("search result offset must be less than or equal to the MARQO_MAX_SEARCH_OFFSET", str(cm.exception))

    @patch.dict('os.environ', {'MARQO_MAX_SEARCH_CONTEXT_DOCS': '2'})
    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    def test_search_max_context_docs_validation_fails(self, mock_get_index):
        """Test that exceeding max context docs raises IllegalRequestedDocCount"""
        mock_get_index.return_value = self.mock_index
        
        from marqo.tensor_search.models.search import SearchContext, SearchContextDocuments
        context = SearchContext(documents=SearchContextDocuments(ids={"doc1": 1, "doc2": 1, "doc3": 1}))
        
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            # Use a dict query to avoid the validation error for string + context
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text={"query": 1.0},  # Use dict query to pass validation
                result_count=5,
                search_method=SearchMethod.TENSOR,
                context=context
            )
        # This should fail on context docs limit, not on string query validation
        self.assertIn("Search context documents limit exceeded", str(cm.exception))

    def test_search_invalid_search_method_fails(self):
        """Test that invalid search method raises InvalidArgError"""
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=5,
                search_method="INVALID_METHOD"
            )
        self.assertIn("Search called with unknown search method", str(cm.exception))

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    def test_lexical_search_ef_search_invalid_arg_error(self, mock_get_index):
        """Test that ef_search parameter with lexical search raises InvalidArgError"""
        mock_get_index.return_value = self.mock_index
        
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            tensor_search.search(
                config=Mock(),
                index_name="test",
                text="query",
                result_count=5,
                search_method=SearchMethod.LEXICAL,
                ef_search=100
            )
        self.assertIn("efSearch is not a valid argument for lexical search", str(cm.exception))

    @patch('marqo.tensor_search.tensor_search.index_meta_cache.get_index')
    def test_lexical_search_approximate_invalid_arg_error(self, mock_get_index):
        """Test that approximate parameter with lexical search raises InvalidArgError"""
        mock_get_index.return_value = self.mock_index

        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            tensor_search.search(
                config=self.config,
                index_name="test-index",
                text="test query",
                search_method=SearchMethod.LEXICAL,
                approximate=True
            )
        
        self.assertIn("approximate is not a valid argument for lexical search", str(cm.exception))

    def test_get_documents_by_ids_empty_collection_fails(self):
        """Test that empty document_ids collection raises InvalidArgError"""
        with self.assertRaises(api_exceptions.InvalidArgError) as cm:
            tensor_search.get_documents_by_ids(
                config=Mock(),
                index_name="test",
                document_ids=[]
            )
        self.assertIn("Can't get empty collection of IDs", str(cm.exception))

    @patch.dict('os.environ', {'MARQO_MAX_RETRIEVABLE_DOCS': '2'})
    def test_get_documents_by_ids_max_docs_limit_fails(self):
        """Test that exceeding max docs limit raises IllegalRequestedDocCount"""
        with self.assertRaises(api_exceptions.IllegalRequestedDocCount) as cm:
            tensor_search.get_documents_by_ids(
                config=Mock(),
                index_name="test",
                document_ids=["doc1", "doc2", "doc3"]
            )
        self.assertIn("documents were requested, which is more than the allowed limit", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
