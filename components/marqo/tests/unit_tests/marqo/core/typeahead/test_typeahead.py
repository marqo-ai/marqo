import unittest
from unittest.mock import Mock, patch

import semver

from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.typeahead import (
    TypeaheadRequest, TypeaheadSuggestion,
    TypeaheadAddQueryRequest, TypeaheadIndexingRequest, TypeaheadIndexingError, TypeaheadStatsResponse,
    TypeaheadGetQueriesResponse
)
from marqo.core.typeahead.typeahead import Typeahead
from marqo.tensor_search import api
from marqo.vespa.models.feed_response import FeedBatchResponse, FeedBatchDocumentResponse
from marqo.vespa.models.get_document_response import GetBatchResponse, GetBatchDocumentResponse, Document
from marqo.vespa.models.query_result import QueryResult, Root, Child, RootFields
from marqo.vespa.models.vespa_document import VespaDocument
from marqo.vespa.vespa_client import VespaClient


class TestTypeaheadIndexQueries(unittest.TestCase):
    """Test cases for the index_queries method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_index_management = Mock(spec=IndexManagement)
        # override index_management in api._config so the version check can also use the mock index_management
        api._config.index_management = self.mock_index_management
        self.typeahead = Typeahead(
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management
        )
        
        # Mock index details
        self.mock_marqo_index = Mock()
        self.mock_marqo_index.typeahead_schema_name = "test_typeahead_schema"
        self.mock_marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.0")  # Supported version
        self.mock_index_management.get_index.return_value = self.mock_marqo_index

    def _hash(self, query):
        return self.typeahead._generate_query_hash(query)

    def test_index_queries_valid_queries(self):
        """Test index_queries with valid queries processes successfully."""
        queries = [
            TypeaheadAddQueryRequest(query="machine learning", popularity=10.0, metadata={"hit_count": 5}),
            TypeaheadAddQueryRequest(query="artificial intelligence", popularity=8.0, metadata={"hit_count": 6})
        ]
        request = TypeaheadIndexingRequest(queries=queries)
        
        # Create real Vespa response objects
        feed_responses = [
            FeedBatchDocumentResponse(id=f"schema::{self._hash('machine learning')}", status=200, message="OK"),
            FeedBatchDocumentResponse(id=f"schema::{self._hash('artificial intelligence')}", status=200, message="OK")
        ]
        vespa_response = FeedBatchResponse(responses=feed_responses, errors=False)
        
        self.mock_vespa_client.feed_batch.return_value = vespa_response
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")
        
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.1]):
            with patch('marqo.core.typeahead.typeahead.time.time', return_value=1234567890):
                result = self.typeahead.index_queries("test_index", request)
        
        # Verify response
        self.assertEqual(result.indexed, 2)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.processing_time_ms, 100)
        
        # Verify vespa client was called with correct documents
        self.mock_vespa_client.feed_batch.assert_called_once()
        call_args = self.mock_vespa_client.feed_batch.call_args
        vespa_docs = call_args[0][0]  # First positional argument
        schema_name = call_args[1]['schema']  # Keyword argument
        
        self.assertEqual(len(vespa_docs), 2)
        self.assertEqual(schema_name, "test_typeahead_schema")
        
        # Verify document structure
        doc1 = vespa_docs[0]
        self.assertIsInstance(doc1, VespaDocument)
        self.assertEqual(doc1.fields["query"], "machine learning")
        self.assertEqual(doc1.fields["popularity"], 10.0)
        self.assertEqual(doc1.fields["metadata"], {"hit_count": 5})
        self.assertEqual(doc1.fields["query_words"], ["machine", "learning"])
        self.assertEqual(doc1.fields["last_updated_at"], 1234567890)

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_index_queries_duplicate_normalization(self, mock_normalize):
        """Test index_queries detects duplicates after normalization."""
        queries = [
            TypeaheadAddQueryRequest(query="Machine Learning", popularity=10.0),
            TypeaheadAddQueryRequest(query="machine learning", popularity=8.0)  # Duplicate after normalization
        ]
        request = TypeaheadIndexingRequest(queries=queries)
        
        # Mock normalization to return same result for both
        mock_normalize.return_value = "machine learning"

        # Create real Vespa response for valid query
        feed_responses = [
            FeedBatchDocumentResponse(id=f"schema::{self._hash('machine learning')}", status=200, message="OK")
        ]
        vespa_response = FeedBatchResponse(responses=feed_responses, errors=False)

        self.mock_vespa_client.feed_batch.return_value = vespa_response
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")
        
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.03]):
            result = self.typeahead.index_queries("test_index", request)
        
        # Should index 1 queries and have 1 duplicate error
        self.assertEqual(result.indexed, 1)
        self.assertEqual(len(result.errors), 1)
        
        error = result.errors[0]
        self.assertIsInstance(error, TypeaheadIndexingError)
        self.assertEqual(error.query, "machine learning")
        self.assertIn("duplicate", error.message.lower())
        self.assertEqual(error.code, 400)

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_index_queries_no_tokens_error(self, mock_normalize):
        """Test index_queries handles queries that produce no tokens."""
        queries = [
            TypeaheadAddQueryRequest(query="valid query", popularity=5.0),
            TypeaheadAddQueryRequest(query="!!!", popularity=3.0)  # Will produce no tokens
        ]
        request = TypeaheadIndexingRequest(queries=queries)
        
        def normalize_side_effect(text):
            if text == "valid query":
                return "valid query"
            elif text == "!!!":
                return ""  # Will split to empty list
            return text
        
        mock_normalize.side_effect = normalize_side_effect
        
        # Create real Vespa response for valid query
        feed_responses = [
            FeedBatchDocumentResponse(id=f"schema::{self._hash('valid query')}", status=200, message="OK")
        ]
        vespa_response = FeedBatchResponse(responses=feed_responses, errors=False)
        
        self.mock_vespa_client.feed_batch.return_value = vespa_response
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")
        
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.04]):
            result = self.typeahead.index_queries("test_index", request)
        
        # Should index 1 valid query and have 1 no-tokens error
        self.assertEqual(result.indexed, 1)
        self.assertEqual(len(result.errors), 1)
        
        error = result.errors[0]
        self.assertIsInstance(error, TypeaheadIndexingError)
        self.assertEqual(error.query, "!!!")
        self.assertIn("No tokens generated", error.message)
        self.assertEqual(error.code, 400)

    def test_index_queries_vespa_response_handling(self):
        """Test index_queries properly handles vespa success and error responses."""
        queries = [
            TypeaheadAddQueryRequest(query="successful query", popularity=5.0),
            TypeaheadAddQueryRequest(query="failed query", popularity=3.0)
        ]
        request = TypeaheadIndexingRequest(queries=queries)
        
        # Create real Vespa response with mixed success/failure
        feed_responses = [
            FeedBatchDocumentResponse(id=f"schema::{self._hash('successful query')}", status=200, message="OK"),
            FeedBatchDocumentResponse(id=f"schema::{self._hash('failed query')}", status=500, message="Internal error")
        ]
        vespa_response = FeedBatchResponse(responses=feed_responses, errors=True)
        
        self.mock_vespa_client.feed_batch.return_value = vespa_response
        
        def translate_response(status, message):
            if status == 200:
                return (200, "OK")
            else:
                return (500, "Internal error")
        
        self.mock_vespa_client.translate_vespa_document_response.side_effect = translate_response
        
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.06]):
            result = self.typeahead.index_queries("test_index", request)
        
        # Should index 1 successful and have 1 error
        self.assertEqual(result.indexed, 1)
        self.assertEqual(len(result.errors), 1)
        
        error = result.errors[0]
        self.assertIsInstance(error, TypeaheadIndexingError)
        self.assertEqual(error.query, "failed query")
        self.assertEqual(error.message, "Internal error")
        self.assertEqual(error.code, 500)

    @patch('marqo.core.typeahead.typeahead.blake3.blake3')
    def test_index_queries_hash_generation(self, mock_blake3):
        """Test index_queries generates consistent document IDs."""
        queries = [TypeaheadAddQueryRequest(query="test query", popularity=1.0)]
        request = TypeaheadIndexingRequest(queries=queries)
        
        # Mock hash generation
        mock_hasher = Mock()
        mock_hasher.digest.return_value = Mock()
        mock_hasher.digest.return_value.hex.return_value = "mockedhash123"
        mock_blake3.return_value = mock_hasher
        
        # Create real Vespa response
        feed_responses = [
            FeedBatchDocumentResponse(id="schema::mockedhash123", status=200, message="OK")
        ]
        vespa_response = FeedBatchResponse(responses=feed_responses, errors=False)
        
        self.mock_vespa_client.feed_batch.return_value = vespa_response
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")
        
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.02]):
            result = self.typeahead.index_queries("test_index", request)
        
        # Verify hash was generated with normalized query
        mock_blake3.assert_called_once_with(b"test query")  # normalized text
        
        # Verify document ID was set correctly
        call_args = self.mock_vespa_client.feed_batch.call_args[0][0]
        doc = call_args[0]
        self.assertEqual(doc.id, "mockedhash123")

    def test_index_queries_processing_time_calculation(self):
        """Test index_queries calculates processing time correctly."""
        queries = [TypeaheadAddQueryRequest(query="test", popularity=1.0)]
        request = TypeaheadIndexingRequest(queries=queries)
        
        # Create minimal Vespa response
        feed_responses = [
            FeedBatchDocumentResponse(id=f"schema::{self._hash('test')}", status=200, message="OK")
        ]
        vespa_response = FeedBatchResponse(responses=feed_responses, errors=False)
        self.mock_vespa_client.feed_batch.return_value = vespa_response
        self.mock_vespa_client.translate_vespa_document_response.return_value = (200, "OK")
        
        # Mock timer to return specific values
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[1.5, 1.7]):  # 0.2 seconds
            result = self.typeahead.index_queries("test_index", request)
        
        # Should convert to milliseconds and round
        self.assertEqual(result.processing_time_ms, 200)


class TestTypeaheadGetSuggestions(unittest.TestCase):
    """Test cases for the get_suggestions method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_index_management = Mock(spec=IndexManagement)
        self.typeahead = Typeahead(
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management
        )
        
        # Common mock index setup
        self.mock_marqo_index = Mock()
        self.mock_marqo_index.typeahead_schema_name = "test_schema"
        self.mock_marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.0")
        
        # Set up the mock_get_index patcher that will be used in most tests
        self.mock_get_index_patcher = patch('marqo.tensor_search.index_meta_cache.get_index')
        self.mock_get_index = self.mock_get_index_patcher.start()
        self.mock_get_index.return_value = self.mock_marqo_index
        
        # Common empty Vespa response
        self.empty_root_fields = RootFields(total_count=0)
        self.empty_root = Root(relevance=0.0, fields=self.empty_root_fields, children=[])
        self.empty_vespa_response = QueryResult(root=self.empty_root)
        
    def tearDown(self):
        """Clean up patches."""
        self.mock_get_index_patcher.stop()

    def test_get_suggestions_for_wildcard_query(self):
        """Test get_suggestions returns top n queries for wildcard query."""
        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="")
        self.typeahead.get_suggestions("test_index", request)

        call_kwargs = self.mock_vespa_client.query.call_args[1]
        yql = call_kwargs['yql']

        # Check YQL uses "WHERE true" for wildcard
        self.assertEqual('SELECT query, metadata FROM test_schema WHERE true', yql)

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_empty_after_normalization(self, mock_normalize):
        """Test get_suggestions returns empty when normalized text is empty."""
        mock_normalize.return_value = ""  # Empty after normalization
        
        request = TypeaheadRequest(q="!!!")
        result = self.typeahead.get_suggestions("test_index", request)
        
        self.assertEqual(len(result.suggestions), 0)
        self.mock_vespa_client.query.assert_not_called()

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_no_tokens(self, mock_normalize):
        """Test get_suggestions returns empty when tokenization produces no tokens."""
        mock_normalize.return_value = "   "  # Only whitespace, will produce no tokens
        
        request = TypeaheadRequest(q="test")
        result = self.typeahead.get_suggestions("test_index", request)
        
        self.assertEqual(len(result.suggestions), 0)
        self.mock_vespa_client.query.assert_not_called()

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_short_tokens_exact_matching(self, mock_normalize):
        """Test get_suggestions uses exact prefix matching for short tokens."""
        mock_normalize.return_value = "ai ml"  # Both tokens are short (< 3 chars)

        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="ai ml", min_fuzzy_match_length=3)

        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.05]):
            result = self.typeahead.get_suggestions("test_index", request)

        call_kwargs = self.mock_vespa_client.query.call_args[1]
        self.assertEqual(
            'SELECT query, metadata FROM test_schema WHERE rank('
            'query_words contains ({prefix:true}"ai") OR '
            'query_words contains ({prefix:true}"ml"), '
            'query_index contains "ai" OR query_index contains "ml")',
            call_kwargs['yql']
        )

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_long_tokens_fuzzy_matching(self, mock_normalize):
        """Test get_suggestions uses fuzzy matching for long tokens."""
        mock_normalize.return_value = "machine learning"  # Both tokens are long (>= 3 chars)

        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="machine learning", fuzzy_edit_distance=2, min_fuzzy_match_length=3)

        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.03]):
            result = self.typeahead.get_suggestions("test_index", request)

        call_kwargs = self.mock_vespa_client.query.call_args[1]
        self.assertEqual(
            'SELECT query, metadata FROM test_schema WHERE rank('
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("machine")) OR '
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("learning")), '
            'query_index contains "machine" OR query_index contains "learning")',
            call_kwargs['yql']
        )

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_mixed_token_lengths(self, mock_normalize):
        """Test get_suggestions handles mix of short and long tokens."""
        mock_normalize.return_value = "ai machine"  # One short, one long token

        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="ai machine", fuzzy_edit_distance=1, min_fuzzy_match_length=3)

        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.04]):
            result = self.typeahead.get_suggestions("test_index", request)

        call_kwargs = self.mock_vespa_client.query.call_args[1]
        self.assertEqual(
            'SELECT query, metadata FROM test_schema WHERE rank('
            'query_words contains ({prefix:true}"ai") OR '
            'query_words contains ({maxEditDistance:1, prefix:true}fuzzy("machine")), '
            'query_index contains "ai" OR query_index contains "machine")',
            call_kwargs['yql']
        )

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_with_popularity_weight(self, mock_normalize):
        """Test get_suggestions includes popularity weight in query features."""
        mock_normalize.return_value = "test"
        
        self.mock_vespa_client.query.return_value = self.empty_vespa_response
        
        request = TypeaheadRequest(q="test", popularity_weight=0.8)
        
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.02]):
            result = self.typeahead.get_suggestions("test_index", request)
        
        # Check query features were included
        call_kwargs = self.mock_vespa_client.query.call_args[1]
        self.assertIn('query_features', call_kwargs)
        self.assertEqual(call_kwargs['query_features']['popularity_weight'], 0.8)

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_with_both_weights(self, mock_normalize):
        """Test get_suggestions includes both weights in query features."""
        mock_normalize.return_value = "test"
        
        self.mock_vespa_client.query.return_value = self.empty_vespa_response
        
        request = TypeaheadRequest(q="test", popularity_weight=0.6, bm25_weight=0.4)
        
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.025]):
            result = self.typeahead.get_suggestions("test_index", request)
        
        # Check both weights were included
        call_kwargs = self.mock_vespa_client.query.call_args[1]
        query_features = call_kwargs['query_features']
        self.assertEqual(query_features['popularity_weight'], 0.6)
        self.assertEqual(query_features['bm25_weight'], 0.4)

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_response_mapping(self, mock_normalize):
        """Test get_suggestions properly maps Vespa response to TypeaheadSuggestions."""
        mock_normalize.return_value = "test"
        
        # Create real Vespa response with hits
        hit1 = Child(
            relevance=0.95,
            fields={
                "query": "test query 1",
                "metadata": {"hit_count": 50}
            }
        )
        hit2 = Child(
            relevance=0.88,
            fields={
                "query": "test query 2", 
                "metadata": {"hit_count": 200}
            }
        )
        
        root_fields = RootFields(total_count=2)
        root = Root(relevance=0.0, fields=root_fields, children=[hit1, hit2])
        vespa_response = QueryResult(root=root)
        
        self.mock_vespa_client.query.return_value = vespa_response
        
        request = TypeaheadRequest(q="test")
        
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.015]):
            result = self.typeahead.get_suggestions("test_index", request)
        
        # Check response structure
        self.assertEqual(len(result.suggestions), 2)
        self.assertEqual(result.processing_time_ms, 15)
        
        # Check first suggestion
        suggestion1 = result.suggestions[0]
        self.assertIsInstance(suggestion1, TypeaheadSuggestion)
        self.assertEqual(suggestion1.suggestion, "test query 1")
        self.assertEqual(suggestion1.score, 0.95)
        self.assertEqual(suggestion1.metadata, {"hit_count": 50})
        
        # Check second suggestion
        suggestion2 = result.suggestions[1]
        self.assertEqual(suggestion2.suggestion, "test query 2")
        self.assertEqual(suggestion2.score, 0.88)
        self.assertEqual(suggestion2.metadata, {"hit_count": 200})

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_vespa_query_construction(self, mock_normalize):
        """Test get_suggestions constructs correct Vespa query parameters."""
        # Override the schema name for this specific test
        self.mock_marqo_index.typeahead_schema_name = "custom_schema"

        mock_normalize.return_value = "machine learning"

        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="machine learning", limit=15)

        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.01]):
            result = self.typeahead.get_suggestions("test_index", request)

        expected_yql = (
            'SELECT query, metadata FROM custom_schema WHERE rank('
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("machine")) OR '
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("learning")), '
            'query_index contains "machine" OR query_index contains "learning")'
        )

        self.mock_vespa_client.query.assert_called_once_with(
            schema="custom_schema",
            yql=expected_yql,
            hits=15,
            ranking="suggestions-rank-profile"
        )

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_processing_time_calculation(self, mock_normalize):
        """Test get_suggestions calculates processing time correctly."""
        mock_normalize.return_value = "test"
        
        self.mock_vespa_client.query.return_value = self.empty_vespa_response
        
        request = TypeaheadRequest(q="test")
        
        # Mock timer to return specific values - 0.1234 seconds
        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[2.0, 2.1234]):
            result = self.typeahead.get_suggestions("test_index", request)
        
        # Should convert to milliseconds and round: 123.4ms -> 123ms
        self.assertEqual(result.processing_time_ms, 123)


    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_match_all_tokens_uses_and_logic(self, mock_normalize):
        """Test get_suggestions uses AND between retrieval terms when matchAllTokens=True."""
        mock_normalize.return_value = "taylor s"

        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="taylor s", match_all_tokens=True)

        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.05]):
            self.typeahead.get_suggestions("test_index", request)

        call_kwargs = self.mock_vespa_client.query.call_args[1]
        self.assertEqual(
            'SELECT query, metadata FROM test_schema WHERE rank('
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("taylor")) AND '
            'query_words contains ({prefix:true}"s"), '
            'query_index contains "taylor" OR query_index contains "s")',
            call_kwargs['yql']
        )

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_match_all_tokens_keeps_fuzzy(self, mock_normalize):
        """Test get_suggestions still uses fuzzy matching for long tokens in matchAllTokens mode."""
        mock_normalize.return_value = "machine learning"

        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="machine learning", match_all_tokens=True, fuzzy_edit_distance=2,
                                   min_fuzzy_match_length=3)

        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.05]):
            self.typeahead.get_suggestions("test_index", request)

        call_kwargs = self.mock_vespa_client.query.call_args[1]
        self.assertEqual(
            'SELECT query, metadata FROM test_schema WHERE rank('
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("machine")) AND '
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("learning")), '
            'query_index contains "machine" OR query_index contains "learning")',
            call_kwargs['yql']
        )

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_match_all_tokens_single_token(self, mock_normalize):
        """Test get_suggestions matchAllTokens mode works with a single token."""
        mock_normalize.return_value = "taylor"

        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="taylor", match_all_tokens=True)

        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.05]):
            self.typeahead.get_suggestions("test_index", request)

        call_kwargs = self.mock_vespa_client.query.call_args[1]
        self.assertEqual(
            'SELECT query, metadata FROM test_schema WHERE rank('
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("taylor")), '
            'query_index contains "taylor")',
            call_kwargs['yql']
        )

    @patch('marqo.core.typeahead.typeahead.normalize_text')
    def test_get_suggestions_default_uses_or_logic(self, mock_normalize):
        """Test get_suggestions uses OR between retrieval terms by default (backward compat)."""
        mock_normalize.return_value = "taylor s"

        self.mock_vespa_client.query.return_value = self.empty_vespa_response

        request = TypeaheadRequest(q="taylor s")

        with patch('marqo.core.typeahead.typeahead.timer', side_effect=[0.0, 0.05]):
            self.typeahead.get_suggestions("test_index", request)

        call_kwargs = self.mock_vespa_client.query.call_args[1]
        self.assertEqual(
            'SELECT query, metadata FROM test_schema WHERE rank('
            'query_words contains ({maxEditDistance:2, prefix:true}fuzzy("taylor")) OR '
            'query_words contains ({prefix:true}"s"), '
            'query_index contains "taylor" OR query_index contains "s")',
            call_kwargs['yql']
        )


class TestTypeaheadDeleteQueries(unittest.TestCase):
    """Test cases for delete query methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_index_management = Mock(spec=IndexManagement)
        self.typeahead = Typeahead(
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management
        )
        
        self.mock_marqo_index = Mock()
        self.mock_marqo_index.typeahead_schema_name = "test_schema"
        self.mock_marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.0")

    def hash(self, query):
        return self.typeahead._generate_query_hash(query)

    def test_delete_all_queries_success(self):
        """Test delete_all_queries removes all documents."""
        self.mock_index_management.get_index.return_value = self.mock_marqo_index
        
        # Mock successful delete response
        self.mock_vespa_client.delete_all_docs.return_value = True
        
        # Execute
        self.typeahead.delete_all_queries("test_index")
        
        # Verify
        self.mock_index_management.get_index.assert_called_once_with(
            index_name="test_index"
        )
        self.mock_vespa_client.delete_all_docs.assert_called_once_with(
            "test_schema"
        )

    def test_delete_queries_success(self):
        """Test delete_queries removes specified queries."""
        self.mock_index_management.get_index.return_value = self.mock_marqo_index

        hash1 = f"{self.hash('query1')}"
        hash2 = f"{self.hash('query2')}"

        # Mock successful delete responses
        successful_response = FeedBatchDocumentResponse(
            id=hash1,
            status=200,
            message="OK",
            path_id="/document/v1/test_schema/test_schema/docid/hash1"
        )
        failed_response = FeedBatchDocumentResponse(
            id=hash2,
            status=404,
            message="Not found",
            path_id="/document/v1/test_schema/test_schema/docid/hash2"
        )
        
        feed_response = FeedBatchResponse(responses=[successful_response, failed_response], errors=True)
        self.mock_vespa_client.delete_batch.return_value = feed_response
        
        # Execute
        result = self.typeahead.delete_queries("test_index", ["query1", "query2"])
        
        # Verify
        self.mock_index_management.get_index.assert_called_once()
        self.mock_vespa_client.delete_batch.assert_called_once()
        
        # Check call arguments for delete_batch
        call_args, kwargs = self.mock_vespa_client.delete_batch.call_args
        self.assertListEqual([hash1, hash2], call_args[0])
        self.assertEqual("test_schema", kwargs["schema"])

        # Check result is returned as-is (TODO to process it properly)
        # self.assertEqual(result, feed_response.__dict__)


class TestTypeaheadStats(unittest.TestCase):
    """Test cases for get_stats method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_index_management = Mock(spec=IndexManagement)
        self.typeahead = Typeahead(
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management
        )
        
        self.mock_marqo_index = Mock()
        self.mock_marqo_index.typeahead_schema_name = "test_schema"
        self.mock_marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.0")

    def test_get_stats_returns_document_count(self):
        """Test get_stats returns the correct document count."""
        self.mock_index_management.get_index.return_value = self.mock_marqo_index
        
        # Mock query response with total_count
        mock_query_response = Mock()
        mock_query_response.total_count = 42
        self.mock_vespa_client.query.return_value = mock_query_response
        
        # Execute
        result = self.typeahead.get_stats("test_index")
        
        # Verify
        self.mock_index_management.get_index.assert_called_once_with(
            index_name="test_index"
        )
        
        # Verify query was called with correct parameters
        self.mock_vespa_client.query.assert_called_once_with(
            schema="test_schema",
            yql="SELECT * FROM test_schema WHERE true",
            hits=0,
            summary="minimal"
        )
        
        # Check result
        self.assertIsInstance(result, TypeaheadStatsResponse)
        self.assertEqual(result.indexed_queries, 42)


class TestTypeaheadGetQueries(unittest.TestCase):
    """Test cases for get_queries method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_index_management = Mock(spec=IndexManagement)
        # override index_management in api._config so the version check can also use the mock index_management
        api._config.index_management = self.mock_index_management
        self.typeahead = Typeahead(
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management
        )
        
        self.mock_marqo_index = Mock()
        self.mock_marqo_index.typeahead_schema_name = "test_schema"
        self.mock_marqo_index.parsed_marqo_version.return_value = semver.VersionInfo.parse("2.24.0")

    def test_get_queries_success(self):
        """Test get_queries returns correct queries."""
        self.mock_index_management.get_index.return_value = self.mock_marqo_index
        
        # Mock batch get response with found documents
        doc1_fields = {
            "query": "test query 1",
            "query_words": ["test", "query", "1"],
            "query_index": "t te tes test q qu que quer query 1",
            "popularity": 1.5,
            "metadata": {"category": 0.8},
            "last_updated_at": 1234567890
        }
        doc2_fields = {
            "query": "test query 2",
            "query_words": ["test", "query", "2"],
            "query_index": "t te tes test q qu que quer query 2",
            "popularity": 0.7,
            "metadata": {"category": 0.5},
            "last_updated_at": 1234567891
        }
        
        found_doc1 = GetBatchDocumentResponse(
            status=200,
            pathId="/document/v1/test_schema/test_schema/docid/hash1",
            id="hash1",
            document=Document(id="hash1", fields=doc1_fields)
        )
        found_doc2 = GetBatchDocumentResponse(
            status=200,
            pathId="/document/v1/test_schema/test_schema/docid/hash2",
            id="hash2",
            document=Document(id="hash2", fields=doc2_fields)
        )
        
        get_batch_response = GetBatchResponse(responses=[found_doc1, found_doc2], errors=False)
        self.mock_vespa_client.get_batch.return_value = get_batch_response
        
        # Execute
        result = self.typeahead.get_queries("test_index", ["test query 1", "test query 2"])
        
        # Verify
        self.mock_vespa_client.get_batch.assert_called_once()
        
        # Check result
        self.assertIsInstance(result, TypeaheadGetQueriesResponse)
        self.assertEqual(len(result.queries), 2)
        
        # Check first query
        query1 = result.queries[0]
        self.assertEqual(query1.query, "test query 1")
        self.assertEqual(query1.popularity, 1.5)
        self.assertEqual(query1.metadata, {"category": 0.8})
        self.assertEqual(query1.last_updated_at, 1234567890)
        
        # Check second query  
        query2 = result.queries[1]
        self.assertEqual(query2.query, "test query 2")
        self.assertEqual(query2.popularity, 0.7)

    def test_get_queries_empty_list(self):
        """Test get_queries with empty query list."""
        self.mock_index_management.get_index.return_value = self.mock_marqo_index

        result = self.typeahead.get_queries("test_index", [])
        
        # Should return empty response without calling Vespa
        self.assertIsInstance(result, TypeaheadGetQueriesResponse)
        self.assertEqual(len(result.queries), 0)
        self.mock_vespa_client.get_batch.assert_not_called()

    def test_get_queries_not_found(self):
        """Test get_queries with queries that don't exist."""
        self.mock_index_management.get_index.return_value = self.mock_marqo_index
        
        # Mock batch get response with no documents found
        not_found_doc = GetBatchDocumentResponse(
            status=404,
            pathId="/document/v1/test_schema/test_schema/docid/hash1",
            id="hash1",
            document=None,
            message="Document not found"
        )
        
        get_batch_response = GetBatchResponse(responses=[not_found_doc], errors=True)
        self.mock_vespa_client.get_batch.return_value = get_batch_response
        
        # Execute
        result = self.typeahead.get_queries("test_index", ["nonexistent query"])
        
        # Verify
        self.assertIsInstance(result, TypeaheadGetQueriesResponse)
        self.assertEqual(len(result.queries), 0)  # No queries found

    def test_get_queries_mixed_found_not_found(self):
        """Test get_queries with mix of found and not found queries."""
        self.mock_index_management.get_index.return_value = self.mock_marqo_index
        
        # Mock response with one found, one not found
        found_doc = GetBatchDocumentResponse(
            status=200,
            pathId="/document/v1/test_schema/test_schema/docid/hash1",
            id="hash1", 
            document=Document(id="hash1", fields={
                "query": "found query",
                "query_words": ["found", "query"],
                "query_index": "f fo fou foun found q qu que quer query",
                "popularity": 1.0,
                "metadata": {},
                "last_updated_at": 1234567890
            })
        )
        not_found_doc = GetBatchDocumentResponse(
            status=404,
            pathId="/document/v1/test_schema/test_schema/docid/hash2",
            id="hash2",
            document=None,
            message="Document not found"
        )
        
        get_batch_response = GetBatchResponse(responses=[found_doc, not_found_doc], errors=True)
        self.mock_vespa_client.get_batch.return_value = get_batch_response
        
        # Execute
        result = self.typeahead.get_queries("test_index", ["found query", "nonexistent query"])
        
        # Verify - should return only the found query
        self.assertIsInstance(result, TypeaheadGetQueriesResponse)
        self.assertEqual(len(result.queries), 1)
        self.assertEqual(result.queries[0].query, "found query")


class TestTypeaheadEscaping(unittest.TestCase):
    """Test cases for the special character escaping functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vespa_client = Mock(spec=VespaClient)
        self.mock_index_management = Mock(spec=IndexManagement)
        self.typeahead = Typeahead(
            vespa_client=self.mock_vespa_client,
            index_management=self.mock_index_management
        )

    def test_escape_token_no_special_chars(self):
        """Test escaping tokens with no special characters."""
        # Simple strings should not be modified
        test_cases = [
            "hello",
            "world",
            "123",
            "test_query",
            "simple-text",
            "with spaces",
            "",
        ]

        for token in test_cases:
            with self.subTest(token=token):
                result = self.typeahead._escape_token(token)
                self.assertEqual(result, token)

    def test_escape_token_quotes(self):
        """Test escaping tokens with double quotes."""
        test_cases = [
            ('"', '\\"'),
            ('hello"world', 'hello\\"world'),
            ('"start', '\\"start'),
            ('end"', 'end\\"'),
            ('""', '\\"\\"'),
            ('say "hello"', 'say \\"hello\\"'),
        ]

        for input_token, expected in test_cases:
            with self.subTest(input_token=input_token):
                result = self.typeahead._escape_token(input_token)
                self.assertEqual(result, expected)

    def test_escape_token_backslashes(self):
        """Test escaping tokens with backslashes."""
        test_cases = [
            ('\\', '\\\\'),
            ('hello\\world', 'hello\\\\world'),
            ('\\start', '\\\\start'),
            ('end\\', 'end\\\\'),
            ('\\\\', '\\\\\\\\'),
            ('path\\to\\file', 'path\\\\to\\\\file'),
        ]

        for input_token, expected in test_cases:
            with self.subTest(input_token=input_token):
                result = self.typeahead._escape_token(input_token)
                self.assertEqual(result, expected)

    def test_escape_token_mixed_special_chars(self):
        """Test escaping tokens with both quotes and backslashes."""
        test_cases = [
            ('"\\', '\\"\\\\'),
            ('"hello\\world"', '\\"hello\\\\world\\"'),
            ('C:\\Program Files\\"test"', 'C:\\\\Program Files\\\\\\"test\\"'),
            ('a"b\\c"d', 'a\\"b\\\\c\\"d'),
        ]

        for input_token, expected in test_cases:
            with self.subTest(input_token=input_token):
                result = self.typeahead._escape_token(input_token)
                self.assertEqual(result, expected)

    def test_escape_token_preserves_other_chars(self):
        """Test that escaping preserves other special characters."""
        # Characters that are NOT in CHARACTERS_TO_BE_ESCAPED_IN_VESPA should be preserved
        test_cases = [
            "hello!world",
            "test@example.com",
            "price$100",
            "50%off",
            "a+b=c",
            "question?",
            "array[0]",
            "function()",
            "hash#tag",
        ]

        for token in test_cases:
            with self.subTest(token=token):
                result = self.typeahead._escape_token(token)
                self.assertEqual(result, token)


if __name__ == "__main__":
    unittest.main()