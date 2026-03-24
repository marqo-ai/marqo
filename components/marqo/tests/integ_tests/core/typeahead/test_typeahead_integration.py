import os
import time
from unittest import mock

from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.marqo_index import *
from marqo.core.models.typeahead import (
    TypeaheadRequest, TypeaheadIndexingRequest, TypeaheadAddQueryRequest
)
from tests.integ_tests.marqo_test import MarqoTestCase


class TestTypeaheadIntegration(MarqoTestCase):
    """Integration tests for typeahead functionality with real Vespa instance."""
    
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        
        # Create a semi-structured index with typeahead
        cls.test_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )
        # simulate an index created prior to 2.23.0
        cls.old_version_index_request = cls.unstructured_marqo_index_request(
            name="old_version_index",
            marqo_version="2.22.0"
        )
        
        cls.indexes = cls.create_indexes([cls.test_index, cls.old_version_index_request])
        cls.test_index_name = cls.test_index.name
        cls.index_220_name = cls.old_version_index_request.name
    
    def setUp(self) -> None:
        self.clear_indexes(self.indexes)
        
        # Also clear typeahead data specifically
        self.config.typeahead.delete_all_queries(self.test_index_name)
        
        # Set device for any inference operations
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()
    
    def tearDown(self) -> None:
        self.device_patcher.stop()
    
    def _create_test_queries(self):
        """Create test query data with various patterns."""
        return [
            TypeaheadAddQueryRequest(query="apple iphone 14", popularity=2.0, metadata={"category": 0.8}),
            TypeaheadAddQueryRequest(query="apple macbook pro", popularity=1.5, metadata={"category": 0.9}),
            TypeaheadAddQueryRequest(query="apple watch series", popularity=1.0, metadata={"category": 0.7}),
            TypeaheadAddQueryRequest(query="samsung galaxy s23", popularity=1.8, metadata={"category": 0.8}),
            TypeaheadAddQueryRequest(query="samsung tablet", popularity=0.5, metadata={"category": 0.6}),
            TypeaheadAddQueryRequest(query="laptop computer", popularity=1.2, metadata={"category": 0.7}),
        ]
    
    def _index_test_queries(self, queries=None):
        """Helper to index queries and return response."""
        if queries is None:
            queries = self._create_test_queries()
        
        request = TypeaheadIndexingRequest(queries=queries)
        return self.config.typeahead.index_queries(self.test_index_name, request)
    
    
    # A. Index Creation Tests
    def test_create_index_with_typeahead_enabled(self):
        """Verify that creating an index automatically creates the typeahead schema."""
        # Get the created index
        marqo_index = self.index_management.get_index(self.test_index_name)
        
        # Verify typeahead schema name is set
        expected_name = f"{marqo_index.schema_name}_typeahead"
        self.assertEqual(marqo_index.typeahead_schema_name, expected_name)
        
        # Verify the typeahead schema exists in Vespa
        stats_response = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(stats_response.indexed_queries, 0)  # Should be empty initially
    
    # B. Query Indexing Tests
    def test_index_single_query(self):
        """Index a single query and verify it's stored."""
        query = TypeaheadAddQueryRequest(query="test query", popularity=1.0, metadata={"test": 0.5})
        request = TypeaheadIndexingRequest(queries=[query])
        
        response = self.config.typeahead.index_queries(self.test_index_name, request)
        
        self.assertEqual(response.indexed, 1)
        self.assertEqual(len(response.errors), 0)
        self.assertGreater(response.processing_time_ms, 0)
        
        # Verify it can be retrieved
        stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(stats.indexed_queries, 1)

        get_result = self.config.typeahead.get_queries(self.test_index_name, ["test query"])
        self.assertEqual(1, len(get_result.queries))
        self.assertEqual(query.query, get_result.queries[0].query)
        self.assertEqual(["test", "query"], get_result.queries[0].query_words)
        self.assertEqual("t te tes test q qu que quer query", get_result.queries[0].query_index)
        self.assertEqual(query.popularity, get_result.queries[0].popularity)
        self.assertEqual(query.metadata, get_result.queries[0].metadata)
        self.assertIsNotNone(get_result.queries[0].last_updated_at)

    def test_index_multiple_queries(self):
        """Index multiple queries in batch."""
        queries = self._create_test_queries()
        response = self._index_test_queries(queries)
        
        self.assertEqual(response.indexed, len(queries))
        self.assertEqual(len(response.errors), 0)
        
        # Verify they can be retrieved
        stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(stats.indexed_queries, len(queries))

    def test_index_duplicate_queries(self):
        """Verify duplicate handling (normalized duplicates should be rejected)."""
        queries = [
            TypeaheadAddQueryRequest(query="Apple iPhone", popularity=1.0),
            TypeaheadAddQueryRequest(query="apple iphone", popularity=2.0),  # Should be detected as duplicate
            TypeaheadAddQueryRequest(query="  apple   iphone  ", popularity=3.0),  # this is not a duplicate. should it?
        ]
        request = TypeaheadIndexingRequest(queries=queries)
        
        response = self.config.typeahead.index_queries(self.test_index_name, request)
        
        # Should have some duplicates detected (2 indexed, 1 error)
        self.assertEqual(response.indexed, 2)
        self.assertEqual(len(response.errors), 1)
        
        # Verify error messages contain duplicate information
        for error in response.errors:
            self.assertIn("duplicate", error.message.lower())

        # Verify it can be retrieved
        stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(stats.indexed_queries, 2)
    
    def test_index_queries_batch_size_limit(self):
        """Test batch size validation."""
        # Set a small batch size limit for testing
        original_value = os.environ.get("MARQO_MAX_DOCUMENTS_BATCH_SIZE")
        os.environ["MARQO_MAX_DOCUMENTS_BATCH_SIZE"] = "2"
        
        try:
            # Create request with more queries than the limit
            queries = [
                TypeaheadAddQueryRequest(query="query1"),
                TypeaheadAddQueryRequest(query="query2"),
                TypeaheadAddQueryRequest(query="query3")  # This exceeds the limit of 2
            ]
            
            # Should raise InvalidArgumentError during model construction
            with self.assertRaises(Exception) as context:
                request = TypeaheadIndexingRequest(queries=queries)
            
            self.assertIn("exceeds limit", str(context.exception))
            
        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop("MARQO_MAX_DOCUMENTS_BATCH_SIZE", None)
            else:
                os.environ["MARQO_MAX_DOCUMENTS_BATCH_SIZE"] = original_value
    
    def test_index_empty_batch(self):
        """Test empty batch validation."""
        # Should raise InvalidArgumentError during model construction
        with self.assertRaises(InvalidArgumentError) as context:
            request = TypeaheadIndexingRequest(queries=[])
        
        self.assertIn("empty index queries request", str(context.exception))
    
    # C. Suggestion Retrieval Tests
    def test_get_basic_suggestions(self):
        """Test basic typeahead suggestions."""
        # First index some queries
        self._index_test_queries()
        
        # Test getting suggestions
        request = TypeaheadRequest(q="app")
        response = self.config.typeahead.get_suggestions(self.test_index_name, request)
        
        # Verify expected suggestions are present
        self.assertEqual(4, len(response.suggestions))
        suggestion_texts = [s.suggestion for s in response.suggestions]
        expected_suggestions = [
            "apple iphone 14",
            "apple macbook pro", 
            "apple watch series",
            "laptop computer",  # fuzzy match
        ]
        self.assertListEqual(expected_suggestions, suggestion_texts)

    def test_get_suggestions_strict_prefix_matching(self):
        """Test prefix matching behavior."""
        self._index_test_queries()
        
        # Test short prefix (should use exact prefix matching, fuzziness if not triggered since the input length < 3)
        request = TypeaheadRequest(q="ap")
        response = self.config.typeahead.get_suggestions(self.test_index_name, request)
        apple_suggestions = [s for s in response.suggestions if "apple" in s.suggestion.lower()]
        self.assertEqual(3, len(response.suggestions))
        self.assertEqual(3, len(apple_suggestions))
    
    def test_get_suggestions_fuzzy_matching(self):
        """Test fuzzy matching for longer queries."""
        self._index_test_queries()
        
        # Test longer query with typo (should use fuzzy matching)
        request = TypeaheadRequest(q="aplle iphone")  # typo in "apple"
        response = self.config.typeahead.get_suggestions(self.test_index_name, request)
        
        # Should still find apple iphone suggestions due to fuzzy matching
        self.assertEqual(3, len(response.suggestions))
        self.assertEqual("apple iphone 14", response.suggestions[0].suggestion)

    def test_get_suggestions_fuzzy_matching_disabled(self):
        """Test fuzzy matching can be disabled for longer queries."""
        self._index_test_queries()

        # Test with fuzziness disabled (edit distance = 0)
        request_no_fuzzy = TypeaheadRequest(q="aplle ihone", fuzzy_edit_distance=0)
        response_no_fuzzy = self.config.typeahead.get_suggestions(self.test_index_name, request_no_fuzzy)
        
        # Should find no suggestions since fuzzy matching is disabled
        self.assertEqual(0, len(response_no_fuzzy.suggestions))
    
    def test_get_suggestions_with_weights(self):
        """Test popularity and BM25 weight parameters."""
        self._index_test_queries()

        # Test that weights actually change the ranking of results
        # From test data: "apple iphone 14" (popularity=2.0), "apple macbook pro" (popularity=1.5), "apple watch series" (popularity=1.0)
        
        # Test with high popularity weight - should favor "apple iphone 14" (highest popularity)
        request_high_popularity = TypeaheadRequest(q="apple macbook", popularity_weight=10.0, bm25_weight=0.1)
        response_high_popularity = self.config.typeahead.get_suggestions(self.test_index_name, request_high_popularity)
        self.assertEqual("apple iphone 14", response_high_popularity.suggestions[0].suggestion)
        
        # Test with high BM25 weight - should favor BM25 scoring over popularity
        request_high_bm25 = TypeaheadRequest(q="apple macbook", popularity_weight=0.1, bm25_weight=10.0)
        response_high_bm25 = self.config.typeahead.get_suggestions(self.test_index_name, request_high_bm25)
        self.assertEqual("apple macbook pro", response_high_bm25.suggestions[0].suggestion)

    def test_get_suggestions_limit(self):
        """Test limiting number of suggestions."""
        self._index_test_queries()
        
        # Test different limits
        for limit in [1, 3]:
            with self.subTest(limit=limit):
                request = TypeaheadRequest(q="app", limit=limit)
                response = self.config.typeahead.get_suggestions(self.test_index_name, request)
                self.assertEqual(len(response.suggestions), limit)
    
    def test_get_suggestions_normalization(self):
        """Test that queries are normalized properly."""
        # Index a query with specific formatting
        query = TypeaheadAddQueryRequest(query="  Apple   iPhone  14  ", popularity=1.0)
        self._index_test_queries([query])
        
        # Test that various input formats return the same result
        test_queries = ["apple iphone", "Apple iPhone", "  apple   iphone  "]
        
        for test_query in test_queries:
            request = TypeaheadRequest(q=test_query)
            response = self.config.typeahead.get_suggestions(self.test_index_name, request)
            self.assertEqual(query.query, response.suggestions[0].suggestion)

    def test_get_suggestions_empty_query(self):
        """Test behavior with short query that doesn't match anything."""
        self._index_test_queries()
        
        # Test with a query that won't match anything  
        request = TypeaheadRequest(q="xyz")
        response = self.config.typeahead.get_suggestions(self.test_index_name, request)
        self.assertEqual(len(response.suggestions), 0)

    def test_get_suggestions_wildcard_query(self):
        """Test behavior with wildcard query that returns top queries by popularity"""
        test_queries = self._create_test_queries()
        self._index_test_queries(test_queries)

        test_cases = [
            # limit, bm25_weight, popularity_weight
            (5, 1.0, 1.0),  # base test case
            (5, 100.0, 1.0),  # test high bm_25 weight does not impact result
            (5, 0.0, 1.0),  # test zero bm_25 weight does not impact result
            (5, 1.0, 100.0),  # test high popularity weight does not impact result
            (2, 1.0, 1.0),  # test different limit
        ]

        for limit, bm25_weight, popularity_weight in test_cases:
            with self.subTest(limit=limit, bm25_weight=bm25_weight, popularity_weight=popularity_weight):
                request = TypeaheadRequest(q="", limit=limit, bm25_weight=bm25_weight,
                                           popularity_weight=popularity_weight)
                response = self.config.typeahead.get_suggestions(self.test_index_name, request)
                self.assertEqual(limit, len(response.suggestions))

                sorted_queries = test_queries.copy()
                sorted_queries.sort(key=lambda s: s.popularity, reverse=True)
                self.assertListEqual([q.query for q in sorted_queries[0:limit]],
                                     [s.suggestion for s in response.suggestions])
    
    def test_get_suggestions_match_all_tokens_filters_unrelated_tokens(self):
        """Test that matchAllTokens=True requires ALL tokens to match, filtering out partial matches.

        With the default OR logic, "taylor s" returns results matching just "s" (e.g., "sport", "sad").
        With matchAllTokens=True (AND logic), only results containing BOTH "taylor" AND "s*" are returned.
        """
        queries = [
            TypeaheadAddQueryRequest(query="taylor swift", popularity=2.0),
            TypeaheadAddQueryRequest(query="taylor", popularity=1.5),
            TypeaheadAddQueryRequest(query="sport", popularity=1.8),
            TypeaheadAddQueryRequest(query="sad", popularity=1.0),
            TypeaheadAddQueryRequest(query="suspense", popularity=0.8),
            TypeaheadAddQueryRequest(query="taylor series math", popularity=0.5),
        ]
        self._index_test_queries(queries)

        # Default OR behavior: "taylor s" matches anything with "taylor" OR "s*"
        request_or = TypeaheadRequest(q="taylor s")
        response_or = self.config.typeahead.get_suggestions(self.test_index_name, request_or)
        or_suggestions = [s.suggestion for s in response_or.suggestions]
        # All 6 queries should be returned (OR matches any token)
        self.assertCountEqual(
            ["taylor swift", "taylor", "sport", "sad", "taylor series math", "suspense"],
            or_suggestions
        )

        # matchAllTokens AND behavior: "taylor s" requires BOTH "taylor" AND "s*"
        request_and = TypeaheadRequest(q="taylor s", match_all_tokens=True)
        response_and = self.config.typeahead.get_suggestions(self.test_index_name, request_and)
        and_suggestions = [s.suggestion for s in response_and.suggestions]
        self.assertListEqual(
            ["taylor swift", "taylor series math"],
            and_suggestions
        )

    def test_get_suggestions_match_all_tokens_with_fuzzy(self):
        """Test that matchAllTokens=True still allows fuzzy matching for typo tolerance."""
        queries = [
            TypeaheadAddQueryRequest(query="taylor swift", popularity=2.0),
            TypeaheadAddQueryRequest(query="samsung galaxy", popularity=1.5),
        ]
        self._index_test_queries(queries)

        # Typo in "taylor" -> "taylro", fuzzy should still match
        request = TypeaheadRequest(q="taylro swi", match_all_tokens=True, fuzzy_edit_distance=2)
        response = self.config.typeahead.get_suggestions(self.test_index_name, request)
        suggestions = [s.suggestion for s in response.suggestions]
        self.assertListEqual(["taylor swift"], suggestions)

    # D. Query Management Tests
    def test_get_queries_by_strings(self):
        """Retrieve specific queries by their query strings."""
        test_queries = self._create_test_queries()
        self._index_test_queries(test_queries)
        
        # Test retrieving specific queries
        query_strings = ["apple iphone 14", "samsung galaxy s23"]
        response = self.config.typeahead.get_queries(self.test_index_name, query_strings)
        
        self.assertEqual(len(response.queries), 2)
        retrieved_queries = {q.query: q for q in response.queries}
        for query_string in query_strings:
            self.assertIn(query_string, retrieved_queries)
        
        # Create a map of expected values from test queries
        expected_queries = {q.query: q for q in test_queries if q.query in query_strings}
        
        # Verify metadata and popularity match exactly what was indexed
        for retrieved_query in response.queries:
            expected_query = expected_queries[retrieved_query.query]
            
            # Verify exact values match
            self.assertEqual(retrieved_query.popularity, expected_query.popularity)
            self.assertEqual(retrieved_query.metadata, expected_query.metadata)
            self.assertIsInstance(retrieved_query.last_updated_at, int)
            self.assertGreater(retrieved_query.last_updated_at, 0)
    
    def test_get_queries_not_found(self):
        """Test behavior when queries don't exist."""
        self._index_test_queries()
        
        # Try to retrieve non-existent queries
        response = self.config.typeahead.get_queries(self.test_index_name, ["nonexistent query"])
        self.assertEqual(len(response.queries), 0)
        
        # Mix of existing and non-existing
        response = self.config.typeahead.get_queries(
            self.test_index_name, 
            ["apple iphone 14", "nonexistent query"]
        )
        self.assertEqual(len(response.queries), 1)
        self.assertEqual(response.queries[0].query, "apple iphone 14")
    
    def test_delete_specific_queries(self):
        """Delete specific queries from typeahead."""
        self._index_test_queries()
        
        # Verify queries exist
        initial_stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertGreater(initial_stats.indexed_queries, 0)
        
        # Delete specific queries
        queries_to_delete = ["apple iphone 14", "samsung galaxy s23"]
        self.config.typeahead.delete_queries(self.test_index_name, queries_to_delete)
        
        # Verify they're gone
        response = self.config.typeahead.get_queries(self.test_index_name, queries_to_delete)
        self.assertEqual(len(response.queries), 0)
        
        # Verify stats are updated
        final_stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(final_stats.indexed_queries, initial_stats.indexed_queries - len(queries_to_delete))
    
    def test_delete_all_queries(self):
        """Delete all queries from typeahead index."""
        self._index_test_queries()
        
        # Verify queries exist
        initial_stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(6, initial_stats.indexed_queries)
        
        # Delete all queries
        self.config.typeahead.delete_all_queries(self.test_index_name)
        
        # Verify all are gone
        time.sleep(1)  # give vespa some time to reach eventual consistency
        final_stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(0, final_stats.indexed_queries)

        # Verify suggestions are empty
        request = TypeaheadRequest(q="apple")
        response = self.config.typeahead.get_suggestions(self.test_index_name, request)
        self.assertEqual(0, len(response.suggestions))
    
    # E. Stats Tests
    def test_get_typeahead_stats(self):
        """Verify stats endpoint returns correct query counts."""
        # Explicitly clear any existing queries
        self.config.typeahead.delete_all_queries(self.test_index_name)
        
        # Initially should be empty
        stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(stats.indexed_queries, 0)
        
        # Index some queries
        test_queries = self._create_test_queries()
        self._index_test_queries(test_queries)
        
        # Stats should reflect indexed queries
        stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(stats.indexed_queries, len(test_queries))
        
        # Delete some queries and verify stats update
        self.config.typeahead.delete_queries(self.test_index_name, ["apple iphone 14"])
        stats = self.config.typeahead.get_stats(self.test_index_name)
        self.assertEqual(stats.indexed_queries, len(test_queries) - 1)

    # F. Version Checking Tests
    def test_typeahead_version_check_with_old_version_index(self):
        """Test that typeahead operations raise UnsupportedFeatureError for indexes created with old versions."""

        # Test get_suggestions raises UnsupportedFeatureError
        request = TypeaheadRequest(q="test query")
        with self.assertRaises(UnsupportedFeatureError) as context:
            self.config.typeahead.get_suggestions(self.index_220_name, request)

        self._assert_version_error_message(context.exception, self.index_220_name, "2.22.0")

        # Test index_queries raises UnsupportedFeatureError
        queries = [TypeaheadAddQueryRequest(query="test query", popularity=1.0)]
        index_request = TypeaheadIndexingRequest(queries=queries)
        with self.assertRaises(UnsupportedFeatureError) as context:
            self.config.typeahead.index_queries(self.index_220_name, index_request)

        self._assert_version_error_message(context.exception, self.index_220_name, "2.22.0")

        # Test delete_all_queries raises UnsupportedFeatureError
        with self.assertRaises(UnsupportedFeatureError) as context:
            self.config.typeahead.delete_all_queries(self.index_220_name)

        self._assert_version_error_message(context.exception, self.index_220_name, "2.22.0")

        # Test delete_queries raises UnsupportedFeatureError
        with self.assertRaises(UnsupportedFeatureError) as context:
            self.config.typeahead.delete_queries(self.index_220_name, ["test query"])

        self._assert_version_error_message(context.exception, self.index_220_name, "2.22.0")

        # Test get_stats raises UnsupportedFeatureError
        with self.assertRaises(UnsupportedFeatureError) as context:
            self.config.typeahead.get_stats(self.index_220_name)

        self._assert_version_error_message(context.exception, self.index_220_name, "2.22.0")

        # Test get_queries raises UnsupportedFeatureError
        with self.assertRaises(UnsupportedFeatureError) as context:
            self.config.typeahead.get_queries(self.index_220_name, ["test query"])

        self._assert_version_error_message(context.exception, self.index_220_name, "2.22.0")

    # G. Special Character Handling Tests
    def test_typeahead_with_special_characters_in_user_input(self):
        """Test typeahead search handles special characters in user input without causing Vespa 500 errors."""
        self._index_test_queries()

        # Test user input queries with special characters that should NOT cause Vespa 500 errors
        user_input_test_cases = [
            # Single special characters
            '"',
            '\\',
            # Queries with quotes
            'a"b"c',
            'a"b"',
            '"b"c',
            '"bc',
            'bc"',
            # Queries with backslashes
            'Path\\to\\file',
            '\\',
            '\\\\',
            'a\\b',
            '\\a',
            'b\\',
            # Mixed special characters
            'Program "with spaces"\\folder',
            '"\\',
            '\\"',
        ]

        for user_query in user_input_test_cases:
            with self.subTest(user_query=user_query):
                request = TypeaheadRequest(q=user_query, limit=10)
                # This should NOT raise a 500 error from Vespa (main goal of the fix)
                try:
                    response = self.config.typeahead.get_suggestions(self.test_index_name, request)
                    # Just verify we get a response without error
                    self.assertIsNotNone(response)
                    self.assertGreaterEqual(len(response.suggestions), 0)
                except Exception as e:
                    self.fail(f"User input query '{user_query}' caused an error: {e}")

    def _assert_version_error_message(self, exception: UnsupportedFeatureError, index_name: str, version: str):
        """Helper method to verify the error message contains expected information."""
        error_message = str(exception)
        self.assertIn("Typeahead functionality is not supported", error_message)
        self.assertIn(index_name, error_message)
        self.assertIn(version, error_message)
        self.assertIn("2.23.0", error_message)
        self.assertIn("recreate the index", error_message)
