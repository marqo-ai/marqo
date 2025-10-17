import os
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from marqo import config
from marqo.core.models.typeahead import (
    TypeaheadResponse, TypeaheadSuggestion,
    TypeaheadIndexingResponse, TypeaheadStatsResponse, TypeaheadIndexingError,
    TypeaheadQuery, TypeaheadGetQueriesResponse
)
from marqo.tensor_search import api


class TestTypeaheadAPIWithTestClient(unittest.TestCase):
    """Test cases for typeahead API endpoints using FastAPI TestClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=config.Config)
        self.mock_typeahead = Mock()
        self.mock_config.typeahead = self.mock_typeahead
        api.app.dependency_overrides[api.get_config] = lambda: self.mock_config
        self.client = TestClient(api.app)

    def tearDown(self):
        # Always clean up overrides so other tests aren't affected
        api.app.dependency_overrides.clear()

    def test_get_suggestions_valid_request(self):
        """Test get_suggestions with valid JSON request."""

        # Create mock response from typeahead service
        mock_suggestions = [
            TypeaheadSuggestion(suggestion="test query", score=0.95, metadata={"key": 2.0})
        ]
        mock_response = TypeaheadResponse(suggestions=mock_suggestions, processing_time_ms=50)
        self.mock_typeahead.get_suggestions.return_value = mock_response

        # Test with valid JSON request
        request_data = {
            "q": "test query",
            "limit": 10,
            "minFuzzyMatchLength": 3,
            "fuzzyEditDistance": 1,
            "popularityWeight": 1.0,
            "bm25Weight": 1.0
        }

        response = self.client.post("/indexes/test_index/suggestions", json=request_data)

        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("suggestions", response_data)
        self.assertIn("processingTimeMs", response_data)
        self.assertEqual(response_data["processingTimeMs"], 50)
        self.assertEqual(len(response_data["suggestions"]), 1)

        # Verify the typeahead service was called with correct parameters
        call_args = self.mock_typeahead.get_suggestions.call_args[0]
        self.assertEqual(call_args[0], "test_index")  # index_name
        request_obj = call_args[1]  # TypeaheadRequest
        self.assertEqual(request_obj.q, "test query")
        self.assertEqual(request_obj.limit, 10)

    def test_get_suggestions_minimal_request(self):
        """Test get_suggestions with minimal JSON request using defaults."""
        
        mock_response = TypeaheadResponse(suggestions=[], processing_time_ms=25)
        self.mock_typeahead.get_suggestions.return_value = mock_response
        
        # Test with minimal JSON request
        request_data = {"q": "test"}
        
        response = self.client.post("/indexes/test_index/suggestions", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        
        # Verify defaults were applied
        call_args = self.mock_typeahead.get_suggestions.call_args[0]
        request_obj = call_args[1]
        self.assertEqual(request_obj.limit, 10)  # default value
        self.assertEqual(request_obj.min_fuzzy_match_length, 3)  # default value
        self.assertEqual(request_obj.fuzzy_edit_distance, 2)  # default value

    def test_get_suggestions_request_with_empty_query(self):
        """Test get_suggestions with minimal JSON request using defaults."""

        mock_response = TypeaheadResponse(suggestions=[], processing_time_ms=25)
        self.mock_typeahead.get_suggestions.return_value = mock_response

        # Test with minimal JSON request
        request_data = {"q": ""}

        response = self.client.post("/indexes/test_index/suggestions", json=request_data)

        self.assertEqual(response.status_code, 200)

        # Verify defaults were applied
        call_args = self.mock_typeahead.get_suggestions.call_args[0]
        request_obj = call_args[1]
        self.assertEqual(request_obj.q, "")  # empty string is allowed
        self.assertEqual(request_obj.limit, 10)  # default value
        self.assertEqual(request_obj.min_fuzzy_match_length, 3)  # default value
        self.assertEqual(request_obj.fuzzy_edit_distance, 2)  # default value

    def test_get_suggestions_invalid_request(self):
        """Test get_suggestions with invalid JSON request triggers validation error."""
        
        # Test with missing required field
        request_data = {"limit": 10}  # missing required 'q' field
        
        response = self.client.post("/indexes/test_index/suggestions", json=request_data)
        
        # Should return validation error
        self.assertEqual(response.status_code, 422)
        response_data = response.json()
        self.assertIn("detail", response_data)
        # Verify it's a validation error for missing 'q' field
        self.assertTrue(any("q" in str(error) for error in response_data["detail"]))

    def test_index_queries_valid_request(self):
        """Test index_queries with valid JSON request."""
        
        # Create mock response from typeahead service
        mock_response = TypeaheadIndexingResponse(indexed=2, errors=[], processing_time_ms=100)
        self.mock_typeahead.index_queries.return_value = mock_response
        
        # Test with valid JSON request
        request_data = {
            "queries": [
                {"query": "query1", "popularity": 1.0, "metadata": {"category": 0.8}},
                {"query": "query2", "popularity": 0.5}
            ]
        }
        
        response = self.client.post("/indexes/test_index/suggestions/queries", json=request_data)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("indexed", response_data)
        self.assertIn("errors", response_data)
        self.assertIn("processingTimeMs", response_data)
        self.assertEqual(response_data["indexed"], 2)
        self.assertEqual(response_data["processingTimeMs"], 100)
        
        # Verify the typeahead service was called correctly
        call_args = self.mock_typeahead.index_queries.call_args[0]
        request_obj = call_args[1]  # TypeaheadIndexRequest
        self.assertEqual(len(request_obj.queries), 2)
        self.assertEqual(request_obj.queries[0].query, "query1")
        self.assertEqual(request_obj.queries[1].popularity, 0.5)

    def test_index_queries_with_errors_response(self):
        """Test index_queries returns errors in response."""
        
        # Create mock response with errors
        errors = [TypeaheadIndexingError(query="bad query", message="Invalid query", code=400)]
        mock_response = TypeaheadIndexingResponse(indexed=1, errors=errors, processing_time_ms=80)
        self.mock_typeahead.index_queries.return_value = mock_response
        
        request_data = {
            "queries": [
                {"query": "good query"},
                {"query": ""}  # This will cause an error
            ]
        }
        
        response = self.client.post("/indexes/test_index/suggestions/queries", json=request_data)
        
        self.assertEqual(response.status_code, 422)
        response_data = response.json()
        self.assertEqual('Value error, query is required and must not be an empty string',
                         response_data['detail'][0]['msg'])

    def test_index_queries_invalid_request(self):
        """Test index_queries with invalid JSON request."""
        
        # Test with invalid structure
        request_data = {"queries": "not a list"}  # should be array
        
        response = self.client.post("/indexes/test_index/suggestions/queries", json=request_data)
        
        self.assertEqual(response.status_code, 422)
        response_data = response.json()
        self.assertIn("detail", response_data)

    def test_delete_all_queries_success(self):
        """Test delete_all_queries endpoint."""
        # Enable batch APIs for this test
        original_value = os.environ.get("MARQO_ENABLE_BATCH_APIS")
        os.environ["MARQO_ENABLE_BATCH_APIS"] = "true"
        
        try:
            self.mock_typeahead.delete_all_queries.return_value = None
            
            response = self.client.delete("/indexes/test_index/suggestions/queries/delete-all")
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), "All queries deleted successfully")
            self.mock_typeahead.delete_all_queries.assert_called_once_with("test_index")
        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop("MARQO_ENABLE_BATCH_APIS", None)
            else:
                os.environ["MARQO_ENABLE_BATCH_APIS"] = original_value

    def test_delete_queries_success(self):
        """Test delete_queries endpoint with request body."""
        self.mock_typeahead.delete_queries.return_value = None
        
        # FastAPI treats List[str] as request body, so send as JSON array
        queries_list = ["query1", "query2"]
        
        response = self.client.request("DELETE", "/indexes/test_index/suggestions/queries", json=queries_list)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "Queries deleted successfully")
        self.mock_typeahead.delete_queries.assert_called_once_with("test_index", ["query1", "query2"])

    def test_get_typeahead_stats_success(self):
        """Test get_typeahead_stats endpoint returns proper JSON response."""
        
        mock_response = TypeaheadStatsResponse(indexed_queries=42)
        self.mock_typeahead.get_stats.return_value = mock_response
        
        response = self.client.get("/indexes/test_index/suggestions/stats")
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        # Should use camelCase alias
        self.assertIn("indexedQueries", response_data)
        self.assertEqual(response_data["indexedQueries"], 42)
        # Should not contain snake_case field name
        self.assertNotIn("indexed_queries", response_data)

    def test_pydantic_model_field_validation(self):
        """Test that Pydantic models validate field types correctly."""
        
        # Test invalid field types
        request_data = {
            "q": "test",
            "limit": "invalid",  # should be int
            "popularityWeight": "not a number"  # should be float
        }
        
        response = self.client.post("/indexes/test_index/suggestions", json=request_data)
        
        self.assertEqual(response.status_code, 422)
        response_data = response.json()
        self.assertIn("detail", response_data)
        # Should have validation errors for the invalid fields
        errors = response_data["detail"]
        self.assertTrue(any("limit" in str(error) for error in errors))

    def test_camel_case_field_aliases(self):
        """Test that camelCase field aliases work correctly in requests."""
        
        mock_response = TypeaheadResponse(suggestions=[], processing_time_ms=30)
        self.mock_typeahead.get_suggestions.return_value = mock_response
        
        # Use camelCase field names in request
        request_data = {
            "q": "test",
            "minFuzzyMatchLength": 5,  # camelCase alias
            "fuzzyEditDistance": 2,    # camelCase alias
            "popularityWeight": 0.8,   # camelCase alias
            "bm25Weight": 1.2         # camelCase alias
        }
        
        response = self.client.post("/indexes/test_index/suggestions", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        
        # Verify the values were correctly mapped from camelCase to snake_case
        call_args = self.mock_typeahead.get_suggestions.call_args[0]
        request_obj = call_args[1]
        self.assertEqual(request_obj.min_fuzzy_match_length, 5)
        self.assertEqual(request_obj.fuzzy_edit_distance, 2)
        self.assertEqual(request_obj.popularity_weight, 0.8)
        self.assertEqual(request_obj.bm25_weight, 1.2)

    def test_get_queries_success(self):
        """Test get_queries endpoint returns correct queries."""
        # Create mock queries
        mock_queries = [
            TypeaheadQuery(
                query="test query 1",
                query_words=["test", "query", "1"],
                query_index="t te tes test q qu que quer query 1",
                popularity=1.5,
                metadata={"category": 0.8},
                last_updated_at=1234567890
            ),
            TypeaheadQuery(
                query="test query 2",
                query_words=["test", "query", "2"],
                query_index="t te tes test q qu que quer query 2",
                popularity=0.7,
                metadata={"category": 0.5},
                last_updated_at=1234567891
            )
        ]
        mock_response = TypeaheadGetQueriesResponse(queries=mock_queries)
        self.mock_typeahead.get_queries.return_value = mock_response
        
        # Send request with list of queries as JSON body
        queries_list = ["test query 1", "test query 2"]
        response = self.client.request("GET", "/indexes/test_index/suggestions/queries", json=queries_list)
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("queries", response_data)
        self.assertEqual(len(response_data["queries"]), 2)
        
        # Check first query with camelCase aliases
        query1 = response_data["queries"][0]
        self.assertEqual(query1["query"], "test query 1")
        self.assertEqual(query1["popularity"], 1.5)
        self.assertEqual(query1["metadata"], {"category": 0.8})
        self.assertEqual(query1["lastUpdatedAt"], 1234567890)  # camelCase alias
        
        # Verify typeahead service was called correctly
        self.mock_typeahead.get_queries.assert_called_once_with("test_index", queries_list)

    def test_get_queries_empty_list(self):
        """Test get_queries endpoint with empty list."""
        mock_response = TypeaheadGetQueriesResponse(queries=[])
        self.mock_typeahead.get_queries.return_value = mock_response
        
        response = self.client.request("GET", "/indexes/test_index/suggestions/queries", json=[])
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["queries"], [])

    def test_get_queries_invalid_request(self):
        """Test get_queries endpoint with invalid request parameters."""
        # Send invalid query parameter type - FastAPI should handle this gracefully
        response = self.client.request("GET", "/indexes/test_index/suggestions/queries", json="123")
        
        # Should still work since strings can be passed
        self.assertTrue(response.status_code in [200, 422])  # Either works or validation error
    
    def test_index_queries_batch_size_limit(self):
        """Test index_queries with batch size exceeding limit."""
        import os
        
        # Set a small batch size limit for testing
        original_value = os.environ.get("MARQO_MAX_DOCUMENTS_BATCH_SIZE")
        os.environ["MARQO_MAX_DOCUMENTS_BATCH_SIZE"] = "2"
        
        try:
            # Create request with more queries than the limit
            request_data = {
                "queries": [
                    {"query": "query1"},
                    {"query": "query2"},
                    {"query": "query3"}  # This exceeds the limit of 2
                ]
            }
            
            response = self.client.post("/indexes/test_index/suggestions/queries", json=request_data)
            
            # Should return validation error (InvalidArgumentError maps to 400)
            self.assertEqual(response.status_code, 400)
            response_data = response.json()
            self.assertIn("message", response_data)
            # Should contain batch size limit error message
            error_message = response_data["message"]
            self.assertIn("exceeds limit", error_message)
            
        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop("MARQO_MAX_DOCUMENTS_BATCH_SIZE", None)
            else:
                os.environ["MARQO_MAX_DOCUMENTS_BATCH_SIZE"] = original_value
    
    def test_index_queries_empty_batch(self):
        """Test index_queries with empty queries list."""
        request_data = {"queries": []}
        
        response = self.client.post("/indexes/test_index/suggestions/queries", json=request_data)
        
        # Should return validation error for empty batch
        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        error_message = response_data["message"]
        self.assertIn("empty index queries request", error_message)


if __name__ == "__main__":
    unittest.main()