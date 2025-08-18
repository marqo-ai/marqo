import json
import uuid
import requests
from tests.marqo_test import MarqoTestCase


class TestTypeahead(MarqoTestCase):
    """Test cases for typeahead functionality using direct HTTP requests."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        cls.unstructured_index_name = "unstructured_typeahead_" + str(uuid.uuid4()).replace('-', '')
        
        cls.create_indexes([
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
            }
        ])
        
        cls.indexes_to_delete = [cls.unstructured_index_name]


    def test_get_suggestions_with_invalid_params(self):
        """Test suggestions endpoint with invalid parameters."""
        invalid_requests = [
            {},  # Missing input
            {"input": ""},  # Empty input
            {"input": "test", "maxSuggestions": -1},  # Invalid maxSuggestions
            {"input": "test", "fuzzyEditDistance": -1},  # Invalid fuzzyEditDistance
        ]
        
        for invalid_request in invalid_requests:
            with self.subTest(invalid_request):
                response = requests.post(
                    f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(invalid_request)
                )
                
                self.assertIn(response.status_code, [400, 422])

    def test_index_queries_with_invalid_format(self):
        """Test indexing queries with invalid format."""
        invalid_requests = [
            {},  # Missing queries
            {"queries": "not a list"},  # queries is not a list
            {"queries": [{"query": "test"}]},  # Missing rank
            {"queries": [{"rank": 1.0}]},  # Missing query
            {"queries": [{"query": "", "rank": 1.0}]},  # Empty query
        ]
        
        for invalid_request in invalid_requests:
            with self.subTest(invalid_request):
                response = requests.post(
                    f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(invalid_request)
                )
                
                self.assertIn(response.status_code, [400, 422])

    def test_suggestions_response_format(self):
        """Test that suggestions response has the correct format when successful."""
        suggestion_request = {
            "input": "test",
            "maxSuggestions": 10,
            "fuzzyEditDistance": 2,
            "minFuzzyMatchLength": 3
        }
        
        response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request)
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("suggestions", response_data)
        self.assertIn("processingTimeMs", response_data)
        self.assertIsInstance(response_data["suggestions"], list)
        self.assertIsInstance(response_data["processingTimeMs"], (int, float))

    def test_add_queries_and_get_suggestions_success(self):
        """Test complete typeahead workflow: index queries, verify stats, get suggestions, delete queries, verify stats again."""
        # First, index some queries with a common prefix
        queries_request = {
            "queries": [
                {"query": "machine learning algorithms", "rank": 10.0},
                {"query": "machine learning basics", "rank": 8.0},
                {"query": "machine learning tutorial", "rank": 6.0},
                {"query": "artificial intelligence", "rank": 9.0},
                {"query": "deep learning", "rank": 7.0}
            ]
        }
        
        # Index the queries
        index_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(queries_request)
        )
        
        # Should successfully index queries
        self.assertEqual(index_response.status_code, 200)
        index_data = index_response.json()
        self.assertEqual(index_data["indexed"], 5)
        self.assertEqual(index_data["errors"], [])
        
        # Wait a moment for indexing to complete
        import time
        time.sleep(2)
        
        # 1. Check stats after adding queries - should show 5 queries
        stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/stats",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(stats_response.status_code, 200)
        stats_data = stats_response.json()
        self.assertIn("indexedQueries", stats_data)
        self.assertEqual(stats_data["indexedQueries"], 5)
        
        # Now get suggestions for a prefix that should match
        suggestion_request = {
            "input": "machine",
            "maxSuggestions": 5
        }
        
        suggestion_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request)
        )
        
        # Should get successful response
        self.assertEqual(suggestion_response.status_code, 200)
        suggestion_data = suggestion_response.json()
        
        # Should have required fields
        self.assertIn("suggestions", suggestion_data)
        self.assertIn("processingTimeMs", suggestion_data)
        
        # Should return at least one suggestion
        suggestions = suggestion_data["suggestions"]
        self.assertIsInstance(suggestions, list)
        self.assertGreaterEqual(len(suggestions), 1)
        
        # Each suggestion should have the required structure
        for suggestion in suggestions:
            self.assertIn("query", suggestion)
            self.assertIn("relevance", suggestion)
            self.assertIsInstance(suggestion["query"], str)
            self.assertIsInstance(suggestion["relevance"], (int, float))
        
        # At least one suggestion should contain "machine"
        machine_suggestions = [s for s in suggestions if "machine" in s["query"].lower()]
        self.assertGreaterEqual(len(machine_suggestions), 1)
        
        # 2. Delete all queries
        delete_response = requests.delete(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(delete_response.status_code, 200)
        delete_data = delete_response.json()
        self.assertIn("deleted", delete_data)
        self.assertIn("message", delete_data)
        self.assertTrue(delete_data["deleted"])
        
        # Wait a moment for deletion to complete
        time.sleep(2)
        
        # 3. Check stats after deletion - should show 0 queries
        final_stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/stats",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(final_stats_response.status_code, 200)
        final_stats_data = final_stats_response.json()
        self.assertIn("indexedQueries", final_stats_data)
        self.assertEqual(final_stats_data["indexedQueries"], 0)
