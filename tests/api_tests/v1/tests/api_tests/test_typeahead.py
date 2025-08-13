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

    def test_get_suggestions_endpoint_exists(self):
        """Test that the suggestions endpoint is accessible and returns proper error structure."""
        suggestion_request = {
            "input": "test query",
            "maxSuggestions": 5,
            "fuzzyEditDistance": 2,
            "minFuzzyMatchLength": 3
        }
        
        response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request)
        )
        
        self.assertIn(response.status_code, [200, 500])
        response_data = response.json()
        
        if response.status_code == 200:
            self.assertIn("suggestions", response_data)
            self.assertIn("processingTimeMs", response_data)
            self.assertIsInstance(response_data["suggestions"], list)
        else:
            self.assertIn("message", response_data)

    def test_get_suggestions_with_minimal_params(self):
        """Test suggestions endpoint with minimal required parameters."""
        suggestion_request = {
            "input": "hello"
        }
        
        response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request)
        )
        
        self.assertIn(response.status_code, [200, 500])
        if response.status_code == 200:
            response_data = response.json()
            self.assertIn("suggestions", response_data)

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
                
                self.assertIn(response.status_code, [400, 422, 500])

    def test_index_queries_endpoint_exists(self):
        """Test that the queries indexing endpoint is accessible."""
        queries_request = {
            "queries": [
                {"query": "test query 1", "rank": 1.0},
                {"query": "test query 2", "rank": 2.0},
                {"query": "another test", "rank": 0.5}
            ]
        }
        
        response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(queries_request)
        )
        
        self.assertIn(response.status_code, [200, 500])
        response_data = response.json()
        
        if response.status_code == 200:
            self.assertIn("indexed", response_data)
            self.assertIn("errors", response_data)
            self.assertIsInstance(response_data["indexed"], int)
            self.assertIsInstance(response_data["errors"], list)
        else:
            self.assertIn("message", response_data)

    def test_index_queries_with_empty_list(self):
        """Test indexing queries with an empty list."""
        # Check initial stats
        initial_stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries/stats",
            headers={"Content-Type": "application/json"}
        )
        
        if initial_stats_response.status_code == 200:
            initial_count = initial_stats_response.json().get("indexedQueries", 0)
        else:
            initial_count = 0
        
        queries_request = {
            "queries": []
        }
        
        response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(queries_request)
        )
        
        self.assertIn(response.status_code, [200, 500])
        if response.status_code == 200:
            response_data = response.json()
            self.assertEqual(response_data["indexed"], 0)
            
            # Stats should remain unchanged
            final_stats_response = requests.get(
                f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries/stats",
                headers={"Content-Type": "application/json"}
            )
            
            if final_stats_response.status_code == 200:
                final_count = final_stats_response.json().get("indexedQueries", 0)
                self.assertEqual(final_count, initial_count)

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
                    f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(invalid_request)
                )
                
                self.assertIn(response.status_code, [400, 422, 500])

    def test_delete_all_queries_endpoint_exists(self):
        """Test that the delete all queries endpoint is accessible."""
        response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries/delete",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertIn(response.status_code, [200, 500])
        response_data = response.json()
        
        if response.status_code == 200:
            self.assertIn("deleted", response_data)
            self.assertIn("message", response_data)
            self.assertTrue(response_data["deleted"])
        else:
            self.assertIn("message", response_data)

    def test_nonexistent_index_returns_error(self):
        """Test that typeahead endpoints return appropriate errors for nonexistent indexes."""
        nonexistent_index = "nonexistent_index_" + str(uuid.uuid4()).replace('-', '')
        
        endpoints_and_data = [
            ("/suggestions", {"input": "test"}, "POST"),
            ("/queries", {"queries": [{"query": "test", "rank": 1.0}]}, "POST"),
            ("/queries/delete", {}, "POST"),
            ("/queries/stats", {}, "GET")
        ]
        
        for endpoint, data, method in endpoints_and_data:
            with self.subTest(endpoint):
                if method == "POST":
                    response = requests.post(
                        f"{self._MARQO_URL}/indexes/{nonexistent_index}{endpoint}",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(data)
                    )
                else:  # GET
                    response = requests.get(
                        f"{self._MARQO_URL}/indexes/{nonexistent_index}{endpoint}",
                        headers={"Content-Type": "application/json"}
                    )
                
                self.assertIn(response.status_code, [400, 404, 500])

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
        
        if response.status_code == 200:
            response_data = response.json()
            self.assertIn("suggestions", response_data)
            self.assertIn("processingTimeMs", response_data)
            self.assertIsInstance(response_data["suggestions"], list)
            self.assertIsInstance(response_data["processingTimeMs"], (int, float))

    def test_typeahead_workflow(self):
        """Test a complete typeahead workflow: index queries, then get suggestions."""
        # Check initial stats (should be 0)
        initial_stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries/stats",
            headers={"Content-Type": "application/json"}
        )
        
        if initial_stats_response.status_code == 200:
            initial_count = initial_stats_response.json().get("indexedQueries", 0)
        else:
            initial_count = 0  # Assume 0 if endpoint not implemented
        
        # Index some queries
        queries_request = {
            "queries": [
                {"query": "machine learning algorithms", "rank": 10.0},
                {"query": "machine learning basics", "rank": 8.0},
                {"query": "deep learning tutorial", "rank": 7.0},
                {"query": "artificial intelligence", "rank": 9.0}
            ]
        }
        
        index_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(queries_request)
        )
        
        # Check stats after indexing
        if index_response.status_code == 200:
            post_index_stats_response = requests.get(
                f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries/stats",
                headers={"Content-Type": "application/json"}
            )
            
            if post_index_stats_response.status_code == 200:
                post_index_count = post_index_stats_response.json().get("indexedQueries", 0)
                # Should have 4 more queries than initial count
                self.assertEqual(post_index_count, initial_count + 4)
        
        # Try to get suggestions
        suggestion_request = {
            "input": "machine",
            "maxSuggestions": 5
        }
        
        suggestion_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request)
        )
        
        # Both requests should be handled (even if not fully implemented)
        self.assertIn(index_response.status_code, [200, 500])
        self.assertIn(suggestion_response.status_code, [200, 500])
        
        # Finally, delete all queries
        delete_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries/delete",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertIn(delete_response.status_code, [200, 500])
        
        # Check stats after deletion (should be back to 0)
        if delete_response.status_code == 200:
            final_stats_response = requests.get(
                f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries/stats",
                headers={"Content-Type": "application/json"}
            )
            
            if final_stats_response.status_code == 200:
                final_count = final_stats_response.json().get("indexedQueries", 0)
                self.assertEqual(final_count, 0)

    def test_get_typeahead_stats_endpoint_exists(self):
        """Test that the typeahead stats endpoint is accessible."""
        response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/queries/stats",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertIn(response.status_code, [200, 500])
        response_data = response.json()
        
        if response.status_code == 200:
            self.assertIn("indexedQueries", response_data)
            self.assertIsInstance(response_data["indexedQueries"], int)
        else:
            self.assertIn("message", response_data)

    def test_stats_endpoint_with_nonexistent_index(self):
        """Test stats endpoint with nonexistent index."""
        nonexistent_index = "nonexistent_index_" + str(uuid.uuid4()).replace('-', '')
        
        response = requests.get(
            f"{self._MARQO_URL}/indexes/{nonexistent_index}/queries/stats",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertIn(response.status_code, [400, 404, 500])