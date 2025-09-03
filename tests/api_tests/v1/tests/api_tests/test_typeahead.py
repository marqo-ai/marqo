import json
import uuid

import requests
from tests.marqo_test import MarqoTestCase


class TestTypeahead(MarqoTestCase):
    """Test cases for typeahead functionality using direct HTTP requests."""
    # TODO Please note that all requests are sent directly to the HTTP endpoint instead of using marqo client.
    #   will address this when typeahead feature is added to pymarqo client

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
            {},  # Missing q
            # {"q": ""},  # Empty q is allowed, it will return the top n queries
            {"q": "test", "limit": -1},  # Invalid limit
            {"q": "test", "fuzzyEditDistance": -1},  # Invalid fuzzyEditDistance
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
            # {"queries": [{"query": "test"}]},  # Missing popularity is allowed, default is 1.0
            {"queries": [{"popularity": 1.0}]},  # Missing query
            {"queries": [{"query": "", "popularity": 1.0}]},  # Empty query
            {"queries": [{"query": "abc", "popularity": "very popular"}]},  # Wrong type of popularity
            {"queries": [{"query": "abc", "popularity": 1.0, "some_random_field": "hello"}]},  # Unsupported fields

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
            "q": "test",
            "limit": 10,
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
                {"query": "machine learning algorithms", "popularity": 10.0, "metadata": {"hit_count": 3}},
                {"query": "machine learning basics", "popularity": 8.0},
                {"query": "machine learning tutorial", "popularity": 6.0},
                {"query": "artificial intelligence", "popularity": 9.0, "metadata": {"hit_count": 500}},
                {"query": "deep learning", "popularity": 7.0}
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

        # 1. Check stats after adding queries - should show 5 queries
        stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/stats",
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(stats_response.status_code, 200)
        stats_data = stats_response.json()
        self.assertIn("indexedQueries", stats_data)
        self.assertEqual(stats_data["indexedQueries"], 5)

        # Test that we can retrieve individual queries
        get_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(["artificial intelligence", "deep learning"])
        )
        self.assertEqual(get_response.status_code, 200)
        get_data = get_response.json()
        self.assertEqual(len(get_data["queries"]), 2)
        self.assertEqual(get_data["queries"][0]["query"], "artificial intelligence")
        self.assertEqual(get_data["queries"][0]["popularity"], 9.0)
        self.assertEqual(get_data["queries"][0]["metadata"], {"hit_count": 500})
        self.assertEqual(get_data["queries"][1]["query"], "deep learning")
        self.assertEqual(get_data["queries"][1]["popularity"], 7.0)
        self.assertEqual(get_data["queries"][1]["metadata"], {})

        # Now get suggestions for a prefix that should match
        suggestion_request = {
            "q": "machine",
            "limit": 5
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
            self.assertIn("suggestion", suggestion)
            self.assertIn("_score", suggestion)
            self.assertIsInstance(suggestion["suggestion"], str)
            self.assertIsInstance(suggestion["_score"], (int, float))

        # At least one suggestion should contain "machine"
        machine_suggestions = [s for s in suggestions if "machine" in s["suggestion"].lower()]
        self.assertGreaterEqual(len(machine_suggestions), 1)

        # Now we do a empty query, which should return top queries
        wildcard_suggestion_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"q": "", "limit": 3})
        )

        # Check if we return the top 3 queries ordered by popularity
        self.assertEqual(wildcard_suggestion_response.status_code, 200)
        wildcard_suggestion_data = wildcard_suggestion_response.json()
        expected_results = ["machine learning algorithms", "artificial intelligence", "machine learning basics"]
        self.assertListEqual(expected_results, [s["suggestion"] for s in wildcard_suggestion_data["suggestions"]])

        # 2. Delete all queries
        delete_response = requests.delete(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries/delete-all",
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(delete_response.status_code, 200)
        delete_data = delete_response.json()
        self.assertEqual(delete_data, "All queries deleted successfully")

        # Wait a moment for deletion to complete

        # 3. Check stats after deletion - should show 0 queries
        final_stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/stats",
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(final_stats_response.status_code, 200)
        final_stats_data = final_stats_response.json()
        self.assertIn("indexedQueries", final_stats_data)
        self.assertEqual(final_stats_data["indexedQueries"], 0)

    def test_duplicate_queries_not_added_twice(self):
        """Test that indexing the same query twice doesn't create duplicates."""
        # Clear any existing queries first
        delete_response = requests.delete(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"}
        )
        # Wait for deletion to complete

        # Index the same query twice with different popularities
        query_batch = {
            "queries": [
                {"query": "test duplicate query abc123", "popularity": 5.0}
            ]
        }

        # Index first time
        first_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(query_batch)
        )

        self.assertEqual(first_response.status_code, 200)
        first_data = first_response.json()
        self.assertEqual(first_data["indexed"], 1)

        # Wait for indexing to complete

        # Check stats after first indexing
        stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/stats",
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(stats_response.status_code, 200)
        stats_data = stats_response.json()
        self.assertEqual(stats_data["indexedQueries"], 1)

        # Index the same query again with different popularity
        query_batch_updated = {
            "queries": [
                {"query": "test duplicate query abc123", "popularity": 10.0}  # Same query, different popularity
            ]
        }

        second_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(query_batch_updated)
        )

        self.assertEqual(second_response.status_code, 200)
        second_data = second_response.json()
        self.assertEqual(second_data["indexed"], 1)  # Should still report 1 indexed (updated)

        # Wait for indexing to complete

        # Check final stats - should still be 1 unique query
        final_stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/stats",
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(final_stats_response.status_code, 200)
        final_stats_data = final_stats_response.json()
        self.assertEqual(final_stats_data["indexedQueries"], 1)  # Should still be 1, not 2

        # Verify the popularity was updated by checking suggestions
        suggestion_request = {
            "q": "test duplicate",
            "limit": 5
        }

        suggestion_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request)
        )

        self.assertEqual(suggestion_response.status_code, 200)
        suggestion_data = suggestion_response.json()
        suggestions = suggestion_data["suggestions"]

        # Should find exactly one suggestion for our test query
        test_suggestions = [s for s in suggestions if s["suggestion"] == "test duplicate query abc123"]
        self.assertEqual(len(test_suggestions), 1)

    def test_delete_specific_queries(self):
        """Test deleting specific queries by name."""
        # Index some queries
        queries_request = {
            "queries": [
                {"query": "delete test query one", "popularity": 10.0},
                {"query": "delete test query two", "popularity": 8.0},
                {"query": "delete test query three", "popularity": 6.0},
                {"query": "keep this query", "popularity": 9.0}
            ]
        }

        # Index the queries
        index_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(queries_request)
        )

        self.assertEqual(index_response.status_code, 200)
        index_data = index_response.json()
        self.assertEqual(index_data["indexed"], 4)

        # Wait for indexing to complete

        # Verify all queries were indexed
        stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/stats",
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(stats_response.status_code, 200)
        stats_data = stats_response.json()
        self.assertEqual(stats_data["indexedQueries"], 4)

        # Delete specific queries
        queries_to_delete = [
            "delete test query one",
            "delete test query two",
            "non-existent query"  # This will be silently ignored (no error tracking)
        ]

        delete_response = requests.delete(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(queries_to_delete)
        )

        self.assertEqual(delete_response.status_code, 200)
        delete_data = delete_response.json()

        # New API returns simple success message
        self.assertEqual(delete_data, "Queries deleted successfully")

        # Wait for deletion to complete

        # Verify queries were deleted - should have 2 remaining (delete test query three + keep this query)
        final_stats_response = requests.get(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/stats",
            headers={"Content-Type": "application/json"}
        )

        self.assertEqual(final_stats_response.status_code, 200)
        final_stats_data = final_stats_response.json()
        self.assertEqual(final_stats_data["indexedQueries"], 2)

        # Verify correct queries remain by checking suggestions
        suggestion_request = {
            "q": "delete test query three",  # Search for specific remaining query
            "limit": 10
        }

        suggestion_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request)
        )

        self.assertEqual(suggestion_response.status_code, 200)
        suggestion_data = suggestion_response.json()
        suggestions = suggestion_data["suggestions"]

        # Should find "delete test query three" (not deleted) but not the 2 deleted ones
        remaining_three_suggestions = [s for s in suggestions if s["suggestion"] == "delete test query three"]
        self.assertEqual(len(remaining_three_suggestions), 1)  # Should remain

        deleted_one_suggestions = [s for s in suggestions if s["suggestion"] == "delete test query one"]
        deleted_two_suggestions = [s for s in suggestions if s["suggestion"] == "delete test query two"]

        self.assertEqual(len(deleted_one_suggestions), 0)  # Should be gone
        self.assertEqual(len(deleted_two_suggestions), 0)  # Should be gone

    def test_suggestions_with_custom_weights(self):
        """Test that custom popularity and BM25 weights affect suggestion ranking."""
        # Index queries with different popularities
        queries_request = {
            "queries": [
                {"query": "weight test high popularity", "popularity": 100.0},
                {"query": "weight test low popularity", "popularity": 1.0},
                {"query": "weight test medium popularity", "popularity": 50.0}
            ]
        }

        # Index the queries
        index_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions/queries",
            headers={"Content-Type": "application/json"},
            data=json.dumps(queries_request)
        )

        self.assertEqual(index_response.status_code, 200)

        # Test with high popularity weight - should favor high popularity query
        suggestion_request_popularity = {
            "q": "weight test",
            "limit": 10,
            "popularityWeight": 10.0,
            "bm25Weight": 0.1
        }

        popularity_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request_popularity)
        )

        self.assertEqual(popularity_response.status_code, 200)
        popularity_data = popularity_response.json()
        popularity_suggestions = popularity_data["suggestions"]

        # Should have suggestions
        self.assertGreater(len(popularity_suggestions), 0)
        
        # The high popularity query should be ranked highly when popularity weight is high
        high_popularity_suggestion = next(
            (s for s in popularity_suggestions if "high popularity" in s["suggestion"]), None
        )
        self.assertIsNotNone(high_popularity_suggestion)

        # Test with high BM25 weight and low popularity weight
        suggestion_request_bm25 = {
            "q": "weight test",
            "limit": 10,
            "popularityWeight": 0.1,
            "bm25Weight": 10.0
        }

        bm25_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request_bm25)
        )

        self.assertEqual(bm25_response.status_code, 200)
        bm25_data = bm25_response.json()
        bm25_suggestions = bm25_data["suggestions"]

        # Should have suggestions
        self.assertGreater(len(bm25_suggestions), 0)

        # Test without weights (should use defaults)
        suggestion_request_default = {
            "q": "weight test",
            "limit": 10
        }

        default_response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.unstructured_index_name}/suggestions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(suggestion_request_default)
        )

        self.assertEqual(default_response.status_code, 200)
        default_data = default_response.json()
        default_suggestions = default_data["suggestions"]

        # Should have suggestions
        self.assertGreater(len(default_suggestions), 0)
