"""
API tests for recency scoring feature using direct HTTP requests.

Tests the recency scoring feature through the HTTP API endpoint.
Note: Using direct HTTP requests instead of marqo client since recency parameters
haven't been added to the client yet.
"""
import json
import uuid
from datetime import datetime, timedelta

import requests
from tests.marqo_test import MarqoTestCase


class TestRecencyScoring(MarqoTestCase):
    """Test cases for recency scoring functionality via HTTP API."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.structured_index_name = "structured_recency_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_name = "unstructured_recency_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["lexical_search"]},
                    {"name": "description", "type": "text", "features": ["lexical_search"]},
                    {"name": "created_at", "type": "long", "features": ["score_modifier"]},
                ],
                "tensorFields": ["title", "description"]
            },
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2"
            }
        ])

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]

    def _add_test_documents(self, index_name):
        """Add test documents with different timestamps."""
        now = datetime.now()

        docs = [
            {
                "_id": "recent",
                "title": "product electronics",
                "description": "test product",
                "created_at": int(now.timestamp())
            },
            {
                "_id": "week-old",
                "title": "product electronics",
                "description": "test product",
                "created_at": int((now - timedelta(days=7)).timestamp())
            },
            {
                "_id": "month-old",
                "title": "product electronics",
                "description": "test product",
                "created_at": int((now - timedelta(days=30)).timestamp())
            }
        ]

        # Add documents using client
        if "unstructured" in index_name:
            self.client.index(index_name).add_documents(docs, tensor_fields=["title", "description"])
        else:
            self.client.index(index_name).add_documents(docs)

        return docs

    def test_basic_exponential_decay_increases_recent_doc_score(self):
        """Test that exponential decay gives higher scores to recent documents."""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index=index_name):
                # Add test documents
                self._add_test_documents(index_name)

                # Search with recency parameters
                search_body = {
                    "q": "product",
                    "searchMethod": "HYBRID",
                    "limit": 10,
                    "recencyParameters": {
                        "recencyField": "created_at",
                        "scale": "7d",
                        "offset": "0d",
                        "decayFunction": "exponential",
                        "decayTo": 0.5,
                        "applyInRankingPhase": "all"
                    }
                }

                response = requests.post(
                    f"{self._MARQO_URL}/indexes/{index_name}/search",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(search_body)
                )

                # Verify response is successful
                self.assertEqual(response.status_code, 200)
                response_data = response.json()

                # Verify response structure
                self.assertIn("hits", response_data)
                self.assertIn("processingTimeMs", response_data)
                self.assertIsInstance(response_data["hits"], list)

                # Verify we got results
                hits = response_data["hits"]
                self.assertGreater(len(hits), 0)

                # Find our specific documents
                recent_doc = next((h for h in hits if h["_id"] == "recent"), None)
                month_old_doc = next((h for h in hits if h["_id"] == "month-old"), None)

                # Verify recent document scores higher than old document
                if recent_doc and month_old_doc:
                    self.assertGreater(
                        recent_doc["_score"],
                        month_old_doc["_score"],
                        "Recent document should score higher than month-old document with recency boost"
                    )

    def test_exponential_decay_with_offset_grace_period(self):
        """Test that offset creates a grace period where documents get full score."""
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index=index_name):
                # Add documents with specific timestamps
                # Use identical content so text relevance is the same
                now = datetime.now()
                docs = [
                    {
                        "_id": "within-offset",
                        "title": "smartphone device gadget",
                        "description": "electronics technology product",
                        "created_at": int((now - timedelta(days=1)).timestamp())
                    },
                    {
                        "_id": "outside-offset",
                        "title": "smartphone device gadget",
                        "description": "electronics technology product",
                        "created_at": int((now - timedelta(days=30)).timestamp())
                    }
                ]

                if "unstructured" in index_name:
                    self.client.index(index_name).add_documents(docs, tensor_fields=["title", "description"])
                else:
                    self.client.index(index_name).add_documents(docs)

                # Search with offset of 3 days, aggressive decay
                search_body = {
                    "q": "smartphone",
                    "searchMethod": "HYBRID",
                    "limit": 10,
                    "recencyParameters": {
                        "recencyField": "created_at",
                        "scale": "7d",
                        "offset": "3d",  # 3-day grace period
                        "decayFunction": "exponential",
                        "decayTo": 0.1,  # More aggressive decay
                        "applyInRankingPhase": "all"
                    }
                }

                response = requests.post(
                    f"{self._MARQO_URL}/indexes/{index_name}/search",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(search_body)
                )

                self.assertEqual(response.status_code, 200)
                hits = response.json()["hits"]

                within_offset = next((h for h in hits if h["_id"] == "within-offset"), None)
                outside_offset = next((h for h in hits if h["_id"] == "outside-offset"), None)

                # Both documents should be found
                self.assertIsNotNone(within_offset, "Document within offset should be in results")
                self.assertIsNotNone(outside_offset, "Document outside offset should be in results")

                # Document within offset should score higher due to recency
                # Since content is identical, recency is the differentiating factor
                if within_offset and outside_offset:
                    self.assertGreater(
                        within_offset["_score"],
                        outside_offset["_score"],
                        f"Document within offset (score={within_offset['_score']:.6f}) should score higher "
                        f"than 30-day old document (score={outside_offset['_score']:.6f})"
                    )

    def test_different_decay_functions_work(self):
        """Test that different decay functions (linear, binary) work through API."""
        index_name = self.structured_index_name
        self._add_test_documents(index_name)

        # Test linear decay
        linear_body = {
            "q": "product",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "14d",
                "offset": "0d",
                "decayFunction": "linear",
                "decayTo": 0.2,
                "applyInRankingPhase": "all"
            }
        }

        response_linear = requests.post(
            f"{self._MARQO_URL}/indexes/{index_name}/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(linear_body)
        )

        self.assertEqual(response_linear.status_code, 200)
        self.assertIn("hits", response_linear.json())

        # Test binary decay
        binary_body = {
            "q": "product",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "7d",
                "offset": "0d",
                "decayFunction": "binary",
                "decayTo": 0.1,
                "applyInRankingPhase": "all"
            }
        }

        response_binary = requests.post(
            f"{self._MARQO_URL}/indexes/{index_name}/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(binary_body)
        )

        self.assertEqual(response_binary.status_code, 200)
        self.assertIn("hits", response_binary.json())

    def test_duration_format_equivalence(self):
        """Test that '7d' and '168h' produce equivalent results."""
        index_name = self.structured_index_name
        self._add_test_documents(index_name)

        # Search with days format
        search_days = {
            "q": "product",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "7d",
                "offset": "0d",
                "decayFunction": "exponential",
                "decayTo": 0.5,
                "applyInRankingPhase": "all"
            }
        }

        response_days = requests.post(
            f"{self._MARQO_URL}/indexes/{index_name}/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_days)
        )

        # Search with hours format (168h = 7d)
        search_hours = {
            "q": "product",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "168h",
                "offset": "0h",
                "decayFunction": "exponential",
                "decayTo": 0.5,
                "applyInRankingPhase": "all"
            }
        }

        response_hours = requests.post(
            f"{self._MARQO_URL}/indexes/{index_name}/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_hours)
        )

        self.assertEqual(response_days.status_code, 200)
        self.assertEqual(response_hours.status_code, 200)

        hits_days = response_days.json()["hits"]
        hits_hours = response_hours.json()["hits"]

        # Should have same number of results
        self.assertEqual(len(hits_days), len(hits_hours))

        # Scores should be approximately equal
        for i in range(len(hits_days)):
            self.assertAlmostEqual(
                hits_days[i]["_score"],
                hits_hours[i]["_score"],
                places=5,
                msg=f"Scores should be equal for equivalent duration formats (doc {i})"
            )
