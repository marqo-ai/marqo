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
        # Recency scoring is only supported for unstructured indexes
        for index_name in [self.unstructured_index_name]:
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
        # Recency scoring is only supported for unstructured indexes
        for index_name in [self.unstructured_index_name]:
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
        # Recency scoring is only supported for unstructured indexes
        index_name = self.unstructured_index_name
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
        # Recency scoring is only supported for unstructured indexes
        index_name = self.unstructured_index_name
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

    def test_structured_index_rejects_recency_parameters(self):
        """Test that structured indexes reject recency parameters with appropriate error."""
        self._add_test_documents(self.structured_index_name)

        search_body = {
            "q": "product",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "7d",
                "decayFunction": "exponential",
                "decayTo": 0.5,
                "applyInRankingPhase": "all"
            }
        }

        response = requests.post(
            f"{self._MARQO_URL}/indexes/{self.structured_index_name}/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_body)
        )

        # Should return an error status
        self.assertNotEqual(response.status_code, 200)
        error_response = response.json()
        # Check error message mentions recency is not supported for structured indexes
        self.assertIn("message", error_response)
        self.assertIn("unstructured", error_response["message"].lower())

    def test_grow_function_penalizes_future_documents(self):
        """Test that growFrom/growFunction penalizes documents with future timestamps."""
        index_name = self.unstructured_index_name

        # Add documents with future timestamps
        now = datetime.now()
        docs = [
            {
                "_id": "present",
                "title": "laptop computer device",
                "description": "tech gadget product",
                "created_at": int(now.timestamp())
            },
            {
                "_id": "near-future",
                "title": "laptop computer device",
                "description": "tech gadget product",
                "created_at": int((now + timedelta(days=3)).timestamp())
            },
            {
                "_id": "far-future",
                "title": "laptop computer device",
                "description": "tech gadget product",
                "created_at": int((now + timedelta(days=30)).timestamp())
            }
        ]

        self.client.index(index_name).add_documents(docs, tensor_fields=["title", "description"])

        # Search with grow function enabled
        search_body = {
            "q": "laptop",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "7d",
                "offset": "0d",
                "decayFunction": "exponential",
                "decayTo": 0.5,
                "growFrom": 0.3,  # Future docs start at 0.3 and grow toward 1.0
                "growFunction": "exponential",
                "growScale": "7d",
                "growOffset": "0d",
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

        present_doc = next((h for h in hits if h["_id"] == "present"), None)
        near_future_doc = next((h for h in hits if h["_id"] == "near-future"), None)
        far_future_doc = next((h for h in hits if h["_id"] == "far-future"), None)

        # All documents should be found
        self.assertIsNotNone(present_doc, "Present document should be in results")
        self.assertIsNotNone(near_future_doc, "Near-future document should be in results")
        self.assertIsNotNone(far_future_doc, "Far-future document should be in results")

        # Present document should score highest (score = 1.0)
        # Far future document should score lowest (penalized most)
        self.assertGreater(
            present_doc["_score"],
            far_future_doc["_score"],
            f"Present document (score={present_doc['_score']:.6f}) should score higher "
            f"than far-future document (score={far_future_doc['_score']:.6f})"
        )

        # Near future should be between present and far future
        self.assertGreater(
            near_future_doc["_score"],
            far_future_doc["_score"],
            f"Near-future document (score={near_future_doc['_score']:.6f}) should score higher "
            f"than far-future document (score={far_future_doc['_score']:.6f})"
        )

    def test_grow_offset_creates_future_plateau(self):
        """Test that growOffset creates a plateau where future docs get score 1.0."""
        index_name = self.unstructured_index_name

        now = datetime.now()
        docs = [
            {
                "_id": "present",
                "title": "camera photography device",
                "description": "digital equipment gear",
                "created_at": int(now.timestamp())
            },
            {
                "_id": "within-grow-offset",
                "title": "camera photography device",
                "description": "digital equipment gear",
                "created_at": int((now + timedelta(days=2)).timestamp())  # Within 3-day offset
            },
            {
                "_id": "beyond-grow-offset",
                "title": "camera photography device",
                "description": "digital equipment gear",
                "created_at": int((now + timedelta(days=10)).timestamp())  # Beyond 3-day offset
            }
        ]

        self.client.index(index_name).add_documents(docs, tensor_fields=["title", "description"])

        # Search with 3-day grow offset (plateau)
        search_body = {
            "q": "camera",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "7d",
                "offset": "0d",
                "decayFunction": "exponential",
                "decayTo": 0.5,
                "growFrom": 0.2,
                "growFunction": "exponential",
                "growScale": "7d",
                "growOffset": "3d",  # 3-day plateau for future timestamps
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

        present = next((h for h in hits if h["_id"] == "present"), None)
        within_offset = next((h for h in hits if h["_id"] == "within-grow-offset"), None)
        beyond_offset = next((h for h in hits if h["_id"] == "beyond-grow-offset"), None)

        self.assertIsNotNone(present)
        self.assertIsNotNone(within_offset)
        self.assertIsNotNone(beyond_offset)

        # Present and within-offset should have similar scores (both get 1.0 recency)
        # Beyond-offset should be penalized
        self.assertGreater(
            within_offset["_score"],
            beyond_offset["_score"],
            f"Doc within grow offset (score={within_offset['_score']:.6f}) should score higher "
            f"than doc beyond offset (score={beyond_offset['_score']:.6f})"
        )

    def test_add_to_score_weight_additive_mode(self):
        """Test that addToScoreWeight applies recency as additive instead of multiplicative."""
        index_name = self.unstructured_index_name

        now = datetime.now()
        docs = [
            {
                "_id": "recent-additive",
                "title": "headphones audio device",
                "description": "music listening gear",
                "created_at": int(now.timestamp())
            },
            {
                "_id": "old-additive",
                "title": "headphones audio device",
                "description": "music listening gear",
                "created_at": int((now - timedelta(days=30)).timestamp())
            }
        ]

        self.client.index(index_name).add_documents(docs, tensor_fields=["title", "description"])

        # Search with additive recency scoring
        search_body = {
            "q": "headphones",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "7d",
                "offset": "0d",
                "decayFunction": "exponential",
                "decayTo": 0.1,
                "addToScoreWeight": 0.5,  # Add recency_score * 0.5 to the base score
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

        recent = next((h for h in hits if h["_id"] == "recent-additive"), None)
        old = next((h for h in hits if h["_id"] == "old-additive"), None)

        self.assertIsNotNone(recent, "Recent document should be in results")
        self.assertIsNotNone(old, "Old document should be in results")

        # Recent document should score higher due to additive recency boost
        self.assertGreater(
            recent["_score"],
            old["_score"],
            f"Recent document (score={recent['_score']:.6f}) should score higher "
            f"than old document (score={old['_score']:.6f}) with additive recency"
        )

        # The difference should be noticeable since we're adding recency_score * 0.5
        # Recent doc: base_score + 1.0 * 0.5
        # Old doc: base_score + 0.1 * 0.5
        # The absolute difference depends on base scores from hybrid search
        score_difference = recent["_score"] - old["_score"]
        self.assertGreater(
            score_difference,
            0.01,  # Should have meaningful difference due to additive boost
            f"Score difference ({score_difference:.6f}) should be meaningful with additive scoring"
        )

    def test_gaussian_decay_function(self):
        """Test that gaussian decay function works correctly."""
        index_name = self.unstructured_index_name
        self._add_test_documents(index_name)

        search_body = {
            "q": "product",
            "searchMethod": "HYBRID",
            "limit": 10,
            "recencyParameters": {
                "recencyField": "created_at",
                "scale": "14d",
                "offset": "0d",
                "decayFunction": "gaussian",
                "decayTo": 0.3,
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

        recent_doc = next((h for h in hits if h["_id"] == "recent"), None)
        month_old_doc = next((h for h in hits if h["_id"] == "month-old"), None)

        # Verify gaussian decay gives recent docs higher scores
        if recent_doc and month_old_doc:
            self.assertGreater(
                recent_doc["_score"],
                month_old_doc["_score"],
                "Gaussian decay should give recent documents higher scores"
            )
