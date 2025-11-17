"""
Comprehensive integration tests for recency scoring feature with Elasticsearch-compatible decay functions.

Tests all decay functions (exponential, linear, gaussian, binary) with various parameter combinations
including the new offset and decay_to parameters.
"""
import unittest
from datetime import datetime, timedelta
import math
from typing import List, Dict, Any

from tests.integ_tests.marqo_test import MarqoTestCase
from marqo.core.models.marqo_index import *
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.search import SearchContext
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.recency_parameters import RecencyParameters


class TestRecencyScoring(MarqoTestCase):
    """
    Comprehensive tests for recency scoring with all decay functions.
    Tests ES-compatible formulas with offset and decay_to parameters.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create a semi-structured index for recency testing
        # Fields will be added automatically when documents are indexed
        cls.test_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        cls.indexes = cls.create_indexes([cls.test_index])
        cls.index = cls.indexes[0]

    def _generate_test_documents(self) -> List[Dict[str, Any]]:
        """Generate test documents with various timestamps for recency testing."""
        now = datetime.now()

        # Create documents with different ages (in days)
        ages_in_days = [0, 1, 3, 5, 7, 10, 14, 20, 30, 60, 90]

        documents = []
        for i, age_days in enumerate(ages_in_days):
            timestamp = (now - timedelta(days=age_days)).timestamp()
            documents.append({
                "_id": f"doc-{i}",
                "title": f"Product {i} - {age_days} days old",
                "description": f"This is a test product released {age_days} days ago",
                "release_timestamp": timestamp,
                "category": "Electronics"
            })

        return documents

    def _add_test_documents(self):
        """Add test documents to the index."""
        documents = self._generate_test_documents()
        add_docs_params = AddDocsParams(
            index_name=self.index.name,
            docs=documents,
            tensor_fields=["title", "description"]
        )
        self.add_documents(self.config, add_docs_params)

    def _search_with_recency(
        self,
        query: str,
        recency_params: RecencyParameters
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid search with recency parameters."""
        search_result = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query,
            search_method=SearchMethod.HYBRID,
            result_count=20,
            offset=0,
            recency_parameters=recency_params
        )
        return search_result['hits']

    def _calculate_expected_exponential_score(
        self,
        age_seconds: float,
        scale_seconds: float,
        offset_seconds: float,
        decay_to: float
    ) -> float:
        """Calculate expected exponential decay score using ES formula."""
        effective_age = max(0, age_seconds - offset_seconds)
        if effective_age == 0:
            return 1.0
        # λ = ln(decay_to) / scale
        # score = max(decay_to, exp(λ × effective_age))
        lambda_val = math.log(decay_to) / scale_seconds
        score = math.exp(lambda_val * effective_age)
        return max(decay_to, score)

    def _calculate_expected_linear_score(
        self,
        age_seconds: float,
        scale_seconds: float,
        offset_seconds: float,
        decay_to: float
    ) -> float:
        """Calculate expected linear decay score using ES formula."""
        effective_age = max(0, age_seconds - offset_seconds)
        if effective_age == 0:
            return 1.0
        # s = scale / (1 - decay_to)
        # score = max(decay_to, (s - effective_age) / s)
        score = (scale_seconds - effective_age * (1.0 - decay_to)) / scale_seconds
        return max(decay_to, score)

    def _calculate_expected_gaussian_score(
        self,
        age_seconds: float,
        scale_seconds: float,
        offset_seconds: float,
        decay_to: float
    ) -> float:
        """Calculate expected gaussian decay score using ES formula."""
        effective_age = max(0, age_seconds - offset_seconds)
        if effective_age == 0:
            return 1.0
        # σ² = -scale² / (2 × ln(decay_to))
        # score = max(decay_to, exp(-effective_age² / (2σ²)))
        score = math.exp(
            -pow(effective_age, 2) * math.log(decay_to) / pow(scale_seconds, 2)
        )
        return max(decay_to, score)

    def _calculate_expected_binary_score(
        self,
        age_seconds: float,
        scale_seconds: float,
        offset_seconds: float,
        decay_to: float
    ) -> float:
        """Calculate expected binary decay score."""
        effective_age = max(0, age_seconds - offset_seconds)
        return 1.0 if effective_age < scale_seconds else decay_to

    def test_exponential_decay_no_offset(self):
        """Test exponential decay with offset=0 (backward compatibility)."""
        self._add_test_documents()

        recency_params = RecencyParameters(
            recency_field="release_timestamp",
            scale=7.0,  # 7 days
            offset=0.0,
            decay_function="exponential",
            decay_to=0.5
        )

        hits = self._search_with_recency("product", recency_params)

        # Verify we got results
        self.assertGreater(len(hits), 0)

        # Verify recency scores are present and reasonable
        for hit in hits:
            recency_score = hit.get('_recency_score')
            self.assertIsNotNone(recency_score, "Recency score should be present")
            self.assertGreaterEqual(recency_score, 0.5, "Score should not be below decay_to")
            self.assertLessEqual(recency_score, 1.0, "Score should not exceed 1.0")

        # Verify newer documents score higher
        # Find 0-day and 30-day documents
        doc_0day = next((h for h in hits if "0 days old" in h['title']), None)
        doc_30day = next((h for h in hits if "30 days old" in h['title']), None)

        if doc_0day and doc_30day:
            self.assertGreaterEqual(
                doc_0day['_recency_score'],
                doc_30day['_recency_score'],
                "Newer document should have higher recency score"
            )

    def test_exponential_decay_with_offset(self):
        """Test exponential decay with grace period (offset > 0)."""
        self._add_test_documents()

        offset_days = 3.0
        scale_days = 7.0

        recency_params = RecencyParameters(
            recency_field="release_timestamp",
            scale=scale_days,
            offset=offset_days,
            decay_function="exponential",
            decay_to=0.3
        )

        hits = self._search_with_recency("product", recency_params)

        # Find documents within and outside offset
        doc_1day = next((h for h in hits if "1 days old" in h['title']), None)
        doc_10day = next((h for h in hits if "10 days old" in h['title']), None)

        if doc_1day:
            # Document within offset should have perfect score
            self.assertAlmostEqual(
                doc_1day['_recency_score'],
                1.0,
                places=2,
                msg="Document within offset should have score of 1.0"
            )

        if doc_10day:
            # Document outside offset should have decayed score
            self.assertLess(
                doc_10day['_recency_score'],
                1.0,
                "Document outside offset should have decayed score"
            )
            self.assertGreaterEqual(
                doc_10day['_recency_score'],
                0.3,
                "Score should not fall below decay_to"
            )

    def test_linear_decay(self):
        """Test linear decay function."""
        self._add_test_documents()

        recency_params = RecencyParameters(
            recency_field="release_timestamp",
            scale=14.0,  # 14 days
            offset=0.0,
            decay_function="linear",
            decay_to=0.2
        )

        hits = self._search_with_recency("product", recency_params)

        # Verify linear decay pattern
        self.assertGreater(len(hits), 0)

        for hit in hits:
            recency_score = hit.get('_recency_score')
            self.assertIsNotNone(recency_score)
            self.assertGreaterEqual(recency_score, 0.2)
            self.assertLessEqual(recency_score, 1.0)

    def test_gaussian_decay(self):
        """Test gaussian decay function."""
        self._add_test_documents()

        recency_params = RecencyParameters(
            recency_field="release_timestamp",
            scale=10.0,  # 10 days
            offset=0.0,
            decay_function="gaussian",
            decay_to=0.4
        )

        hits = self._search_with_recency("product", recency_params)

        self.assertGreater(len(hits), 0)

        for hit in hits:
            recency_score = hit.get('_recency_score')
            self.assertIsNotNone(recency_score)
            self.assertGreaterEqual(recency_score, 0.4)
            self.assertLessEqual(recency_score, 1.0)

    def test_binary_decay(self):
        """Test binary decay (step function)."""
        self._add_test_documents()

        threshold_days = 7.0

        recency_params = RecencyParameters(
            recency_field="release_timestamp",
            scale=threshold_days,
            offset=0.0,
            decay_function="binary",
            decay_to=0.1
        )

        hits = self._search_with_recency("product", recency_params)

        # Find documents on either side of threshold
        doc_5day = next((h for h in hits if "5 days old" in h['title']), None)
        doc_10day = next((h for h in hits if "10 days old" in h['title']), None)

        if doc_5day:
            # Before threshold
            self.assertAlmostEqual(
                doc_5day['_recency_score'],
                1.0,
                places=2,
                msg="Document before threshold should have score of 1.0"
            )

        if doc_10day:
            # After threshold
            self.assertAlmostEqual(
                doc_10day['_recency_score'],
                0.1,
                places=2,
                msg="Document after threshold should have score of decay_to"
            )

    def test_binary_decay_with_offset(self):
        """Test binary decay with offset."""
        self._add_test_documents()

        offset_days = 2.0
        scale_days = 5.0

        recency_params = RecencyParameters(
            recency_field="release_timestamp",
            scale=scale_days,
            offset=offset_days,
            decay_function="binary",
            decay_to=0.2
        )

        hits = self._search_with_recency("product", recency_params)

        # Documents 0-2 days: score = 1.0 (within offset)
        # Documents 2-7 days: score = 1.0 (within offset+scale)
        # Documents 7+ days: score = 0.2 (beyond offset+scale)

        doc_1day = next((h for h in hits if "1 days old" in h['title']), None)
        doc_5day = next((h for h in hits if "5 days old" in h['title']), None)
        doc_10day = next((h for h in hits if "10 days old" in h['title']), None)

        if doc_1day:
            self.assertAlmostEqual(doc_1day['_recency_score'], 1.0, places=2)

        if doc_5day:
            self.assertAlmostEqual(doc_5day['_recency_score'], 1.0, places=2)

        if doc_10day:
            self.assertAlmostEqual(doc_10day['_recency_score'], 0.2, places=2)

    def test_decay_to_floor_behavior(self):
        """Test that scores floor at decay_to value."""
        self._add_test_documents()

        recency_params = RecencyParameters(
            recency_field="release_timestamp",
            scale=5.0,  # Small scale to ensure decay
            offset=0.0,
            decay_function="exponential",
            decay_to=0.6
        )

        hits = self._search_with_recency("product", recency_params)

        # Find very old document
        doc_90day = next((h for h in hits if "90 days old" in h['title']), None)

        if doc_90day:
            # Should be at floor (decay_to)
            self.assertAlmostEqual(
                doc_90day['_recency_score'],
                0.6,
                places=2,
                msg="Very old document should hit decay_to floor"
            )

    def test_score_at_offset_plus_scale(self):
        """Verify score reaches exactly decay_to at distance offset+scale."""
        self._add_test_documents()

        offset_days = 2.0
        scale_days = 5.0
        decay_to = 0.5

        # Test with exponential
        recency_params = RecencyParameters(
            recency_field="release_timestamp",
            scale=scale_days,
            offset=offset_days,
            decay_function="exponential",
            decay_to=decay_to
        )

        hits = self._search_with_recency("product", recency_params)

        # Document at 7 days (offset=2 + scale=5) should score exactly decay_to
        doc_7day = next((h for h in hits if "7 days old" in h['title']), None)

        if doc_7day:
            self.assertAlmostEqual(
                doc_7day['_recency_score'],
                decay_to,
                places=2,
                msg=f"Document at offset+scale should score exactly {decay_to}"
            )

    def test_decay_to_validation(self):
        """Test that decay_to must be in range (0.0, 1.0]."""
        # Valid values
        valid_params = [
            {"decay_to": 0.1},
            {"decay_to": 0.5},
            {"decay_to": 1.0},
        ]

        for params in valid_params:
            try:
                RecencyParameters(
                    recency_field="release_timestamp",
                    scale=7.0,
                    **params
                )
            except Exception as e:
                self.fail(f"Valid decay_to={params['decay_to']} should not raise: {e}")

        # Invalid values - decay_to cannot be 0 or negative
        invalid_values = [0.0, -0.1]

        for decay_to in invalid_values:
            with self.assertRaises(Exception, msg=f"decay_to={decay_to} should raise validation error"):
                RecencyParameters(
                    recency_field="release_timestamp",
                    scale=7.0,
                    decay_to=decay_to
                )

    def test_offset_validation(self):
        """Test that offset must be >= 0."""
        # Valid values
        valid_offsets = [0.0, 1.0, 10.0]

        for offset in valid_offsets:
            try:
                RecencyParameters(
                    recency_field="release_timestamp",
                    scale=7.0,
                    offset=offset
                )
            except Exception as e:
                self.fail(f"Valid offset={offset} should not raise: {e}")

        # Invalid values - negative offset
        with self.assertRaises(Exception, msg="Negative offset should raise validation error"):
            RecencyParameters(
                recency_field="release_timestamp",
                scale=7.0,
                offset=-1.0
            )

    def test_all_decay_functions_comparison(self):
        """Compare all decay functions with same parameters to verify different behavior."""
        self._add_test_documents()

        scale_days = 10.0
        offset_days = 2.0
        decay_to = 0.5

        decay_functions = ["exponential", "linear", "gaussian", "binary"]
        results = {}

        for decay_func in decay_functions:
            recency_params = RecencyParameters(
                recency_field="release_timestamp",
                scale=scale_days,
                offset=offset_days,
                decay_function=decay_func,
                decay_to=decay_to
            )

            hits = self._search_with_recency("product", recency_params)

            # Extract recency scores for 14-day old document
            doc_14day = next((h for h in hits if "14 days old" in h['title']), None)
            if doc_14day:
                results[decay_func] = doc_14day['_recency_score']

        # Verify we got results for all functions
        self.assertEqual(len(results), 4, "Should have results for all 4 decay functions")

        # Verify different functions produce different scores (except possibly at boundaries)
        # All should floor at decay_to
        for func, score in results.items():
            self.assertGreaterEqual(
                score,
                decay_to,
                f"{func} should not go below decay_to"
            )

        # Binary should be at floor (14 days > offset+scale = 12 days)
        self.assertAlmostEqual(
            results["binary"],
            decay_to,
            places=2,
            msg="Binary should be at decay_to for document beyond threshold"
        )


if __name__ == '__main__':
    unittest.main()
