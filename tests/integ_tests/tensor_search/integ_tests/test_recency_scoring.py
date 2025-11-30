"""
Comprehensive integration tests for recency scoring feature.

Tests all decay functions (exponential, linear, gaussian, binary) with various parameter
combinations, ranking phases, hybrid search configurations, and feature combinations.
"""
import time
import unittest
from datetime import datetime, timedelta

import math

from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.recency_parameters import RecencyParameters
from marqo.tensor_search.models.relevance_cutoff_model import (
    RelevanceCutoffModel, RelevanceCutoffMethod, MeanStdParameters
)
from marqo.tensor_search.models.sort_by_model import SortByModel, SortByField
from tests.integ_tests.marqo_test import MarqoTestCase


class TestRecencyScoring(MarqoTestCase):
    """
    Comprehensive integration tests for recency scoring feature.

    Test coverage:
    - All decay functions (exponential, linear, gaussian, binary)
    - Scale/offset combinations
    - decay_to floor values
    - Apply in ranking phase options
    - Retrieval/ranking method combinations
    - Feature combinations (relevance cutoff, sortBy, collapsing)
    - Negative cases (wrong index type, search method, etc.)
    - Parameter validation
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # 1. Main semi-structured index for most tests
        cls.main_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        # 2. Semi-structured index with collapse field
        cls.collapse_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            collapse_fields=[CollapseField(name="parent_id", minGroups=3)]
        )

        # 3. Structured index for negative test
        cls.structured_index_request = cls.structured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2'),
            fields=[
                FieldRequest(name="title", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch]),
                FieldRequest(name="description", type=FieldType.Text,
                             features=[FieldFeature.LexicalSearch]),
                FieldRequest(name="timestamp", type=FieldType.Float),
            ],
            tensor_fields=["title"]
        )

        cls.indexes = cls.create_indexes([
            cls.main_index_request,
            cls.collapse_index_request,
            cls.structured_index_request
        ])

        cls.main_index = cls.indexes[0]
        cls.collapse_index = cls.indexes[1]
        cls.structured_index = cls.indexes[2]

    # ============== Helper Methods ==============
    def _generate_shared_documents(self) -> List[Dict[str, Any]]:
        """Generate shared documents with various ages and attributes."""
        now = datetime.now()

        # Document ages: 0, 1, 3, 5, 7, 10, 14, 30, 60, 90 days
        ages_in_days = [0, 1, 3, 5, 7, 10, 14, 30, 60, 90]

        documents = []
        for i, age_days in enumerate(ages_in_days):
            timestamp = (now - timedelta(days=age_days)).timestamp()
            documents.append({
                "_id": f"doc-{age_days}d",
                "title": "product item",
                "description": f"test product {age_days} days old",
                "timestamp": timestamp,
                "sort_value": 100 - age_days,  # Higher for newer docs
                "parent_id": f"group-{chr(65 + i % 5)}",  # A-E rotation
                "mult": 1.0 + (i % 3) * 0.5,  # 1.0, 1.5, 2.0
            })

        # Special: Document without timestamp field
        documents.append({
            "_id": "doc-no-ts",
            "title": "product item",
            "description": "product without timestamp",
            "sort_value": 0,
            "parent_id": "group-F",
            "mult": 1.0,
        })

        return documents

    def _add_shared_documents(self, index=None):
        """Add shared documents to the specified or main index."""
        if index is None:
            index = self.main_index
        documents = self._generate_shared_documents()
        add_docs_params = AddDocsParams(
            index_name=index.name,
            docs=documents,
            tensor_fields=["title", "description"]
        )
        self.add_documents(self.config, add_docs_params)
        # Allow time for Vespa to fully index the documents including score_modifiers
        time.sleep(1)

    def _add_docs_to_structured_index(self):
        """Add documents to the structured index for negative tests."""
        now = datetime.now()
        documents = [
            {
                "_id": "struct-doc-1",
                "title": "product item",
                "description": "test product",
                "timestamp": now.timestamp(),
            }
        ]
        add_docs_params = AddDocsParams(
            index_name=self.structured_index.name,
            docs=documents,
            # Note: tensor_fields must not be specified for structured indexes
        )
        self.add_documents(self.config, add_docs_params)

    def _search_with_recency(
        self,
        query: str,
        params: RecencyParameters,
        index=None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with recency parameters."""
        if index is None:
            index = self.main_index
        result = tensor_search.search(
            config=self.config,
            index_name=index.name,
            text=query,
            search_method=SearchMethod.HYBRID,
            recency_parameters=params,
            result_count=20
        )
        return result['hits']

    # ============== Score Calculation Helpers ==============
    def _parse_duration_to_seconds(self, duration: str) -> float:
        """Parse duration string (e.g., '7d', '24h') to seconds."""
        if duration.endswith('d'):
            return float(duration[:-1]) * 24 * 60 * 60
        elif duration.endswith('h'):
            return float(duration[:-1]) * 60 * 60
        else:
            raise ValueError(f"Invalid duration format: {duration}")

    def _calculate_expected_score(
        self,
        age_seconds: float,
        scale: str,
        offset: str,
        decay_function: str,
        decay_to: float
    ) -> float:
        """Calculate expected recency score using ES-compatible formulas."""
        scale_seconds = self._parse_duration_to_seconds(scale)
        offset_seconds = self._parse_duration_to_seconds(offset)

        effective_age = max(0.0, age_seconds - offset_seconds)

        if effective_age == 0:
            return 1.0

        if decay_function == "exponential":
            # λ = ln(decay_to) / scale
            # score = max(decay_to, exp(λ × effective_age))
            lambda_val = math.log(decay_to) / scale_seconds
            score = math.exp(lambda_val * effective_age)
        elif decay_function == "linear":
            # score = max(decay_to, (scale - effective_age × (1 - decay_to)) / scale)
            score = (scale_seconds - effective_age * (1.0 - decay_to)) / scale_seconds
        elif decay_function == "gaussian":
            # σ² = -scale² / (2 × ln(decay_to))
            # score = max(decay_to, exp(-effective_age² / (2σ²)))
            # Simplified: score = max(decay_to, exp(effective_age² × ln(decay_to) / scale²))
            score = math.exp(
                pow(effective_age, 2) * math.log(decay_to) / pow(scale_seconds, 2)
            )
        elif decay_function == "binary":
            # score = 1.0 if effective_age < scale else decay_to
            score = 1.0 if effective_age < scale_seconds else decay_to
        else:
            raise ValueError(f"Unknown decay function: {decay_function}")

        return max(decay_to, score)

    def _get_doc_age_seconds(self, hit: Dict) -> Optional[float]:
        """Get the age in seconds for a document based on its timestamp field."""
        doc_id = hit.get('_id')
        if doc_id == "doc-no-ts":
            return None  # No timestamp

        # Use the actual timestamp from the document
        timestamp = hit.get('timestamp')
        if timestamp is not None:
            current_time = datetime.now().timestamp()
            return max(0, current_time - timestamp)
        return None

    # ============== Verification Helpers ==============

    def _verify_basic_recency_behavior(
        self,
        hits: List[Dict],
        decay_to: float,
        scale: str = "7d",
        offset: str = "0d",
        decay_function: str = "exponential"
    ):
        """Verify recency scores match expected values within 3 decimal places."""
        self.assertGreater(len(hits), 0, "Should have results")

        for hit in hits:
            actual_score = hit.get('_recency_score')
            doc_id = hit.get('_id')

            self.assertIsNotNone(actual_score, f"Recency score should be present for {doc_id}")

            # Calculate expected score
            age_seconds = self._get_doc_age_seconds(hit)
            if age_seconds is not None:
                expected_score = self._calculate_expected_score(
                    age_seconds, scale, offset, decay_function, decay_to
                )
                self.assertAlmostEqual(
                    actual_score,
                    expected_score,
                    places=3,
                    msg=f"Score mismatch for {doc_id}: expected {expected_score:.4f}, got {actual_score:.4f}"
                )
            elif doc_id == "doc-no-ts":
                # Document without timestamp should get decay_to
                self.assertAlmostEqual(
                    actual_score,
                    decay_to,
                    places=3,
                    msg=f"Doc without timestamp should have decay_to score"
                )

        # Verify newer docs score higher than older docs
        doc_0d = self._get_doc_by_id(hits, "doc-0d")
        doc_30d = self._get_doc_by_id(hits, "doc-30d")
        if doc_0d and doc_30d:
            self.assertGreater(
                doc_0d['_recency_score'],
                doc_30d['_recency_score'],
                "Newer doc should have higher recency score"
            )

    def _verify_decay_to_floor(self, hits: List[Dict], decay_to: float):
        """Verify that old documents floor at decay_to value."""
        doc_90d = self._get_doc_by_id(hits, "doc-90d")
        if doc_90d:
            self.assertAlmostEqual(
                doc_90d['_recency_score'],
                decay_to,
                places=3,
                msg=f"Very old doc should be at decay_to floor ({decay_to})"
            )

    def _verify_offset_behavior(
        self,
        hits: List[Dict],
        offset: str,
        scale: str = "7d",
        decay_function: str = "exponential",
        decay_to: float = 0.5
    ):
        """Verify documents within offset have score ~1.0 and verify exact scores."""
        offset_seconds = self._parse_duration_to_seconds(offset)

        for hit in hits:
            doc_id = hit.get('_id')
            actual_score = hit.get('_recency_score')
            age_seconds = self._get_doc_age_seconds(hit)

            if age_seconds is not None:
                expected_score = self._calculate_expected_score(
                    age_seconds, scale, offset, decay_function, decay_to
                )
                self.assertAlmostEqual(
                    actual_score,
                    expected_score,
                    places=3,
                    msg=f"Score mismatch for {doc_id} with offset={offset}"
                )

                # Additional check: docs within offset should have score 1.0
                if age_seconds < offset_seconds:
                    self.assertAlmostEqual(
                        actual_score,
                        1.0,
                        places=3,
                        msg=f"Doc {doc_id} within offset should have score 1.0"
                    )

    def _get_doc_by_id(self, hits: List[Dict], doc_id: str) -> Optional[Dict]:
        """Find document by ID in search results."""
        return next((h for h in hits if h.get('_id') == doc_id), None)

    # ============== Core Decay Function Tests ==============

    def test_decay_functions(self):
        """Test all decay functions work correctly."""
        self._add_shared_documents()

        for decay_func in ["exponential", "linear", "gaussian", "binary"]:
            with self.subTest(function=decay_func):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function=decay_func,
                    decay_to=0.5
                )
                hits = self._search_with_recency("product", params)
                self._verify_basic_recency_behavior(
                    hits, decay_to=0.5, scale="7d", offset="0d", decay_function=decay_func
                )

    def test_scale_offset_combinations(self):
        """Test representative scale/offset combinations."""
        self._add_shared_documents()

        test_cases = [
            ("7d", "0d"),   # No offset
            ("7d", "3d"),   # Offset < scale
            ("14d", "7d"),  # Large scale with offset
            ("3d", "0d"),   # Short scale
        ]

        for scale, offset in test_cases:
            with self.subTest(scale=scale, offset=offset):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale=scale,
                    offset=offset,
                    decay_function="exponential",
                    decay_to=0.5
                )
                hits = self._search_with_recency("product", params)
                self._verify_basic_recency_behavior(
                    hits, decay_to=0.5, scale=scale, offset=offset, decay_function="exponential"
                )
                self._verify_offset_behavior(hits, offset, scale=scale)

    def test_decay_to_values(self):
        """Test various decay_to floor values."""
        self._add_shared_documents()

        for decay_to in [0.1, 0.3, 0.5, 0.8]:
            with self.subTest(decay_to=decay_to):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    decay_to=decay_to
                )
                hits = self._search_with_recency("product", params)
                self._verify_basic_recency_behavior(hits, decay_to=decay_to)
                self._verify_decay_to_floor(hits, decay_to)

    def test_missing_field_uses_decay_to(self):
        """Documents without timestamp field get decay_to score."""
        self._add_shared_documents()

        for decay_to in [0.2, 0.5]:
            with self.subTest(decay_to=decay_to):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    decay_to=decay_to
                )
                hits = self._search_with_recency("product", params)
                doc_no_ts = self._get_doc_by_id(hits, "doc-no-ts")

                self.assertIsNotNone(doc_no_ts, "Document without timestamp should be found")
                self.assertAlmostEqual(
                    doc_no_ts['_recency_score'],
                    decay_to,
                    places=2,
                    msg="Doc without timestamp should have decay_to as recency score"
                )

    # ============== Apply in Ranking Phase Tests ==============

    def test_apply_in_ranking_phase_options(self):
        """Test all apply_in_ranking_phase options."""
        self._add_shared_documents()

        for phase in ["all", "only-global", "exclude-global"]:
            with self.subTest(phase=phase):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    decay_to=0.5,
                    apply_in_ranking_phase=phase
                )
                hits = self._search_with_recency("product", params)
                self._verify_basic_recency_behavior(hits, decay_to=0.5)

    # ============== Retrieval/Ranking Method Combinations ==============

    def test_retrieval_ranking_combinations(self):
        """Test recency with all hybrid parameter combinations."""
        self._add_shared_documents()

        test_cases = [
            (RetrievalMethod.Disjunction, RankingMethod.RRF),
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
        ]

        params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5
        )

        for retrieval, ranking in test_cases:
            with self.subTest(retrieval=retrieval.value, ranking=ranking.value):
                hybrid_params = HybridParameters(
                    retrievalMethod=retrieval,
                    rankingMethod=ranking
                )
                search_result = tensor_search.search(
                    config=self.config,
                    index_name=self.main_index.name,
                    text="product",
                    search_method=SearchMethod.HYBRID,
                    recency_parameters=params,
                    hybrid_parameters=hybrid_params,
                    result_count=10
                )
                self._verify_basic_recency_behavior(search_result['hits'], decay_to=0.5)

    # ============== Feature Combination Tests ==============
    def test_with_relevance_cutoff(self):
        """Test recency + relevance cutoff."""
        self._add_shared_documents()

        cutoff_configs = [
            (RelevanceCutoffMethod.MeanStdDev, MeanStdParameters(stdDevFactor=2.0)),
            (RelevanceCutoffMethod.GapDetection, None),
        ]

        for method, cutoff_params in cutoff_configs:
            with self.subTest(method=method.value):
                recency_params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    decay_to=0.5
                )
                relevance_cutoff = RelevanceCutoffModel(
                    method=method,
                    parameters=cutoff_params
                )

                search_result = tensor_search.search(
                    config=self.config,
                    index_name=self.main_index.name,
                    text="product",
                    search_method=SearchMethod.HYBRID,
                    recency_parameters=recency_params,
                    relevance_cutoff=relevance_cutoff,
                    result_count=20
                )

                hits = search_result['hits']
                self.assertGreater(len(hits), 0, "Should have results")

                # Verify recency scores present on remaining results
                for hit in hits:
                    self.assertIsNotNone(
                        hit.get('_recency_score'),
                        "Recency score should be present"
                    )

    def test_with_sort_by_exclude_global(self):
        """Test recency + sortBy (requires exclude-global)."""
        self._add_shared_documents()

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.3,
            apply_in_ranking_phase="exclude-global"
        )
        sort_by = SortByModel(
            fields=[SortByField(field_name="sort_value", order="desc")],
            min_sort_candidates=10
        )

        search_result = tensor_search.search(
            config=self.config,
            index_name=self.main_index.name,
            text="product",
            search_method=SearchMethod.HYBRID,
            recency_parameters=recency_params,
            sort_by=sort_by,
            hybrid_parameters=HybridParameters(rerankDepthTensor=10),
            result_count=10
        )

        hits = search_result['hits']
        self.assertGreater(len(hits), 0, "Should have results")

        # Verify sorted correctly
        sort_values = [h['sort_value'] for h in hits if 'sort_value' in h]
        self.assertEqual(
            sort_values,
            sorted(sort_values, reverse=True),
            "Results should be sorted by sort_value descending"
        )

        # Verify recency scores present
        for hit in hits:
            self.assertIsNotNone(
                hit.get('_recency_score'),
                "Recency score should be present"
            )

    def test_with_collapsing_field(self):
        """Test recency + collapsing field."""
        # Add documents to collapse index
        self._add_shared_documents(index=self.collapse_index)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5
        )

        search_result = tensor_search.search(
            config=self.config,
            index_name=self.collapse_index.name,
            text="product",
            search_method=SearchMethod.HYBRID,
            recency_parameters=recency_params,
            collapse_field_name="parent_id",
            result_count=10
        )

        hits = search_result['hits']
        self.assertGreater(len(hits), 0, "Should have results")

        # Verify collapsing worked (unique parent_ids)
        parent_ids = [h['parent_id'] for h in hits if 'parent_id' in h]
        self.assertEqual(
            len(parent_ids),
            len(set(parent_ids)),
            "Each result should have unique parent_id (collapsed)"
        )

        # Verify recency scores present
        for hit in hits:
            self.assertIsNotNone(
                hit.get('_recency_score'),
                "Recency score should be present"
            )

    # ============== Negative Case Tests ==============

    def test_structured_index_not_supported(self):
        """Recency should fail on structured indexes."""
        self._add_docs_to_structured_index()

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5
        )

        with self.assertRaises(UnsupportedFeatureError) as ctx:
            tensor_search.search(
                config=self.config,
                index_name=self.structured_index.name,
                text="product",
                search_method=SearchMethod.HYBRID,
                recency_parameters=recency_params,
                result_count=10
            )

        self.assertIn(
            "unstructured",
            str(ctx.exception).lower(),
            "Error should mention unstructured indexes"
        )

    def test_non_hybrid_search_method_not_supported(self):
        """Recency requires HYBRID search method (validated at API layer)."""
        from marqo.tensor_search.models.api_models import BulkSearchQueryEntity

        for search_method in ["TENSOR", "LEXICAL"]:
            with self.subTest(search_method=search_method):
                with self.assertRaises(ValueError) as ctx:
                    BulkSearchQueryEntity(
                        index=self.main_index.name,
                        q="product",
                        searchMethod=search_method,
                        recencyParameters={
                            "recencyField": "timestamp",
                            "scale": "7d",
                            "decayFunction": "exponential",
                            "decayTo": 0.5,
                        }
                    )

                self.assertIn(
                    "hybrid",
                    str(ctx.exception).lower(),
                    "Error should mention HYBRID search"
                )

    def test_sort_by_with_non_exclude_global_fails(self):
        """sortBy + recency should fail unless exclude-global."""
        from marqo.tensor_search.models.api_models import BulkSearchQueryEntity

        for phase in ["all", "only-global"]:
            with self.subTest(phase=phase):
                with self.assertRaises(ValueError) as ctx:
                    BulkSearchQueryEntity(
                        index=self.main_index.name,
                        q="product",
                        searchMethod="HYBRID",
                        recencyParameters={
                            "recencyField": "timestamp",
                            "scale": "7d",
                            "decayFunction": "exponential",
                            "decayTo": 0.5,
                            "applyInRankingPhase": phase
                        },
                        sortBy={"fields": [{"fieldName": "sort_value"}]}
                    )

                self.assertIn(
                    "sortby",
                    str(ctx.exception).lower(),
                    "Error should mention sortBy"
                )

    # ============== Validation Tests ==============

    def test_decay_to_validation(self):
        """Test decay_to must be in (0.0, 1.0]."""
        valid_values = [0.1, 0.5, 1.0]
        invalid_values = [0.0, -0.1, 1.1]

        for val in valid_values:
            with self.subTest(decay_to=val, valid=True):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    decay_function="exponential",
                    decay_to=val
                )
                self.assertEqual(params.decay_to, val)

        for val in invalid_values:
            with self.subTest(decay_to=val, valid=False):
                with self.assertRaises(Exception):
                    RecencyParameters(
                        recency_field="timestamp",
                        scale="7d",
                        decay_function="exponential",
                        decay_to=val
                    )

    def test_duration_format_validation(self):
        """Test scale/offset duration string formats."""
        valid_formats = ["1d", "7d", "24h", "168h"]
        invalid_formats = ["-1d", "abc"]

        for fmt in valid_formats:
            with self.subTest(format=fmt, valid=True):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale=fmt,
                    offset="0d",
                    decay_function="exponential",
                    decay_to=0.5
                )
                self.assertIsNotNone(params)

        for fmt in invalid_formats:
            with self.subTest(format=fmt, valid=False):
                with self.assertRaises(Exception):
                    RecencyParameters(
                        recency_field="timestamp",
                        scale=fmt,
                        offset="0d",
                        decay_function="exponential",
                        decay_to=0.5
                    )


if __name__ == '__main__':
    unittest.main()
