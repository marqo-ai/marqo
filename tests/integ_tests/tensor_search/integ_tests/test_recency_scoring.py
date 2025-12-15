"""
Comprehensive integration tests for recency scoring feature.

Tests all decay functions (exponential, linear, gaussian, binary) with various parameter
combinations, ranking phases, hybrid search configurations, and feature combinations.
"""
import time
import unittest
from datetime import datetime, timedelta

import math
import pytest

from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.recency_parameters import RecencyParameters
from marqo.tensor_search.models.relevance_cutoff_model import (
    RelevanceCutoffModel, RelevanceCutoffMethod
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

        # 4. Index specifically for relevance cutoff testing (uses different documents)
        cls.relevance_cutoff_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        cls.indexes = cls.create_indexes([
            cls.main_index_request,
            cls.collapse_index_request,
            cls.structured_index_request,
            cls.relevance_cutoff_index_request
        ])

        cls.main_index = cls.indexes[0]
        cls.collapse_index = cls.indexes[1]
        cls.structured_index = cls.indexes[2]
        cls.relevance_cutoff_index = cls.indexes[3]

    # ============== Helper Methods ==============
    def _generate_shared_documents(self) -> List[Dict[str, Any]]:
        """Generate shared documents with various ages and attributes.

        Price groupings for sortBy tie-breaker testing:
        - Price 100: doc-0d, doc-3d, doc-7d (newer docs should rank first within group)
        - Price 80: doc-1d, doc-5d, doc-14d
        - Price 60: doc-10d, doc-30d
        - Price 40: doc-60d, doc-90d
        """
        now = datetime.now()

        # Document ages: 0, 1, 3, 5, 7, 10, 14, 30, 60, 90 days
        ages_in_days = [0, 1, 3, 5, 7, 10, 14, 30, 60, 90]

        # Price groups - documents with same price will test tie-breaking by recency
        price_map = {
            0: 100, 3: 100, 7: 100,      # Group 1: same price, different ages
            1: 80, 5: 80, 14: 80,        # Group 2: same price, different ages
            10: 60, 30: 60,              # Group 3: same price, different ages
            60: 40, 90: 40,              # Group 4: same price, different ages
        }

        documents = []
        for i, age_days in enumerate(ages_in_days):
            timestamp = (now - timedelta(days=age_days)).timestamp()
            documents.append({
                "_id": f"doc-{age_days}d",
                "title": "product item",
                "description": f"test product {age_days} days old",
                "timestamp": timestamp,
                "price": price_map[age_days],  # Deliberate duplicates for tie-breaker testing
                "parent_id": f"group-{chr(65 + i % 5)}",  # A-E rotation
                "mult": 1.0 + (i % 3) * 0.5,  # 1.0, 1.5, 2.0
            })

        # Special: Document without timestamp field
        documents.append({
            "_id": "doc-no-ts",
            "title": "product item",
            "description": "product without timestamp",
            "price": 20,  # Unique price for this doc
            "parent_id": "group-F",
            "mult": 1.0,
        })

        return documents

    def _generate_relevance_cutoff_documents(self) -> List[Dict[str, Any]]:
        """Generate documents for deterministic relevance cutoff testing.

        Query: "machine learning artificial intelligence algorithms"

        Document categories by relevance:
        - HIGH: Contains ALL 5 query words (h1-h10) - 10 docs
        - MEDIUM: Contains EXACTLY 3 of 5 query words (m1-m3) - 3 docs
        - LOW: Contains exactly 1 query word (l1-l2) - 2 docs
        - IRRELEVANT: Contains 0 query words (i1-i5) - 5 docs

        Total: 20 documents with varying ages
        Expected probe candidates: 13 (HIGH + MEDIUM, excludes LOW and IRRELEVANT)
        """
        now = datetime.now()

        # HIGH RELEVANCE (10 docs) - Contains ALL 5 query words
        # Ages distributed: 0, 1, 3, 5, 7, 10, 14, 21, 28, 30 days
        high_relevance = [
            {"_id": "h1",
             "content": "Machine learning algorithms in artificial intelligence enable systems to adapt.",
             "timestamp": (now - timedelta(days=0)).timestamp()},
            {"_id": "h2",
             "content": "Artificial intelligence relies on machine learning algorithms to build models.",
             "timestamp": (now - timedelta(days=1)).timestamp()},
            {"_id": "h3",
             "content": "Researchers develop artificial intelligence machine learning algorithms.",
             "timestamp": (now - timedelta(days=3)).timestamp()},
            {"_id": "h4",
             "content": "Scalable artificial intelligence frameworks integrate machine learning algorithms.",
             "timestamp": (now - timedelta(days=5)).timestamp()},
            {"_id": "h5",
             "content": "Modern artificial intelligence and machine learning algorithms optimize workflows.",
             "timestamp": (now - timedelta(days=7)).timestamp()},
            {"_id": "h6",
             "content": "Sophisticated artificial intelligence machine learning algorithms optimize mining.",
             "timestamp": (now - timedelta(days=10)).timestamp()},
            {"_id": "h7",
             "content": "Cutting-edge artificial intelligence machine learning algorithms accelerate processing.",
             "timestamp": (now - timedelta(days=14)).timestamp()},
            {"_id": "h8",
             "content": "Enterprise artificial intelligence solutions embed machine learning algorithms.",
             "timestamp": (now - timedelta(days=21)).timestamp()},
            {"_id": "h9",
             "content": "Robust artificial intelligence machine learning algorithms improve quality.",
             "timestamp": (now - timedelta(days=28)).timestamp()},
            {"_id": "h10",
             "content": "Innovative artificial intelligence and machine learning algorithms revolutionize analytics.",
             "timestamp": (now - timedelta(days=30)).timestamp()},
        ]

        # MEDIUM RELEVANCE (3 docs) - Contains exactly 3 of 5 query words
        medium_relevance = [
            {"_id": "m1",
             "content": "Machine learning algorithms process financial time series for forecasting.",
             "timestamp": (now - timedelta(days=7)).timestamp()},
            {"_id": "m2",
             "content": "Artificial intelligence algorithms underpin recommendation engines.",
             "timestamp": (now - timedelta(days=14)).timestamp()},
            {"_id": "m3",
             "content": "Artificial intelligence learning models adapt to new user behaviors.",
             "timestamp": (now - timedelta(days=21)).timestamp()},
        ]

        # LOW RELEVANCE (2 docs) - Contains exactly 1 query word
        low_relevance = [
            {"_id": "l1",
             "content": "Engineers use machine tools for precise cutting operations.",
             "timestamp": (now - timedelta(days=1)).timestamp()},
            {"_id": "l2",
             "content": "Innovators encourage collaborative learning environments to foster growth.",
             "timestamp": (now - timedelta(days=5)).timestamp()},
        ]

        # IRRELEVANT (5 docs) - Contains 0 query words
        irrelevant = [
            {"_id": "i1",
             "content": "Bright morning sunlight streamed through the quiet study room.",
             "timestamp": (now - timedelta(days=0)).timestamp()},
            {"_id": "i2",
             "content": "Surprising weather patterns emerged across the town.",
             "timestamp": (now - timedelta(days=3)).timestamp()},
            {"_id": "i3",
             "content": "Vibrant wildflowers adorned the rolling hills during summer.",
             "timestamp": (now - timedelta(days=10)).timestamp()},
            {"_id": "i4",
             "content": "Chilly autumn breeze painted golden leaves across streets.",
             "timestamp": (now - timedelta(days=20)).timestamp()},
            {"_id": "i5",
             "content": "The ancient manuscript revealed hidden stories from forgotten civilizations.",
             "timestamp": (now - timedelta(days=30)).timestamp()},
        ]

        return high_relevance + medium_relevance + low_relevance + irrelevant

    def _add_relevance_cutoff_documents(self):
        """Add documents designed for relevance cutoff testing."""
        documents = self._generate_relevance_cutoff_documents()
        add_docs_params = AddDocsParams(
            index_name=self.relevance_cutoff_index.name,
            docs=documents,
            tensor_fields=["content"]
        )
        self.add_documents(self.config, add_docs_params)

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
                    scale="8d",
                    offset="0d",
                    decay_function=decay_func,
                    decay_to=0.5
                )
                hits = self._search_with_recency("product", params)
                self._verify_basic_recency_behavior(
                    hits, decay_to=0.5, scale="8d", offset="0d", decay_function=decay_func
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
    @pytest.mark.skip_for_multinode(
        "Multi-nodes will return different lexical results so we can not assert on the results.")
    def test_with_relevance_cutoff(self):
        """Test recency + relevance cutoff interaction.

        Verifies:
        1. Relevance cutoff probe query runs with recency DISABLED (pure relevance)
           - Probe candidates count is deterministic based on semantic/lexical relevance
           - If recency was applied to probe, results would vary based on document ages
        2. Returned docs have correctly calculated recency scores

        Uses dedicated document set with predictable relevance distribution:
        - HIGH (10 docs): Contains all 5 query words
        - MEDIUM (3 docs): Contains 3 of 5 query words
        - LOW (2 docs): Contains 1 query word
        - IRRELEVANT (5 docs): Contains 0 query words
        """
        self._add_relevance_cutoff_documents()

        QUERY = "machine learning artificial intelligence algorithms"
        # HIGH (10) + MEDIUM (3) should pass semantic relevance threshold
        # LOW and IRRELEVANT should be filtered out by relevance cutoff

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.3,
            apply_in_ranking_phase="only-global",  # apply recency at phase-1 ranking defies the purpose of cutoff
        )

        # Use relative_max_score with moderate threshold to get HIGH relevance docs
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters={"relativeScoreFactor": 0.5}
        )

        search_result = tensor_search.search(
            config=self.config,
            index_name=self.relevance_cutoff_index.name,
            text=QUERY,
            search_method=SearchMethod.HYBRID,
            recency_parameters=recency_params,
            relevance_cutoff=relevance_cutoff,
            result_count=20
        )

        hits = search_result['hits']

        # 1. Verify probe candidates - proves recency was NOT used in probe
        probe_candidates = search_result.get('_probeCandidates')
        self.assertEqual(15, probe_candidates, )

        # 2. Verify relevant candidates based on threshold
        relevant_candidates = search_result.get('_relevantCandidates')
        self.assertEqual(13, relevant_candidates, "Relevant candidates should cover high and medium relevant docs")

        for hit in hits:
            actual_recency = hit.get('_recency_score')
            doc_id = hit.get('_id')

            # 3. Verify returned docs are from HIGH or MEDIUM relevance categories
            # (LOW and IRRELEVANT should be filtered out by relevance cutoff)
            self.assertTrue(
                doc_id.startswith('h') or doc_id.startswith('m'),
                f"Only HIGH/MEDIUM relevance docs should be returned, got {doc_id}"
            )

            # 4. Verify recency scores ARE applied to returned results
            timestamp = hit.get('timestamp')
            if timestamp is not None:
                current_time = datetime.now().timestamp()
                age_seconds = max(0, current_time - timestamp)
                expected_score = self._calculate_expected_score(
                    age_seconds, scale="7d", offset="0d",
                    decay_function="exponential", decay_to=0.3
                )
                self.assertAlmostEqual(
                    actual_recency, expected_score, places=3,
                    msg=f"Recency score mismatch for {doc_id}"
                )

    @pytest.mark.skip_for_multinode(
        "Multi-nodes will return different lexical results so we can not assert on the results.")
    def test_with_sort_by_exclude_global(self):
        """Test recency + sortBy with recency as tie-breaker for equal prices.

        Documents have deliberate price duplicates:
        - Price 100: doc-0d, doc-3d, doc-7d
        - Price 80: doc-1d, doc-5d, doc-14d
        - Price 60: doc-10d, doc-30d
        - Price 40: doc-60d, doc-90d
        - Price 20: doc-no-ts

        When sorted by price desc, documents with same price should be
        ordered by recency (newer docs first) as a tie-breaker.

        Uses scale=120d to ensure all docs (up to 90 days old) have
        distinct recency scores for proper tie-breaking.
        """
        self._add_shared_documents()

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",  # Large scale so all docs have distinct recency scores
            offset="0d",
            decay_function="exponential",
            decay_to=0.3,
            apply_in_ranking_phase="exclude-global"
        )
        sort_by = SortByModel(
            fields=[SortByField(field_name="price", order="desc")],
            min_sort_candidates=20
        )

        search_result = tensor_search.search(
            config=self.config,
            index_name=self.main_index.name,
            text="product",
            search_method=SearchMethod.HYBRID,
            recency_parameters=recency_params,
            sort_by=sort_by,
            hybrid_parameters=HybridParameters(rerankDepthTensor=20),
            result_count=15
        )

        hits = search_result['hits']

        # 1. Verify exact order of hits
        # Sorted by price desc, with recency as tie-breaker (newer docs first)
        expected_order = [
            "doc-0d", "doc-3d", "doc-7d",      # Price 100: newest to oldest
            "doc-1d", "doc-5d", "doc-14d",     # Price 80: newest to oldest
            "doc-10d", "doc-30d",              # Price 60: newest to oldest
            "doc-60d", "doc-90d",              # Price 40: newest to oldest
            "doc-no-ts",                       # Price 20: no timestamp
        ]
        actual_order = [hit['_id'] for hit in hits]
        self.assertListEqual(
            expected_order,
            actual_order,
            "Results should be sorted by price desc, with recency as tie-breaker"
        )

        # 2. Verify recency scores are calculated correctly for each doc
        self._verify_basic_recency_behavior(
            hits,
            decay_to=0.3,
            scale="120d",
            offset="0d",
            decay_function="exponential"
        )

    def test_with_collapsing_field(self):
        """Test recency + collapsing field picks most recent variant per parent.

        Document structure (parent_id uses A-E rotation):
        - group-A: doc-0d (newest), doc-10d
        - group-B: doc-1d (newest), doc-14d
        - group-C: doc-3d (newest), doc-30d
        - group-D: doc-5d (newest), doc-60d
        - group-E: doc-7d (newest), doc-90d
        - group-F: doc-no-ts (only variant)

        With recency boosting, the most recent variant should be selected
        for each parent group when collapsing.

        Uses scale=120d to ensure all documents (up to 90 days old) have
        distinct recency scores for proper variant selection.
        """
        # Add documents to collapse index
        self._add_shared_documents(index=self.collapse_index)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",  # Large scale so all docs have distinct recency scores
            offset="0d",
            decay_function="exponential",
            decay_to=0.3
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

        # 1. Verify collapsing worked (unique parent_ids)
        parent_ids = [h['parent_id'] for h in hits if 'parent_id' in h]
        self.assertEqual(
            len(parent_ids),
            len(set(parent_ids)),
            "Each result should have unique parent_id (collapsed)"
        )

        # 2. Verify recency scores present
        for hit in hits:
            self.assertIsNotNone(
                hit.get('_recency_score'),
                "Recency score should be present"
            )

        # 3. Verify the most recent variant is selected for each parent group
        # Expected: newest variant should be picked for each group
        expected_newest_variant = {
            "group-A": "doc-0d",   # 0d is newer than 10d
            "group-B": "doc-1d",   # 1d is newer than 14d
            "group-C": "doc-3d",   # 3d is newer than 30d
            "group-D": "doc-5d",   # 5d is newer than 60d
            "group-E": "doc-7d",   # 7d is newer than 90d
            "group-F": "doc-no-ts",  # Only variant
        }

        for hit in hits:
            parent_id = hit.get('parent_id')
            doc_id = hit.get('_id')
            if parent_id in expected_newest_variant:
                expected_doc = expected_newest_variant[parent_id]
                self.assertEqual(
                    doc_id, expected_doc,
                    f"For {parent_id}, expected newest variant {expected_doc} but got {doc_id}"
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
                        sortBy={"fields": [{"fieldName": "price"}]}
                    )

                self.assertIn(
                    "'sortBy' cannot be used with 'recencyParameters' with global-phase reranking in hybrid search",
                    str(ctx.exception)
                )

    # ============== Additive Recency Scoring Tests ==============

    def test_additive_recency_scoring(self):
        """Test additive recency scoring with addToScoreWeight parameter.

        When addToScoreWeight is provided, recency is applied additively:
        final_score = modified_score + (recency_score * addToScoreWeight)

        Instead of multiplicatively:
        final_score = modified_score * recency_score
        """
        self._add_shared_documents()

        # Test with different addToScoreWeight values
        weight_values = [0.1, 0.5, 1.0, 2.0]

        for weight in weight_values:
            with self.subTest(addToScoreWeight=weight):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    decay_to=0.5,
                    add_to_score_weight=weight
                )
                hits = self._search_with_recency("product", params)

                self.assertGreater(len(hits), 0, "Should have results")

                # Verify recency scores are calculated correctly
                for hit in hits:
                    actual_recency = hit.get('_recency_score')
                    doc_id = hit.get('_id')

                    self.assertIsNotNone(actual_recency, f"Recency score should be present for {doc_id}")

                    # Verify recency score is within valid range [decay_to, 1.0]
                    self.assertGreaterEqual(actual_recency, 0.5, f"Recency score for {doc_id} should be >= decay_to")
                    self.assertLessEqual(actual_recency, 1.0, f"Recency score for {doc_id} should be <= 1.0")

                # Verify newer docs score higher than older docs
                doc_0d = self._get_doc_by_id(hits, "doc-0d")
                doc_30d = self._get_doc_by_id(hits, "doc-30d")
                if doc_0d and doc_30d:
                    self.assertGreater(
                        doc_0d['_recency_score'],
                        doc_30d['_recency_score'],
                        "Newer doc should have higher recency score"
                    )

    def test_additive_recency_vs_multiplicative(self):
        """Test that additive and multiplicative recency produce different final scores.

        With additive mode (addToScoreWeight > 0), very old documents get boosted more
        relative to their base score compared to multiplicative mode where they get
        penalized more heavily.
        """
        self._add_shared_documents()

        base_params = {
            "recency_field": "timestamp",
            "scale": "7d",
            "offset": "0d",
            "decay_function": "exponential",
            "decay_to": 0.3
        }

        # Get results without additive (multiplicative mode - default)
        multiplicative_params = RecencyParameters(**base_params)
        multiplicative_hits = self._search_with_recency("product", multiplicative_params)

        # Get results with additive mode
        additive_params = RecencyParameters(
            **base_params,
            add_to_score_weight=0.5
        )
        additive_hits = self._search_with_recency("product", additive_params)

        # Both should return results
        self.assertGreater(len(multiplicative_hits), 0, "Multiplicative should have results")
        self.assertGreater(len(additive_hits), 0, "Additive should have results")

        # Both should have the same recency scores (recency calculation is the same)
        for mult_hit, add_hit in zip(multiplicative_hits, additive_hits):
            if mult_hit['_id'] == add_hit['_id']:
                self.assertAlmostEqual(
                    mult_hit.get('_recency_score', 0),
                    add_hit.get('_recency_score', 0),
                    places=3,
                    msg=f"Recency scores should be the same for {mult_hit['_id']}"
                )

    def test_additive_recency_with_hybrid_search_methods(self):
        """Test additive recency works with different hybrid search configurations."""
        self._add_shared_documents()

        test_cases = [
            (RetrievalMethod.Disjunction, RankingMethod.RRF),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
        ]

        params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            add_to_score_weight=0.5
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

                hits = search_result['hits']
                self.assertGreater(len(hits), 0, "Should have results")

                # Verify recency scores are present
                for hit in hits:
                    self.assertIsNotNone(
                        hit.get('_recency_score'),
                        f"Recency score should be present for {hit['_id']}"
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


    # ============== Grow Parameter Tests ==============

    def _generate_future_documents(self) -> List[Dict[str, Any]]:
        """Generate documents with future timestamps for grow parameter testing.

        Document ages (relative to now):
        - Past documents: -7d, -3d, -1d (negative = in the past)
        - Current: 0d
        - Future documents: +1d, +3d, +7d, +14d, +30d (positive = in the future)
        """
        now = datetime.now()

        # Mix of past, present, and future documents
        time_offsets_days = [-7, -3, -1, 0, 1, 3, 7, 14, 30]

        documents = []
        for offset_days in time_offsets_days:
            timestamp = (now + timedelta(days=offset_days)).timestamp()
            if offset_days < 0:
                doc_id = f"doc-past-{abs(offset_days)}d"
            elif offset_days == 0:
                doc_id = "doc-now"
            else:
                doc_id = f"doc-future-{offset_days}d"

            documents.append({
                "_id": doc_id,
                "title": "event announcement",
                "description": f"event scheduled for {offset_days} days from now",
                "timestamp": timestamp,
            })

        return documents

    def _add_future_documents(self, index=None):
        """Add future timestamp documents to the specified or main index."""
        if index is None:
            index = self.main_index
        documents = self._generate_future_documents()
        add_docs_params = AddDocsParams(
            index_name=index.name,
            docs=documents,
            tensor_fields=["title", "description"]
        )
        self.add_documents(self.config, add_docs_params)

    def _calculate_expected_grow_score(
        self,
        future_age_seconds: float,
        grow_scale: str,
        grow_offset: str,
        grow_function: str,
        grow_from: float
    ) -> float:
        """Calculate expected grow score using the same formulas as decay (mirrored)."""
        scale_seconds = self._parse_duration_to_seconds(grow_scale)
        offset_seconds = self._parse_duration_to_seconds(grow_offset)

        # Future age after subtracting offset (plateau zone)
        effective_future_age = max(0.0, future_age_seconds - offset_seconds)

        if effective_future_age == 0:
            return 1.0

        if grow_function == "exponential":
            # Mirror of decay: score approaches grow_from at scale
            score = 1.0 - (1.0 - grow_from) * math.exp(
                math.log(1.0 - grow_from) * effective_future_age / scale_seconds
            )
        elif grow_function == "linear":
            score = 1.0 - (1.0 - grow_from) * effective_future_age / scale_seconds
        elif grow_function == "gaussian":
            score = 1.0 - (1.0 - grow_from) * (
                1.0 - math.exp(pow(effective_future_age, 2) * math.log(grow_from) / pow(scale_seconds, 2))
            )
        elif grow_function == "binary":
            score = 1.0 if effective_future_age < scale_seconds else grow_from
        else:
            raise ValueError(f"Unknown grow function: {grow_function}")

        return max(grow_from, score)

    def test_grow_disabled_by_default(self):
        """Test future timestamps get score 1.0 when growFrom is not specified."""
        self._add_future_documents()

        # Without grow_from, future timestamps should get score 1.0
        params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5
            # No grow_from specified
        )
        hits = self._search_with_recency("event", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Future documents should have score 1.0 when grow is disabled
        for hit in hits:
            doc_id = hit.get('_id')
            recency_score = hit.get('_recency_score')
            self.assertIsNotNone(recency_score, f"Recency score should be present for {doc_id}")

            if doc_id.startswith("doc-future"):
                self.assertAlmostEqual(
                    recency_score, 1.0, places=2,
                    msg=f"Future doc {doc_id} should have score 1.0 when grow disabled"
                )

    def test_grow_functions(self):
        """Test all grow functions work correctly for future timestamps."""
        self._add_future_documents()

        for grow_func in ["exponential", "linear", "gaussian", "binary"]:
            with self.subTest(function=grow_func):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    decay_to=0.5,
                    grow_from=0.3,
                    grow_function=grow_func,
                    grow_scale="14d",
                    grow_offset="0d"
                )
                hits = self._search_with_recency("event", params)

                self.assertGreater(len(hits), 0, "Should have results")

                # Verify future documents have grow scores applied
                for hit in hits:
                    doc_id = hit.get('_id')
                    recency_score = hit.get('_recency_score')
                    self.assertIsNotNone(recency_score, f"Recency score should be present for {doc_id}")

                    if doc_id.startswith("doc-future"):
                        # Score should be between grow_from and 1.0
                        self.assertGreaterEqual(
                            recency_score, 0.3,
                            f"Future doc {doc_id} score should be >= grow_from"
                        )
                        self.assertLessEqual(
                            recency_score, 1.0,
                            f"Future doc {doc_id} score should be <= 1.0"
                        )

    def test_grow_with_decay(self):
        """Test combined grow (future) and decay (past) behavior."""
        self._add_future_documents()

        params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="14d",
            grow_offset="0d"
        )
        hits = self._search_with_recency("event", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Categorize documents
        past_docs = [h for h in hits if h['_id'].startswith("doc-past")]
        future_docs = [h for h in hits if h['_id'].startswith("doc-future")]
        now_doc = self._get_doc_by_id(hits, "doc-now")

        # Verify now doc has score ~1.0
        if now_doc:
            self.assertAlmostEqual(
                now_doc.get('_recency_score'), 1.0, places=1,
                msg="Current timestamp should have score ~1.0"
            )

        # Verify past docs use decay (score between decay_to and 1.0)
        for hit in past_docs:
            recency_score = hit.get('_recency_score')
            self.assertGreaterEqual(recency_score, 0.5, f"Past doc {hit['_id']} should use decay_to as floor")
            self.assertLessEqual(recency_score, 1.0, f"Past doc {hit['_id']} should be <= 1.0")

        # Verify future docs use grow (score between grow_from and 1.0)
        for hit in future_docs:
            recency_score = hit.get('_recency_score')
            self.assertGreaterEqual(recency_score, 0.3, f"Future doc {hit['_id']} should use grow_from as floor")
            self.assertLessEqual(recency_score, 1.0, f"Future doc {hit['_id']} should be <= 1.0")

    def test_grow_defaults_to_decay_function(self):
        """Test growFunction defaults to decayFunction when not specified."""
        self._add_future_documents()

        # When grow_function is not specified, it should default to decay_function
        params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="linear",  # Using linear decay
            decay_to=0.5,
            grow_from=0.3,
            # grow_function not specified - should default to "linear"
            grow_scale="14d",
            grow_offset="0d"
        )
        hits = self._search_with_recency("event", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Verify future documents have grow scores applied
        for hit in hits:
            doc_id = hit.get('_id')
            recency_score = hit.get('_recency_score')

            if doc_id.startswith("doc-future"):
                # Score should be between grow_from and 1.0
                self.assertGreaterEqual(
                    recency_score, 0.3,
                    f"Future doc {doc_id} score should be >= grow_from"
                )

    def test_grow_defaults_to_scale(self):
        """Test growScale defaults to scale when not specified."""
        self._add_future_documents()

        # When grow_scale is not specified, it should default to scale
        params = RecencyParameters(
            recency_field="timestamp",
            scale="14d",  # Using 14d scale
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.3,
            grow_function="exponential",
            # grow_scale not specified - should default to "14d"
            grow_offset="0d"
        )
        hits = self._search_with_recency("event", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Verify future documents have grow scores applied
        future_14d = self._get_doc_by_id(hits, "doc-future-14d")
        if future_14d:
            # At future_age = scale, score should be close to grow_from
            recency_score = future_14d.get('_recency_score')
            # Allow some tolerance since we're testing the score at scale
            self.assertLess(
                recency_score, 0.5,
                f"Future doc at scale should have score closer to grow_from"
            )

    def test_grow_offset_creates_plateau(self):
        """Test growOffset creates plateau zone where score = 1.0."""
        self._add_future_documents()

        # With grow_offset=7d, documents with timestamps between now() and now()+7d
        # should have score 1.0 (plateau zone)
        params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="14d",
            grow_offset="7d"  # Plateau zone: now() to now()+7d
        )
        hits = self._search_with_recency("event", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Documents in plateau zone (1d, 3d, 7d) should have score ~1.0
        plateau_docs = ["doc-future-1d", "doc-future-3d", "doc-future-7d"]
        for doc_id in plateau_docs:
            hit = self._get_doc_by_id(hits, doc_id)
            if hit:
                recency_score = hit.get('_recency_score')
                self.assertAlmostEqual(
                    recency_score, 1.0, places=1,
                    msg=f"Doc {doc_id} within plateau should have score ~1.0"
                )

        # Documents beyond plateau (14d, 30d) should have score < 1.0
        beyond_plateau_docs = ["doc-future-14d", "doc-future-30d"]
        for doc_id in beyond_plateau_docs:
            hit = self._get_doc_by_id(hits, doc_id)
            if hit:
                recency_score = hit.get('_recency_score')
                self.assertLess(
                    recency_score, 1.0,
                    msg=f"Doc {doc_id} beyond plateau should have score < 1.0"
                )

    def test_grow_binary_function(self):
        """Test binary grow function creates step function at scale."""
        self._add_future_documents()

        params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.2,
            grow_function="binary",
            grow_scale="10d",  # Step at 10 days
            grow_offset="0d"
        )
        hits = self._search_with_recency("event", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Documents within scale (1d, 3d, 7d) should have score 1.0
        within_scale_docs = ["doc-future-1d", "doc-future-3d", "doc-future-7d"]
        for doc_id in within_scale_docs:
            hit = self._get_doc_by_id(hits, doc_id)
            if hit:
                recency_score = hit.get('_recency_score')
                self.assertAlmostEqual(
                    recency_score, 1.0, places=1,
                    msg=f"Doc {doc_id} within binary scale should have score 1.0"
                )

        # Documents beyond scale (14d, 30d) should have score = grow_from
        beyond_scale_docs = ["doc-future-14d", "doc-future-30d"]
        for doc_id in beyond_scale_docs:
            hit = self._get_doc_by_id(hits, doc_id)
            if hit:
                recency_score = hit.get('_recency_score')
                self.assertAlmostEqual(
                    recency_score, 0.2, places=1,
                    msg=f"Doc {doc_id} beyond binary scale should have score ~grow_from"
                )

    def test_grow_parameters_validation(self):
        """Test grow parameter validation."""
        # Valid grow_from values
        valid_grow_from = [0.01, 0.5, 1.0]
        for val in valid_grow_from:
            with self.subTest(grow_from=val, valid=True):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    decay_function="exponential",
                    decay_to=0.5,
                    grow_from=val
                )
                self.assertEqual(params.grow_from, val)

        # Invalid grow_from values
        invalid_grow_from = [0.0, -0.1, 1.1]
        for val in invalid_grow_from:
            with self.subTest(grow_from=val, valid=False):
                with self.assertRaises(Exception):
                    RecencyParameters(
                        recency_field="timestamp",
                        scale="7d",
                        decay_function="exponential",
                        decay_to=0.5,
                        grow_from=val
                    )


if __name__ == '__main__':
    unittest.main()
