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
        time.sleep(1)  # Allow time for indexing

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
        # TODO rewrite this test
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
                        sortBy={"fields": [{"fieldName": "price"}]}
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
