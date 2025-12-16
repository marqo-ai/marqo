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
        """Generate shared documents with various ages (past and future) and attributes.

        Naming convention:
        - Past docs: doc-Xd where X is days ago (e.g., doc-0d = today, doc-7d = 7 days ago)
        - Future docs: doc+Xd where X is days in future (e.g., doc+1d = 1 day from now)

        Price groupings for sortBy tie-breaker testing:
        - Price 120: doc+1d, doc+3d, doc+7d, doc+14d, doc+30d (future docs)
        - Price 100: doc-0d, doc-3d, doc-7d (past docs - newer)
        - Price 80: doc-1d, doc-5d, doc-14d
        - Price 60: doc-10d, doc-30d
        - Price 40: doc-60d, doc-90d
        - Price 20: doc-no-ts

        Parent groupings for collapsing test (future and past docs mixed):
        - group-A: doc-0d, doc-10d, doc+1d
        - group-B: doc-1d, doc-14d, doc+3d
        - group-C: doc-3d, doc-30d, doc+7d
        - group-D: doc-5d, doc-60d, doc+14d
        - group-E: doc-7d, doc-90d, doc+30d
        - group-F: doc-no-ts
        """
        now = datetime.now()

        # Explicit config for each document: (price, parent_id, mult)
        # age > 0 means days in past, age < 0 means days in future
        doc_configs = {
            # Future docs (negative ages = future timestamps) - Price 120
            # Mixed into same groups as past docs
            -30: (120, "group-E", 1.0),   # doc+30d - with doc-7d, doc-90d
            -14: (120, "group-D", 1.5),   # doc+14d - with doc-5d, doc-60d
            -7:  (120, "group-C", 2.0),   # doc+7d - with doc-3d, doc-30d
            -3:  (120, "group-B", 1.0),   # doc+3d - with doc-1d, doc-14d
            -1:  (120, "group-A", 1.5),   # doc+1d - with doc-0d, doc-10d

            # Past docs (positive ages = past timestamps)
            0:   (100, "group-A", 2.0),   # doc-0d (today)
            1:   (80,  "group-B", 1.0),   # doc-1d
            3:   (100, "group-C", 1.5),   # doc-3d
            5:   (80,  "group-D", 2.0),   # doc-5d
            7:   (100, "group-E", 1.0),   # doc-7d
            10:  (60,  "group-A", 1.5),   # doc-10d
            14:  (80,  "group-B", 2.0),   # doc-14d
            30:  (60,  "group-C", 1.0),   # doc-30d
            60:  (40,  "group-D", 1.5),   # doc-60d
            90:  (40,  "group-E", 2.0),   # doc-90d
        }

        documents = []

        for age_days, (price, parent_id, mult) in doc_configs.items():
            # timestamp = now - age_days (negative age = future timestamp)
            timestamp = (now - timedelta(days=age_days)).timestamp()

            # Doc ID format: doc+Xd for future, doc-Xd for past
            if age_days < 0:
                doc_id = f"doc+{abs(age_days)}d"
            else:
                doc_id = f"doc-{age_days}d"

            documents.append({
                "_id": doc_id,
                "title": "product item",
                "description": f"test product {age_days} days old",
                "timestamp": timestamp,
                "price": price,
                "parent_id": parent_id,
                "mult": mult,
            })

        # Special: Document without timestamp field
        documents.append({
            "_id": "doc-no-ts",
            "title": "product item",
            "description": "product without timestamp",
            "price": 20,
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
        decay_to: float,
        grow_enabled: bool = False,
        grow_from: float = 0.5,
        grow_function: str = None,  # Defaults to decay_function
        grow_scale: str = None,  # Defaults to scale
        grow_offset: str = None,  # Defaults to "0d"
    ) -> float:
        """Calculate expected recency score matching the rank profile logic.

        Unified logic handles both decay (past) and grow (future) timestamps:
        - age_seconds > 0: past document, use decay logic
        - age_seconds < 0: future document, use grow logic (if enabled) or return 1.0
        - age_seconds == 0: current document, score = 1.0
        """
        scale_seconds = self._parse_duration_to_seconds(scale)
        offset_seconds = self._parse_duration_to_seconds(offset)

        # Handle grow parameter defaults
        if grow_function is None:
            grow_function = decay_function
        if grow_scale is None:
            grow_scale = scale
        if grow_offset is None:
            grow_offset = "0d"

        grow_scale_seconds = self._parse_duration_to_seconds(grow_scale)
        grow_offset_seconds = self._parse_duration_to_seconds(grow_offset)

        # Past document (age >= 0): use decay logic
        if age_seconds >= 0:
            # Check if beyond decay range (very old)
            if age_seconds >= scale_seconds + offset_seconds:
                return decay_to

            # Check if within offset grace period
            if age_seconds < offset_seconds:
                return 1.0

            # Calculate decay score
            effective_age = age_seconds - offset_seconds
            return self._calculate_function_score(
                effective_age, scale_seconds, decay_to, decay_function
            )

        # Future document (age < 0): use grow logic if enabled
        if not grow_enabled:
            return 1.0

        future_age = -age_seconds  # Convert to positive

        # Check if within grow offset plateau zone
        if future_age <= grow_offset_seconds:
            return 1.0

        # Check if beyond grow range (far future)
        if future_age >= grow_offset_seconds + grow_scale_seconds:
            return grow_from

        # Calculate grow score
        effective_future_age = future_age - grow_offset_seconds
        return self._calculate_function_score(
            effective_future_age, grow_scale_seconds, grow_from, grow_function
        )

    def _calculate_function_score(
        self,
        effective_age: float,
        scale_seconds: float,
        floor_value: float,
        function_type: str
    ) -> float:
        """Calculate score using specified decay/grow function.

        Both decay and grow use the same mathematical functions:
        - At effective_age=0: score = 1.0
        - At effective_age=scale: score = floor_value
        - Beyond scale: score = floor_value (clamped)
        """
        if effective_age == 0:
            return 1.0

        if function_type == "exponential":
            # λ = ln(floor_value) / scale
            # score = exp(λ × effective_age)
            score = math.exp(math.log(floor_value) * effective_age / scale_seconds)
        elif function_type == "linear":
            # score = (scale - effective_age × (1 - floor_value)) / scale
            score = (scale_seconds - effective_age * (1.0 - floor_value)) / scale_seconds
        elif function_type == "gaussian":
            # score = exp(effective_age² × ln(floor_value) / scale²)
            score = math.exp(
                pow(effective_age, 2) * math.log(floor_value) / pow(scale_seconds, 2)
            )
        elif function_type == "binary":
            # score = 1.0 if effective_age < scale else floor_value
            score = 1.0 if effective_age < scale_seconds else floor_value
        else:
            raise ValueError(f"Unknown function type: {function_type}")

        return max(floor_value, score)

    def _get_doc_age_seconds(self, hit: Dict) -> Optional[float]:
        """Get the age in seconds for a document based on its timestamp field.

        Returns:
            - Positive value: past document (timestamp < now)
            - Negative value: future document (timestamp > now)
            - None: document has no timestamp field
        """
        doc_id = hit.get('_id')
        if doc_id == "doc-no-ts":
            return None  # No timestamp

        # Use the actual timestamp from the document
        timestamp = hit.get('timestamp')
        if timestamp is not None:
            current_time = datetime.now().timestamp()
            return current_time - timestamp  # Can be negative for future docs
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
        - Price 120: doc+1d, doc+3d, doc+7d, doc+14d, doc+30d (future docs)
        - Price 100: doc-0d, doc-3d, doc-7d
        - Price 80: doc-1d, doc-5d, doc-14d
        - Price 60: doc-10d, doc-30d
        - Price 40: doc-60d, doc-90d
        - Price 20: doc-no-ts

        When sorted by price desc, documents with same price should be
        ordered by recency (newer docs first) as a tie-breaker.
        Future docs all have score 1.0 (grow disabled), so their relative
        order within price=120 group is non-deterministic.

        Uses scale=120d to ensure all past docs (up to 90 days old) have
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
            result_count=20
        )

        hits = search_result['hits']
        actual_order = [hit['_id'] for hit in hits]

        # 1. Verify price ordering (groups should be in correct order)
        # Future docs (price 120) should come first - order within group is non-deterministic
        future_docs = {"doc+1d", "doc+3d", "doc+7d", "doc+14d", "doc+30d"}
        price_100_docs = {"doc-0d", "doc-3d", "doc-7d"}
        price_80_docs = {"doc-1d", "doc-5d", "doc-14d"}
        price_60_docs = {"doc-10d", "doc-30d"}
        price_40_docs = {"doc-60d", "doc-90d"}

        # Verify docs are grouped by price (first 5 should be future, etc.)
        self.assertEqual(set(actual_order[:5]), future_docs, "First 5 docs should be price 120 (future)")
        self.assertEqual(set(actual_order[5:8]), price_100_docs, "Next 3 docs should be price 100")
        self.assertEqual(set(actual_order[8:11]), price_80_docs, "Next 3 docs should be price 80")
        self.assertEqual(set(actual_order[11:13]), price_60_docs, "Next 2 docs should be price 60")
        self.assertEqual(set(actual_order[13:15]), price_40_docs, "Next 2 docs should be price 40")
        self.assertEqual(actual_order[15], "doc-no-ts", "Last doc should be price 20")

        # 2. Verify recency-based ordering within past doc groups (distinct scores)
        # Price 100 group: doc-0d should be first, then doc-3d, then doc-7d
        price_100_order = actual_order[5:8]
        self.assertEqual(price_100_order, ["doc-0d", "doc-3d", "doc-7d"], "Price 100 group should be ordered by recency")

        # Price 80 group: doc-1d should be first, then doc-5d, then doc-14d
        price_80_order = actual_order[8:11]
        self.assertEqual(price_80_order, ["doc-1d", "doc-5d", "doc-14d"], "Price 80 group should be ordered by recency")

        # Price 60 group: doc-10d should be first, then doc-30d
        price_60_order = actual_order[11:13]
        self.assertEqual(price_60_order, ["doc-10d", "doc-30d"], "Price 60 group should be ordered by recency")

        # Price 40 group: doc-60d should be first, then doc-90d
        price_40_order = actual_order[13:15]
        self.assertEqual(price_40_order, ["doc-60d", "doc-90d"], "Price 40 group should be ordered by recency")

        # 3. Verify recency scores are calculated correctly for each doc
        self._verify_basic_recency_behavior(
            hits,
            decay_to=0.3,
            scale="120d",
            offset="0d",
            decay_function="exponential"
        )

    def test_with_collapsing_field(self):
        """Test recency + collapsing field picks highest scoring variant per parent.

        Document structure (future and past docs mixed in same groups):
        - group-A: doc-0d (today, score=1.0), doc-10d, doc+1d
        - group-B: doc-1d (closest to now), doc-14d, doc+3d
        - group-C: doc-3d (closest to now), doc-30d, doc+7d
        - group-D: doc-5d (closest to now), doc-60d, doc+14d
        - group-E: doc-7d (closest to now), doc-90d, doc+30d
        - group-F: doc-no-ts (only variant, score=0.3)

        With recency boosting and grow enabled, the variant closest to now
        should be selected for each parent group when collapsing:
        - Past docs closest to now have highest scores (~1.0)
        - Future docs have lower scores due to grow function
        - Old past docs have lowest scores due to decay

        Uses scale=120d for decay (slower) and grow_scale=60d for grow (faster)
        to ensure past docs closest to now win over equidistant future docs.
        """
        # Add documents to collapse index
        self._add_shared_documents(index=self.collapse_index)

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",  # Large scale so past docs decay slowly
            offset="0d",
            decay_function="exponential",
            decay_to=0.3,
            # Enable grow so future docs have distinct scores (closer = higher)
            grow_from=0.2,
            grow_function="exponential",
            grow_scale="60d",  # Faster decay for future docs
            grow_offset="0d"
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

        # 3. Verify the highest scoring variant is selected for each parent group
        # Past docs closest to now win because decay is slower than grow
        expected_winner = {
            "group-A": "doc-0d",   # 0d (score=1.0) beats doc-10d and doc+1d
            "group-B": "doc-1d",   # 1d ago beats doc-14d and doc+3d
            "group-C": "doc-3d",   # 3d ago beats doc-30d and doc+7d
            "group-D": "doc-5d",   # 5d ago beats doc-60d and doc+14d
            "group-E": "doc-7d",   # 7d ago beats doc-90d and doc+30d
            "group-F": "doc-no-ts",  # Only variant
        }

        for hit in hits:
            parent_id = hit.get('parent_id')
            doc_id = hit.get('_id')

            if parent_id in expected_winner:
                expected_doc = expected_winner[parent_id]
                self.assertEqual(
                    doc_id, expected_doc,
                    f"For {parent_id}, expected winner {expected_doc} but got {doc_id}"
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

    # NOTE: Version check tests for grow and addToScoreWeight are in unit tests
    # (test_hybrid_search.py::TestRecencyValidation) because integration tests
    # cannot create indexes with old schema versions.

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

    def test_grow_params_all_or_nothing_validation(self):
        """Test that grow parameters must be either all provided or all omitted."""
        # Partial combinations should fail
        partial_cases = [
            ("only_grow_from", {"grow_from": 0.5}),
            ("missing_grow_offset", {"grow_from": 0.5, "grow_function": "exponential", "grow_scale": "7d"}),
            ("missing_grow_scale", {"grow_from": 0.5, "grow_function": "exponential", "grow_offset": "0d"}),
        ]

        for test_name, grow_params in partial_cases:
            with self.subTest(test_name):
                with self.assertRaises(Exception) as ctx:
                    RecencyParameters(
                        recency_field="timestamp",
                        scale="7d",
                        decay_function="exponential",
                        decay_to=0.5,
                        **grow_params
                    )
                self.assertIn("all provided or all omitted", str(ctx.exception).lower())

        # All provided should work
        with self.subTest("all_provided"):
            params = RecencyParameters(
                recency_field="timestamp",
                scale="7d",
                decay_function="exponential",
                decay_to=0.5,
                grow_from=0.5,
                grow_function="exponential",
                grow_scale="7d",
                grow_offset="0d"
            )
            self.assertEqual(params.grow_from, 0.5)

    def test_grow_from_validation(self):
        """Test grow_from must be in (0.0, 1.0]."""
        valid_values = [0.01, 0.5, 1.0]
        invalid_values = [0.0, -0.1, 1.1]

        for val in valid_values:
            with self.subTest(grow_from=val, valid=True):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    decay_function="exponential",
                    decay_to=0.5,
                    grow_from=val,
                    grow_function="exponential",
                    grow_scale="7d",
                    grow_offset="0d"
                )
                self.assertEqual(params.grow_from, val)

        for val in invalid_values:
            with self.subTest(grow_from=val, valid=False):
                with self.assertRaises(Exception):
                    RecencyParameters(
                        recency_field="timestamp",
                        scale="7d",
                        decay_function="exponential",
                        decay_to=0.5,
                        grow_from=val,
                        grow_function="exponential",
                        grow_scale="7d",
                        grow_offset="0d"
                    )

    def test_grow_function_validation(self):
        """Test grow_function must be a valid function type."""
        valid_functions = ["exponential", "linear", "gaussian", "binary"]
        invalid_functions = ["invalid", "exp", "lin"]

        for func in valid_functions:
            with self.subTest(grow_function=func, valid=True):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    decay_function="exponential",
                    decay_to=0.5,
                    grow_from=0.3,
                    grow_function=func,
                    grow_scale="7d",
                    grow_offset="0d"
                )
                self.assertEqual(params.grow_function, func)

        for func in invalid_functions:
            with self.subTest(grow_function=func, valid=False):
                with self.assertRaises(Exception):
                    RecencyParameters(
                        recency_field="timestamp",
                        scale="7d",
                        decay_function="exponential",
                        decay_to=0.5,
                        grow_from=0.3,
                        grow_function=func,
                        grow_scale="7d",
                        grow_offset="0d"
                    )

    def test_grow_scale_validation(self):
        """Test grow_scale must be a valid duration format."""
        valid_formats = ["1d", "7d", "24h", "168h"]
        invalid_formats = ["-1d", "abc", "0d"]

        for fmt in valid_formats:
            with self.subTest(grow_scale=fmt, valid=True):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    decay_function="exponential",
                    decay_to=0.5,
                    grow_from=0.3,
                    grow_function="exponential",
                    grow_scale=fmt,
                    grow_offset="0d"
                )
                self.assertEqual(params.grow_scale, fmt)

        for fmt in invalid_formats:
            with self.subTest(grow_scale=fmt, valid=False):
                with self.assertRaises(Exception):
                    RecencyParameters(
                        recency_field="timestamp",
                        scale="7d",
                        decay_function="exponential",
                        decay_to=0.5,
                        grow_from=0.3,
                        grow_function="exponential",
                        grow_scale=fmt,
                        grow_offset="0d"
                    )

    def test_grow_offset_validation(self):
        """Test grow_offset must be a valid duration format."""
        valid_formats = ["0d", "1d", "7d", "24h"]
        invalid_formats = ["-1d", "abc"]

        for fmt in valid_formats:
            with self.subTest(grow_offset=fmt, valid=True):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    decay_function="exponential",
                    decay_to=0.5,
                    grow_from=0.3,
                    grow_function="exponential",
                    grow_scale="7d",
                    grow_offset=fmt
                )
                self.assertEqual(params.grow_offset, fmt)

        for fmt in invalid_formats:
            with self.subTest(grow_offset=fmt, valid=False):
                with self.assertRaises(Exception):
                    RecencyParameters(
                        recency_field="timestamp",
                        scale="7d",
                        decay_function="exponential",
                        decay_to=0.5,
                        grow_from=0.3,
                        grow_function="exponential",
                        grow_scale="7d",
                        grow_offset=fmt
                    )

    # ============== Grow Parameter Tests ==============

    def _verify_grow_behavior(
        self,
        hits: List[Dict],
        decay_to: float,
        grow_from: float,
        scale: str = "7d",
        offset: str = "0d",
        decay_function: str = "exponential",
        grow_function: str = None,
        grow_scale: str = None,
        grow_offset: str = None,
    ):
        """Verify recency scores match expected values for both decay and grow.

        Uses _calculate_expected_score which handles both past (decay) and future (grow).
        """
        if grow_function is None:
            grow_function = decay_function
        if grow_scale is None:
            grow_scale = scale

        for hit in hits:
            actual_score = hit.get('_recency_score')
            doc_id = hit.get('_id')

            self.assertIsNotNone(actual_score, f"Recency score should be present for {doc_id}")

            age_seconds = self._get_doc_age_seconds(hit)
            if age_seconds is not None:
                expected_score = self._calculate_expected_score(
                    age_seconds=age_seconds,
                    scale=scale,
                    offset=offset,
                    decay_function=decay_function,
                    decay_to=decay_to,
                    grow_enabled=True,
                    grow_from=grow_from,
                    grow_function=grow_function,
                    grow_scale=grow_scale,
                    grow_offset=grow_offset,
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

    def test_grow_disabled_by_default(self):
        """Test future timestamps get score 1.0 when growFrom is not specified.

        Shared documents include:
        - Past docs: doc-0d, doc-1d, doc-3d, ... doc-90d (use decay)
        - Future docs: doc+1d, doc+3d, doc+7d, doc+14d, doc+30d (all score 1.0)
        """
        self._add_shared_documents()

        # Without grow_from, future timestamps should get score 1.0
        params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5
            # No grow_from specified - grow disabled
        )
        hits = self._search_with_recency("product", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Verify all future documents have score exactly 1.0
        for hit in hits:
            doc_id = hit.get('_id')
            recency_score = hit.get('_recency_score')
            self.assertIsNotNone(recency_score, f"Recency score should be present for {doc_id}")

            if doc_id.startswith("doc+"):
                self.assertAlmostEqual(
                    recency_score, 1.0, places=3,
                    msg=f"Future doc {doc_id} should have score 1.0 when grow disabled"
                )

        # Verify past docs still use decay correctly
        self._verify_basic_recency_behavior(
            [h for h in hits if not h['_id'].startswith("doc+")],
            decay_to=0.5,
            scale="7d",
            offset="0d",
            decay_function="exponential"
        )

    def test_grow_functions(self):
        """Test all grow functions work correctly with precise score verification.

        Fixed decay params: scale=120d, decay_to=0.3, offset=0d, exponential
        Tests each grow function: exponential, linear, gaussian, binary
        """
        self._add_shared_documents()

        for grow_func in ["exponential", "linear", "gaussian", "binary"]:
            with self.subTest(function=grow_func):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="120d",  # Large decay scale so past docs are distinct
                    offset="0d",
                    decay_function="exponential",
                    decay_to=0.3,
                    grow_from=0.2,
                    grow_function=grow_func,
                    grow_scale="60d",  # Grow scale covers future docs
                    grow_offset="0d"
                )
                hits = self._search_with_recency("product", params)

                self.assertGreater(len(hits), 0, "Should have results")

                # Verify exact scores for all documents
                self._verify_grow_behavior(
                    hits,
                    decay_to=0.3,
                    grow_from=0.2,
                    scale="120d",
                    offset="0d",
                    decay_function="exponential",
                    grow_function=grow_func,
                    grow_scale="60d",
                    grow_offset="0d"
                )

    def test_grow_with_decay(self):
        """Test combined grow (future) and decay (past) with precise score verification.

        Verifies:
        - doc-0d has score ~1.0 (current)
        - Past docs have decay scores in [decay_to, 1.0]
        - Future docs have grow scores in [grow_from, 1.0]
        """
        self._add_shared_documents()

        params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="60d",
            grow_offset="0d"
        )
        hits = self._search_with_recency("product", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Verify all scores precisely
        self._verify_grow_behavior(
            hits,
            decay_to=0.5,
            grow_from=0.3,
            scale="120d",
            offset="0d",
            decay_function="exponential",
            grow_function="exponential",
            grow_scale="60d",
            grow_offset="0d"
        )

        # Additional category checks
        doc_0d = self._get_doc_by_id(hits, "doc-0d")
        if doc_0d:
            self.assertAlmostEqual(
                doc_0d.get('_recency_score'), 1.0, places=2,
                msg="doc-0d should have score ~1.0"
            )

        # Verify decay/grow floor bounds
        for hit in hits:
            doc_id = hit.get('_id')
            score = hit.get('_recency_score')

            if doc_id.startswith("doc-") and doc_id != "doc-0d" and doc_id != "doc-no-ts":
                self.assertGreaterEqual(score, 0.5, f"Past doc {doc_id} should be >= decay_to")
                self.assertLessEqual(score, 1.0, f"Past doc {doc_id} should be <= 1.0")
            elif doc_id.startswith("doc+"):
                self.assertGreaterEqual(score, 0.3, f"Future doc {doc_id} should be >= grow_from")
                self.assertLessEqual(score, 1.0, f"Future doc {doc_id} should be <= 1.0")

    def test_grow_with_linear_function(self):
        """Test grow with linear function produces correct scores.

        Use linear grow function and verify future docs have expected scores.
        """
        self._add_shared_documents()

        params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",
            offset="0d",
            decay_function="linear",
            decay_to=0.5,
            grow_from=0.3,
            grow_function="linear",
            grow_scale="60d",
            grow_offset="0d"
        )
        hits = self._search_with_recency("product", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Verify scores with linear grow function
        self._verify_grow_behavior(
            hits,
            decay_to=0.5,
            grow_from=0.3,
            scale="120d",
            offset="0d",
            decay_function="linear",
            grow_function="linear",
            grow_scale="60d",
            grow_offset="0d"
        )

    def test_grow_scale_at_boundary(self):
        """Test grow_scale boundary - doc at exactly scale gets grow_from score.

        Use grow_scale=30d and verify doc+30d has score ~grow_from.
        """
        self._add_shared_documents()

        params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="30d",
            grow_offset="0d"
        )
        hits = self._search_with_recency("product", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Verify scores
        self._verify_grow_behavior(
            hits,
            decay_to=0.5,
            grow_from=0.3,
            scale="120d",
            offset="0d",
            decay_function="exponential",
            grow_function="exponential",
            grow_scale="30d",
            grow_offset="0d"
        )

        # Specifically check doc+30d - at exactly scale, score should be at grow_from
        doc_30d_future = self._get_doc_by_id(hits, "doc+30d")
        if doc_30d_future:
            recency_score = doc_30d_future.get('_recency_score')
            self.assertAlmostEqual(
                recency_score, 0.3, places=2,
                msg="Future doc at exactly scale should have score ~grow_from"
            )

    def test_grow_offset_creates_plateau(self):
        """Test growOffset creates plateau zone where score = 1.0.

        With grow_offset=10d:
        - doc+1d, doc+3d, doc+7d should be in plateau (score = 1.0)
        - doc+14d, doc+30d should be beyond plateau (score < 1.0)
        """
        self._add_shared_documents()

        params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="60d",
            grow_offset="10d"  # Plateau zone: now() to now()+10d
        )
        hits = self._search_with_recency("product", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Verify scores with grow_offset plateau
        self._verify_grow_behavior(
            hits,
            decay_to=0.5,
            grow_from=0.3,
            scale="120d",
            offset="0d",
            decay_function="exponential",
            grow_function="exponential",
            grow_scale="60d",
            grow_offset="10d"
        )

        # Documents in plateau zone (1d, 3d, 7d < 10d) should have score 1.0
        plateau_docs = ["doc+1d", "doc+3d", "doc+7d"]
        for doc_id in plateau_docs:
            hit = self._get_doc_by_id(hits, doc_id)
            if hit:
                recency_score = hit.get('_recency_score')
                self.assertAlmostEqual(
                    recency_score, 1.0, places=2,
                    msg=f"Doc {doc_id} within plateau should have score 1.0"
                )

        # Documents beyond plateau (14d, 30d > 10d) should have score < 1.0
        beyond_plateau_docs = ["doc+14d", "doc+30d"]
        for doc_id in beyond_plateau_docs:
            hit = self._get_doc_by_id(hits, doc_id)
            if hit:
                recency_score = hit.get('_recency_score')
                self.assertLess(
                    recency_score, 1.0,
                    msg=f"Doc {doc_id} beyond plateau should have score < 1.0"
                )

    def test_grow_binary_function(self):
        """Test binary grow function creates step function at scale.

        With grow_scale=10d:
        - doc+1d, doc+3d, doc+7d (< 10d) should have score 1.0
        - doc+14d, doc+30d (>= 10d) should have score = grow_from
        """
        self._add_shared_documents()

        params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            grow_from=0.2,
            grow_function="binary",
            grow_scale="10d",  # Step at 10 days
            grow_offset="0d"
        )
        hits = self._search_with_recency("product", params)

        self.assertGreater(len(hits), 0, "Should have results")

        # Verify exact scores with binary grow
        self._verify_grow_behavior(
            hits,
            decay_to=0.5,
            grow_from=0.2,
            scale="120d",
            offset="0d",
            decay_function="exponential",
            grow_function="binary",
            grow_scale="10d",
            grow_offset="0d"
        )

        # Documents within scale (< 10d) should have score 1.0
        within_scale_docs = ["doc+1d", "doc+3d", "doc+7d"]
        for doc_id in within_scale_docs:
            hit = self._get_doc_by_id(hits, doc_id)
            if hit:
                recency_score = hit.get('_recency_score')
                self.assertAlmostEqual(
                    recency_score, 1.0, places=2,
                    msg=f"Doc {doc_id} within binary scale should have score 1.0"
                )

        # Documents beyond scale (>= 10d) should have score = grow_from
        beyond_scale_docs = ["doc+14d", "doc+30d"]
        for doc_id in beyond_scale_docs:
            hit = self._get_doc_by_id(hits, doc_id)
            if hit:
                recency_score = hit.get('_recency_score')
                self.assertAlmostEqual(
                    recency_score, 0.2, places=2,
                    msg=f"Doc {doc_id} beyond binary scale should have score ~grow_from"
                )

    # ============== Add To Score Weight Tests ==============

    def test_add_to_score_weight_values(self):
        """Test addToScoreWeight with various weight values.

        Fixed params: scale=7d, decay_to=0.5, grow_from=0.3, exponential
        Tests weights: [0.1, 1.0, 10.0, 100.0] (must be > 0.0)

        Verifies:
        - Recency scores are calculated identically regardless of weight
        - Different weights only affect final _score, not _recency_score
        """
        self._add_shared_documents()

        weight_values = [0.1, 1.0, 10.0, 100.0]  # All must be > 0.0
        results_by_weight = {}

        for weight in weight_values:
            with self.subTest(addToScoreWeight=weight):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    decay_to=0.5,
                    grow_from=0.3,
                    grow_function="exponential",
                    grow_scale="7d",
                    grow_offset="0d",
                    add_to_score_weight=weight
                )
                hits = self._search_with_recency("product", params)

                self.assertGreater(len(hits), 0, "Should have results")
                results_by_weight[weight] = {h['_id']: h for h in hits}

                # Verify recency scores match expected values
                self._verify_grow_behavior(
                    hits,
                    decay_to=0.5,
                    grow_from=0.3,
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    grow_function="exponential",
                    grow_scale="7d",
                    grow_offset="0d"
                )

        # Verify recency scores are identical across all weight values
        base_results = results_by_weight[0.1]
        for weight in [1.0, 10.0, 100.0]:
            for doc_id, base_hit in base_results.items():
                if doc_id in results_by_weight[weight]:
                    other_hit = results_by_weight[weight][doc_id]
                    self.assertAlmostEqual(
                        base_hit.get('_recency_score', 0),
                        other_hit.get('_recency_score', 0),
                        places=3,
                        msg=f"Recency scores should be identical for {doc_id} across weights"
                    )


if __name__ == '__main__':
    unittest.main()
