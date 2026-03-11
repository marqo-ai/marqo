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

from pydantic.v1 import ValidationError

from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.models.recency_parameters import RecencyParameters, ApplyInRankingPhase
from marqo.tensor_search.models.relevance_cutoff_model import (
    RelevanceCutoffModel, RelevanceCutoffMethod
)
from marqo.tensor_search.models.sort_by_model import SortByModel, SortByField
from tests.integ_tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.collapse_model import CollapseModel


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

        # 5. 2.24.7 does not support recency feature
        cls.old_index_2247_request = cls.unstructured_marqo_index_request(
            marqo_version="2.24.7",
            schema_template_version="2.24.7",
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        # 5. 2.24.8 does not support grow params and add_to_score_weight
        cls.old_index_2248_request = cls.unstructured_marqo_index_request(
            marqo_version="2.24.8",
            schema_template_version="2.24.8",
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        cls.indexes = cls.create_indexes([
            cls.main_index_request,
            cls.collapse_index_request,
            cls.structured_index_request,
            cls.relevance_cutoff_index_request,
            cls.old_index_2247_request,
            cls.old_index_2248_request
        ])

        cls.main_index = cls.indexes[0]
        cls.collapse_index = cls.indexes[1]
        cls.structured_index = cls.indexes[2]
        cls.relevance_cutoff_index = cls.indexes[3]
        cls.old_index_2247 = cls.indexes[4]
        cls.old_index_2248 = cls.indexes[5]

    # ============== Helper Methods ==============
    def _generate_shared_documents(self) -> List[Dict[str, Any]]:
        """Generate shared documents with various ages (past and future) and attributes.

        Naming convention:
        - Past docs: doc-Xd where X is days ago (e.g., doc-0d = today, doc-7d = 7 days ago)
        - Future docs: doc+Xd where X is days in future (e.g., doc+1d = 1 day from now)
        """
        now = datetime.now()

        # Explicit config for each document: (price, parent_id, mult)
        # age > 0 means days in past, age < 0 means days in future
        doc_configs = {
            # Future docs (negative ages = future timestamps)
            # Mixed into same groups as past docs
            -30: (80, "group-C", 1.0),   # doc+30d - with doc-3d, doc-30d: C
            -14: (40, "group-D", 1.5),   # doc+14d - with doc-5d, doc-60d: D
            -7:  (60, "group-E", 2.0),   # doc+7d - with doc-7d, doc-90d: E
            -3:  (100, "group-A", 1.0),   # doc+3d - with doc-0d, doc-10d: A
            -1:  (80, "group-B", 1.5),   # doc+1d - with doc-1d, doc-14d: B

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
        # TODO add future documents
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
        grow_from: float = None,
        grow_function: str = None,
        grow_scale: str = None,
        grow_offset: str = None,
    ) -> float:
        """Calculate expected recency score matching the rank profile logic.

        Unified logic handles both decay (past) and grow (future) timestamps:
        - age_seconds > 0: past document, use decay logic
        - age_seconds < 0: future document, use grow logic (if enabled) or return 1.0
        - age_seconds == 0: current document, score = 1.0
        """
        scale_seconds = self._parse_duration_to_seconds(scale)
        offset_seconds = self._parse_duration_to_seconds(offset)

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

        # Future document (age < 0): use grow logic if enabled, return 1.0 if not
        if grow_from is None: # if not enabled
            return 1.0

        future_age = -age_seconds  # Convert to positive
        grow_scale_seconds = self._parse_duration_to_seconds(grow_scale)
        grow_offset_seconds = self._parse_duration_to_seconds(grow_offset)

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

    def _get_doc_age_seconds(self, hit: Dict, center: Optional[float] = None) -> Optional[float]:
        """Get the age in seconds for a document based on its timestamp field.

        Args:
            hit: Search result document
            center: Fixed reference timestamp (Unix epoch seconds). Uses now() if None.

        Returns:
            - Positive value: past document (timestamp < reference)
            - Negative value: future document (timestamp > reference)
            - None: document has no timestamp field
        """
        doc_id = hit.get('_id')
        if doc_id == "doc-no-ts":
            return None  # No timestamp

        # Use the actual timestamp from the document
        timestamp = hit.get('timestamp')
        if timestamp is not None:
            reference_time = center if center is not None else datetime.now().timestamp()
            return reference_time - timestamp  # Can be negative for future docs
        return None

    # ============== Verification Helpers ==============

    def _verify_recency_behavior(
        self,
        hits: List[Dict],
        recency_params: RecencyParameters
    ):
        """Verify recency scores match expected values within 3 decimal places."""
        self.assertGreater(len(hits), 0, "Should have results")

        for hit in hits:
            actual_score = hit.get('_recency_score')
            doc_id = hit.get('_id')

            self.assertIsNotNone(actual_score, f"Recency score should be present for {doc_id}")

            # Calculate expected score
            age_seconds = self._get_doc_age_seconds(hit, center=recency_params.center)
            if age_seconds is not None:
                expected_score = self._calculate_expected_score(
                    age_seconds, recency_params.scale, recency_params.offset, recency_params.decay_function, recency_params.decay_to,
                    recency_params.grow_from, recency_params.grow_function, recency_params.grow_scale, recency_params.grow_offset
                )
                self.assertAlmostEqual(
                    expected_score,
                    actual_score,
                    places=3,
                    msg=f"Score mismatch for {doc_id}: expected {expected_score:.4f}, got {actual_score:.4f}"
                )
            elif doc_id == "doc-no-ts":
                # Document without timestamp should get decay_to
                self.assertAlmostEqual(
                    recency_params.decay_to,
                    actual_score,
                    places=3,
                    msg=f"Doc without timestamp should have decay_to score"
                )

    def _get_doc_by_id(self, hits: List[Dict], doc_id: str) -> Optional[Dict]:
        """Find document by ID in search results."""
        return next((h for h in hits if h.get('_id') == doc_id), None)

    def _extract_scores(self, hits: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Extract score map including field values for modifier calculation.

        Returns:
            Dict mapping doc_id -> {lexical, tensor, score, recency, mult, timestamp}
        """
        result = {}
        for hit in hits:
            doc_id = hit['_id']
            result[doc_id] = {
                'lexical': hit.get('_lexical_score'),
                'tensor': hit.get('_tensor_score'),
                'score': hit.get('_score'),
                'recency': hit.get('_recency_score'),
                # Field values for global score modifier calculation
                'mult': hit.get('mult'),
                'timestamp': hit.get('timestamp'),
            }
        return result

    def _calculate_rrf_score(
        self,
        scores: Dict[str, Dict[str, float]],
        alpha: float = 0.5,
        k: int = 60
    ) -> Dict[str, float]:
        """Calculate RRF scores from lexical and tensor scores.

        RRF (Reciprocal Rank Fusion) formula:
        - tensor_rrf = alpha * (1.0 / (tensor_rank + k))
        - lexical_rrf = (1 - alpha) * (1.0 / (lexical_rank + k))
        - combined = tensor_rrf + lexical_rrf (if doc in both)

        Args:
            scores: Dict mapping doc_id -> {lexical, tensor, recency}
            alpha: Weight for tensor vs lexical (default 0.5)
            k: RRF constant (default 60)

        Returns:
            Dict mapping doc_id -> calculated RRF score
        """
        # Extract and rank by lexical scores (descending)
        lexical_scores = [
            (doc_id, s['lexical'])
            for doc_id, s in scores.items()
            if s['lexical'] is not None
        ]
        lexical_scores.sort(key=lambda x: x[1], reverse=True)
        lexical_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(lexical_scores)}

        # Extract and rank by tensor scores (descending)
        tensor_scores = [
            (doc_id, s['tensor'])
            for doc_id, s in scores.items()
            if s['tensor'] is not None
        ]
        tensor_scores.sort(key=lambda x: x[1], reverse=True)
        tensor_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(tensor_scores)}

        # Calculate RRF for each document
        rrf_scores = {}
        all_doc_ids = set(lexical_ranks.keys()) | set(tensor_ranks.keys())

        for doc_id in all_doc_ids:
            rrf = 0.0

            # Tensor contribution
            if doc_id in tensor_ranks and alpha > 0:
                rrf += alpha * (1.0 / (tensor_ranks[doc_id] + k))

            # Lexical contribution
            if doc_id in lexical_ranks and alpha < 1.0:
                rrf += (1.0 - alpha) * (1.0 / (lexical_ranks[doc_id] + k))

            rrf_scores[doc_id] = rrf

        return rrf_scores

    def _calculate_global_modifiers(
        self,
        doc_scores: Dict[str, float],
        multiply_weights: Dict[str, float],
        add_weights: Dict[str, float]
    ) -> tuple:
        """Calculate global score modifiers from document field values.

        Formula from Vespa schema template:
        - mult_modifier = product of (weight * field_value) for each field
        - add_modifier = sum of (weight * field_value) for each field

        Args:
            doc_scores: Dict with document field values (e.g., {'mult': 2.0, 'timestamp': 1.7e9})
            multiply_weights: Dict of {field_name: weight} for multiply_score_by
            add_weights: Dict of {field_name: weight} for add_to_score

        Returns:
            (mult_modifier, add_modifier) tuple
        """
        # Multiplicative: product of (weight * field_value)
        mult_modifier = 1.0
        for field_name, weight in multiply_weights.items():
            field_value = doc_scores.get(field_name)
            if field_value is not None:
                mult_modifier *= (weight * field_value)

        # Additive: sum of (weight * field_value)
        add_modifier = 0.0
        for field_name, weight in add_weights.items():
            field_value = doc_scores.get(field_name)
            if field_value is not None:
                add_modifier += (weight * field_value)

        return mult_modifier, add_modifier

    def _verify_phase_score_changes(
        self,
        baseline: Dict[str, Dict],
        recency: Dict[str, Dict],
        phase: str,
        add_weight: Optional[float],
        multiply_weights: Optional[Dict[str, float]] = None,
        add_weights: Optional[Dict[str, float]] = None,
        rerank_depth: Optional[int] = None,
    ):
        """Verify scores changed according to apply_in_ranking_phase setting.

        Verifies both:
        1. Which scores changed based on phase setting
        2. The actual score values match expected calculation:
           - Multiplicative (add_weight=None): expected = original * recency
           - Additive (add_weight provided): expected = original + recency * add_weight

        Phase behavior:
        - 'all': recency applied to lexical, tensor, AND global RRF score
        - 'exclude-global': recency applied to lexical and tensor only
        - 'only-global': recency applied to global RRF score only

        Global score modifiers are calculated from document field values:
        - mult_modifier = product of (weight * field_value) for each field in multiply_weights
        - add_modifier = sum of (weight * field_value) for each field in add_weights
        - base_score = rrf * mult_modifier + add_modifier

        rerank_depth: If provided, only the top N documents (by RRF score) get global
        phase processing (modifiers + recency). Documents beyond this depth keep raw RRF.
        """
        common_docs = set(baseline.keys()) & set(recency.keys())
        self.assertGreater(len(common_docs), 0, "Should have common docs")

        # Find docs where recency was actually applied (score != 1.0)
        affected_docs = [
            doc_id for doc_id in common_docs
            if recency[doc_id]['recency'] is not None
            and abs(recency[doc_id]['recency'] - 1.0) > 0.01
        ]
        self.assertGreater(len(affected_docs), 0, "Should have docs affected by recency")

        def calculate_expected_recency(original: float, recency_score: float) -> float:
            """Calculate expected score after recency is applied."""
            if add_weight is not None:
                return original + recency_score * add_weight
            else:
                return original * recency_score

        # Calculate RRF scores and ranks (needed for rerank_depth logic)
        rrf_scores = self._calculate_rrf_score(recency)

        # Create a ranking of documents by RRF score (for rerank_depth check)
        # Rank is 1-indexed (top doc has rank 1)
        rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        doc_rrf_rank = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(rrf_ranked)}

        for doc_id in affected_docs:
            b = baseline[doc_id]
            r = recency[doc_id]
            recency_score = r['recency']

            # Skip if missing score data
            if b['lexical'] is None or b['tensor'] is None or b['score'] is None:
                continue

            # Check if this doc is within rerank_depth (gets global phase processing)
            doc_rank = doc_rrf_rank.get(doc_id, float('inf'))
            within_rerank_depth = rerank_depth is None or doc_rank <= rerank_depth

            if phase == "all":
                # Lexical and tensor scores should be modified by recency
                expected_lexical = calculate_expected_recency(b['lexical'], recency_score)
                expected_tensor = calculate_expected_recency(b['tensor'], recency_score)

                self.assertAlmostEqual(
                    expected_lexical, r['lexical'], places=3,
                    msg=f"Doc {doc_id}: lexical score mismatch. "
                    f"Expected {expected_lexical:.6f}, got {r['lexical']:.6f}"
                )
                self.assertAlmostEqual(
                    expected_tensor, r['tensor'], places=3,
                    msg=f"Doc {doc_id}: tensor score mismatch. "
                    f"Expected {expected_tensor:.6f}, got {r['tensor']:.6f}"
                )

                # Calculate expected RRF from modified lexical/tensor scores
                expected_rrf = rrf_scores.get(doc_id, 0)

                if within_rerank_depth:
                    # Apply global score modifiers if present
                    if multiply_weights or add_weights:
                        mult_mod, add_mod = self._calculate_global_modifiers(
                            r, multiply_weights or {}, add_weights or {}
                        )
                        base_score = expected_rrf * mult_mod + add_mod
                    else:
                        base_score = expected_rrf

                    # Apply recency to final score (global phase)
                    expected_final = calculate_expected_recency(base_score, recency_score)
                else:
                    # Beyond rerank_depth: no global modifiers or recency applied to RRF
                    expected_final = expected_rrf

                self.assertAlmostEqual(
                    expected_final, r['score'], places=3,
                    msg=f"Doc {doc_id} (rank {doc_rank}): RRF score mismatch. "
                    f"Expected {expected_final:.6f}, got {r['score']:.6f}"
                )

            elif phase == "exclude-global":
                # Lexical and tensor modified by recency, but NOT the global RRF score
                expected_lexical = calculate_expected_recency(b['lexical'], recency_score)
                expected_tensor = calculate_expected_recency(b['tensor'], recency_score)

                self.assertAlmostEqual(
                    expected_lexical, r['lexical'], places=3,
                    msg=f"Doc {doc_id}: lexical score mismatch. "
                    f"Expected {expected_lexical:.6f}, got {r['lexical']:.6f}"
                )
                self.assertAlmostEqual(
                    expected_tensor, r['tensor'], places=3,
                    msg=f"Doc {doc_id}: tensor score mismatch. "
                    f"Expected {expected_tensor:.6f}, got {r['tensor']:.6f}"
                )

                # Calculate expected RRF from modified lexical/tensor scores
                # NO recency applied to RRF in global phase
                expected_rrf = rrf_scores.get(doc_id, 0)

                if within_rerank_depth:
                    # Apply global score modifiers if present (but no recency)
                    if multiply_weights or add_weights:
                        mult_mod, add_mod = self._calculate_global_modifiers(
                            r, multiply_weights or {}, add_weights or {}
                        )
                        expected_final = expected_rrf * mult_mod + add_mod
                    else:
                        expected_final = expected_rrf
                else:
                    # Beyond rerank_depth: no global modifiers applied
                    expected_final = expected_rrf

                self.assertAlmostEqual(
                    expected_final, r['score'], places=3,
                    msg=f"Doc {doc_id} (rank {doc_rank}): RRF score mismatch. "
                    f"Expected {expected_final:.6f}, got {r['score']:.6f}"
                )

            elif phase == "only-global":
                # Lexical and tensor should remain unchanged
                self.assertAlmostEqual(
                    b['lexical'], r['lexical'], places=5,
                    msg=f"Doc {doc_id}: lexical should be unchanged. "
                    f"Baseline {b['lexical']:.6f}, got {r['lexical']:.6f}"
                )
                self.assertAlmostEqual(
                    b['tensor'], r['tensor'], places=5,
                    msg=f"Doc {doc_id}: tensor should be unchanged. "
                    f"Baseline {b['tensor']:.6f}, got {r['tensor']:.6f}"
                )

                if within_rerank_depth:
                    # RRF score should be modified by recency
                    # (baseline already has modifiers applied, so we just apply recency)
                    expected_score = calculate_expected_recency(b['score'], recency_score)
                else:
                    # Beyond rerank_depth: no recency applied, score equals baseline
                    expected_score = b['score']

                self.assertAlmostEqual(
                    expected_score, r['score'], places=3,
                    msg=f"Doc {doc_id} (rank {doc_rank}): RRF score mismatch. "
                        f"Expected {expected_score:.6f}, got {r['score']:.6f}"
                )

    # ============== Core Decay Function Tests ==============

    def test_decay_and_grow_functions(self):
        """Test all decay functions work correctly."""
        self._add_shared_documents()

        for decay_and_grow_func in ["exponential", "linear", "gaussian", "binary"]:
            with self.subTest(function=decay_and_grow_func):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="8d",
                    offset="0d",
                    decay_function=decay_and_grow_func,
                    decay_to=0.5,
                    grow_function=decay_and_grow_func,
                    grow_from=0.2,
                    grow_scale="3d",
                    grow_offset="1d",
                )
                hits = self._search_with_recency("product", params)
                self._verify_recency_behavior(hits, params)

    def test_scale_offset_combinations(self):
        """Test representative scale/offset combinations."""
        self._add_shared_documents()

        test_cases = [
            ("7d", "0d"),   # No offset
            ("3d", "0d"),  # Short scale
            ("14d", "7d"),  # Large scale with offset
        ]

        for scale, offset in test_cases:
            with self.subTest(scale=scale, offset=offset):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale=scale,
                    offset=offset,
                    decay_function="exponential",
                    decay_to=0.5,
                    grow_from=0.2,
                    grow_function="linear",
                    grow_scale=scale,
                    grow_offset=offset,
                )
                hits = self._search_with_recency("product", params)
                self._verify_recency_behavior(hits, params)

    def test_decay_to_grow_from_values(self):
        """Test various decay_to floor values."""
        self._add_shared_documents()

        for decay_to_grow_from in [0.1, 0.3, 0.5, 0.8]:
            with self.subTest(decay_to_grow_from=decay_to_grow_from):
                params = RecencyParameters(
                    recency_field="timestamp",
                    scale="7d",
                    offset="0d",
                    decay_function="exponential",
                    decay_to=decay_to_grow_from,
                    grow_from=decay_to_grow_from,
                    grow_function="linear",
                    grow_scale="3d",
                    grow_offset="1d",
                )
                hits = self._search_with_recency("product", params)
                self._verify_recency_behavior(hits, params)

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
        self._verify_recency_behavior([h for h in hits if not h['_id'].startswith("doc+")], params)

    # ============== Apply in Ranking Phase and Add To Score Weight Tests ==============
    @pytest.mark.skip_for_multinode(
        "Multi-nodes will return different lexical results so we can not assert on the results.")
    def test_apply_in_ranking_phase_with_score_modifiers(self):
        """Comprehensive test for apply_in_ranking_phase and add_to_score_weight.

        Verifies that recency scoring is applied correctly in different ranking phases:
        - 'all': Recency modifies lexical_score, tensor_score, AND rrf_score
        - 'exclude-global': Recency modifies lexical_score and tensor_score only
        - 'only-global': Recency modifies rrf_score only

        Also tests:
        - With and without score modifiers
        - Different add_to_score_weight values (multiplicative vs additive)
        """
        self._add_shared_documents()

        # Test configurations
        # (apply_in_ranking_phase, add_to_score_weight, use_score_modifiers, rerank_depth)
        test_cases = [
            # Standard cases without rerank_depth limit
            ("all", None, True, None),           # Multiplicative, with score mods
            ("all", None, False, None),          # Multiplicative, without score mods
            ("all", 10.0, True, None),           # Additive, with score mods
            ("exclude-global", None, True, None),
            ("exclude-global", None, False, None),
            ("exclude-global", 10.0, True, None),
            ("only-global", None, True, None),
            ("only-global", None, False, None),
            ("only-global", 10.0, True, None),

            # These verify that docs beyond rerank_depth don't get global processing, only top 5 get modifiers+recency in global
            ("all", None, True, 5),
            ("exclude-global", None, True, 5),
            ("only-global", None, True, 5),
        ]

        from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists

        for phase, add_weight, use_score_mods, rerank_depth in test_cases:
            with self.subTest(phase=phase, add_to_score_weight=add_weight, score_mods=use_score_mods, rerank_depth=rerank_depth):
                # 1. Build score modifiers (if enabled)
                score_mods = None
                multiply_weights = None
                add_weights = None
                if use_score_mods:
                    score_mods = ScoreModifierLists(
                        multiply_score_by=[{"field_name": "mult", "weight": 5.0}],
                        add_to_score=[{"field_name": "timestamp", "weight": 1e-8}]
                    )
                    # Extract weights for verification
                    multiply_weights = {"mult": 5.0}
                    add_weights = {"timestamp": 1e-8}

                # 2. Baseline search: WITH score mods, WITHOUT recency
                baseline_result = tensor_search.search(
                    config=self.config,
                    index_name=self.main_index.name,
                    text="product",
                    search_method=SearchMethod.HYBRID,
                    hybrid_parameters=HybridParameters(
                        scoreModifiersTensor=score_mods,
                        scoreModifiersLexical=score_mods,
                    ),
                    score_modifiers=score_mods,
                    result_count=20,
                    rerank_depth=rerank_depth,  # Controls global phase processing (score modifiers + recency)
                )
                baseline_scores = self._extract_scores(baseline_result['hits'])

                # 3. Recency search: WITH score mods AND recency
                recency_params = RecencyParameters(
                    recency_field="timestamp",
                    scale="60d",  # Moderate scale for varied scores
                    offset="0d",
                    decay_function="exponential",
                    decay_to=0.3,
                    grow_from=0.2,
                    grow_function="linear",
                    grow_scale="45d",
                    grow_offset="1d",
                    apply_in_ranking_phase=phase,
                    add_to_score_weight=add_weight
                )
                recency_result = tensor_search.search(
                    config=self.config,
                    index_name=self.main_index.name,
                    text="product",
                    search_method=SearchMethod.HYBRID,
                    hybrid_parameters=HybridParameters(
                        scoreModifiersTensor=score_mods,
                        scoreModifiersLexical=score_mods,
                    ),
                    recency_parameters=recency_params,
                    score_modifiers=score_mods,
                    result_count=20,
                    rerank_depth=rerank_depth,  # Controls global phase processing (score modifiers + recency)
                )
                recency_scores = self._extract_scores(recency_result['hits'])

                # 4. Verify scores changed according to phase setting
                self._verify_phase_score_changes(
                    baseline_scores, recency_scores, phase, add_weight,
                    multiply_weights=multiply_weights,
                    add_weights=add_weights,
                    rerank_depth=rerank_depth
                )

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
            decay_to=0.5,
            grow_from=0.2,
            grow_function="linear",
            grow_scale="3d",
            grow_offset="1d"
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
                self._verify_recency_behavior(search_result['hits'], params)

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

        When sorted by price desc, documents with same price should be
        ordered by recency (newer docs first) as a tie-breaker.
        """
        self._add_shared_documents()

        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="120d",  # Large scale so all docs have distinct recency scores
            offset="0d",
            decay_function="exponential",
            decay_to=0.3,
            # faster grow
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="90d",
            grow_offset="0d",
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
        expected_order = [
            "doc-0d", "doc-3d", "doc+3d", "doc-7d",  # price=100
            "doc-1d", "doc+1d", "doc-5d", "doc-14d", "doc+30d",  # price=80
            "doc+7d", "doc-10d", "doc-30d",  # price=60
            "doc+14d", "doc-60d", "doc-90d",  # price=40
            "doc-no-ts",  # price=20
        ]

        self.assertListEqual(expected_order, actual_order)

        # 2. Verify recency scores are calculated correctly for each doc
        self._verify_recency_behavior(hits, recency_params)

    @pytest.mark.skip_for_multinode(
        "Multi-nodes will return different lexical results so we can not assert on the results.")
    def test_with_collapsing_field(self):
        """Test recency + collapsing field picks highest scoring variant per parent.

        Document structure (future and past docs mixed in same groups):
        - group-A: doc-0d (today, score=1.0), doc-10d, doc+3d (offset-2d)
        - group-B: doc+1d (closest to now, with offset-2), doc-1d, doc-14d
        - group-C: doc-3d (closest to now), doc-30d, doc+30d (offset-2d)
        - group-D: doc-5d (closest to now), doc-60d, doc+14d (offset-2d)
        - group-E: doc+7d (closest to now, with offset-2), doc-7d, doc-90d, doc+7d
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
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="60d",  # Faster decay for future docs
            grow_offset="2d",  # Two days plateau before release date
        )

        search_result = tensor_search.search(
            config=self.config,
            index_name=self.collapse_index.name,
            text="product",
            search_method=SearchMethod.HYBRID,
            recency_parameters=recency_params,
            collapse=CollapseModel(name="parent_id"),
            result_count=10,
            hybrid_parameters=HybridParameters(
                rerankDepthTensor=20
            )
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
            "group-A": "doc-0d",   # 0d (score=1.0) beats doc-10d and doc+3d
            "group-B": "doc+1d",   # 1d away (with growOffset=2, score=1.0) beats doc-14d and doc-1d
            "group-C": "doc-3d",   # 3d ago beats doc-30d and doc+30d
            "group-D": "doc-5d",   # 5d ago beats doc-60d and doc+14d
            "group-E": "doc-7d",   # 7d ago beats doc-90d and doc+7d (with growOffset=2, but with quicker decay)
            "group-F": "doc-no-ts",  # Only variant
        }

        for hit in hits:
            parent_id = hit.get('parent_id')
            doc_id = hit.get('_id')

            if parent_id in expected_winner:
                expected_doc = expected_winner[parent_id]
                self.assertEqual(
                    expected_doc, doc_id,
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
        for search_method in ["TENSOR", "LEXICAL"]:
            with self.subTest(search_method=search_method):
                with self.assertRaises(ValidationError) as ctx:
                    SearchQuery(
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

    def test_apply_to_subqueries_non_hybrid_fails(self):
        """applyToSubqueries requires HYBRID search method."""
        for search_method in ["TENSOR", "LEXICAL"]:
            with self.subTest(search_method=search_method):
                with self.assertRaises(ValidationError) as ctx:
                    SearchQuery(
                        q="product",
                        searchMethod=search_method,
                        recencyParameters={
                            "recencyField": "timestamp",
                            "scale": "7d",
                            "decayTo": 0.5,
                            "applyToSubqueries": ["tensor"],
                        }
                    )
                self.assertIn("HYBRID", str(ctx.exception))

    def test_apply_to_subqueries_non_disjunction_fails(self):
        """applyToSubqueries requires disjunction retrieval method."""
        with self.assertRaises(ValidationError) as ctx:
            SearchQuery(
                q="product",
                searchMethod="HYBRID",
                hybridParameters={"retrievalMethod": "lexical", "rankingMethod": "lexical"},
                recencyParameters={
                    "recencyField": "timestamp",
                    "scale": "7d",
                    "decayTo": 0.5,
                    "applyToSubqueries": ["tensor"],
                }
            )
        self.assertIn("disjunction", str(ctx.exception).lower())

    def test_sort_by_with_non_exclude_global_fails(self):
        """sortBy + recency should fail unless exclude-global."""
        for phase in ["all", "only-global"]:
            with self.subTest(phase=phase):
                with self.assertRaises(ValidationError) as ctx:
                    SearchQuery(
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

    def test_recency_on_old_index_2247_not_supported(self):
        """Recency parameters should fail on index created with schema version 2.24.7."""
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
                index_name=self.old_index_2247.name,
                text="product",
                search_method=SearchMethod.HYBRID,
                recency_parameters=recency_params,
                result_count=10
            )

        # Verify error message mentions minimum version requirement
        self.assertIn("2.24.8", str(ctx.exception), "Error should mention minimum version 2.24.8")

    def test_grow_params_on_old_index_2248_not_supported(self):
        """Grow parameters should fail on index created with schema version 2.24.8."""
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            # Grow parameters - not supported on 2.24.8
            grow_from=0.3,
            grow_function="exponential",
            grow_scale="7d",
            grow_offset="0d"
        )

        with self.assertRaises(UnsupportedFeatureError) as ctx:
            tensor_search.search(
                config=self.config,
                index_name=self.old_index_2248.name,
                text="product",
                search_method=SearchMethod.HYBRID,
                recency_parameters=recency_params,
                result_count=10
            )

        # Verify error message mentions growFrom and minimum version
        error_message = str(ctx.exception)
        self.assertIn("growFrom", error_message, "Error should mention growFrom parameter")

    def test_add_to_score_weight_on_old_index_2248_not_supported(self):
        """addToScoreWeight parameter should fail on index created with schema version 2.24.8."""
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.5,
            add_to_score_weight=1.0  # Not supported on 2.24.8
        )

        with self.assertRaises(UnsupportedFeatureError) as ctx:
            tensor_search.search(
                config=self.config,
                index_name=self.old_index_2248.name,
                text="product",
                search_method=SearchMethod.HYBRID,
                recency_parameters=recency_params,
                result_count=10
            )

        # Verify error message mentions addToScoreWeight and minimum version
        error_message = str(ctx.exception)
        self.assertIn("addToScoreWeight", error_message, "Error should mention addToScoreWeight parameter")

    # ============== Parameter Validation Tests ==============

    def test_recency_parameter_validation_errors(self):
        """Test validation errors for RecencyParameters.

        Covers error paths in recency_parameters.py and duration_parser.py:
        - Empty recency_field
        - Invalid scale/offset format
        - Zero or negative scale/offset values
        - Partial grow parameters
        """
        validation_cases = [
            # (description, kwargs, expected_error_substring)
            (
                "empty recency_field",
                {"recency_field": "", "scale": "7d", "decay_function": "exponential", "decay_to": 0.5},
                "recency_field cannot be empty"
            ),
            (
                "whitespace recency_field",
                {"recency_field": "   ", "scale": "7d", "decay_function": "exponential", "decay_to": 0.5},
                "recency_field cannot be empty"
            ),
            (
                "invalid scale format",
                {"recency_field": "timestamp", "scale": "invalid", "decay_function": "exponential", "decay_to": 0.5},
                "Invalid scale format"
            ),
            (
                "zero scale",
                {"recency_field": "timestamp", "scale": "0d", "decay_function": "exponential", "decay_to": 0.5},
                "scale must be greater than 0"
            ),
            (
                "invalid offset format",
                {"recency_field": "timestamp", "scale": "7d", "offset": "bad", "decay_function": "exponential", "decay_to": 0.5},
                "Invalid offset format"
            ),
            (
                "negative offset",
                {"recency_field": "timestamp", "scale": "7d", "offset": "-1d", "decay_function": "exponential", "decay_to": 0.5},
                "Invalid offset format"  # Regex doesn't match negative values
            ),
            (
                "invalid grow_scale format",
                {"recency_field": "timestamp", "scale": "7d", "decay_function": "exponential", "decay_to": 0.5,
                 "grow_from": 0.5, "grow_function": "exponential", "grow_scale": "invalid", "grow_offset": "0d"},
                "Invalid grow_scale format"
            ),
            (
                "zero grow_scale",
                {"recency_field": "timestamp", "scale": "7d", "decay_function": "exponential", "decay_to": 0.5,
                 "grow_from": 0.5, "grow_function": "exponential", "grow_scale": "0d", "grow_offset": "0d"},
                "grow_scale must be greater than 0"
            ),
            (
                "invalid grow_offset format",
                {"recency_field": "timestamp", "scale": "7d", "decay_function": "exponential", "decay_to": 0.5,
                 "grow_from": 0.5, "grow_function": "exponential", "grow_scale": "7d", "grow_offset": "bad"},
                "Invalid grow_offset format"
            ),
            (
                "negative grow_offset",
                {"recency_field": "timestamp", "scale": "7d", "decay_function": "exponential", "decay_to": 0.5,
                 "grow_from": 0.5, "grow_function": "exponential", "grow_scale": "7d", "grow_offset": "-1d"},
                "Invalid grow_offset format"  # Regex doesn't match negative values
            ),
            (
                "partial grow params - missing grow_function",
                {"recency_field": "timestamp", "scale": "7d", "decay_function": "exponential", "decay_to": 0.5,
                 "grow_from": 0.5, "grow_scale": "7d", "grow_offset": "0d"},
                "Grow parameters must be either all provided or all omitted"
            ),
            (
                "partial grow params - missing grow_scale and grow_offset",
                {"recency_field": "timestamp", "scale": "7d", "decay_function": "exponential", "decay_to": 0.5,
                 "grow_from": 0.5, "grow_function": "exponential"},
                "Grow parameters must be either all provided or all omitted"
            ),
            (
                "negative center value",
                {"recency_field": "timestamp", "center": -1.0},
                "ensure this value is greater than or equal to 0"
            ),
            (
                "invalid apply_to_subqueries value",
                {"recency_field": "timestamp", "apply_to_subqueries": ["invalid"]},
                "unexpected value; permitted:"
            ),
            (
                "mixed invalid apply_to_subqueries",
                {"recency_field": "timestamp", "apply_to_subqueries": ["tensor", "bad"]},
                "unexpected value; permitted:"
            ),
        ]

        for description, kwargs, expected_error in validation_cases:
            with self.subTest(case=description):
                with self.assertRaises(ValidationError) as ctx:
                    RecencyParameters(**kwargs)
                self.assertIn(
                    expected_error,
                    str(ctx.exception),
                    f"Error for '{description}' should contain '{expected_error}'"
                )

    def test_apply_to_subqueries_deduplicates(self):
        """Duplicate values in applyToSubqueries are deduplicated."""
        params = RecencyParameters(
            recency_field="timestamp",
            apply_to_subqueries=["tensor", "tensor"]
        )
        self.assertEqual(["tensor"], params.apply_to_subqueries)

        params2 = RecencyParameters(
            recency_field="timestamp",
            apply_to_subqueries=["lexical", "tensor", "lexical"]
        )
        self.assertEqual(["lexical", "tensor"], params2.apply_to_subqueries)

    @pytest.mark.skip_for_multinode(
        "Multi-nodes will return different lexical results so we can not assert on the results.")
    def test_apply_to_subqueries_controls_recency_application(self):
        """Test that applyToSubqueries controls which subquery scores are modified by recency.

        Uses apply_in_ranking_phase="exclude-global" to isolate the subquery effect —
        recency only modifies first-phase subquery scores, not the global RRF score.
        Uses rerankDepthTensor=20 to ensure all documents appear in the tensor result set.

        Baseline: no recency parameters at all.
        For each variant, verifies:
        - _recency_score is present and correctly calculated
        - []: _tensor_score and _lexical_score unchanged from baseline
        - ["tensor"]: _tensor_score = baseline * recency, _lexical_score unchanged
        - ["lexical"]: _lexical_score = baseline * recency, _tensor_score unchanged
        - ["tensor", "lexical"]: both scores = baseline * recency
        """
        self._add_shared_documents()

        recency_base_params = dict(
            recency_field="timestamp",
            scale="60d",
            offset="0d",
            decay_function="exponential",
            decay_to=0.3,
            grow_from=0.2,
            grow_function="linear",
            grow_scale="45d",
            grow_offset="1d",
            apply_in_ranking_phase="exclude-global",
        )

        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            rerankDepthTensor=20,
        )

        # Baseline: no recency at all
        baseline_result = tensor_search.search(
            config=self.config,
            index_name=self.main_index.name,
            text="product",
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=hybrid_params,
            result_count=20,
        )
        baseline_scores = self._extract_scores(baseline_result['hits'])

        # (apply_to, recency_on_tensor, recency_on_lexical)
        test_cases = [
            ([], False, False),
            (["tensor"], True, False),
            (["lexical"], False, True),
            (["tensor", "lexical"], True, True),
        ]

        for apply_to, recency_on_tensor, recency_on_lexical in test_cases:
            with self.subTest(apply_to_subqueries=apply_to):
                recency_params = RecencyParameters(
                    **recency_base_params,
                    apply_to_subqueries=apply_to,
                )
                result = tensor_search.search(
                    config=self.config,
                    index_name=self.main_index.name,
                    text="product",
                    search_method=SearchMethod.HYBRID,
                    hybrid_parameters=hybrid_params,
                    recency_parameters=recency_params,
                    result_count=20,
                )
                hits = result['hits']
                self.assertGreater(len(hits), 0)

                # 1. Verify _recency_score is present and correct
                self._verify_recency_behavior(hits, recency_params)

                variant_scores = self._extract_scores(hits)

                # Verify baseline and variant return the same doc IDs
                baseline_ids = set(baseline_scores)
                variant_ids = set(variant_scores)
                self.assertEqual(baseline_ids, variant_ids,
                    f"Baseline and variant should return the same doc IDs. "
                    f"Only in baseline: {baseline_ids - variant_ids}, "
                    f"Only in variant: {variant_ids - baseline_ids}")

                for doc_id in variant_scores:
                    b = baseline_scores[doc_id]
                    v = variant_scores[doc_id]
                    recency = v['recency']

                    # 2. Check tensor scores (skip if baseline has no tensor score)
                    if b['tensor'] is not None:
                        if recency_on_tensor:
                            expected_tensor = b['tensor'] * recency
                            self.assertAlmostEqual(
                                expected_tensor, v['tensor'], places=5,
                                msg=f"Doc {doc_id}: tensor score should be baseline * recency"
                            )
                        else:
                            self.assertAlmostEqual(
                                b['tensor'], v['tensor'], places=5,
                                msg=f"Doc {doc_id}: tensor score should match baseline"
                            )

                    # 3. Check lexical scores (skip if baseline has no lexical score)
                    if b['lexical'] is not None:
                        if recency_on_lexical:
                            expected_lexical = b['lexical'] * recency
                            self.assertAlmostEqual(
                                expected_lexical, v['lexical'], places=5,
                                msg=f"Doc {doc_id}: lexical score should be baseline * recency"
                            )
                        else:
                            self.assertAlmostEqual(
                                b['lexical'], v['lexical'], places=5,
                                msg=f"Doc {doc_id}: lexical score should match baseline"
                            )

    @pytest.mark.skip_for_multinode(
        "Multi-nodes will return different lexical results so we can not assert on the results.")
    def test_center_produces_reproducible_scores(self):
        """Test that center parameter produces the same scores across multiple queries.

        Uses center=now-20min so all three documents are in the past relative to center,
        covering a range of ages: recent (20min), medium (4d), and old (14d).
        """
        now = datetime.now()
        documents = [
            {"_id": "doc-recent", "title": "recent document about technology",
             "timestamp": (now - timedelta(hours=1)).timestamp()},
            {"_id": "doc-medium", "title": "medium age document about technology",
             "timestamp": (now - timedelta(days=4)).timestamp()},
            {"_id": "doc-old", "title": "old document about technology",
             "timestamp": (now - timedelta(days=14)).timestamp()},
        ]
        self.add_documents(
            self.config,
            AddDocsParams(index_name=self.main_index.name, docs=documents, tensor_fields=["title"]),
        )

        fixed_center = (now - timedelta(minutes=20)).timestamp()
        recency_params = RecencyParameters(
            recency_field="timestamp", scale="7d", decay_to=0.5, center=fixed_center
        )

        results_1 = tensor_search.search(
            config=self.config, index_name=self.main_index.name,
            text="technology", search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction, rankingMethod=RankingMethod.RRF),
            recency_parameters=recency_params, result_count=10,
        )

        time.sleep(2)

        results_2 = tensor_search.search(
            config=self.config, index_name=self.main_index.name,
            text="technology", search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction, rankingMethod=RankingMethod.RRF),
            recency_parameters=recency_params, result_count=10,
        )

        # Verify recency scores are correctly calculated for both runs
        self._verify_recency_behavior(results_1['hits'], recency_params)
        self._verify_recency_behavior(results_2['hits'], recency_params)

        # Verify both runs produce identical ordering and scores
        self.assertEqual(len(results_1["hits"]), len(results_2["hits"]))
        for hit1, hit2 in zip(results_1["hits"], results_2["hits"]):
            self.assertEqual(hit1["_id"], hit2["_id"])
            self.assertEqual(hit1["_recency_score"], hit2["_recency_score"],
                             f"Recency score mismatch for {hit1['_id']}")
            self.assertAlmostEqual(hit1["_score"], hit2["_score"], places=5)


if __name__ == '__main__':
    unittest.main()
