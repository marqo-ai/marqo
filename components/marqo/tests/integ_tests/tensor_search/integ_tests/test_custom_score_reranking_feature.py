"""
Integration tests for the custom score reranking feature

Follows custom_score_rerank_plan_final.md (Integration Test Structure): query "tuxedo",
index with model open_clip/ViT-B-16-SigLIP/webli, 4 fields (lex_retrieval_field,
lex_ranking_field, tensor_retrieval_field, tensor_ranking_field). Docs are defined so
base RRF order is deterministic (doc1..doc5) and add_to_score with bm25 lex_ranking_field
or closeness tensor_ranking_field reverses order to (doc5..doc1). Each test shows:
(1) custom score modified final score, (2) order changes deterministically,
(3) _score vs _pre_rerank_score known by modifier and field scores.
"""
import os
from typing import Any, Dict, List
from unittest import mock

from marqo.core.constants import (
    MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX,
    MARQO_CUSTOM_SCORE_RERANKERS_MINIMUM_VERSION,
    MARQO_DOC_PRE_RERANK_SCORE,
)
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.marqo_index import CollapseField, DistanceMetric, FieldFeature, FieldType, Model
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.api_models import ScoreModifierLists, SearchQuery
from marqo.tensor_search.models.relevance_cutoff_model import (
    RelevanceCutoffModel,
    RelevanceCutoffMethod,
    RelativeMaxScoreParameters,
)
from marqo.tensor_search.models.recency_parameters import RecencyParameters
from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
from marqo.tensor_search.models.sort_by_model import SortByModel, SortByField
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase
from marqo.core.exceptions import InvalidArgumentError, UnsupportedFeatureError

import unittest
import pytest
import time

# --- Test Data ---
TENSOR_FIELDS_PLAN = ["tensor_retrieval_field", "tensor_ranking_field"]
DOCS_TUXEDO_PLAN = [
    {
        # (1) In BOTH tensor and lexical
        "_id": "doc1",
        "lex_retrieval_field": "tuxedo tuxedo tuxedo",  # VERY HIGH lexical score
        "tensor_retrieval_field": "tuxedo",             # VERY HIGH tensor score
        "lex_ranking_field": "tuxedo",                  # lowest bm25 score for global reranking
        "tensor_ranking_field": "unrelated",            # lowest closeness score for global reranking
    },
    {
        # (2) In ONLY tensor (medium strength)
        "_id": "doc2",
        "lex_retrieval_field": "no match",              # no lexical match
        "tensor_retrieval_field": "suit",                # MEDIUM tensor score
        "lex_ranking_field": "tuxedo tuxedo",           # 2nd lowest bm25 score for global reranking
        "tensor_ranking_field": "rainbow tie",          # 2nd lowest closeness score for global reranking
    },
    {
        # (3) In ONLY lexical (medium strength)
        "_id": "doc3",
        "lex_retrieval_field": "tuxedo tuxedo",         # MEDIUM lexical score
        # No retrieval tensor match at all
        "lex_ranking_field": "tuxedo tuxedo tuxedo",    # 3rd lowest bm25 score for global reranking
        "tensor_ranking_field": "shorts",               # 3rd lowest closeness score for global reranking
    },
    {
        # (4) In ONLY tensor (lower strength)
        "_id": "doc4",
        "lex_retrieval_field": "no match",              # no lexical match
        "tensor_retrieval_field": "shorts",             # LOWER tensor score (but it's still clothes)
        "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo",  # 4th lowest bm25 score for global reranking
        "tensor_ranking_field": "suit",                      # 4th lowest closeness score for global reranking
    },
    {
        # (5) In ONLY lexical (lower strength)
        "_id": "doc5",
        "lex_retrieval_field": "tuxedo",                # LOW lexical score
        # No retrieval tensor match at all
        "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo tuxedo",  # highest bm25 score for global reranking
        "tensor_ranking_field": "tuxedo",               # highest closeness score for global reranking
    },
]

# This doc structure allows certain docs to come to the surface depending on which aggregate was chosen.
# Ranking fields only; retrieval fields are added when indexing so all docs match the query.
DOCS_TUXEDO_FOR_LEXICAL_AGGREGATES = [
    {
        # If max aggregate is chosen, this doc should come to the top (one field has highest BM25)
        "_id": "strongest_max",
        "lex_ranking_field_1": "tuxedo tuxedo tuxedo",
        "lex_ranking_field_2": "no match",
    },
{
        # A doc that should end up in the middle, whether aggregate method is sum, avg, or max.
        # Note that avg divides by all fields in the index, not just that in this doc. That's why this doc
        # doesn't have the highest avg
        "_id": "middle_of_both",
        "lex_ranking_field_1": "tuxedo tuxedo",
        "lex_ranking_field_2": "tuxedo tuxedo",
    },
    {
        # If sum/avg aggregate is chosen, this doc should come to the top (many fields with good BM25)
        "_id": "strongest_sum_avg",
        "lex_ranking_field_1": "tuxedo",
        "lex_ranking_field_2": "tuxedo",
        "lex_ranking_field_3": "tuxedo",
        "lex_ranking_field_4": "tuxedo",
        "lex_ranking_field_5": "tuxedo",
        "lex_ranking_field_6": "tuxedo",
    }
]

# Closeness aggregate test: separate index, docs use CLOSENESS_TUXEDO terms so sum/max/avg differ per doc.
DOCS_TUXEDO_FOR_CLOSENESS_AGGREGATES = [
    {
        # If max aggregate is chosen: one field has highest closeness (tuxedo 1.0), rest low
        "_id": "strongest_max",
        "tensor_ranking_field_1": "tuxedo",         # Adds 1.0 closeness score
        "tensor_ranking_field_2": "unrelated",      # Adds 0.4584048390388489 closeness score
    },
    {
        # A doc that should end up in the middle, whether aggregate method is sum, avg, or max.
        "_id": "middle_of_both",
        "tensor_ranking_field_1": "black tuxedo",   # Adds 0.7219897508621216 closeness score
        "tensor_ranking_field_2": "black tuxedo",   # Adds 0.7219897508621216 closeness score
        "tensor_ranking_field_3": "black tuxedo",   # Adds 0.7219897508621216 closeness score
    },
    {
        # If sum/avg aggregate is chosen: many fields with medium-high closeness
        "_id": "strongest_sum_avg",
        "tensor_ranking_field_1": "rainbow tie",    # Adds 0.5811693072319031 closeness score
        "tensor_ranking_field_2": "rainbow tie",    # Adds 0.5811693072319031 closeness score
        "tensor_ranking_field_3": "rainbow tie",    # Adds 0.5811693072319031 closeness score
        "tensor_ranking_field_4": "rainbow tie",    # Adds 0.5811693072319031 closeness score
        "tensor_ranking_field_5": "rainbow tie",    # Adds 0.5811693072319031 closeness score
        "tensor_ranking_field_6": "rainbow tie",    # Adds 0.5811693072319031 closeness score
    },
]

HYBRID_PARAMS_TUXEDO = HybridParameters(
    retrievalMethod=RetrievalMethod.Disjunction,
    rankingMethod=RankingMethod.RRF,
    alpha=0.5001,
    rrfK=60,
    searchableAttributesTensor=["tensor_retrieval_field"],
    searchableAttributesLexical=["lex_retrieval_field"],
    verbose=True
)

BASE_RRF_ORDER = ["doc1", "doc2", "doc3", "doc4", "doc5"]
REVERSED_ORDER = ["doc5", "doc4", "doc3", "doc2", "doc1"]

# Closeness (prenormalized-angular) to "tuxedo" with model open_clip/ViT-B-16-SigLIP-512/webli (from plan).
OLD_CLOSENESS_TUXEDO = {
    "tuxedo": 1.0,
    "black tuxedo": 0.9290061705548538,
    "black tie": 0.9105825129267998,
    "suit": 0.901995477338818,
    "shorts": 0.8311302085908341,
    "backpack": 0.8264572877032847,
    "floral dress": 0.825913938286213,
    "suede shoes": 0.8106355248508749,
    "rainbow tie": 0.7955299917394352,
    "unrelated": 0.5882339267201514,
}

# Closeness (prenormalized-angular) to "tuxedo" with model open_clip/ViT-B-16-SigLIP-512/webli (from plan).
# Using new closeness (copied from vespa)
NEW_CLOSENESS_TUXEDO = {
    "tuxedo": 1.0,
    "black tuxedo": 0.7219897508621216,
    "black tie": 0.6926929950714111,
    "suit": 0.6790624260902405,
    "shorts": 0.6071039438247681,
    "backpack": 0.6020216941833496,
    "floral dress": 0.6027941703796387,
    "suede shoes": 0.5905405879020691,
    "rainbow tie": 0.5811693072319031,
    "unrelated": 0.4584048390388489,
}

# Helpers for TestCustomScoreRerankingWithOtherFeatures (same tuxedo index/model as main tests).
def _tuxedo_docs_with_extras(*, popularity=None, category=None, parent_id=None, timestamp=None):
    """Return list of docs from DOCS_TUXEDO_PLAN with optional extra fields for modifier/collapse/facets tests."""
    docs = [dict(d) for d in DOCS_TUXEDO_PLAN]
    if popularity is not None:
        for i, d in enumerate(docs):
            d["popularity"] = popularity[i] if isinstance(popularity, (list, tuple)) else popularity
    if category is not None:
        for i, d in enumerate(docs):
            d["category"] = category[i] if isinstance(category, (list, tuple)) else category
    if parent_id is not None:
        for i, d in enumerate(docs):
            d["parent_id"] = parent_id[i] if isinstance(parent_id, (list, tuple)) else parent_id
    if timestamp is not None:
        for i, d in enumerate(docs):
            d["timestamp"] = timestamp[i] if isinstance(timestamp, (list, tuple)) else timestamp
    return docs


class TestCustomScoreRerankingFeature(MarqoTestCase):
    """
    Plan-based integration tests: query "tuxedo", index open_clip/ViT-B-16-SigLIP-512/webli,
    4 fields, 5 docs. Base RRF order doc1..doc5; add_to_score bm25/closeness reverses to doc5..doc1.
    Each test shows: (1) modifier changed final score, (2) order deterministic, (3) _score vs _pre_rerank_score.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # TODO: Change hardcoded values. Original planned open_clip/ViT-B-16-SigLIP-512/webli is no longer there
        model = Model(name="open_clip/ViT-B-16-SigLIP/webli")
        index_request = cls.unstructured_marqo_index_request(model=model)
        index_request_empty = cls.unstructured_marqo_index_request(model=model)
        index_request_bm25_aggregates = cls.unstructured_marqo_index_request(model=model)
        index_request_closeness_aggregates = cls.unstructured_marqo_index_request(model=model)
        index_request_lexical_only = cls.unstructured_marqo_index_request(model=model)

        cls.indexes = cls.create_indexes([
            index_request,
            index_request_empty,
            index_request_bm25_aggregates,
            index_request_closeness_aggregates,
            index_request_lexical_only,
        ])
        cls.index = cls.indexes[0]
        cls.index_number_only = cls.indexes[1]
        cls.index_bm25_aggregates = cls.indexes[2]
        cls.index_closeness_aggregates = cls.indexes[3]
        cls.index_lexical_only = cls.indexes[4]


    def _add_tuxedo_docs(self) -> None:
        """Add the 5 plan docs to the index (used by each test)."""
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=DOCS_TUXEDO_PLAN,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )

    def test_custom_score_rerank_raises_when_schema_version_below_minimum(self):
        """Custom score rerankers on an index with schema version < 2.26.0 raises UnsupportedFeatureError."""
        import marqo.tensor_search.index_meta_cache as index_meta_cache
        real_get_index = index_meta_cache.get_index

        def get_index_old_schema(index_management, index_name, force_refresh=False):
            idx = real_get_index(index_management, index_name, force_refresh)
            if index_name == self.index.name:
                idx = idx.copy(deep=True, update={"schema_template_version": "2.25.0"})
                idx.clear_cache()  # recompute index_supports_* from new schema_template_version
            return idx

        with mock.patch("marqo.tensor_search.index_meta_cache.get_index", get_index_old_schema):
            with self.assertRaises(UnsupportedFeatureError) as ctx:
                tensor_search.search(
                    config=self.config,
                    index_name=self.index.name,
                    text="x",
                    search_method="HYBRID",
                    hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                    score_modifiers=ScoreModifierLists(
                        add_to_score=[{"field_name": f"marqo__score_bm25_sum", "weight": 1.0}]
                    ),
                    result_count=5,
                )
        msg = str(ctx.exception)
        self.assertIn(str(MARQO_CUSTOM_SCORE_RERANKERS_MINIMUM_VERSION), msg)
        self.assertIn("2.25.0", msg)

    def _assert_pre_rerank_score_matches_baseline(
        self, res_with_rerank, res_no_rerank, tolerance=1e-5
    ):
        """Assert each hit's _pre_rerank_score equals the baseline (no modifiers) score for the same doc."""
        scores_without = {h["_id"]: h["_score"] for h in res_no_rerank["hits"]}
        for hit in res_with_rerank["hits"]:
            doc_id = hit["_id"]
            self.assertIn(
                MARQO_DOC_PRE_RERANK_SCORE,
                hit,
                msg=f"Hit {doc_id} should have _pre_rerank_score when custom score reranking is used",
            )
            self.assertIn(doc_id, scores_without, msg=f"Doc {doc_id} should appear in baseline search")
            self.assertAlmostEqual(
                hit[MARQO_DOC_PRE_RERANK_SCORE],
                scores_without[doc_id],
                delta=tolerance,
                msg=f"Doc {doc_id}: _pre_rerank_score should equal baseline score",
            )

    def _custom_score_add_contributions_by_doc_id(
        self, add_to_score_ops: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Run hybrid search with only the given marqo__score_* add_to_score modifiers (same query/index as plan).
        Returns doc_id -> (_score - _pre_rerank_score) for each hit.

        Used to compare single-modifier vs combined-modifier behavior in one test method (no state shared
        across pytest test cases, which avoids ordering flakiness).
        """
        res = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(add_to_score=add_to_score_ops),
            result_count=10,
        )
        return {
            h["_id"]: h["_score"] - h[MARQO_DOC_PRE_RERANK_SCORE]
            for h in res["hits"]
        }

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_base_rrf_order_deterministic(self):
        """Base RRF (no modifiers) returns all 5 docs; doc1 (in both tensor and lexical) ranks first."""
        self._add_tuxedo_docs()
        res = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        ids = [h["_id"] for h in res["hits"]]
        self.assertEqual(ids, BASE_RRF_ORDER, msg="Base RRF order must be doc1, doc2, doc3, doc4, doc5")

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_non_existent_custom_score_field_add_to_score_leaves_score_unchanged(self):
        """add_to_score with a non-existent custom score field does not error; scores stay exactly as baseline."""
        self._add_tuxedo_docs()
        res_baseline = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        res_with_modifier = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {"field_name": "marqo__score_bm25_field_non_existent_field", "weight": 1.0}
                ]
            ),
            result_count=10,
        )
        baseline_scores = {h["_id"]: h["_score"] for h in res_baseline["hits"]}
        for hit in res_with_modifier["hits"]:
            self.assertAlmostEqual(
                hit["_score"],
                baseline_scores[hit["_id"]],
                places=9,
                msg=f"Doc {hit['_id']}: score with non-existent add_to_score should equal baseline",
            )

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_non_existent_custom_score_field_multiply_score_by_leaves_score_unchanged(self):
        """multiply_score_by with a non-existent custom score field does not error; scores stay exactly as baseline."""
        self._add_tuxedo_docs()
        res_baseline = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        res_with_modifier = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                multiply_score_by=[
                    {"field_name": "marqo__score_bm25_field_non_existent_field", "weight": 2.0}
                ]
            ),
            result_count=10,
        )
        baseline_scores = {h["_id"]: h["_score"] for h in res_baseline["hits"]}
        for hit in res_with_modifier["hits"]:
            self.assertAlmostEqual(
                hit["_score"],
                baseline_scores[hit["_id"]],
                places=9,
                msg=f"Doc {hit['_id']}: score with non-existent multiply_score_by should equal baseline",
            )

    def test_document_with_marqo_reserved_field_name_cannot_be_created(self):
        """A document cannot be created if it contains a field whose name is the reserved custom score key.
        Field names starting with marqo__ are protected; the document is rejected at add time."""
        doc_with_reserved_field = {
            "_id": "doc_reserved_field",
            "marqo__score_bm25_field_lex_ranking_field": "attempted value",
            "lex_retrieval_field": "a",
            "tensor_retrieval_field": "a",
            "lex_ranking_field": "a",
            "tensor_ranking_field": "a",
        }
        res = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=[doc_with_reserved_field],
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )
        self.assertTrue(res.errors, "Adding a doc with marqo__ field name should report errors")
        self.assertEqual(1, len(res.items))
        item = res.items[0]
        self.assertNotEqual(200, item.status, "Document with reserved field name should fail")
        error_message = (item.message or item.error or "")
        self.assertIn(
            "marqo__",
            error_message,
            msg="Error should mention the reserved prefix marqo__",
        )
        self.assertIn(
            "must not start",
            error_message,
            msg="Error should state that field name must not start with reserved prefix",
        )

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_rrf_with_bm25_single_field_modifies_scores_and_reverses_order(self):
        """
        add_to_score with bm25 lex_ranking_field: (1) modifies final score so doc with highest
        BM25 in lex_ranking_field ranks highest; (2) order reverses to doc5, doc4, doc3, doc2, doc1;
        (3) _score differs from _pre_rerank_score by the added normalized BM25 contribution.
        """
        self._add_tuxedo_docs()
        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        res_with_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        self._assert_pre_rerank_score_matches_baseline(res_with_rerank, res_no_rerank)
        ids = [h["_id"] for h in res_with_rerank["hits"]]
        self.assertEqual(REVERSED_ORDER, ids, msg="add_to_score bm25 lex_ranking_field: order must be doc5, doc4, doc3, doc2, doc1")
        scores_with = {h["_id"]: h["_score"] for h in res_with_rerank["hits"]}
        self.assertGreater(scores_with["doc5"], scores_with["doc1"], msg="doc5 (highest bm25 lex_ranking_field) should have higher score than doc1")
        any_changed = any(h["_score"] != h[MARQO_DOC_PRE_RERANK_SCORE] for h in res_with_rerank["hits"])
        self.assertTrue(any_changed, msg="Custom score modifier must change at least one doc's score")

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_rrf_with_closeness_retrieval_vector_single_field_modifies_scores_and_reverses_order(self):
        """
        add_to_score with closeness tensor_ranking_field (weight 1.0): order reverses to doc5..doc1.
        Final _score = _pre_rerank_score + weight * raw_closeness (plan: custom_score_rerank_plan_final.md;
        CLOSENESS_TUXEDO / DOC_TENSOR_RANKING_CLOSENESS). We assert the formula using each hit's
        observed contribution (score - pre_rerank) because Vespa's closeness output can differ slightly
        from the plan's hardcoded values; contribution order must still match plan (doc5 highest → doc1 lowest).
        """
        self._add_tuxedo_docs()
        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        res_with_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_closeness_retrieval_vector_field_tensor_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        self._assert_pre_rerank_score_matches_baseline(res_with_rerank, res_no_rerank)
        ids = [h["_id"] for h in res_with_rerank["hits"]]
        self.assertEqual(REVERSED_ORDER, ids, msg="add_to_score closeness: order must be doc5, doc4, doc3, doc2, doc1")
        # Backend adds weight * (normalized or raw) closeness; Vespa's output can differ from plan's hardcoded values.
        # Assert contribution order: doc5 (highest closeness) gets largest add, then doc4, ..., doc1.
        for hit in res_with_rerank["hits"]:
            contribution = hit["_score"] - hit[MARQO_DOC_PRE_RERANK_SCORE]
            self.assertGreaterEqual(
                contribution,
                0.0,
                msg=f"Doc {hit['_id']}: add_to_score contribution should be non-negative for weight 1",
            )
        contributions = [res_with_rerank["hits"][i]["_score"] - res_with_rerank["hits"][i][MARQO_DOC_PRE_RERANK_SCORE] for i in range(len(ids))]
        for i in range(len(contributions) - 1):
            self.assertGreaterEqual(
                contributions[i],
                contributions[i + 1],
                msg=f"Closeness contributions should decrease in order doc5..doc1: {contributions}",
            )

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_rrf_with_multiple_custom_score_modifiers_add_to_score(self):
        """
        add_to_score with both bm25 lex_ranking_field and closeness tensor_ranking_field (each weight 1.0).

        BM25-only and closeness-only contributions are measured in this test; combined contribution
        must equal their sum exactly per doc
        """
        self._add_tuxedo_docs()
        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        bm25_only = [{"field_name": "marqo__score_bm25_field_lex_ranking_field", "weight": 1.0}]
        closeness_only = [
            {"field_name": "marqo__score_closeness_retrieval_vector_field_tensor_ranking_field", "weight": 1.0}
        ]
        contrib_bm25 = self._custom_score_add_contributions_by_doc_id(bm25_only)
        contrib_closeness = self._custom_score_add_contributions_by_doc_id(closeness_only)
        res_both = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(add_to_score=bm25_only + closeness_only),
            result_count=10,
        )
        self._assert_pre_rerank_score_matches_baseline(res_both, res_no_rerank)
        ids = [h["_id"] for h in res_both["hits"]]
        self.assertEqual(REVERSED_ORDER, ids, msg="With both modifiers order must be doc5, doc4, doc3, doc2, doc1")
        contrib_both = {
            h["_id"]: h["_score"] - h[MARQO_DOC_PRE_RERANK_SCORE] for h in res_both["hits"]
        }
        for doc_id in BASE_RRF_ORDER:
            self.assertIn(doc_id, contrib_bm25)
            self.assertIn(doc_id, contrib_closeness)
            self.assertIn(doc_id, contrib_both)
            expected_sum = contrib_bm25[doc_id] + contrib_closeness[doc_id]
            self.assertAlmostEqual(
                contrib_both[doc_id],
                expected_sum,
                places=12,
                msg=(
                    f"Doc {doc_id}: combined add_to_score contribution must equal bm25 + closeness "
                    f"({contrib_both[doc_id]} vs {contrib_bm25[doc_id]} + {contrib_closeness[doc_id]})"
                ),
            )

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_custom_score_rerank_with_attributes_to_retrieve(self):
        """
        Summary-features (bm25(*), ranking_closeness_metric_*) are rank-profile outputs, not
        document attributes; they are not controlled by attributes_to_retrieve. With
        attributes_to_retrieve set, custom score rerank (BM25 or closeness) must still reverse
        order (doc5..doc1), and each hit must contain only _id, _score, _highlights, and the
        requested attributes.
        """
        docs = [dict(d) for d in DOCS_TUXEDO_PLAN]
        for d in docs:
            d["other_field"] = f"val_{d['_id']}"
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )
        with self.subTest(modifier="bm25"):
            # All results should only have this 1 field
            requested_attrs = ["lex_ranking_field"]
            res = tensor_search.search(
                config=self.config,
                index_name=self.index.name,
                text="tuxedo",
                search_method="HYBRID",
                hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[{"field_name": "marqo__score_bm25_field_lex_ranking_field", "weight": 1.0}]
                ),
                result_count=10,
                attributes_to_retrieve=requested_attrs,
            )
            ids = [h["_id"] for h in res["hits"]]
            self.assertEqual(
                REVERSED_ORDER, ids,
                msg="BM25 add_to_score must reverse order with attributes_to_retrieve",
            )
            allowed_keys = {"_id", "_score", "_highlights"}.union(requested_attrs)
            for hit in res["hits"]:
                self.assertLessEqual(
                    set(hit.keys()), allowed_keys,
                    msg=f"Hit {hit.get('_id')} should only have requested attributes and metadata; keys: {sorted(hit.keys())}",
                )
                for attr in requested_attrs:
                    self.assertIn(attr, hit, msg=f"Requested attribute {attr} must be present in hit {hit.get('_id')}")
        with self.subTest(modifier="closeness"):
            requested_attrs = ["other_field"]
            res = tensor_search.search(
                config=self.config,
                index_name=self.index.name,
                text="tuxedo",
                search_method="HYBRID",
                hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        {
                            "field_name": "marqo__score_closeness_retrieval_vector_field_tensor_ranking_field",
                            "weight": 1.0,
                        }
                    ]
                ),
                result_count=10,
                attributes_to_retrieve=requested_attrs,
            )
            ids = [h["_id"] for h in res["hits"]]
            self.assertEqual(
                REVERSED_ORDER, ids,
                msg="Closeness add_to_score must reverse order (doc5..doc1) even when only other_field is retrieved",
            )
            allowed_keys = {"_id", "_score", "_highlights"}.union(requested_attrs)
            for hit in res["hits"]:
                self.assertLessEqual(
                    set(hit.keys()), allowed_keys,
                    msg=f"Hit {hit.get('_id')} should only have other_field and metadata; keys: {sorted(hit.keys())}",
                )
                for attr in requested_attrs:
                    self.assertIn(attr, hit, msg=f"Requested attribute {attr} must be present in hit {hit.get('_id')}")
                self.assertEqual(hit["other_field"], f"val_{hit['_id']}")

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_closeness_weighted_exact_final_score(self):
        """
        add_to_score with closeness and weight 2.0 or -1.0: final _score must equal
        _pre_rerank_score + weight * contribution (plan: weight × anticipated closeness; see
        DOC_TENSOR_RANKING_CLOSENESS). We use contribution from a weight=1 run so the assertion
        is exact regardless of Vespa's raw closeness values, and proves the formula is applied correctly.
        """
        self._add_tuxedo_docs()
        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        # Get per-doc contribution (raw closeness effect) from weight=1 run.
        res_w1 = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_closeness_retrieval_vector_field_tensor_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        contribution_by_id = {
            hit["_id"]: hit["_score"] - hit[MARQO_DOC_PRE_RERANK_SCORE]
            for hit in res_w1["hits"]
        }
        for weight in (2.0, -1.0):
            res = tensor_search.search(
                config=self.config,
                index_name=self.index.name,
                text="tuxedo",
                search_method="HYBRID",
                hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        {
                            "field_name": f"marqo__score_closeness_retrieval_vector_field_tensor_ranking_field",
                            "weight": weight,
                        }
                    ]
                ),
                result_count=10,
            )
            self._assert_pre_rerank_score_matches_baseline(res, res_no_rerank)
            for hit in res["hits"]:
                doc_id = hit["_id"]
                pre = hit[MARQO_DOC_PRE_RERANK_SCORE]
                contribution = contribution_by_id[doc_id]
                expected = pre + weight * contribution
                self.assertAlmostEqual(
                    hit["_score"],
                    expected,
                    delta=1e-5,
                    msg=f"Doc {doc_id} weight={weight}: expected _score = pre_rerank + {weight} * contribution = {expected}",
                )

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_rrf_with_bm25_multiply_score_by_affects_scores(self):
        """
        multiply_score_by with bm25 lex_ranking_field (weight 1.0): each doc's score is multiplied
        by its normalized BM25 (value / max). Doc5 has the highest BM25 so its multiplier is 1.0
        (score unchanged). Doc1 has the lowest BM25 so its multiplier is min/max (small but > 0).
        The multiplier increases monotonically from doc1 to doc5.
        """
        self._add_tuxedo_docs()
        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        res_with_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                multiply_score_by=[
                    {
                        "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        self._assert_pre_rerank_score_matches_baseline(res_with_rerank, res_no_rerank)
        pre_rerank = {h["_id"]: h[MARQO_DOC_PRE_RERANK_SCORE] for h in res_with_rerank["hits"]}
        scores_with = {h["_id"]: h["_score"] for h in res_with_rerank["hits"]}

        # Doc5 (max BM25) keeps its full score (multiplied by 1.0)
        self.assertEqual(scores_with["doc5"], pre_rerank["doc5"],
                         msg="doc5 (max BM25) should keep full score (multiplied by 1.0)")

        # Doc1 (min BM25) has a reduced but non-zero score (divide-by-max never gives 0)
        self.assertGreater(scores_with["doc1"], 0.0,
                           msg="doc1 score must be > 0 with divide-by-max normalization")
        self.assertLess(scores_with["doc1"], pre_rerank["doc1"],
                        msg="doc1 (min BM25) should have reduced score")

        # The multiplier (score / pre_rerank) increases from doc1 to doc5
        # BM25 order: doc1 < doc2 < doc3 < doc4 < doc5
        multipliers = {}
        for doc_id in ["doc1", "doc2", "doc3", "doc4", "doc5"]:
            multipliers[doc_id] = scores_with[doc_id] / pre_rerank[doc_id]
        for i, (a, b) in enumerate(zip(
            ["doc1", "doc2", "doc3", "doc4"],
            ["doc2", "doc3", "doc4", "doc5"],
        )):
            self.assertLess(
                multipliers[a], multipliers[b],
                msg=f"Multiplier for {a} ({multipliers[a]:.4f}) should be < {b} ({multipliers[b]:.4f})",
            )


    def test_all_bm25_aggregates_sum_max_avg(self):
        """
        BM25 sum/max/avg aggregates: use a dedicated index and DOCS_TUXEDO_FOR_LEXICAL_AGGREGATES.
        Assert strict order for each aggregate. Assert that aggregate was normalized.
        """

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index_bm25_aggregates.name,
                docs=DOCS_TUXEDO_FOR_LEXICAL_AGGREGATES,
                tensor_fields=["lex_ranking_field_1"]   # Making this a tensor field just so hybrid search works
            ),
        )
        res_baseline = tensor_search.search(
            config=self.config,
            index_name=self.index_bm25_aggregates.name,
            hybrid_parameters=HybridParameters(verbose=True),
            text="tuxedo",
            search_method="HYBRID",
            result_count=10,
        )

        # Collect baseline scores for comparison
        for hit in res_baseline["hits"]:
            if hit["_id"] == "strongest_max":
                baseline_score_strongest_max = hit["_score"]
            elif hit["_id"] == "middle_of_both":
                baseline_score_middle = hit["_score"]
            elif hit["_id"] == "strongest_sum_avg":
                baseline_score_strongest_sum_avg = hit["_score"]

        for agg in ("sum", "max", "avg"):
            with self.subTest(aggregate=agg):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.index_bm25_aggregates.name,
                    hybrid_parameters=HybridParameters(verbose=True),
                    text="tuxedo",
                    search_method="HYBRID",
                    score_modifiers=ScoreModifierLists(
                        add_to_score=[
                            # High weight because the numbers are small.
                            {"field_name": f"marqo__score_bm25_{agg}", "weight": 1000.0}
                        ]
                    ),
                    result_count=10,
                )

                # Basic assertions. List length and base score
                self.assertEqual(len(res["hits"]), 3)
                self._assert_pre_rerank_score_matches_baseline(res, res_baseline)

                # Assert order is correct for each aggregate type
                if agg in ("sum", "avg"):
                    expected_order = ["strongest_sum_avg", "middle_of_both", "strongest_max"]
                    # Top hit has max normalized score (1.0) so gets +1000 to original score
                    self.assertEqual(res["hits"][0]["_score"], baseline_score_strongest_sum_avg + 1000.0)
                    # Bottom hit has a small positive normalized score (not 0, since we use divide-by-max)
                    self.assertGreater(res["hits"][-1]["_score"], baseline_score_strongest_max)

                else:  # max
                    expected_order = ["strongest_max", "middle_of_both", "strongest_sum_avg"]
                    self.assertEqual(res["hits"][0]["_score"], baseline_score_strongest_max + 1000.0)
                    self.assertGreater(res["hits"][-1]["_score"], baseline_score_strongest_sum_avg)

                # Confirm order is correct
                ids = [h["_id"] for h in res["hits"]]
                self.assertEqual(ids, expected_order, msg=f"add_to_score bm25_{agg}: order must be {expected_order}")


    def test_all_closeness_aggregates_sum_max_avg(self):
        """
        Closeness sum/max/avg aggregates: use a dedicated index and DOCS_TUXEDO_FOR_CLOSENESS_AGGREGATES
        Assert strict order for each aggregate. Assert that aggregate was normalized.
        """

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index_closeness_aggregates.name,
                docs=DOCS_TUXEDO_FOR_CLOSENESS_AGGREGATES,
                tensor_fields=["tensor_ranking_field_1", "tensor_ranking_field_2", "tensor_ranking_field_3",
                               "tensor_ranking_field_4", "tensor_ranking_field_5", "tensor_ranking_field_6"]
            ),
        )
        res_baseline = tensor_search.search(
            config=self.config,
            index_name=self.index_closeness_aggregates.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(verbose=True),
            result_count=10,
        )

        # Collect baseline scores for comparison
        for hit in res_baseline["hits"]:
            if hit["_id"] == "strongest_max":
                baseline_score_strongest_max = hit["_score"]
            elif hit["_id"] == "middle_of_both":
                baseline_score_middle = hit["_score"]
            elif hit["_id"] == "strongest_sum_avg":
                baseline_score_strongest_sum_avg = hit["_score"]

        for agg in ("sum", "max", "avg"):
            with self.subTest(aggregate=agg):
                res = tensor_search.search(
                    config=self.config,
                    index_name=self.index_closeness_aggregates.name,
                    text="tuxedo",
                    search_method="HYBRID",
                    hybrid_parameters=HybridParameters(verbose=True),
                    score_modifiers=ScoreModifierLists(
                        add_to_score=[
                            # High weight because the numbers are small.
                            {"field_name": f"marqo__score_closeness_retrieval_vector_{agg}", "weight": 1000.0}
                        ]
                    ),
                    result_count=10,
                )

                # Basic assertions. List length and base score
                self.assertEqual(len(res["hits"]), 3)
                self._assert_pre_rerank_score_matches_baseline(res, res_baseline)

                # Assert order is correct for each aggregate type
                if agg in ("sum", "avg"):
                    expected_order = ["strongest_sum_avg", "middle_of_both", "strongest_max"]
                else:  # max
                    expected_order = ["strongest_max", "middle_of_both", "strongest_sum_avg"]

                # Confirm order is correct
                ids = [h["_id"] for h in res["hits"]]
                self.assertEqual(ids, expected_order, msg=f"add_to_score closeness_{agg}: order must be {expected_order}")

                # Top hit has max normalized score (1.0) so gets +weight to pre_rerank_score
                self.assertEqual(
                    res["hits"][0]["_score"],
                    res["hits"][0][MARQO_DOC_PRE_RERANK_SCORE] + 1000.0,
                )
                # All hits have custom score contribution >= 0 (add_to_score never decreases score)
                for hit in res["hits"]:
                    self.assertGreaterEqual(
                        hit["_score"], hit[MARQO_DOC_PRE_RERANK_SCORE],
                        msg=f"Hit {hit['_id']}: score should be >= pre_rerank_score",
                    )


    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_custom_score_rerank_different_weights_affect_order_and_scores(self):
        """
        add_to_score with weight -1.0 reverses order (doc1 first, doc5 last); with weight 2.0
        order is doc5..doc1. _score differs from _pre_rerank_score by weight * normalized modifier.
        """
        self._add_tuxedo_docs()
        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        res_neg = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {"field_name": f"marqo__score_bm25_field_lex_ranking_field", "weight": -1.0}
                ]
            ),
            result_count=10,
        )
        self._assert_pre_rerank_score_matches_baseline(res_neg, res_no_rerank)
        ids_neg = [h["_id"] for h in res_neg["hits"]]
        self.assertEqual(ids_neg, BASE_RRF_ORDER, msg="Weight -1.0: order must be doc1, doc2, doc3, doc4, doc5")
        # With negative weight, at least docs with non-zero bm25 contribution should have score != pre_rerank
        any_changed = any(h["_score"] != h[MARQO_DOC_PRE_RERANK_SCORE] for h in res_neg["hits"])
        self.assertTrue(any_changed, msg="Weight -1.0 must change at least one doc's score")

        res_double = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {"field_name": f"marqo__score_bm25_field_lex_ranking_field", "weight": 2.0}
                ]
            ),
            result_count=10,
        )
        self._assert_pre_rerank_score_matches_baseline(res_double, res_no_rerank)
        ids_double = [h["_id"] for h in res_double["hits"]]
        self.assertEqual(ids_double, REVERSED_ORDER, msg="Weight 2.0 should give reversed order doc5..doc1")

    def test_hybrid_search_with_custom_score_raises_when_index_has_no_lexical_fields(self):
        """
        Hybrid search with custom score modifier fails when the index has no lexically searchable
        fields (number-only index). The index does not support hybrid, so we expect
        InvalidArgumentError indicating hybrid search cannot be performed.
        """
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index_number_only.name,
                docs=[{"_id": "seed", "numeric_field": 1}],
                tensor_fields=[],
            ),
        )

        with self.assertRaises(InvalidArgumentError) as ctx:
            tensor_search.search(
                config=self.config,
                index_name=self.index_number_only.name,
                text="anything",
                search_method="HYBRID",
                score_modifiers=ScoreModifierLists(
                    add_to_score=[{"field_name": "marqo__score_bm25_sum", "weight": 1.0}]
                ),
                result_count=5,
            )
        self.assertIn("hybrid search cannot be performed", str(ctx.exception))

    def test_hybrid_search_with_custom_score_raises_when_index_has_no_tensor_fields(self):
        """
        Hybrid search with custom score modifier fails when the index has no tensor fields
        (lexical-only index). The index does not support hybrid, so we expect
        InvalidArgumentError indicating hybrid search cannot be performed.
        """
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index_lexical_only.name,
                docs=[{"_id": "seed_lex", "lex_retrieval_field": "anything"}],
                tensor_fields=[],
            ),
        )
        params_lexical_only = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5001,
            rrfK=60,
            searchableAttributesTensor=[],
            searchableAttributesLexical=["lex_retrieval_field"],
        )
        with self.assertRaises(InvalidArgumentError) as ctx:
            tensor_search.search(
                config=self.config,
                index_name=self.index_lexical_only.name,
                text="anything",
                search_method="HYBRID",
                hybrid_parameters=params_lexical_only,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        {"field_name": "marqo__score_closeness_retrieval_vector_sum", "weight": 1.0}
                    ]
                ),
                result_count=5,
            )
        self.assertIn("hybrid search cannot be performed", str(ctx.exception))


class TestCustomScoreRerankStructuredIndexUnsupported(MarqoTestCase):
    """Custom score reranking must raise UnsupportedFeatureError on structured indexes."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        index_request = cls.structured_marqo_index_request(
            model=Model(name="hf/all-MiniLM-L6-v2"),
            fields=[
                FieldRequest(
                    name="title",
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch],
                ),
            ],
            tensor_fields=["title"],
        )
        cls.indexes = cls.create_indexes([index_request])
        cls.index = cls.indexes[0]

    def test_custom_score_rerank_on_structured_index_raises_unsupported_feature_error(self):
        """Using marqo__score_* modifiers on a structured index must raise UnsupportedFeatureError."""
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=[{"_id": "1", "title": "widget"}],
            ),
        )
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            tensor_search.search(
                config=self.config,
                index_name=self.index.name,
                text="widget",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    alpha=0.5,
                    rrfK=60,
                    searchableAttributesTensor=["title"],
                    searchableAttributesLexical=["title"],
                ),
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        {
                            "field_name": f"marqo__score_bm25_field_title",
                            "weight": 1.0,
                        }
                    ]
                ),
                result_count=5,
            )
        self.assertIn("only supported for semi-structured", str(ctx.exception).lower())


class TestCustomScoreRerankAllDistanceMetrics(MarqoTestCase):
    """
    Custom score reranking (closeness add_to_score) must work for every distance metric
    used for vector search. One index per metric; same tuxedo docs; assert doc5 ranks first.
    """

    # Metrics that support standard closeness-style ranking (exclude Geodegrees, Hamming for simplicity).
    DISTANCE_METRICS = [
        DistanceMetric.Angular,
        DistanceMetric.PrenormalizedAngular,
        DistanceMetric.Euclidean,
        DistanceMetric.DotProduct,
    ]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        model = Model(name="open_clip/ViT-B-16-SigLIP/webli")
        requests = [
            cls.unstructured_marqo_index_request(model=model, distance_metric=metric)
            for metric in cls.DISTANCE_METRICS
        ]
        cls.indexes = cls.create_indexes(requests)
        cls.index_by_metric = {cls.DISTANCE_METRICS[i]: cls.indexes[i] for i in range(len(cls.DISTANCE_METRICS))}

    def test_custom_score_rerank_closeness_per_distance_metric(self):
        """For each distance metric, closeness add_to_score must apply: doc5 first, 5 hits, _pre_rerank_score present."""
        for metric in self.DISTANCE_METRICS:
            with self.subTest(distance_metric=metric.value):
                index = self.index_by_metric[metric]
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name,
                        docs=DOCS_TUXEDO_PLAN,
                        tensor_fields=TENSOR_FIELDS_PLAN,
                    ),
                )
                res = tensor_search.search(
                    config=self.config,
                    index_name=index.name,
                    text="tuxedo",
                    search_method="HYBRID",
                    hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                    score_modifiers=ScoreModifierLists(
                        add_to_score=[
                            {
                                "field_name": f"marqo__score_closeness_retrieval_vector_field_tensor_ranking_field",
                                "weight": 1.0,
                            }
                        ]
                    ),
                    result_count=10,
                )
                self.assertEqual(len(res["hits"]), 5, msg=f"distance_metric={metric.value}: expect 5 hits")
                ids = [h["_id"] for h in res["hits"]]
                self.assertEqual(REVERSED_ORDER, ids, msg=f"distance_metric={metric.value}: order must be doc5, doc4, doc3, doc2, doc1")
                for hit in res["hits"]:
                    self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit, msg=f"distance_metric={metric.value}: each hit must have _pre_rerank_score")


class TestCustomScoreRerankingWithOtherFeatures(MarqoTestCase):
    """
    Integration tests that custom score reranking does not break other features.
    Uses the same tuxedo index and model (open_clip/ViT-B-16-SigLIP-512/webli, DOCS_TUXEDO_PLAN)
    as the main tests for deterministic retrieval/ranking and exact score assertions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        index_request = cls.unstructured_marqo_index_request(
            model=Model(name="open_clip/ViT-B-16-SigLIP/webli"),
        )
        collapse_index_request = cls.unstructured_marqo_index_request(
            model=Model(name="open_clip/ViT-B-16-SigLIP/webli"),
            collapse_fields=[CollapseField(name="parent_id", minGroups=2)],
        )
        cls.indexes = cls.create_indexes([index_request, collapse_index_request])
        cls.index = cls.indexes[0]
        cls.collapse_index = cls.indexes[1]

    def _add_tuxedo_docs(self, **extras):
        """Add tuxedo plan docs (optionally with popularity, category, etc.) to self.index."""
        docs = _tuxedo_docs_with_extras(**extras)
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )

    def _assert_pre_rerank_score_matches_baseline(
        self, res_with_rerank, res_no_rerank, tolerance=1e-5
    ):
        """Assert each hit's _pre_rerank_score equals the baseline (no modifiers) score for the same doc."""
        scores_without = {h["_id"]: h["_score"] for h in res_no_rerank["hits"]}
        for hit in res_with_rerank["hits"]:
            doc_id = hit["_id"]
            self.assertIn(
                MARQO_DOC_PRE_RERANK_SCORE,
                hit,
                msg=f"Hit {doc_id} should have _pre_rerank_score when custom score reranking is used",
            )
            self.assertIn(doc_id, scores_without, msg=f"Doc {doc_id} should appear in baseline search")
            self.assertAlmostEqual(
                hit[MARQO_DOC_PRE_RERANK_SCORE],
                scores_without[doc_id],
                delta=tolerance,
                msg=f"Doc {doc_id}: _pre_rerank_score should equal baseline score",
            )

    def _custom_score_add_contributions_by_doc_id(
        self, add_to_score_ops: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Run hybrid search with only the given marqo__score_* add_to_score modifiers (same query/index as plan).
        Returns doc_id -> (_score - _pre_rerank_score) for each hit.
        """
        res = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(add_to_score=add_to_score_ops),
            result_count=10,
        )
        return {
            h["_id"]: h["_score"] - h[MARQO_DOC_PRE_RERANK_SCORE]
            for h in res["hits"]
        }

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_popularity_bm25_and_closeness_together_exact_score(self):
        """
        add_to_score with popularity (global), BM25 custom (lex_ranking_field), and closeness custom
        (tensor_ranking_field), each weight 1.0. For every hit assert:
        _score == _pre_rerank_score + popularity + bm25_contribution + closeness_contribution.
        """
        popularity_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # doc1..doc5
        docs = _tuxedo_docs_with_extras(popularity=popularity_values)
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )
        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        bm25_only = [{"field_name": "marqo__score_bm25_field_lex_ranking_field", "weight": 1.0}]
        closeness_only = [
            {"field_name": "marqo__score_closeness_retrieval_vector_field_tensor_ranking_field", "weight": 1.0}
        ]
        contrib_bm25 = self._custom_score_add_contributions_by_doc_id(bm25_only)
        contrib_closeness = self._custom_score_add_contributions_by_doc_id(closeness_only)
        popularity_by_id = {f"doc{i + 1}": popularity_values[i] for i in range(5)}

        add_to_score = [
            {"field_name": "popularity", "weight": 1.0},
            *bm25_only,
            *closeness_only,
        ]
        res = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(add_to_score=add_to_score),
            result_count=10,
        )
        self._assert_pre_rerank_score_matches_baseline(res, res_no_rerank)
        self.assertEqual(len(res["hits"]), 5, msg="All 5 plan docs must be returned")
        for hit in res["hits"]:
            doc_id = hit["_id"]
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)
            pre = hit[MARQO_DOC_PRE_RERANK_SCORE]
            pop = popularity_by_id[doc_id]
            self.assertIn(doc_id, contrib_bm25)
            self.assertIn(doc_id, contrib_closeness)
            expected_score = pre + pop + contrib_bm25[doc_id] + contrib_closeness[doc_id]
            self.assertAlmostEqual(
                hit["_score"],
                expected_score,
                places=12,
                msg=(
                    f"Doc {doc_id}: _score must equal _pre_rerank_score + popularity + bm25 + closeness "
                    f"({hit['_score']} vs {pre} + {pop} + {contrib_bm25[doc_id]} + {contrib_closeness[doc_id]})"
                ),
            )

    @pytest.mark.skip_for_multinode("For multinode: exact score may not match with replicas")
    def test_pre_rerank_score_returned_with_only_global_score_modifiers(self):
        """
        Tests that using global score modifiers (popularity) only, no custom. Modifier must affect order;
        when _pre_rerank_score is present, assert it equals baseline and _score = _pre_rerank_score + popularity.
        """
        docs = _tuxedo_docs_with_extras(popularity=[0.1, 0.2, 0.3, 0.4, 0.5])  # doc1=0.1 .. doc5=0.5
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )
        res_baseline = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=10,
        )
        res_with_modifier = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(add_to_score=[{"field_name": "popularity", "weight": 1.0}]),
            result_count=10,
        )
        self.assertEqual(len(res_with_modifier["hits"]), 5)
        # Higher popularity should rank higher (doc5 has 0.5, doc4 0.4, ..., doc1 has 0.1).
        ids_with_modifier = [h["_id"] for h in res_with_modifier["hits"]]
        self.assertEqual(ids_with_modifier, REVERSED_ORDER, msg="Order by popularity must be doc5, doc4, doc3, doc2, doc1")
        baseline_by_id = {h["_id"]: h["_score"] for h in res_baseline["hits"]}
        for hit in res_with_modifier["hits"]:
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)
            self.assertAlmostEqual(
                hit[MARQO_DOC_PRE_RERANK_SCORE], baseline_by_id[hit["_id"]], delta=1e-5
            )
            pop = next(d["popularity"] for d in docs if d["_id"] == hit["_id"])
            self.assertAlmostEqual(
                hit["_score"], hit[MARQO_DOC_PRE_RERANK_SCORE] + pop, delta=1e-5
            )

    def test_custom_score_rerank_with_rerank_depth_tensor(self):
        """
        rerankDepthTensor=10 with limit=5: tensor YQL must contain targetHits:10; custom score
        rerank applied (doc5 first for closeness, all hits have _pre_rerank_score).
        """
        self._add_tuxedo_docs()
        rerank_depth_tensor, limit, offset = 10, 5, 0
        captured_query = {}
        original_query = self.config.vespa_client.query

        def capture_then_query(**kwargs):
            captured_query.clear()
            captured_query.update(kwargs)
            return original_query(**kwargs)

        hybrid_params_rerank = HybridParameters(
            **{**HYBRID_PARAMS_TUXEDO.dict(), "rerankDepthTensor": rerank_depth_tensor}
        )
        with mock.patch.object(self.config.vespa_client, "query", capture_then_query):
            res = tensor_search.search(
                config=self.config,
                index_name=self.index.name,
                text="tuxedo",
                search_method="HYBRID",
                hybrid_parameters=hybrid_params_rerank,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        {
                            "field_name": f"marqo__score_closeness_retrieval_vector_field_tensor_ranking_field",
                            "weight": 1.0,
                        }
                    ]
                ),
                result_count=limit,
                offset=offset,
            )
        tensor_yql = captured_query.get("marqo__yql.tensor") or ""
        self.assertIn(f"targetHits:{rerank_depth_tensor}", tensor_yql)
        ids = [h["_id"] for h in res["hits"]]
        self.assertEqual(REVERSED_ORDER, ids, msg="add_to_score closeness with rerankDepthTensor: order must be doc5, doc4, doc3, doc2, doc1")
        for hit in res["hits"]:
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_custom_score_rerank_only_affects_first_rerank_depth_hits(self):
        """
        rerank_depth=3, result_count=5: top 3 hits have _pre_rerank_score and modified scores;
        hits 4 and 5 are excess (no _pre_rerank_score, score = baseline).
        """
        self._add_tuxedo_docs()
        res_baseline = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            result_count=5,
        )
        baseline_scores = {h["_id"]: h["_score"] for h in res_baseline["hits"]}
        self.assertEqual(len(baseline_scores), 5)
        res_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=5,
            rerank_depth=3,
        )
        self.assertEqual(len(res_rerank["hits"]), 5)
        ids_rerank = [h["_id"] for h in res_rerank["hits"]]
        # Only top 3 docs are reranked, so the order must be 3,2,1,4,5
        self.assertEqual(ids_rerank, ["doc3", "doc2", "doc1", "doc4", "doc5"], msg="rerank_depth=3: order must be doc3, doc2, doc1, doc4, doc5")
        for i in range(3):
            hit = res_rerank["hits"][i]
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)
            self.assertAlmostEqual(hit[MARQO_DOC_PRE_RERANK_SCORE], baseline_scores[hit["_id"]], delta=1e-5)
        for i in range(3, 5):
            hit = res_rerank["hits"][i]
            self.assertNotIn(MARQO_DOC_PRE_RERANK_SCORE, hit)
            self.assertAlmostEqual(hit["_score"], baseline_scores[hit["_id"]], delta=1e-5)

    def test_custom_score_rerank_with_facets(self):
        """
        Facets are a separate query in parallel. (1) Main results have custom score applied.
        (2) The facets query sent to Vespa must be identical whether or not custom score
        reranking is used (capture via mock and assert marqo__yql.facets is the same).
        """
        docs = _tuxedo_docs_with_extras(category=("A", "B", "A", "B", "A"))
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )
        facets_params = FacetsParameters(
            fields={"category": FieldFacetsConfiguration(type="string")}
        )
        original_query = self.config.vespa_client.query

        def run_search(score_modifiers):
            captured = {}
            def capture_then_query(**kwargs):
                captured.clear()
                captured["marqo__yql.facets"] = kwargs.get("marqo__yql.facets")
                return original_query(**kwargs)
            with mock.patch.object(self.config.vespa_client, "query", capture_then_query):
                return tensor_search.search(
                    config=self.config,
                    index_name=self.index.name,
                    text="tuxedo",
                    search_method="HYBRID",
                    hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                    score_modifiers=score_modifiers,
                    result_count=5,
                    facets=facets_params,
                ), captured.get("marqo__yql.facets")

        res_no_rerank, facets_yql_no_rerank = run_search(None)
        res_with_rerank, facets_yql_with_rerank = run_search(
            ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                        "weight": 1.0,
                    }
                ]
            )
        )
        self.assertEqual(len(res_with_rerank["hits"]), 5)
        self.assertIn("facets", res_with_rerank)
        self.assertIn("category", res_with_rerank["facets"])
        for hit in res_with_rerank["hits"]:
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)
        self.assertIsNotNone(facets_yql_no_rerank)
        self.assertIsNotNone(facets_yql_with_rerank)
        self.assertEqual(
            facets_yql_no_rerank,
            facets_yql_with_rerank,
            msg="Facets YQL must be identical with or without custom score reranking",
        )
        self.assertGreater(len(res_with_rerank["facets"]["category"]), 0)

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes")
    def test_custom_score_rerank_no_redundant_bm25_terms_in_rank(self):
        """
        When the main lexical term already includes a field that is also requested in add_to_score
        and multiply_score_by, the query sent to Vespa must not duplicate that field in rank().
        (1) Capture the Vespa query and assert lexical YQL has no redundant BM25 term.
        (2) Result order must still be correct (doc5 first with add_to_score bm25 lex_ranking_field).
        """
        self._add_tuxedo_docs()
        # Main lexical term includes both lex_retrieval_field and lex_ranking_field; custom score
        # requests bm25 for lex_ranking_field only -> redundant; simplification should omit extra
        # term for lexical retriever so the field appears only once in the lexical YQL.
        hybrid_params_redundant = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5001,
            rrfK=60,
            searchableAttributesTensor=["tensor_retrieval_field"],
            searchableAttributesLexical=["lex_retrieval_field", "lex_ranking_field"],
        )
        captured_query = {}
        original_query = self.config.vespa_client.query

        def capture_then_query(**kwargs):
            captured_query.clear()
            captured_query.update(kwargs)
            return original_query(**kwargs)

        with mock.patch.object(self.config.vespa_client, "query", capture_then_query):
            res = tensor_search.search(
                config=self.config,
                index_name=self.index.name,
                text="tuxedo",
                search_method="HYBRID",
                hybrid_parameters=hybrid_params_redundant,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        {
                            "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                            "weight": 1.0,
                        }
                    ],
                    multiply_score_by=[
                        {
                            "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                            "weight": 1.0,
                        }
                    ],
                ),
                result_count=10,
            )

        lexical_yql = captured_query.get("marqo__yql.lexical") or ""
        tensor_yql = captured_query.get("marqo__yql.tensor") or ""

        # Lexical retriever: main term already has lex_ranking_field, so no extra rank() term.
        # So lexical YQL must not contain a second (redundant) contains for lex_ranking_field.
        # Main term has two fields -> two "contains" for the phrase; no extra term -> still two.
        self.assertEqual(
            lexical_yql.count('contains "tuxedo"'),
            2,
            msg="Lexical YQL must have exactly two contains (one per field in main term), no redundant rank() term",
        )
        # Tensor retriever has no main lexical term, so it still needs the BM25 term in rank().
        self.assertIn("rank(", tensor_yql, msg="Tensor YQL must still use rank() for BM25 custom score")

        # Results must still be in correct order: add_to_score bm25 lex_ranking_field -> doc5, doc4, doc3, doc2, doc1.
        ids = [h["_id"] for h in res["hits"]]
        self.assertEqual(REVERSED_ORDER, ids, msg="add_to_score bm25 lex_ranking_field: order must be doc5, doc4, doc3, doc2, doc1")
        for hit in res["hits"]:
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)

    # Pagination with custom score reranking: offset must be applied after global reranking.
    # Backend currently applies offset before/during the pipeline, so this test would fail.
    # Will be fixed in a separate feature; skipping until then.
    @pytest.mark.skip(reason="Pagination after custom score reranking will be fixed in a separate feature")
    def test_custom_score_rerank_with_pagination(self):
        """
        Pagination happens after reranking. With custom score reranking, offset must
        skip the correct number of hits from the reranked list. We keep result_count
        fixed so the backend uses the same pipeline; then offset=2 must return hits
        that are exactly positions 2 and 3 of the offset=0 response.
        """
        docs = [dict(d) for d in DOCS_TUXEDO_PLAN] + [
            {"_id": "doc6", "lex_retrieval_field": "tuxedo", "tensor_retrieval_field": "tuxedo", "lex_ranking_field": "tuxedo " * 2, "tensor_ranking_field": "tuxedo"}
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )
        score_modifiers = ScoreModifierLists(
            add_to_score=[
                {"field_name": f"marqo__score_bm25_field_lex_ranking_field", "weight": 1.0}
            ]
        )
        res_full = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                score_modifiers=score_modifiers,
                result_count=6,
                offset=0,
        )
        full_ids = [h["_id"] for h in res_full["hits"]]
        self.assertEqual(len(full_ids), 6, msg="Exactly 6 docs in index")
        res_offset_2 = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=score_modifiers,
            result_count=6,
            offset=2,
        )
        self.assertEqual(len(res_offset_2["hits"]), 4, msg="result_count=6 offset=2 returns 4 hits")
        # Pagination: first two hits of (offset=2) must equal positions 2 and 3 of (offset=0)
        self.assertEqual(
            [res_offset_2["hits"][0]["_id"], res_offset_2["hits"][1]["_id"]],
            full_ids[2:4],
            "offset=2 must return the same hits as positions 2–3 of the full list",
        )
        # Custom score reranking was applied (hits have pre-rerank score)
        for hit in res_full["hits"][:2] + res_offset_2["hits"][:2]:
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)

    def test_custom_score_rerank_with_collapse_fields(self):
        """
        Collapsing happens during fusion; reranking applies to the fused list.
        With collapse and custom score reranking, we get one result per group
        and custom score is applied to the reranked (then collapsed) list.
        """
        docs = _tuxedo_docs_with_extras(parent_id=("g1", "g1", "g2", "g2", "g1"))[:4]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.collapse_index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )
        res = tensor_search.search(
            config=self.config,
            index_name=self.collapse_index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
            collapse=CollapseModel(name="parent_id"),
        )
        self.assertIn("hits", res)
        # After collapse we get exactly one per parent_id (2 groups, 4 docs).
        self.assertEqual(len(res["hits"]), 2, msg="Collapse: exactly 2 groups")
        seen_parents = set()
        for hit in res["hits"]:
            pid = hit.get("parent_id")
            if pid is not None:
                self.assertNotIn(pid, seen_parents, msg="Collapse: one hit per parent_id")
                seen_parents.add(pid)
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)

    @pytest.mark.skip_for_multinode("Lexical/RRF scores can differ between nodes")
    def test_collapse_sort_by_with_custom_score_rerank_both_features(self):
        """
        Show that collapse.sortBy feature (higher price = better) works alongside custom score rerank:

        - Relevance collapse winners per group are g1_lo, g2_lo (strong hybrid with rerankers). Expensive variants
          g1_hi, g2_hi have weak retrieval so they would get negligible custom BM25 add if ranked.
        - Group order follows reranked relevance among those winners: BM25(lex_ranking) favors
          g2_lo over g1_lo → order [g2_lo, g1_lo].
        - sortBy price desc picks strictly higher price per group → visible docs g2_hi, g1_hi.
        - merge_hit copies all _* metadata from the relevance hit: _score and _pre_rerank_score
          stay with g2_lo / g1_lo while _id (and fields like price) come from the hi variants.
        """
        # g*_lo: win per-group relevance. g*_hi: weak RRF, high price (strictly better on desc sort).
        # Only tensor_retrieval_field is needed for hybrid retrieval; lex_ranking_field for custom BM25 rerank.
        docs = [
            {
                "_id": "g1_lo",
                "parent_id": "p1",
                "price": 1.0,
                "lex_retrieval_field": "tuxedo tuxedo tuxedo",
                "tensor_retrieval_field": "tuxedo",
                "lex_ranking_field": "nomatch",
            },
            {
                "_id": "g1_hi",
                "parent_id": "p1",
                "price": 100.0,
                "lex_retrieval_field": "other",
                "tensor_retrieval_field": "unrelated",
                "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo tuxedo",
            },
            {
                "_id": "g2_lo",
                "parent_id": "p2",
                "price": 2.0,
                "lex_retrieval_field": "tuxedo tuxedo",
                "tensor_retrieval_field": "tuxedo",
                "lex_ranking_field": "tuxedo tuxedo tuxedo tuxedo tuxedo tuxedo tuxedo tuxedo tuxedo tuxedo",
            },
            {
                "_id": "g2_hi",
                "parent_id": "p2",
                "price": 200.0,
                "lex_retrieval_field": "suit",
                "tensor_retrieval_field": "suit",
                "lex_ranking_field": "tuxedo tuxedo tuxedo",
            },
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.collapse_index.name,
                docs=docs,
                tensor_fields=["tensor_retrieval_field"],
            ),
        )
        hybrid = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5001,
            rrfK=60,
            searchableAttributesTensor=["tensor_retrieval_field"],
            searchableAttributesLexical=["lex_retrieval_field"],
        )
        collapse_rel = CollapseModel(name="parent_id")
        collapse_sort_desc = CollapseModel(
            name="parent_id",
            sort_by=CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price", order="desc")]
            ),
        )
        rerank_mods = ScoreModifierLists(
            add_to_score=[
                {
                    "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_lex_ranking_field",
                    "weight": 4.0,
                }
            ]
        )

        res_rel_rerank = tensor_search.search(
            config=self.config,
            index_name=self.collapse_index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=hybrid,
            collapse=collapse_rel,
            score_modifiers=rerank_mods,
            result_count=10,
        )
        lo_by_id = {h["_id"]: h for h in res_rel_rerank["hits"]}
        self.assertCountEqual(["g1_lo", "g2_lo"], list(lo_by_id.keys()))
        rerank_order = [h["_id"] for h in res_rel_rerank["hits"]]
        self.assertEqual(
            ["g2_lo", "g1_lo"],
            rerank_order,
            msg="custom BM25 on lex_ranking: g2_lo >> g1_lo (nomatch)",
        )

        res_combined = tensor_search.search(
            config=self.config,
            index_name=self.collapse_index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=hybrid,
            collapse=collapse_sort_desc,
            score_modifiers=rerank_mods,
            result_count=10,
        )
        self.assertEqual(len(res_combined["hits"]), 2)
        combined_order = [h["_id"] for h in res_combined["hits"]]
        self.assertEqual(
            ["g2_hi", "g1_hi"],
            combined_order,
            msg="group order = reranked relevance winners; body = highest price per group",
        )
        for hid, orig in [("g1_hi", "g1_lo"), ("g2_hi", "g2_lo")]:
            hit = next(h for h in res_combined["hits"] if h["_id"] == hid)
            self.assertEqual(hit.get("_originalId"), orig)
            self.assertAlmostEqual(
                hit["_score"],
                lo_by_id[orig]["_score"],
                delta=1e-4,
                msg=f"{hid}: _score from relevance winner {orig}",
            )
            self.assertAlmostEqual(
                hit[MARQO_DOC_PRE_RERANK_SCORE],
                lo_by_id[orig][MARQO_DOC_PRE_RERANK_SCORE],
                delta=1e-4,
                msg=f"{hid}: _pre_rerank_score copied like _score from {orig}",
            )
            self.assertEqual(hit["price"], 100.0 if hid == "g1_hi" else 200.0)

    def test_custom_score_rerank_with_relevance_cutoff(self):
        """
        Custom scores do not change or use targetHits; relevance cutoff (and its probe
        lexical query) should be unaffected. Both together: cutoff is applied and custom
        score reranking is applied to the returned hits.
        Probe lexical query sent to Vespa must be identical with and without custom score
        modifiers (custom score must not change the relevance-cutoff probe).
        """
        self._add_tuxedo_docs()
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            probe_depth=50,
            parameters=RelativeMaxScoreParameters(relative_score_factor=0.5),
        )
        captured_queries = []
        original_query = self.config.vespa_client.query

        def capture_then_query(**kwargs):
            captured_queries.append(dict(kwargs))
            return original_query(**kwargs)

        with mock.patch.object(self.config.vespa_client, "query", capture_then_query):
            tensor_search.search(
                config=self.config,
                index_name=self.index.name,
                text="tuxedo",
                search_method="HYBRID",
                hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        {
                            "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                            "weight": 1.0,
                        }
                    ]
                ),
                result_count=10,
                relevance_cutoff=relevance_cutoff,
            )
            tensor_search.search(
                config=self.config,
                index_name=self.index.name,
                text="tuxedo",
                search_method="HYBRID",
                hybrid_parameters=HYBRID_PARAMS_TUXEDO,
                result_count=10,
                relevance_cutoff=relevance_cutoff,
            )
        self.assertEqual(len(captured_queries), 2, msg="Must capture both requests")
        captured_with_custom_score = captured_queries[0]
        captured_without_custom_score = captured_queries[1]

        # Probe uses marqo__yql.lexical.probe when relevance_cutoff is set (no custom-score extra terms).
        # Both requests must send the same probe YQL so the relevance-cutoff probe is unchanged.
        probe_yql_with = captured_with_custom_score.get("marqo__yql.lexical.probe")
        probe_yql_without = captured_without_custom_score.get("marqo__yql.lexical.probe")
        self.assertIsNotNone(probe_yql_with, msg="Probe YQL must be sent when relevance_cutoff is set")
        self.assertIsNotNone(probe_yql_without, msg="Probe YQL must be sent when relevance_cutoff is set")
        self.assertEqual(
            probe_yql_with,
            probe_yql_without,
            msg="Probe lexical YQL must be identical with and without custom score rerank",
        )
        # Main lexical with custom score can have extra rank() terms; probe must not.
        main_lexical_with = captured_with_custom_score.get("marqo__yql.lexical") or ""
        self.assertNotEqual(
            main_lexical_with,
            probe_yql_with,
            msg="With custom score BM25, main lexical YQL should include extra rank() term; probe must not",
        )

        # Run again (without patch) to assert hit count and pre-rerank score
        res = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
            relevance_cutoff=relevance_cutoff,
        )
        self.assertIn("hits", res)
        self.assertEqual(
            len(res["hits"]), 3,
            msg="Relevance cutoff returns at least one hit; count is determined by probe",
        )
        for hit in res["hits"]:
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)

    def test_custom_score_rerank_with_recency_boost(self):
        """
        Recency and custom score reranking both apply in the global phase with existing
        global score modifier application. They should work independently together.
        """
        now = time.time()
        docs = [
            {**DOCS_TUXEDO_PLAN[0], "_id": "old", "timestamp": now - 30 * 86400},
            {**DOCS_TUXEDO_PLAN[1], "_id": "new", "timestamp": now - 1 * 86400},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS_PLAN,
            ),
        )
        recency_params = RecencyParameters(
            recency_field="timestamp",
            scale="7d",
            offset="0d",
            decay_to=0.5,
        )
        res = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HYBRID_PARAMS_TUXEDO,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
            recency_parameters=recency_params,
        )
        self.assertIn("hits", res)
        self.assertEqual(len(res["hits"]), 2, msg="Exactly 2 docs (old, new)")
        for hit in res["hits"]:
            self.assertIn(MARQO_DOC_PRE_RERANK_SCORE, hit)

    def test_sort_by_with_custom_score_modifiers_raises(self):
        """
        sort_by and scoreModifiers (global score modifiers, including custom score reranking)
        cannot be used together; API validation should error out.
        """
        payload = {
            "q": "tuxedo",
            "searchMethod": SearchMethod.HYBRID,
            "limit": 5,
            "hybridParameters": HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
            ),
            "sortBy": SortByModel(fields=[SortByField(field_name="variantTitle")]),
            "scoreModifiers": ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"marqo__score_bm25_field_lex_ranking_field",
                        "weight": 1.0,
                    }
                ]
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            SearchQuery(**payload)
        self.assertIn("sortBy", str(ctx.exception))
        self.assertIn("scoreModifiers", str(ctx.exception))

    def test_custom_score_rerank_for_hit_only_in_lexical(self):
        """
        Retrieve on field_a, custom score rerank on both field_a and field_b.
        lexical_only_doc has field_a and field_b but NOT exclusive_tensor_field,
        so it appears only in lexical results (not tensor). Confirm that both
        BM25 custom score rerankers are properly applied to lexical_only_doc.
        """
        docs = [
            {
                "_id": "lexical_only_doc",
                "field_a": "tuxedo tuxedo tuxedo",      # will be top bm25 hit
                "field_b": "tuxedo tuxedo",             # will be top bm25 hit
                # NO exclusive_tensor_field -> won't appear in tensor results
            },
            {
                "_id": "both_doc",
                "field_a": "tuxedo",
                "field_b": "tuxedo",
                "exclusive_tensor_field": "tuxedo",
            },
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=["exclusive_tensor_field"],
            ),
        )

        res_baseline = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                searchableAttributesLexical=["field_a"],
                searchableAttributesTensor=["exclusive_tensor_field"],
            ),
            result_count=10,
        )

        res = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="tuxedo",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                searchableAttributesLexical=["field_a"],
                searchableAttributesTensor=["exclusive_tensor_field"],
            ),
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {"field_name": "marqo__score_bm25_field_field_a", "weight": 1000.0},
                    {"field_name": "marqo__score_bm25_field_field_b", "weight": 1000.0},
                ]
            ),
            result_count=10,
        )

        # Both docs should be in results
        ids = [h["_id"] for h in res["hits"]]
        self.assertIn("lexical_only_doc", ids)
        self.assertIn("both_doc", ids)

        # Both custom score rerankers should boost lexical_only_doc's score above its baseline
        for hit in res["hits"]:
            if hit["_id"] == "lexical_only_doc":
                baseline = next(
                    h["_score"] for h in res_baseline["hits"] if h["_id"] == "lexical_only_doc"
                )
                # Score should be significantly above baseline (both rerankers contributing)
                self.assertAlmostEqual(
                    hit["_score"], baseline + 2000,
                    msg="lexical_only_doc should have +1000 from each field",
                )
                # Pre-rerank score should match baseline
                self.assertAlmostEqual(
                    hit[MARQO_DOC_PRE_RERANK_SCORE], baseline, places=5,
                    msg="pre_rerank_score should match baseline",
                )


if __name__ == "__main__":
    unittest.main()
