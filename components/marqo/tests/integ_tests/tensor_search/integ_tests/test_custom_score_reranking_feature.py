"""
Integration tests for the custom score reranking feature (Part C of the plan).

Semi-structured index only. Field layout: variantTitle and variantDescription are text
(lexical for BM25). tensorField1 and tensorField2 are tensor fields (for closeness);
only these are passed in tensor_fields when adding docs.
"""
import os
from unittest import mock

from marqo.core.constants import MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.marqo_index import Model
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.api_models import ScoreModifierLists
from tests.integ_tests.marqo_test import MarqoTestCase

import unittest

# Only tensor fields; variantTitle and variantDescription are text (lexical) in semi-structured.
TENSOR_FIELDS = ["tensorField1", "tensorField2"]


def _doc_bm25(variant_title: str, variant_description: str = "", tensor_text: str = "widget") -> dict:
    """Doc with all index fields; tensor fields share same text so all docs are retrieved."""
    return {
        "variantTitle": variant_title,
        "variantDescription": variant_description,
        "tensorField1": tensor_text,
        "tensorField2": tensor_text,
    }


def _doc_closeness(variant_title: str, tensor_field1: str, tensor_field2: str = None) -> dict:
    """Doc for closeness tests; variantTitle matches query so lexical returns all."""
    if tensor_field2 is None:
        tensor_field2 = tensor_field1
    return {
        "variantTitle": variant_title,
        "variantDescription": "",
        "tensorField1": tensor_field1,
        "tensorField2": tensor_field2,
    }


class TestCustomScoreRerankingFeature(MarqoTestCase):
    """Integration tests that RRF + custom score reranker rerank and modify scores correctly. Semi-structured only."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        index_request = cls.unstructured_marqo_index_request(
            model=Model(name="hf/all-MiniLM-L6-v2"),
        )
        cls.indexes = cls.create_indexes([index_request])
        cls.index = cls.indexes[0]

    def setUp(self) -> None:
        super().setUp()
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_rrf_with_bm25_single_field_modifies_scores(self):
        """
        Shows a custom score reranker changes the final document score.
        BM25 scores of variantTitle are added to RRF score; docs with more "widget" terms get a higher score boost.
        """
        docs = [
            {"_id": "low_bm25", **_doc_bm25("widget")},
            {"_id": "mid_bm25", **_doc_bm25("widget widget")},
            {"_id": "high_bm25", **_doc_bm25("widget widget widget")},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS,
            ),
        )

        query = "widget"
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )

        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            result_count=10,
        )
        self.assertIn("hits", res_no_rerank)
        self.assertEqual(len(res_no_rerank["hits"]), 3, "Expect 3 hits for 3 docs")

        res_with_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_variantTitle",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        self.assertIn("hits", res_with_rerank)
        self.assertGreaterEqual(len(res_with_rerank["hits"]), 3)

        scores_without = {h["_id"]: h["_score"] for h in res_no_rerank["hits"]}
        scores_with = {h["_id"]: h["_score"] for h in res_with_rerank["hits"]}

        tolerance = 1e-5
        expected_deltas = {"low_bm25": 0.0, "mid_bm25": 0.726, "high_bm25": 1.0}
        for doc_id, expected in expected_deltas.items():
            self.assertIn(doc_id, scores_without, msg=f"Missing {doc_id} in search without reranker")
            self.assertIn(doc_id, scores_with, msg=f"Missing {doc_id} in search with reranker")
            delta = scores_with[doc_id] - scores_without[doc_id]
            tol = 0.001 if doc_id == "mid_bm25" else tolerance
            self.assertAlmostEqual(
                delta,
                expected,
                delta=tol,
                msg=f"Doc {doc_id}: expected score delta {expected}, got {delta}",
            )
        deltas = {d: scores_with[d] - scores_without[d] for d in expected_deltas}
        self.assertLess(deltas["low_bm25"], deltas["mid_bm25"])
        self.assertLess(deltas["mid_bm25"], deltas["high_bm25"])

        ids_with_rerank = [h["_id"] for h in res_with_rerank["hits"]]
        self.assertEqual(
            ids_with_rerank[0],
            "high_bm25",
            msg="Doc with most 'widget' in variantTitle should rank first when using bm25_field_variantTitle reranker",
        )

    def test_rrf_with_bm25_custom_score_changes_order(self):
        """
        Shows a custom score reranker can change the order of results.
        Search only variantDescription; rank by bm25_field_variantTitle. Two docs with opposite BM25 (high in one field, low in the other). Without reranker the variantDescription-high doc is first; with reranker the variantTitle-high doc is first.
        """
        docs_order = [
            {"_id": "desc_first", **_doc_bm25("x", "widget widget widget")},
            {"_id": "title_first", **_doc_bm25("widget widget widget", "widget")},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs_order,
                tensor_fields=TENSOR_FIELDS,
            ),
        )
        hybrid_params_desc = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
            searchableAttributesLexical=["variantDescription"],
            searchableAttributesTensor=TENSOR_FIELDS,
        )
        res_no_rerank_desc = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="widget",
            search_method="HYBRID",
            hybrid_parameters=hybrid_params_desc,
            result_count=10,
        )
        res_with_rerank_desc = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="widget",
            search_method="HYBRID",
            hybrid_parameters=hybrid_params_desc,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_variantTitle",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        ids_no = [h["_id"] for h in res_no_rerank_desc["hits"]]
        ids_with = [h["_id"] for h in res_with_rerank_desc["hits"]]
        self.assertEqual(ids_no[0], "desc_first", msg="Without reranker, doc with most 'widget' in variantDescription ranks first")
        self.assertEqual(ids_with[0], "title_first", msg="With bm25_field_variantTitle reranker, doc with most 'widget' in variantTitle ranks first; order reversed")

    def test_bm25_multiple_terms_and_and_or(self):
        """
        Custom score reranker with multi-term queries: OR terms (unquoted) and AND term (double-quoted).
        With OR query the doc matching both terms ranks first; with AND query only the doc containing the quoted term matches and ranks first.
        """
        docs = [
            {"_id": "only_foo", **_doc_bm25("foo")},
            {"_id": "only_bar", **_doc_bm25("bar")},
            {"_id": "foo_bar_both", **_doc_bm25("foo bar")},
            {"_id": "foo_bar_unique", **_doc_bm25("foo bar unique")},  # only one with "unique"
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS,
            ),
        )
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )

        # OR: "foo bar" – terms are OR; doc with both should rank first when boosting by bm25 variantTitle
        res_or = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="foo bar",
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_variantTitle",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        self.assertIn("hits", res_or)
        self.assertGreaterEqual(len(res_or["hits"]), 3)
        ids_or = [h["_id"] for h in res_or["hits"]]
        self.assertEqual(ids_or[0], "foo_bar_both", msg="Doc matching both OR terms should rank first with bm25_field reranker")

        # AND: "unique" in double quotes is required; only foo_bar_unique contains "unique"
        res_and = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text='foo bar "unique"',
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_variantTitle",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        self.assertIn("hits", res_and)
        self.assertGreaterEqual(len(res_and["hits"]), 1)
        ids_and = [h["_id"] for h in res_and["hits"]]
        self.assertEqual(ids_and[0], "foo_bar_unique", msg="Only doc with AND term 'unique' should rank first")

    def test_bm25_aggregate_add_to_score(self):
        """
        BM25 aggregate (bm25_sum) sums BM25 across variantTitle and variantDescription.
        Both fields get the same content per doc so the sum doubles; with weight 2.0, score deltas are 0, ~1.45, 2.
        """
        docs = [
            {"_id": "low_bm25", **_doc_bm25("widget", "widget")},
            {"_id": "mid_bm25", **_doc_bm25("widget widget", "widget widget")},
            {"_id": "high_bm25", **_doc_bm25("widget widget widget", "widget widget widget")},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS,
            ),
        )
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )

        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="widget",
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            result_count=10,
        )
        res_with_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="widget",
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {"field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_sum", "weight": 2.0}
                ]
            ),
            result_count=10,
        )
        scores_without = {h["_id"]: h["_score"] for h in res_no_rerank["hits"]}
        scores_with = {h["_id"]: h["_score"] for h in res_with_rerank["hits"]}
        # Two fields with same BM25 → sum is 2x; min-max normalized still 0, ~0.726, 1; weight 2 → deltas 0, ~1.45, 2
        tolerance = 1e-5
        expected_deltas = {"low_bm25": 0.0, "mid_bm25": 1.45, "high_bm25": 2.0}
        for doc_id, expected in expected_deltas.items():
            self.assertIn(doc_id, scores_without)
            self.assertIn(doc_id, scores_with)
            delta = scores_with[doc_id] - scores_without[doc_id]
            tol = 0.02 if doc_id == "mid_bm25" else tolerance
            self.assertAlmostEqual(delta, expected, delta=tol, msg=f"bm25_sum delta for {doc_id}: expected {expected}, got {delta}")
        ids_with_rerank = [h["_id"] for h in res_with_rerank["hits"]]
        self.assertEqual(ids_with_rerank[0], "high_bm25", msg="high_bm25 should rank first with bm25_sum reranker")

    def test_custom_score_rerank_different_weights_exact_scores(self):
        """
        Custom score add with negative weight (-1.0) and weight > 1 (2.0); asserts exact score deltas.
        Same three-doc setup: deltas are weight * normalized BM25, so -1.0 gives 0, ~-0.726, -1 and 2.0 gives 0, ~1.45, 2.
        """
        docs = [
            {"_id": "low_bm25", **_doc_bm25("widget")},
            {"_id": "mid_bm25", **_doc_bm25("widget widget")},
            {"_id": "high_bm25", **_doc_bm25("widget widget widget")},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS,
            ),
        )
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )

        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="widget",
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            result_count=10,
        )
        scores_without = {h["_id"]: h["_score"] for h in res_no_rerank["hits"]}

        # Weight -1.0: deltas 0, ~-0.726, -1
        res_neg = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="widget",
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_variantTitle",
                        "weight": -1.0,
                    }
                ]
            ),
            result_count=10,
        )
        scores_neg = {h["_id"]: h["_score"] for h in res_neg["hits"]}
        tol_mid = 0.01  # mid doc normalized BM25 varies slightly with schema
        self.assertAlmostEqual(scores_neg["low_bm25"] - scores_without["low_bm25"], 0.0, delta=1e-5)
        self.assertAlmostEqual(scores_neg["mid_bm25"] - scores_without["mid_bm25"], -0.726, delta=tol_mid)
        self.assertAlmostEqual(scores_neg["high_bm25"] - scores_without["high_bm25"], -1.0, delta=1e-5)

        # Weight 2.0: deltas 0, ~1.45, 2
        res_double = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text="widget",
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_variantTitle",
                        "weight": 2.0,
                    }
                ]
            ),
            result_count=10,
        )
        scores_double = {h["_id"]: h["_score"] for h in res_double["hits"]}
        self.assertAlmostEqual(scores_double["low_bm25"] - scores_without["low_bm25"], 0.0, delta=1e-5)
        self.assertAlmostEqual(scores_double["mid_bm25"] - scores_without["mid_bm25"], 1.45, delta=tol_mid)
        self.assertAlmostEqual(scores_double["high_bm25"] - scores_without["high_bm25"], 2.0, delta=1e-5)

    def test_rrf_with_closeness_retrieval_vector_single_field_modifies_scores(self):
        """
        Shows a custom score reranker changes the final document score.
        Closeness of tensorField1 to the query vector is added to RRF score; the doc with query text in tensorField1 gets the highest boost (5 docs so all positions are exercised).
        """
        query_phrase = "exact match phrase"
        docs = [
            {"_id": "d0", **_doc_closeness(query_phrase, "totally unrelated content A")},
            {"_id": "d1", **_doc_closeness(query_phrase, "somewhat related content B")},
            {"_id": "d2", **_doc_closeness(query_phrase, "partially related phrase C")},
            {"_id": "d3", **_doc_closeness(query_phrase, "exact match phrase")},
            {"_id": "d4", **_doc_closeness(query_phrase, "exact match phrase")},  # tie with d3 for max closeness
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS,
            ),
        )
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )

        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query_phrase,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            result_count=10,
        )
        res_with_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query_phrase,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_field_tensorField1",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        self.assertIn("hits", res_no_rerank)
        self.assertIn("hits", res_with_rerank)
        self.assertEqual(len(res_no_rerank["hits"]), 5, "Expect 5 hits for 5 docs")
        self.assertGreaterEqual(len(res_with_rerank["hits"]), 5)

        scores_without = {h["_id"]: h["_score"] for h in res_no_rerank["hits"]}
        scores_with = {h["_id"]: h["_score"] for h in res_with_rerank["hits"]}
        deltas = {doc_id: scores_with[doc_id] - scores_without[doc_id] for doc_id in scores_without if doc_id in scores_with}
        # Docs with query text in tensorField1 (d3, d4) should get the largest score increase
        self.assertIn("d3", deltas)
        self.assertIn("d4", deltas)
        max_delta = max(deltas.values())
        self.assertGreaterEqual(deltas["d3"], max_delta - 1e-5, msg="d3 (query text in tensorField1) should get among the highest add-to-score delta")
        self.assertGreaterEqual(deltas["d4"], max_delta - 1e-5, msg="d4 (query text in tensorField1) should get among the highest add-to-score delta")
        # At least one of the "far" docs should have strictly smaller delta than d3/d4
        far_deltas = [deltas["d0"], deltas["d1"], deltas["d2"]]
        self.assertLess(min(far_deltas), max_delta, msg="At least one less-close doc should have smaller delta than the closest docs")
        ids_with_rerank = [h["_id"] for h in res_with_rerank["hits"]]
        self.assertIn(ids_with_rerank[0], ("d3", "d4"), msg="Top doc should be one with query text in tensorField1")

    def test_rrf_with_closeness_retrieval_vector_custom_score_changes_order(self):
        """
        Shows a custom score reranker can change the order of results.
        Query "dogs". Tensor search uses only tensorField2 (weird order); reranker uses closeness of tensorField1 so order becomes: dogs (closest), puppies, unrelated.
        """
        query = "dogs"
        # tensorField1: unrelated = far, puppies = closer, dogs = closest. tensorField2 is used for search only and is chosen so order without rerank is wrong.
        docs_order = [
            {"_id": "doc_dogs", **_doc_closeness(query, "dogs", "unrelated")},       # field1 closest; field2 far so without rerank ranks last
            {"_id": "doc_puppies", **_doc_closeness(query, "puppies", "puppies")},   # field1 mid; field2 mid
            {"_id": "doc_unrelated", **_doc_closeness(query, "unrelated", "dogs")},  # field1 far; field2 = "dogs" so without rerank ranks first
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs_order,
                tensor_fields=TENSOR_FIELDS,
            ),
        )
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
            searchableAttributesTensor=["tensorField2"],
        )

        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            result_count=10,
        )
        res_with_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {
                        "field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_field_tensorField1",
                        "weight": 1.0,
                    }
                ]
            ),
            result_count=10,
        )
        ids_before = [h["_id"] for h in res_no_rerank["hits"]]
        ids_after = [h["_id"] for h in res_with_rerank["hits"]]
        self.assertEqual(len(ids_before), 3)
        self.assertEqual(len(ids_after), 3)
        # Without rerank: only tensorField2 is searched, so doc_unrelated (field2="dogs") first, doc_puppies (field2="puppies") second, doc_dogs (field2="unrelated") last
        self.assertEqual(ids_before[0], "doc_unrelated", msg="Without rerank: tensorField2 is searchable so doc with field2='dogs' ranks first")
        self.assertEqual(ids_before[1], "doc_puppies")
        self.assertEqual(ids_before[2], "doc_dogs", msg="Without rerank: doc with field2='unrelated' ranks last")
        # With rerank: order by closeness of tensorField1 to "dogs" → dogs, puppies, unrelated
        self.assertEqual(ids_after[0], "doc_dogs", msg="With rerank: doc with tensorField1='dogs' (closest) ranks first")
        self.assertEqual(ids_after[1], "doc_puppies", msg="With rerank: doc with tensorField1='puppies' (closer) ranks second")
        self.assertEqual(ids_after[2], "doc_unrelated", msg="With rerank: doc with tensorField1='unrelated' (far) ranks last")
        self.assertNotEqual(ids_before, ids_after, msg="Order before and after reranking must differ")

    def test_closeness_retrieval_vector_aggregate_add_to_score(self):
        """
        closeness_retrieval_vector_sum sums closeness across tensorField1 and tensorField2.
        Both fields have the same content per doc so the sum doubles; with weight 1.0 the closest doc gets a larger score increase and ranks first.
        """
        query_phrase = "exact match phrase"
        docs = [
            {"_id": "far", **_doc_closeness(query_phrase, "unrelated topic", "unrelated topic")},
            {"_id": "close", **_doc_closeness(query_phrase, query_phrase, query_phrase)},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS,
            ),
        )
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )

        res_no_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query_phrase,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            result_count=10,
        )
        res_with_rerank = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query_phrase,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {"field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_sum", "weight": 1.0}
                ]
            ),
            result_count=10,
        )
        self.assertIn("hits", res_with_rerank)
        self.assertGreaterEqual(len(res_with_rerank["hits"]), 2)
        ids_with_rerank = [h["_id"] for h in res_with_rerank["hits"]]
        self.assertEqual(ids_with_rerank[0], "close", msg="Closest doc should rank first with closeness_retrieval_vector_sum")
        scores_without = {h["_id"]: h["_score"] for h in res_no_rerank["hits"]}
        scores_with = {h["_id"]: h["_score"] for h in res_with_rerank["hits"]}
        delta_far = scores_with["far"] - scores_without["far"]
        delta_close = scores_with["close"] - scores_without["close"]
        self.assertGreater(delta_close, delta_far, msg="Closer doc should get larger score increase from closeness_retrieval_vector_sum")

    def test_closeness_retrieval_vector_combined(self):
        """
        One query using add_to_score (single field + aggregate) and multiply_score_by (single field) with different weights.
        Two docs: one with query text in both tensor fields (closeness ~1), one with different text (closeness ~0). Asserts order and that all modifier kinds applied (add single, add sum, mult single).
        """
        query_phrase = "exact match phrase"
        docs = [
            {"_id": "far", **_doc_closeness(query_phrase, "unrelated A", "unrelated B")},
            {"_id": "close", **_doc_closeness(query_phrase, query_phrase, query_phrase)},
        ]
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index.name,
                docs=docs,
                tensor_fields=TENSOR_FIELDS,
            ),
        )
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )

        # Combined: add_to_score single field (tensorField1) weight 1.0, add_to_score sum weight 0.5, multiply_score_by tensorField2 weight 1.0
        res = tensor_search.search(
            config=self.config,
            index_name=self.index.name,
            text=query_phrase,
            search_method="HYBRID",
            hybrid_parameters=hybrid_params,
            score_modifiers=ScoreModifierLists(
                add_to_score=[
                    {"field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_field_tensorField1", "weight": 1.0},
                    {"field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_sum", "weight": 0.5},
                ],
                multiply_score_by=[
                    {"field_name": f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_field_tensorField2", "weight": 1.0},
                ],
            ),
            result_count=10,
        )
        self.assertIn("hits", res)
        self.assertGreaterEqual(len(res["hits"]), 2)
        ids = [h["_id"] for h in res["hits"]]
        self.assertEqual(ids[0], "close", msg="Close doc (high closeness on both fields) should rank first with combined add + mult")
        scores = {h["_id"]: h["_score"] for h in res["hits"]}
        self.assertGreater(scores["close"], scores["far"], msg="Close doc should have strictly higher score than far doc")
        # Far doc gets multiply by ~0 (closeness ~0 on tensorField2) so its score is heavily reduced; close gets add + mult both positive
        self.assertGreater(scores["far"], 0, msg="Far doc should still have positive score (closeness > 0 in practice)")


if __name__ == "__main__":
    unittest.main()
