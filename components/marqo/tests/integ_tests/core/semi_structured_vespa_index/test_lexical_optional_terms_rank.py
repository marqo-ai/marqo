"""Integration tests for lexical optional/required terms using rank().

Tests that the rank() operator correctly handles mixed required (quoted) and
optional (unquoted) terms so that:
- Only required terms affect recall (matching)
- Optional terms contribute to BM25 scoring
- Various combinations with hybrid search, custom score rerank, and lexicalOperand work correctly
"""

import unittest
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.tensor_search import tensor_search
from tests.integ_tests.marqo_test import MarqoTestCase


class TestLexicalOptionalTermsWithRank(MarqoTestCase):
    """Integration tests for the rank() operator with mixed required/optional lexical terms."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Create an unstructured index with auto-generated name (uses default random/small model)
        index_request = cls.unstructured_marqo_index_request()
        cls.create_indexes([index_request])
        cls.index_name = index_request.name

        # Add test documents
        # doc1: has "required" only
        # doc2: has "required" and "optional"
        # doc3: has "optional" only
        # doc4: has neither
        docs = [
            {"_id": "doc1", "content": "This document has the required term"},
            {"_id": "doc2", "content": "This document has both required and optional terms"},
            {"_id": "doc3", "content": "This document has the optional term only"},
            {"_id": "doc4", "content": "This document has nothing relevant"},
        ]

        cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=docs,
                index_name=cls.index_name,
                tensor_fields=["content"],
            ),
        )

    def setUp(self):
        # Don't clear documents between tests
        pass

    def test_basic_recall_with_mixed_terms(self):
        """Documents with only 'required' should be returned even without 'optional'."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"required" optional',
            search_method="LEXICAL",
            result_count=10,
        )
        result_ids = {hit["_id"] for hit in res["hits"]}
        # doc1 and doc2 have "required" - both should be returned
        self.assertIn("doc1", result_ids, "Doc with only 'required' should be found")
        self.assertIn("doc2", result_ids, "Doc with both terms should be found")
        # doc3 has only "optional" - should NOT be returned (optional terms don't affect recall)
        self.assertNotIn("doc3", result_ids,
                         "Doc with only 'optional' should NOT be found when 'required' is quoted")

    def test_bm25_scoring_optional_boosts(self):
        """Documents matching both required+optional should score higher than required-only."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"required" optional',
            search_method="LEXICAL",
            result_count=10,
        )
        scores = {hit["_id"]: hit["_score"] for hit in res["hits"]}
        if "doc1" in scores and "doc2" in scores:
            self.assertGreater(
                scores["doc2"], scores["doc1"],
                "Doc with both 'required' and 'optional' should score higher"
            )

    def test_or_only_unchanged(self):
        """Optional-only query should return docs matching any term (unchanged behavior)."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="required optional",
            search_method="LEXICAL",
            result_count=10,
        )
        result_ids = {hit["_id"] for hit in res["hits"]}
        # All docs with either 'required' or 'optional' should be found
        self.assertIn("doc1", result_ids)
        self.assertIn("doc2", result_ids)
        self.assertIn("doc3", result_ids)

    def test_and_only_unchanged(self):
        """All-required query should require all terms (unchanged behavior)."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"required" "optional"',
            search_method="LEXICAL",
            result_count=10,
        )
        result_ids = {hit["_id"] for hit in res["hits"]}
        # Only doc2 has both required and optional
        self.assertIn("doc2", result_ids)
        # doc1 has only required, doc3 has only optional
        self.assertNotIn("doc1", result_ids)
        self.assertNotIn("doc3", result_ids)

    def test_hybrid_lexical_tensor_with_mixed_terms(self):
        """Hybrid search with mixed terms should use rank() for lexical retrieval."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"required" optional',
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
            ),
            result_count=10,
        )
        result_ids = {hit["_id"] for hit in res["hits"]}
        # doc1 and doc2 should be returned (have "required")
        # doc3 may or may not be found (tensor may retrieve it)
        self.assertIn("doc1", result_ids)
        self.assertIn("doc2", result_ids)

    def test_lexical_operand_and_keeps_old_behavior(self):
        """With lexicalOperand='and', all terms become required (old AND behavior)."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"required" optional',
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
                lexicalOperand="and",
            ),
            result_count=10,
        )
        result_ids = {hit["_id"] for hit in res["hits"]}
        # With AND, both "required" AND "optional" are required for lexical matching
        # doc2 has both terms
        self.assertIn("doc2", result_ids)

    def test_lexical_operand_weakand_uses_rank(self):
        """With lexicalOperand='weakAnd', rank() is used for mixed terms."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"required" optional',
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
                lexicalOperand="weakAnd",
            ),
            result_count=10,
        )
        result_ids = {hit["_id"] for hit in res["hits"]}
        # doc1 and doc2 should be returned (have "required")
        self.assertIn("doc1", result_ids)
        self.assertIn("doc2", result_ids)

    def test_rerank_depth_lexical_with_mixed_terms(self):
        """OR-only query with rerankDepthLexical still uses weakAnd with targetHits."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text="required optional",
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
                rerankDepthLexical=100,
            ),
            result_count=10,
        )
        # Should return results without error
        self.assertIn("hits", res)

    def test_rerank_depth_with_lexical_operand_and_raises(self):
        """rerankDepthLexical with lexicalOperand='and' should raise validation error."""
        with self.assertRaises(Exception):
            tensor_search.search(
                config=self.config,
                index_name=self.index_name,
                text="required optional",
                search_method="HYBRID",
                hybrid_parameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    alpha=0.5,
                    rerankDepthLexical=100,
                    lexicalOperand="and",
                ),
                result_count=10,
            )

    def test_track_total_hits_with_mixed_terms(self):
        """trackTotalHits with mixed terms should work correctly."""
        res = tensor_search.search(
            config=self.config,
            index_name=self.index_name,
            text='"required" optional',
            search_method="HYBRID",
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
            ),
            result_count=10,
            track_total_hits=True,
        )
        self.assertIn("hits", res)
        # totalHits should be present
        self.assertIn("totalHits", res)


if __name__ == "__main__":
    unittest.main()
