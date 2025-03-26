import unittest
from pydantic import ValidationError

from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import SearchQuery, CustomVectorQuery


class TestSearchQueryModel(unittest.TestCase):
    def test_tensor_query_string(self):
        q = "dogs"
        sq = SearchQuery(q=q, searchMethod="TENSOR")
        self.assertEqual(sq.q, q)
        self.assertEqual(sq.searchMethod, SearchMethod.TENSOR)

    def test_tensor_query_dict(self):
        q = {"dogs": 2.0, "cats": -1.0}
        sq = SearchQuery(q=q, searchMethod="TENSOR")
        self.assertEqual(sq.q, q)

    def test_tensor_query_custom_vector(self):
        custom_query = CustomVectorQuery(customVector=CustomVectorQuery.CustomVector(
            content="dogs", vector=[0.1, 0.2, 0.3]))
        sq = SearchQuery(q=custom_query, searchMethod="TENSOR")
        self.assertEqual(sq.q, custom_query)

    def test_tensor_query_missing_query_and_context_raises(self):
        with self.assertRaises(ValueError):
            SearchQuery(searchMethod="TENSOR")

    def test_lexical_query_missing_q_raises(self):
        with self.assertRaises(ValueError):
            SearchQuery(searchMethod="LEXICAL")

    def test_lexical_query_valid(self):
        sq = SearchQuery(q="dogs", searchMethod="LEXICAL")
        self.assertEqual(sq.q, "dogs")
        self.assertEqual(sq.searchMethod, SearchMethod.LEXICAL)

    def test_hybrid_query_with_only_queryTensor(self):
        sq = SearchQuery(searchMethod="HYBRID", hybridParameters=HybridParameters(queryTensor={"dogs": 1.0}))
        self.assertEqual(sq.searchMethod, SearchMethod.HYBRID)

    def test_hybrid_query_with_only_queryLexical(self):
        sq = SearchQuery(searchMethod="HYBRID", hybridParameters=HybridParameters(queryLexical="dogs"))
        self.assertEqual(sq.searchMethod, SearchMethod.HYBRID)

    def test_hybrid_query_with_q_and_tensor_fails(self):
        with self.assertRaises(ValueError):
            SearchQuery(q="dogs", searchMethod="HYBRID",
                        hybridParameters=HybridParameters(queryTensor={"dogs": 1.0}))

    def test_hybrid_query_with_q_and_lexical_fails(self):
        with self.assertRaises(ValueError):
            SearchQuery(q="dogs", searchMethod="HYBRID",
                        hybridParameters=HybridParameters(queryLexical="dogs"))

    def test_hybrid_query_without_q_context_or_params_fails(self):
        with self.assertRaises(ValueError):
            SearchQuery(searchMethod="HYBRID")

    def test_invalid_rerank_depth_not_rrf(self):
        with self.assertRaises(ValueError):
            SearchQuery(
                q="test", rerankDepth=5, searchMethod="HYBRID",
                hybridParameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Tensor,
                    rankingMethod=RankingMethod.Tensor
                )
            )

    def test_valid_rerank_depth_with_rrf(self):
        sq = SearchQuery(
            q="test", rerankDepth=5, searchMethod="HYBRID",
            hybridParameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF
            )
        )
        self.assertEqual(sq.rerankDepth, 5)

    def test_negative_rerank_depth_raises(self):
        with self.assertRaises(ValueError):
            SearchQuery(
                q="test", rerankDepth=-5, searchMethod="HYBRID",
                hybridParameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical,
                    rankingMethod=RankingMethod.RRF
                )
            )

    def test_hybrid_params_only_allowed_for_hybrid(self):
        with self.assertRaises(ValueError):
            SearchQuery(
                q="test", searchMethod="TENSOR",
                hybridParameters=HybridParameters(queryLexical="dogs")
            )

    def test_search_method_defaults_to_tensor(self):
        sq = SearchQuery(q="dogs")
        self.assertEqual(sq.searchMethod, SearchMethod.TENSOR)

    def test_image_and_media_headers_conflict(self):
        with self.assertRaises(ValueError):
            SearchQuery(
                q="dogs", imageDownloadHeaders={"Auth": "token"},
                mediaDownloadHeaders={"Auth": "token"}
            )


if __name__ == "__main__":
    unittest.main()
