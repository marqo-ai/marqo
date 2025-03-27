from integ_tests.marqo_test import MarqoTestCase
from pydantic import ValidationError

from marqo.core.models.hybrid_parameters import RankingMethod, RetrievalMethod, HybridParameters
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import SearchQuery


class TestSearchQuery(MarqoTestCase):
    def test_search_query_ExpectedErrorRaisedForInvalidSearchMethod(self):
        """Test that the ValidationError is raised when an incorrect search method is provided."""
        invalid_search_methods = [
            ("", "Empty string"),
            (1, "Integer"),
            ([], "List"),
            ({"searchMethod": "LEXICAL"}, "Dictionary"),
        ]
        for search_method, search_method_type in invalid_search_methods:
            with self.subTest(search_method_type=search_method_type):
                with self.assertRaises(ValidationError) as cm:
                    _ = SearchQuery(q="test", search_method=search_method)
                self.assertIn("search_method", str(cm.exception))

    def test_search_query_CanAcceptDifferentSearchMethods(self):
        """Test that the SearchQuery can accept different search methods."""
        valid_search_methods = [
            ("lexical", SearchMethod.LEXICAL, "lowercase lexical"),
            ("teNsor", SearchMethod.TENSOR, "mixed case tensor"),
            ("hybrid", SearchMethod.HYBRID, "mixed case hybrid"),
            (None, SearchMethod.TENSOR, "None"),
        ]
        for search_method, expected_search_method, search_method_type in valid_search_methods:
            with self.subTest(search_method_type=search_method_type):
                search_query = SearchQuery(q="test", searchMethod=search_method)
                self.assertEqual(expected_search_method, search_query.searchMethod)

        # A special case for no search method provided
        search_query = SearchQuery(q="test")
        self.assertEqual(SearchMethod.TENSOR, search_query.searchMethod)

    def test_search_query_rerank_depth_fails_if_not_hybrid_search_rrf(self):
        """
        Tests that creating a search query with rerank_depth fails if not using
        hybrid search with the RRF rankingMethod.
        """
        # TODO: Remove this test when rerank_depth is supported for tensor, lexical, tensor/lexical, lexical/tensor search.

        # Non-hybrid search
        for search_method in [SearchMethod.LEXICAL]:
            with self.assertRaises(ValueError) as e:
                _ = SearchQuery(q="test", searchMethod=search_method, rerankDepth=5)
            self.assertIn("not supported for 'LEXICAL' search method", str(e.exception))

        # Hybrid search with non-RRF rankingMethod
        for retrieval_method, ranking_method in [
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical)
        ]:
            with self.assertRaises(ValueError) as e:
                _ = SearchQuery(
                    q="test", rerankDepth=5,
                    searchMethod=SearchMethod.HYBRID,
                    hybridParameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method
                    )
                )
            self.assertIn("only supported for 'HYBRID' search with the 'RRF' rankingMethod", str(e.exception))

    def test_search_query_rerank_depth_fails_if_negative(self):
        """
        Tests that creating a search query with rerank_depth fails if the value is negative.
        """
        with self.assertRaises(ValueError) as e:
            _ = SearchQuery(q="test", searchMethod=SearchMethod.HYBRID, rerankDepth=-5)
        self.assertIn("rerankDepth cannot be negative", str(e.exception))

    def test_search_query_rerank_depth_default_value(self):
        """
        Tests that rerank_depth is set to None if not provided.
        """
        search_query = SearchQuery(q="test", searchMethod=SearchMethod.HYBRID, limit=10, offset=5, rerankDepth=20)
        self.assertEqual(20, search_query.rerankDepth)

        search_query = SearchQuery(q="test", searchMethod=SearchMethod.HYBRID, limit=10, offset=5)
        self.assertEqual(None, search_query.rerankDepth)


    def test_hybrid_search_without_queries_or_context_fails(self):
        with self.assertRaises(ValueError) as e:
            _ = SearchQuery(searchMethod=SearchMethod.HYBRID)

        self.assertIn("One of Query(q), context, hybridParameters.queryTensor, or hybridParameters.queryTensor is required for HYBRID search but all are missing", str(e.exception))

    def test_query_with_tensor_query_fails(self):
        with self.assertRaises(ValueError) as e:
            _ = SearchQuery(q="test", searchMethod=SearchMethod.HYBRID, hybridParameters=HybridParameters(queryTensor="test"))

        self.assertIn("Query(q) cannot be provided for HYBRID search when hybridParameters.queryTensor or hybridParameters.queryLexical is provided", str(e.exception))

    def test_query_with_lexical_query_fails(self):
        with self.assertRaises(ValueError) as e:
            _ = SearchQuery(q="test", searchMethod=SearchMethod.HYBRID, hybridParameters=HybridParameters(queryLexical="test"))

        self.assertIn("Query(q) cannot be provided for HYBRID search when hybridParameters.queryTensor or hybridParameters.queryLexical is provided", str(e.exception))
