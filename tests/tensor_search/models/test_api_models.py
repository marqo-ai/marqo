from unittest import mock

import marqo.core.exceptions as core_exceptions
from marqo.s2_inference.errors import ImageDownloadError
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from marqo.core.models.add_docs_params import AddDocsParams
from tests.marqo_test import MarqoTestCase, TestImageUrls
from marqo import exceptions as base_exceptions
from marqo.core.models.marqo_query import MarqoLexicalQuery
from marqo.core.models.score_modifier import ScoreModifierType, ScoreModifier
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.core.models.hybrid_parameters import RankingMethod, RetrievalMethod, HybridParameters
from pydantic import ValidationError
import marqo.api.exceptions as api_exceptions


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

    def test_search_query_rerank_count_fails_if_not_hybrid_search_rrf(self):
        """
        Tests that creating a search query with rerank_count fails if not using
        hybrid search with the RRF rankingMethod.
        """
        # TODO: Remove this test when rerank_count is supported for tensor, lexical, tensor/lexical, lexical/tensor search.

        # Non-hybrid search
        for search_method in [SearchMethod.LEXICAL, SearchMethod.TENSOR]:
            with self.assertRaises(ValueError) as e:
                _ = SearchQuery(q="test", searchMethod=search_method, rerankCount=5)
            self.assertIn("only supported for 'HYBRID' search", str(e.exception))

        # Hybrid search with non-RRF rankingMethod
        for retrieval_method, ranking_method in [
            (RetrievalMethod.Lexical, RankingMethod.Lexical),
            (RetrievalMethod.Tensor, RankingMethod.Tensor),
            (RetrievalMethod.Lexical, RankingMethod.Tensor),
            (RetrievalMethod.Tensor, RankingMethod.Lexical)
        ]:
            with self.assertRaises(ValueError) as e:
                _ = SearchQuery(
                    q="test", rerankCount=5,
                    searchMethod=SearchMethod.HYBRID,
                    hybridParameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method
                    )
                )
            self.assertIn("only supported for 'HYBRID' search with the 'RRF' rankingMethod", str(e.exception))

    def test_search_query_rerank_count_default_value(self):
        """
        Tests that rerank_count is set to None if not provided.
        """
        search_query = SearchQuery(q="test", searchMethod=SearchMethod.HYBRID, limit=10, offset=5, rerankCount=20)
        self.assertEqual(20, search_query.rerankCount)

        search_query = SearchQuery(q="test", searchMethod=SearchMethod.HYBRID, limit=10, offset=5)
        self.assertEqual(None, search_query.rerankCount)