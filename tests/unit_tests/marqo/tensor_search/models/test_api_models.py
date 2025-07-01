import unittest
from pydantic.v1 import ValidationError

from marqo.tensor_search.models.api_models import SearchQuery, CustomVectorQuery
from marqo.tensor_search.enums import SearchMethod
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor


class TestSearchQuery(unittest.TestCase):

    def test_search_query_with_all_parameters(self):
        """Test SearchQuery creation with all parameters set to valid values."""
        custom_vector_query = CustomVectorQuery(
            customVector=CustomVectorQuery.CustomVector(
                content="test content",
                vector=[0.1, 0.2, 0.3, 0.4]
            )
        )

        hybrid_parameters = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.7,
            rrfK=100
        )

        facets = FacetsParameters(
            fields={
                "category": FieldFacetsConfiguration(type="string", maxResults=10)
            }
        )

        context = SearchContext(
            tensor=[SearchContextTensor(vector=[0.1, 0.2], weight=1.0)]
        )

        search_query = SearchQuery(
            q=custom_vector_query,
            searchableAttributes=["title", "description"],
            searchMethod=SearchMethod.HYBRID,
            limit=20,
            offset=5,
            rerankDepth=100,
            efSearch=200,
            approximate=True,
            approximateThreshold=0.85,
            showHighlights=False,
            reRanker="test_reranker",
            filter="category:electronics",
            attributesToRetrieve=["title", "price"],
            boost={"title": 1.5},
            mediaDownloadHeaders={"Authorization": "Bearer token"},
            context=context,
            textQueryPrefix="search:",
            hybridParameters=hybrid_parameters,
            facets=facets,
            trackTotalHits=True,
            language="en"
        )

        # Verify key attributes
        self.assertEqual(search_query.searchMethod, SearchMethod.HYBRID)
        self.assertEqual(search_query.limit, 20)
        self.assertEqual(search_query.approximateThreshold, 0.85)
        self.assertIsNotNone(search_query.hybridParameters)
        self.assertIsNotNone(search_query.facets)
        self.assertEqual(search_query.language, "en")

    def test_search_query_required_parameters_only(self):
        """Test SearchQuery with only required parameters."""
        # For tensor search, either q or context is required
        search_query = SearchQuery(
            q="test query",
            searchMethod=SearchMethod.TENSOR
        )

        # Verify defaults
        self.assertEqual(search_query.searchMethod, SearchMethod.TENSOR)
        self.assertEqual(search_query.limit, 10)
        self.assertEqual(search_query.offset, 0)
        self.assertTrue(search_query.showHighlights)
        self.assertIsNone(search_query.hybridParameters)

    def test_hybrid_parameters_validation(self):
        """Test that hybrid parameters are only allowed for hybrid search."""
        hybrid_parameters = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF
        )

        # Should fail for tensor search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                hybridParameters=hybrid_parameters
            )
        self.assertIn("Hybrid parameters can only be provided for 'HYBRID' search", str(cm.exception))

    def test_facets_validation(self):
        """Test that facets are only allowed for hybrid search."""
        facets = FacetsParameters(
            fields={"category": FieldFacetsConfiguration(type="string")}
        )

        # Should fail for tensor search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                facets=facets
            )
        self.assertIn("Facets can only be provided for 'HYBRID' search", str(cm.exception))

    def test_track_total_hits_validation(self):
        """Test that trackTotalHits is only allowed for hybrid search."""
        # Should fail for tensor search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                trackTotalHits=True
            )
        self.assertIn("trackTotalHits can only be provided for 'HYBRID' search", str(cm.exception))

    def test_approximate_threshold_validation(self):
        """Test approximate threshold validation."""
        # Should fail for lexical search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.LEXICAL,
                approximateThreshold=0.5
            )
        self.assertIn("'approximateThreshold' is only valid for 'HYBRID' and 'TENSOR' search methods",
                      str(cm.exception))

        # Should fail when approximate=False
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                approximate=False,
                approximateThreshold=0.5
            )
        self.assertIn("'approximateThreshold' cannot be set when 'approximate' is False", str(cm.exception))

        # Should fail for invalid range
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                approximateThreshold=1.5
            )
        self.assertIn("'approximateThreshold' must be between 0 and 1", str(cm.exception))

    def test_query_and_context_validation(self):
        """Test validation of query and context requirements."""
        # Lexical search requires query
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(searchMethod=SearchMethod.LEXICAL)
        self.assertIn("Query(q) is required for lexical search", str(cm.exception))

        # Tensor search requires either query or context
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(searchMethod=SearchMethod.TENSOR)
        self.assertIn("One of Query(q) or context is required for TENSOR search", str(cm.exception))

    def test_rerank_depth_validation(self):
        """Test rerank depth validation."""
        # Should fail for lexical search
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.LEXICAL,
                rerankDepth=10
            )
        self.assertIn("'rerankDepth' is currently not supported for 'LEXICAL' search method", str(cm.exception))

        # Should fail for negative values
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                rerankDepth=-1
            )
        self.assertIn("rerankDepth cannot be negative", str(cm.exception))

    def test_image_download_headers_validation(self):
        """Test validation of image download headers."""
        # Should fail when both headers are set
        with self.assertRaises(ValidationError) as cm:
            SearchQuery(
                q="test",
                image_download_headers={"header1": "value1"},
                mediaDownloadHeaders={"header2": "value2"}
            )
        self.assertIn("Cannot set both imageDownloadHeaders", str(cm.exception))

        # Should work when imageDownloadHeaders is set and mediaDownloadHeaders is copied
        search_query = SearchQuery(
            q="test",
            image_download_headers={"Authorization": "Bearer token"}
        )
        self.assertEqual(search_query.mediaDownloadHeaders, {"Authorization": "Bearer token"})

    def test_language_field_validation_with_all_search_modes(self):
        """Test language field behavior across all search modes."""
        
        test_cases = [
            {
                "search_method": SearchMethod.TENSOR,
                "language": "en",
                "should_fail": True,
                "expected_error": "language parameter is not supported for TENSOR search method"
            },
            {
                "search_method": SearchMethod.LEXICAL,
                "language": "fr",
                "should_fail": False,
                "expected_error": None
            },
            {
                "search_method": SearchMethod.HYBRID,
                "language": "es",
                "should_fail": False,
                "expected_error": None
            }
        ]
        
        for case in test_cases:
            with self.subTest(search_method=case["search_method"]):
                if case["should_fail"]:
                    with self.assertRaises(ValidationError) as cm:
                        SearchQuery(
                            q="test query",
                            searchMethod=case["search_method"],
                            language=case["language"]
                        )
                    self.assertIn(case["expected_error"], str(cm.exception))
                    self.assertIn("Language specification only applies to lexical and hybrid search", str(cm.exception))
                else:
                    search_query = SearchQuery(
                        q="test query",
                        searchMethod=case["search_method"],
                        language=case["language"]
                    )
                    self.assertEqual(search_query.language, case["language"])
                    self.assertEqual(search_query.searchMethod, case["search_method"])


if __name__ == '__main__':
    unittest.main()
