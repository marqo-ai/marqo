import unittest
from pydantic.v1 import ValidationError

from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import SearchQuery, CustomVectorQuery
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration, RangeConfiguration


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
        sq = SearchQuery(searchMethod="HYBRID", hybridParameters=HybridParameters(
            queryTensor={"dogs": 1.0},
            retrievalMethod=RetrievalMethod.Tensor,
            rankingMethod=RankingMethod.Tensor
        ))
        self.assertEqual(sq.searchMethod, SearchMethod.HYBRID)

    def test_hybrid_query_with_only_queryLexical(self):
        sq = SearchQuery(searchMethod="HYBRID", hybridParameters=HybridParameters(
            queryLexical="dogs",
            retrievalMethod=RetrievalMethod.Lexical,
            rankingMethod=RankingMethod.Lexical
        ))
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

    def test_facets_only_allowed_for_hybrid(self):
        with self.assertRaises(ValueError):
            SearchQuery(
                q="test", searchMethod="TENSOR",
                facets=FacetsParameters(fields={
                    "price": FieldFacetsConfiguration(type="number")
                })
            )

    def test_facets_valid_for_hybrid(self):
        sq = SearchQuery(
            q="test", searchMethod="HYBRID",
            facets=FacetsParameters(fields={
                "price": FieldFacetsConfiguration(type="number")
            }),
            hybridParameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical
            )
        )
        self.assertIsNotNone(sq.facets)

    def test_facets_exclude_terms_without_filter_fails(self):
        with self.assertRaises(ValueError):
            SearchQuery(
                q="test", searchMethod="HYBRID",
                facets=FacetsParameters(fields={
                    "category": FieldFacetsConfiguration(
                        type="string",
                        excludeTerms=["electronics"]
                    )
                }),
                hybridParameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical,
                    rankingMethod=RankingMethod.Lexical
                )
            )

    def test_facets_exclude_terms_with_matching_filter(self):
        sq = SearchQuery(
            q="test", searchMethod="HYBRID",
            facets=FacetsParameters(fields={
                "category": FieldFacetsConfiguration(
                    type="string",
                    excludeTerms=["category:electronics"]
                )
            }),
            filter="category:electronics AND price:>100",
            hybridParameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Lexical
            )
        )
        self.assertIsNotNone(sq.facets)

    def test_facets_exclude_terms_with_non_matching_filter_fails(self):
        with self.assertRaises(ValueError):
            SearchQuery(
                q="test", searchMethod="HYBRID",
                facets=FacetsParameters(fields={
                    "category": FieldFacetsConfiguration(
                        type="string",
                        excludeTerms=["electronics", "books"]
                    )
                }),
                filter="category:electronics AND price:>100",
                hybridParameters=HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical,
                    rankingMethod=RankingMethod.Lexical
                )
            )

class TestRangeConfiguration(unittest.TestCase):
    def test_valid_range(self):
        RangeConfiguration.validate({"from": 0, "to": 10})

    def test_valid_range_with_name(self):
        RangeConfiguration.validate({"from": 0,"to": 10, "name": "test_range"})

    def test_range_same_value_fails(self):
        with self.assertRaises(ValueError):
            RangeConfiguration.validate({"from": 10, "to": 10})

    def test_invalid_range_values(self):
        with self.assertRaises(ValueError):
            RangeConfiguration.validate({"from": 10, "to": 5})

    def test_partial_range(self):
        RangeConfiguration.validate({"from": 0})
        RangeConfiguration.validate({"to": 10})

class TestFieldFacetsConfiguration(unittest.TestCase):
    def test_valid_string_type(self):
        fc = FieldFacetsConfiguration(type="string")
        self.assertEqual(fc.type, "string")

    def test_valid_array_type(self):
        fc = FieldFacetsConfiguration(type="array")
        self.assertEqual(fc.type, "array")

    def test_valid_number_type(self):
        fc = FieldFacetsConfiguration(type="number")
        self.assertEqual(fc.type, "number")

    def test_invalid_type(self):
        with self.assertRaises(ValidationError):
            FieldFacetsConfiguration(type="invalid")

    def test_valid_max_results(self):
        fc = FieldFacetsConfiguration(type="string", maxResults=100)
        self.assertEqual(fc.max_results, 100)

    def test_invalid_max_results_zero(self):
        with self.assertRaises(ValueError):
            FieldFacetsConfiguration(type="string", maxResults=0)

    def test_invalid_max_results_negative(self):
        with self.assertRaises(ValueError):
            FieldFacetsConfiguration(type="string", maxResults=-1)

    def test_invalid_max_results_too_large(self):
        with self.assertRaises(ValueError):
            FieldFacetsConfiguration(type="string", maxResults=10001)

    def test_ranges_only_for_number_type(self):
        with self.assertRaises(ValueError):
            FieldFacetsConfiguration(
                type="string",
                ranges=[{"from": 0, "to": 10}]
            )

    def test_valid_ranges_for_number_type(self):
        fc = FieldFacetsConfiguration(
            type="number",
            ranges=[
                {"from": 0, "to": 10},
                {"from": 10, "to": 20}
            ]
        )
        self.assertEqual(len(fc.ranges), 2)

    def test_overlapping_ranges(self):
        with self.assertRaises(ValueError):
            FieldFacetsConfiguration(
                type="number",
                ranges=[
                    {"from": 0, "to": 15},
                    {"from": 10, "to": 20}
                ]
            )

    def test_ranges_overlapping_with_to_none(self):
        with self.assertRaises(ValueError):
            FieldFacetsConfiguration(
                type="number",
                ranges=[
                    {"from": 0, "to": None},
                    {"from": 10, "to": 20}
                ]
            )

    def test_ranges_overlapping_with_from_none(self):
        with self.assertRaises(ValueError):
            FieldFacetsConfiguration(
                type="number",
                ranges=[
                    {"from": None, "to": 10},
                    {"from": 5, "to": 20}
                ]
            )

class TestFacetsParameters(unittest.TestCase):
    def test_valid_facets_parameters(self):
        fp = FacetsParameters(
            fields={
                "price": FieldFacetsConfiguration(type="number"),
                "category": FieldFacetsConfiguration(type="string")
            }
        )
        self.assertEqual(len(fp.fields), 2)

    def test_valid_max_depth(self):
        fp = FacetsParameters(
            fields={"category": FieldFacetsConfiguration(type="string")},
            maxDepth=5
        )
        self.assertEqual(fp.max_depth, 5)

    def test_invalid_max_depth(self):
        with self.assertRaises(ValueError):
            FacetsParameters(
                fields={"category": FieldFacetsConfiguration(type="string")},
                maxDepth=0
            )

    def test_valid_max_results(self):
        fp = FacetsParameters(
            fields={"category": FieldFacetsConfiguration(type="string")},
            maxResults=100
        )
        self.assertEqual(fp.max_results, 100)

    def test_invalid_max_results(self):
        with self.assertRaises(ValueError):
            FacetsParameters(
                fields={"category": FieldFacetsConfiguration(type="string")},
                maxResults=0
            )

    def test_valid_order(self):
        fp = FacetsParameters(
            fields={"category": FieldFacetsConfiguration(type="string")},
            order="asc"
        )
        self.assertEqual(fp.order, "asc")

    def test_invalid_order(self):
        with self.assertRaises(ValidationError):
            FacetsParameters(
                fields={"category": FieldFacetsConfiguration(type="string")},
                order="invalid"
            )

class TestHybridParametersValidation(unittest.TestCase):
    """Tests for HybridParameters validation logic."""

    def test_rerank_depth_lexical_valid_with_lexical_retrieval(self):
        """Test rerankDepthLexical is valid with lexical retrieval method."""
        hp = HybridParameters(
            retrievalMethod=RetrievalMethod.Lexical,
            rankingMethod=RankingMethod.Tensor,
            rerankDepthLexical=100
        )
        self.assertEqual(hp.rerankDepthLexical, 100)

    def test_rerank_depth_lexical_valid_with_disjunction_retrieval(self):
        """Test rerankDepthLexical is valid with disjunction retrieval method."""
        hp = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            rerankDepthLexical=50
        )
        self.assertEqual(hp.rerankDepthLexical, 50)

    def test_rerank_depth_lexical_invalid_with_tensor_retrieval(self):
        """Test rerankDepthLexical fails with tensor retrieval method."""
        with self.assertRaises(ValueError) as ctx:
            HybridParameters(
                retrievalMethod=RetrievalMethod.Tensor,
                rankingMethod=RankingMethod.Tensor,
                rerankDepthLexical=100
            )
        self.assertIn("rerankDepthLexical", str(ctx.exception))
        self.assertIn("lexical", str(ctx.exception).lower())

    def test_rerank_depth_lexical_must_be_at_least_1(self):
        """Test rerankDepthLexical must be >= 1."""
        with self.assertRaises(ValidationError):
            HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Tensor,
                rerankDepthLexical=0
            )

    def test_weak_and_parameters_valid_with_rerank_depth_lexical(self):
        """Test weakAndParameters is valid when rerankDepthLexical is set."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        hp = HybridParameters(
            retrievalMethod=RetrievalMethod.Lexical,
            rankingMethod=RankingMethod.Tensor,
            rerankDepthLexical=100,
            weakAndParameters=WeakAndParameters(
                stopwordLimit=0.5,
                adjustTarget=0.3
            )
        )
        self.assertIsNotNone(hp.weakAndParameters)
        self.assertEqual(hp.weakAndParameters.stopwordLimit, 0.5)

    def test_weak_and_parameters_invalid_without_rerank_depth_lexical(self):
        """Test weakAndParameters fails when rerankDepthLexical is not set."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        with self.assertRaises(ValueError) as ctx:
            HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                weakAndParameters=WeakAndParameters(
                    stopwordLimit=0.5
                )
            )
        self.assertIn("weakAndParameters", str(ctx.exception))
        self.assertIn("rerankDepthLexical", str(ctx.exception))


class TestWeakAndParameters(unittest.TestCase):
    """Tests for WeakAndParameters model."""

    def test_valid_weak_and_parameters(self):
        """Test valid WeakAndParameters with all fields."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        params = WeakAndParameters(
            stopwordLimit=0.5,
            adjustTarget=0.3,
            allowDropAll=True,
            filterThreshold=0.1
        )
        self.assertEqual(params.stopwordLimit, 0.5)
        self.assertEqual(params.adjustTarget, 0.3)
        self.assertTrue(params.allowDropAll)
        self.assertEqual(params.filterThreshold, 0.1)

    def test_weak_and_parameters_optional_fields(self):
        """Test WeakAndParameters with no fields (all optional)."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        params = WeakAndParameters()
        self.assertIsNone(params.stopwordLimit)
        self.assertIsNone(params.adjustTarget)
        self.assertIsNone(params.allowDropAll)
        self.assertIsNone(params.filterThreshold)

    def test_weak_and_parameters_stopword_limit_range(self):
        """Test stopwordLimit must be between 0 and 1."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        # Valid boundary values
        WeakAndParameters(stopwordLimit=0)
        WeakAndParameters(stopwordLimit=1)
        WeakAndParameters(stopwordLimit=0.5)

        # Invalid values
        with self.assertRaises(ValidationError):
            WeakAndParameters(stopwordLimit=-0.1)
        with self.assertRaises(ValidationError):
            WeakAndParameters(stopwordLimit=1.1)

    def test_weak_and_parameters_adjust_target_range(self):
        """Test adjustTarget must be between 0 and 1."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        # Valid boundary values
        WeakAndParameters(adjustTarget=0)
        WeakAndParameters(adjustTarget=1)

        # Invalid values
        with self.assertRaises(ValidationError):
            WeakAndParameters(adjustTarget=-0.1)
        with self.assertRaises(ValidationError):
            WeakAndParameters(adjustTarget=1.1)

    def test_weak_and_parameters_filter_threshold_range(self):
        """Test filterThreshold must be between 0 and 1."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        # Valid boundary values
        WeakAndParameters(filterThreshold=0)
        WeakAndParameters(filterThreshold=1)

        # Invalid values
        with self.assertRaises(ValidationError):
            WeakAndParameters(filterThreshold=-0.1)
        with self.assertRaises(ValidationError):
            WeakAndParameters(filterThreshold=1.1)

    def test_convert_to_vespa_query_dict_all_fields(self):
        """Test convert_to_vespa_query_dict with all fields set."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        params = WeakAndParameters(
            stopwordLimit=0.5,
            adjustTarget=0.3,
            allowDropAll=True,
            filterThreshold=0.1
        )
        result = params.convert_to_vespa_query_dict()
        self.assertEqual(result["ranking.matching.weakand.stopwordLimit"], 0.5)
        self.assertEqual(result["ranking.matching.weakand.adjustTarget"], 0.3)
        self.assertTrue(result["ranking.matching.weakand.allowDropAll"])
        self.assertEqual(result["ranking.matching.filterThreshold"], 0.1)

    def test_convert_to_vespa_query_dict_partial_fields(self):
        """Test convert_to_vespa_query_dict excludes None fields."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        params = WeakAndParameters(
            stopwordLimit=0.5,
            allowDropAll=False
        )
        result = params.convert_to_vespa_query_dict()
        self.assertEqual(result["ranking.matching.weakand.stopwordLimit"], 0.5)
        self.assertFalse(result["ranking.matching.weakand.allowDropAll"])
        self.assertNotIn("ranking.matching.weakand.adjustTarget", result)
        self.assertNotIn("ranking.matching.filterThreshold", result)

    def test_convert_to_vespa_query_dict_empty(self):
        """Test convert_to_vespa_query_dict with no fields returns empty dict."""
        from marqo.core.models.hybrid_parameters import WeakAndParameters
        params = WeakAndParameters()
        result = params.convert_to_vespa_query_dict()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()

