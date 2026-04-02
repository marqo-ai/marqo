import unittest
from pydantic.v1 import ValidationError

from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod, WeakAndParameters, LexicalOperand
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

    def test_rerank_depth_lexical_validation(self):
        """Test rerankDepthLexical validation with different retrieval methods."""
        # Valid cases: rerankDepthLexical with lexical or disjunction retrieval
        valid_cases = [
            ("lexical_retrieval", RetrievalMethod.Lexical, RankingMethod.Tensor, 100),
            ("disjunction_retrieval", RetrievalMethod.Disjunction, RankingMethod.RRF, 50),
        ]
        for name, retrieval, ranking, depth in valid_cases:
            with self.subTest(name):
                hp = HybridParameters(
                    retrievalMethod=retrieval,
                    rankingMethod=ranking,
                    rerankDepthLexical=depth
                )
                self.assertEqual(hp.rerankDepthLexical, depth)

        # Invalid: rerankDepthLexical with tensor retrieval
        with self.subTest("tensor_retrieval_fails"):
            with self.assertRaises(ValueError) as ctx:
                HybridParameters(
                    retrievalMethod=RetrievalMethod.Tensor,
                    rankingMethod=RankingMethod.Tensor,
                    rerankDepthLexical=100
                )
            self.assertIn("rerankDepthLexical", str(ctx.exception))

        # Invalid: rerankDepthLexical must be >= 1
        with self.subTest("must_be_at_least_1"):
            with self.assertRaises(ValidationError):
                HybridParameters(
                    retrievalMethod=RetrievalMethod.Lexical,
                    rankingMethod=RankingMethod.Tensor,
                    rerankDepthLexical=0
                )

    def test_weak_and_parameters_validation(self):
        """Test weakAndParameters requires rerankDepthLexical to be set."""
        # Valid: weakAndParameters with rerankDepthLexical
        with self.subTest("valid_with_rerank_depth"):
            hp = HybridParameters(
                retrievalMethod=RetrievalMethod.Lexical,
                rankingMethod=RankingMethod.Tensor,
                rerankDepthLexical=100,
                weakAndParameters=WeakAndParameters(stopwordLimit=0.5, adjustTarget=0.3)
            )
            self.assertIsNotNone(hp.weakAndParameters)
            self.assertEqual(hp.weakAndParameters.stopwordLimit, 0.5)

        # Invalid: weakAndParameters without rerankDepthLexical
        with self.subTest("invalid_without_rerank_depth"):
            with self.assertRaises(ValueError) as ctx:
                HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    weakAndParameters=WeakAndParameters(stopwordLimit=0.5)
                )
            self.assertIn("weakAndParameters", str(ctx.exception))
            self.assertIn("rerankDepthLexical", str(ctx.exception))


class TestLexicalOperand(unittest.TestCase):
    """Tests for lexicalOperand parameter in HybridParameters."""

    def test_lexical_operand_valid_values(self):
        """Test that all valid lexicalOperand values are accepted."""
        for operand in ['or', 'and', 'weakAnd']:
            with self.subTest(operand=operand):
                hp = HybridParameters(lexicalOperand=operand)
                self.assertEqual(hp.lexicalOperand, operand)

    def test_lexical_operand_none_by_default(self):
        """Test that lexicalOperand defaults to None."""
        hp = HybridParameters()
        self.assertIsNone(hp.lexicalOperand)

    def test_lexical_operand_invalid_value(self):
        """Test that invalid lexicalOperand values are rejected."""
        with self.assertRaises(ValidationError):
            HybridParameters(lexicalOperand="invalid")

    def test_lexical_operand_works_with_any_retrieval_method(self):
        """Test that lexicalOperand works with different retrieval/ranking combos."""
        test_cases = [
            ("disjunction_rrf", RetrievalMethod.Disjunction, RankingMethod.RRF),
            ("lexical_lexical", RetrievalMethod.Lexical, RankingMethod.Lexical),
            ("lexical_tensor", RetrievalMethod.Lexical, RankingMethod.Tensor),
        ]
        for name, retrieval, ranking in test_cases:
            with self.subTest(name):
                hp = HybridParameters(
                    retrievalMethod=retrieval,
                    rankingMethod=ranking,
                    lexicalOperand='or'
                )
                self.assertEqual(hp.lexicalOperand, 'or')


class TestWeakAndParameters(unittest.TestCase):
    """Tests for WeakAndParameters model."""

    def test_weak_and_parameters_creation(self):
        """Test WeakAndParameters creation with various field combinations."""
        test_cases = [
            ("all_fields", {"stopwordLimit": 0.5, "adjustTarget": 0.3, "allowDropAll": True, "filterThreshold": 0.1},
             {"stopwordLimit": 0.5, "adjustTarget": 0.3, "allowDropAll": True, "filterThreshold": 0.1}),
            ("no_fields", {},
             {"stopwordLimit": None, "adjustTarget": None, "allowDropAll": None, "filterThreshold": None}),
            ("partial_fields", {"stopwordLimit": 0.5, "allowDropAll": False},
             {"stopwordLimit": 0.5, "adjustTarget": None, "allowDropAll": False, "filterThreshold": None}),
        ]

        for name, input_params, expected in test_cases:
            with self.subTest(name):
                params = WeakAndParameters(**input_params)
                for field, value in expected.items():
                    self.assertEqual(getattr(params, field), value)

    def test_weak_and_parameters_field_ranges(self):
        """Test field range validation (0 to 1) for stopwordLimit, adjustTarget, filterThreshold."""
        fields = ["stopwordLimit", "adjustTarget", "filterThreshold"]

        for field in fields:
            # Valid boundary values
            for valid_value in [0, 0.5, 1]:
                with self.subTest(field=field, value=valid_value, expected="valid"):
                    WeakAndParameters(**{field: valid_value})  # Should not raise

            # Invalid values
            for invalid_value in [-0.1, 1.1]:
                with self.subTest(field=field, value=invalid_value, expected="invalid"):
                    with self.assertRaises(ValidationError):
                        WeakAndParameters(**{field: invalid_value})

    def test_convert_to_vespa_query_dict(self):
        """Test convert_to_vespa_query_dict with various field combinations."""
        test_cases = [
            ("all_fields",
             {"stopwordLimit": 0.5, "adjustTarget": 0.3, "allowDropAll": True, "filterThreshold": 0.1},
             {"ranking.matching.weakand.stopwordLimit": 0.5, "ranking.matching.weakand.adjustTarget": 0.3,
              "ranking.matching.weakand.allowDropAll": True, "ranking.matching.filterThreshold": 0.1}),
            ("partial_fields",
             {"stopwordLimit": 0.5, "allowDropAll": False},
             {"ranking.matching.weakand.stopwordLimit": 0.5, "ranking.matching.weakand.allowDropAll": False}),
            ("empty", {}, {}),
        ]

        for name, input_params, expected in test_cases:
            with self.subTest(name):
                params = WeakAndParameters(**input_params)
                result = params.convert_to_vespa_query_dict()
                self.assertEqual(result, expected)


class TestCollapseModel(unittest.TestCase):
    """Tests for CollapseModel and CollapseSortBy validation logic."""

    def test_always_fetch_variants_true(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        model = CollapseModel(
            name="parent_id",
            sort_by=CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price")],
                alwaysFetchVariants=True,
            )
        )
        self.assertTrue(model.sort_by.always_fetch_variants)

    def test_always_fetch_variants_default_false(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        model = CollapseModel(
            name="parent_id",
            sort_by=CollapseSortBy(fields=[CollapseSortByField(fieldName="price")])
        )
        self.assertFalse(model.sort_by.always_fetch_variants)

    def test_no_sort_by_valid(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel
        model = CollapseModel(name="parent_id")
        self.assertIsNone(model.sort_by)

    def test_always_fetch_variants_false_with_sort_by(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        model = CollapseModel(
            name="parent_id",
            sort_by=CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price")],
                alwaysFetchVariants=False,
            )
        )
        self.assertFalse(model.sort_by.always_fetch_variants)

    def test_num_threads_per_search(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        model = CollapseModel(
            name="parent_id",
            sort_by=CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price")],
                numThreadsPerSearch=4,
            )
        )
        self.assertEqual(4, model.sort_by.num_threads_per_search)

    def test_disable_if_main_sort_by_fields(self):
        from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField
        model = CollapseModel(
            name="parent_id",
            sort_by=CollapseSortBy(
                fields=[CollapseSortByField(fieldName="price")],
                disableIfMainSortByFields=["price", "date"],
            )
        )
        self.assertEqual({"price", "date"}, model.sort_by.disable_if_main_sort_by_fields)

    def test_sort_by_requires_fields(self):
        from marqo.tensor_search.models.collapse_model import CollapseSortBy
        with self.assertRaises(ValidationError):
            CollapseSortBy(alwaysFetchVariants=True)


if __name__ == "__main__":
    unittest.main()

