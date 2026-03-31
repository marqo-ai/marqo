from unittest import TestCase

from pydantic.v1 import ValidationError

from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.models.relevance_cutoff_model import RelevanceCutoffModel, RelevanceCutoffMethod, \
    RelativeMaxScoreParameters
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists, ScoreModifierOperator
from marqo.tensor_search.models.sort_by_model import SortByModel, SortByField, SortOrder, SortMissingPolicy


class TestSortByModels(TestCase):
    def test_valid_sort_by_model_single_field(self):
        sort_by_model = SortByModel(fields=[
            SortByField(fieldName="price", order=SortOrder.Asc, missing=SortMissingPolicy.First)
        ])
        self.assertEqual(len(sort_by_model.fields), 1)
        self.assertEqual(sort_by_model.fields[0].field_name, "price")
        self.assertEqual(sort_by_model.fields[0].order, SortOrder.Asc)
        self.assertEqual(sort_by_model.fields[0].missing, SortMissingPolicy.First)

    def test_valid_sort_by_model_multiple_fields(self):
        sort_by_model = SortByModel(fields=[
            SortByField(fieldName="rating"),
            SortByField(fieldName="price", order=SortOrder.Asc)
        ])
        self.assertEqual(len(sort_by_model.fields), 2)
        self.assertEqual(sort_by_model.fields[0].field_name, "rating")
        self.assertEqual(sort_by_model.fields[0].order, SortOrder.Desc)  # default
        self.assertEqual(sort_by_model.fields[1].order, SortOrder.Asc)

    def test_sort_by_with_sort_depth_and_sort_candidates(self):
        sort_by_model = SortByModel(
            fields=[SortByField(fieldName="price")],
            sortDepth=5,
            min_sort_candidates=15
        )
        self.assertEqual(sort_by_model.sort_depth, 5)
        self.assertEqual(sort_by_model.min_sort_candidates, 15)

    def test_sort_by_invalid_in_tensor_search(self):
        with self.assertRaises(ValueError) as e:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.TENSOR,
                sortBy=SortByModel(fields=[SortByField(fieldName="price")])
            )
        self.assertIn("sortBy can only be provided for 'HYBRID' search", str(e.exception))

    def test_sort_by_invalid_with_score_modifiers(self):
        with self.assertRaises(ValidationError) as e:
            SearchQuery(
                q="test",
                searchMethod=SearchMethod.HYBRID,
                sortBy=SortByModel(fields=[SortByField(fieldName="price")]),
                scoreModifiers=ScoreModifierLists(multiply_score_by=
                                                  [ScoreModifierOperator(field_name="popularity", weight=2)])
            )
        self.assertIn("'sortBy' cannot be used with 'scoreModifiers'", str(e.exception))

    def test_sort_by_sort_candidates_defaulting(self):
        query = SearchQuery(
            q="test",
            searchMethod=SearchMethod.HYBRID,
            sortBy=SortByModel(fields=[SortByField(fieldName="price")]),
            limit=10,
            offset=5
        )
        # 3 * limit = 30 vs offset + limit = 15 -> max is 30
        self.assertEqual(query.sort_by.min_sort_candidates, 30)

    def test_sort_by_rejected_in_lexical_search(self):
        with self.assertRaises(ValueError) as e:
            SearchQuery(
                q="keyword",
                searchMethod=SearchMethod.LEXICAL,
                sortBy=SortByModel(fields=[SortByField(fieldName="date")])
            )
        self.assertIn("sortBy can only be provided for 'HYBRID' search", str(e.exception))

    def test_sort_by_with_only_required_fields(self):
        sort_by_model = SortByModel(fields=[SortByField(fieldName="score")])
        self.assertEqual(sort_by_model.fields[0].field_name, "score")
        self.assertEqual(sort_by_model.fields[0].order, SortOrder.Desc)  # default
        self.assertEqual(sort_by_model.fields[0].missing, SortMissingPolicy.Last)  # default

    def test_sort_by_with_empty_fields_should_raise(self):
        with self.assertRaises(ValidationError) as e:
            SortByModel(fields=[])
        self.assertIn("ensure this value has at least 1 items", str(e.exception))

    def test_sort_by_with_three_fields(self):
        sort_by_model = SortByModel(fields=[
            SortByField(fieldName="rating"),
            SortByField(fieldName="price"),
            SortByField(fieldName="popularity")
        ])
        self.assertEqual(len(sort_by_model.fields), 3)

    def test_sort_by_with_more_than_three_fields_should_raise(self):
        with self.assertRaises(ValidationError) as e:
            SortByModel(fields=[
                SortByField(fieldName="rating"),
                SortByField(fieldName="price"),
                SortByField(fieldName="popularity"),
                SortByField(fieldName="updated_at")
            ])
        self.assertIn("at most 3 items", str(e.exception))

    def test_sort_by_sort_candidates_defaulting_offset_greater_than_triple_limit(self):
        query = SearchQuery(
            q="test",
            searchMethod=SearchMethod.HYBRID,
            sortBy=SortByModel(fields=[SortByField(fieldName="views")]),
            limit=10,
            offset=50
        )
        self.assertEqual(query.sort_by.min_sort_candidates, 60)  # max(3*10, 10+50)

    def test_sort_by_explicit_min_sort_candidates_preserved_greater_than_offset_plus_limit_and_three_times_limit(self):
        """Ensure that if minSortCandidates is explicitly set, it is preserved, when
        it's larger than offset + limit and greater than 3 * limit."""
        query = SearchQuery(
            q="explicit test",
            searchMethod=SearchMethod.HYBRID,
            sortBy=SortByModel(fields=[SortByField(fieldName="value")], min_sort_candidates=77),
            limit=5,
            offset=2
        )
        self.assertEqual(query.sort_by.min_sort_candidates, 77)

    def test_sort_by_explicit_min_sort_candidates_preserved_greater_than_offset_plus_limit_smaller_than_three_times_limit(self):
        """Ensure that if minSortCandidates is explicitly set, it is preserved, when
        it's larger than offset + limit but smaller than 3 * limit."""
        query = SearchQuery(
            q="explicit test",
            searchMethod=SearchMethod.HYBRID,
            sortBy=SortByModel(fields=[SortByField(fieldName="value")], min_sort_candidates=11),
            limit=5,
            offset=2
        )
        self.assertEqual(query.sort_by.min_sort_candidates, 11)

    def test_sort_by_explicit_sort_candidates_adjusted_to_offset_plus_limit(self):
        """Ensure minSortCandidates is adjusted to offset + limit when it's smaller."""
        query = SearchQuery(
            q="explicit test",
            searchMethod=SearchMethod.HYBRID,
            sortBy=SortByModel(fields=[SortByField(fieldName="value")], min_sort_candidates=77),
            limit=50,
            offset=29
        )
        self.assertEqual(79, query.sort_by.min_sort_candidates)

    def test_sort_by_sort_candidates_with_default_limit_and_offset(self):
        query = SearchQuery(
            q="default",
            searchMethod=SearchMethod.HYBRID,
            sortBy=SortByModel(fields=[SortByField(fieldName="value")]),
        )
        self.assertEqual(30, query.sort_by.min_sort_candidates)

    def test_sort_by_sort_candidates_can_be_none_if_relevance_cutoff_is_provided(self):
        query = SearchQuery(
            q="default",
            searchMethod=SearchMethod.HYBRID,
            sortBy=SortByModel(fields=[SortByField(fieldName="value")]),
            relevanceCutoff=RelevanceCutoffModel(
                method=RelevanceCutoffMethod.RelativeMaxScore,
                parameters=RelativeMaxScoreParameters(relativeScoreFactor=0.9)
            )
        )

        self.assertEqual(None, query.sort_by.min_sort_candidates)

    def test_sort_by_invalid_sort_candidates_below_one(self):
        with self.assertRaises(ValidationError) as e:
            SortByModel(fields=[SortByField(fieldName="foo")], min_sort_candidates=0)
        self.assertIn("greater than or equal to 1", str(e.exception))

    def test_sort_by_invalid_sort_depth_below_one(self):
        with self.assertRaises(ValidationError) as e:
            SortByModel(fields=[SortByField(fieldName="bar")], sortDepth=0)
        self.assertIn("greater than or equal to 1", str(e.exception))

    def test_sort_by_invalid_field_name_should_raise(self):
        with self.assertRaises(ValidationError) as e:
            SortByField(fieldName="test/invalid")
        self.assertIn("Field name must match [a-zA-Z_][a-zA-Z0-9_]*", str(e.exception))