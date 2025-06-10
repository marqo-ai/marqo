import random
from unittest import TestCase

from unittest.mock import MagicMock

from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoHybridQuery
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.tensor_search.models.sort_by_model import SortByModel
from marqo.version import get_version


class TestSemiStructuredIndexToVespaQuerySortBy(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.marqo_index = MagicMock(spec=SemiStructuredMarqoIndex)
        cls.marqo_index.parsed_marqo_version.return_value = get_version()
        cls.marqo_index.schema_name = "test_sort_by_index"
        cls.index = SemiStructuredVespaIndex(cls.marqo_index)
        cls.hybrid_query = MarqoHybridQuery(
            index_name = "test_index",
            vector_query = None,
            filter=None,
            limit=10,
            offset=0,
            attributes_to_retrieve=["title", "description", "price"],
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF
            ),
            score_modifiers_lexical=None,
            score_modifiers_tensor=None,
            or_phrases=[],
            and_phrases=[],
            sort_by=None
        )

    def test_sort_by_multiple_fields_desc(self):
        """Test sorting by two fields with descending and ascending orders."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "price", "order": "desc"},
                {"field_name": "rating", "order": "asc"}
            ],
            sortDepth=3,
            minSortCandidates=50
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']

        self.assertEqual(2, len(sort_fields))
        self.assertEqual("price", sort_fields[0]["field_name"])
        self.assertEqual("desc", sort_fields[0]["order"].value)
        self.assertEqual("rating", sort_fields[1]["field_name"])
        self.assertEqual("asc", sort_fields[1]["order"].value)
        self.assertEqual(3, r['marqo__hybrid.sortBy.sortDepth'])
        self.assertEqual(50, r['marqo__hybrid.sortBy.minSortCandidates'])

    def test_sort_by_single_field_no_optional(self):
        """Test sorting by a single field with no optional params."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "title", "order": "asc"}
            ],
            minSortCandidates=30
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']
        sort_depth = r["marqo__hybrid.sortBy.sortDepth"]
        min_sort_candidates = r["marqo__hybrid.sortBy.minSortCandidates"]

        self.assertEqual(1, len(sort_fields))
        self.assertEqual("title", sort_fields[0]["field_name"])
        self.assertEqual("asc", sort_fields[0]["order"].value)
        self.assertEqual(None, sort_depth)
        self.assertEqual(30, min_sort_candidates)

    def test_sort_by_with_missing_first(self):
        """Test a field with missing='first' and all optional params."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "description", "order": "asc", "missing": "first"}
            ],
            sortDepth=2,
            minSortCandidates=20
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']

        self.assertEqual(1, len(sort_fields))
        self.assertEqual("description", sort_fields[0]["field_name"])
        self.assertEqual("asc", sort_fields[0]["order"].value)
        self.assertEqual("first", sort_fields[0]["missing"].value)
        self.assertEqual(2, r['marqo__hybrid.sortBy.sortDepth'])
        self.assertEqual(20, r['marqo__hybrid.sortBy.minSortCandidates'])

    def test_sort_by_none(self):
        """Test that no sort_by results in no sort fields present."""
        self.hybrid_query.sort_by = None
        r = self.index._to_vespa_hybrid_query(self.hybrid_query)

        self.assertNotIn("marqo__hybrid.sortBy.fields", r)
        self.assertNotIn("marqo__hybrid.sortBy.sortDepth", r)
        self.assertNotIn("marqo__hybrid.sortBy.minSortCandidates", r)

    def test_sort_by_three_fields_mixed_order_and_missing(self):
        """Test three fields with mixed order and missing policies."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "price", "order": "desc", "missing": "last"},
                {"field_name": "rating", "order": "asc"},
                {"field_name": "stock", "order": "desc", "missing": "first"}
            ],
            sortDepth=4,
            minSortCandidates=100
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        fields = r["marqo__hybrid.sortBy.fields"]

        self.assertEqual(3, len(fields))
        self.assertEqual("price", fields[0]["field_name"])
        self.assertEqual("desc", fields[0]["order"].value)
        self.assertEqual("last", fields[0]["missing"].value)

        self.assertEqual("rating", fields[1]["field_name"])
        self.assertEqual("asc", fields[1]["order"].value)
        self.assertEqual("last", fields[1]["missing"].value)  # Default missing policy

        self.assertEqual("stock", fields[2]["field_name"])
        self.assertEqual("desc", fields[2]["order"].value)
        self.assertEqual("first", fields[2]["missing"].value)

        self.assertEqual(4, r["marqo__hybrid.sortBy.sortDepth"])
        self.assertEqual(100, r["marqo__hybrid.sortBy.minSortCandidates"])

    def test_query_features_sort_field_weights_3_fields(self):
        """A fuzzy test to ensure that query_features are correctly populated with sort field weights."""
        test_fields = [
            {"field_name": "alpha", "order": "asc"},
            {"field_name": "beta", "order": "desc"},
            {"field_name": "gamma", "order": "asc"}
        ]
        for _ in range(20):
            random.shuffle(test_fields)
            self.hybrid_query.sort_by = SortByModel(
                fields=test_fields
            )

            r = self.index._to_vespa_hybrid_query(self.hybrid_query)
            query_features = r["query_features"]
            for i, field in enumerate(self.hybrid_query.sort_by.fields):
                field_name = field.field_name
                self.assertIn(f"marqo__sort_field_weights_{i}", query_features)
                self.assertIn(field_name, query_features[f"marqo__sort_field_weights_{i}"])
                self.assertEqual(1, query_features[f"marqo__sort_field_weights_{i}"][field_name])

    def test_query_features_sort_field_weights_2_fields(self):
        """A fuzzy test to ensure that query_features are correctly populated with sort field weights."""
        test_fields = [
            {"field_name": "alpha", "order": "asc"},
            {"field_name": "beta", "order": "desc"},
        ]
        for _ in range(20):
            random.shuffle(test_fields)
            self.hybrid_query.sort_by = SortByModel(
                fields=test_fields
            )

            r = self.index._to_vespa_hybrid_query(self.hybrid_query)
            query_features = r["query_features"]
            for i, field in enumerate(self.hybrid_query.sort_by.fields):
                field_name = field.field_name
                self.assertIn(f"marqo__sort_field_weights_{i}", query_features)
                self.assertIn(field_name, query_features[f"marqo__sort_field_weights_{i}"])
                self.assertEqual(1, query_features[f"marqo__sort_field_weights_{i}"][field_name])
            self.assertEqual({}, query_features[f"marqo__sort_field_weights_{2}"])

    def test_query_features_sort_field_weights_1_field(self):
        """A fuzzy test to ensure that query_features are correctly populated with sort field weights."""
        test_fields = [
            {"field_name": "alpha", "order": "asc"},
        ]

        self.hybrid_query.sort_by = SortByModel(
            fields=test_fields
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        query_features = r["query_features"]

        self.assertEqual({"alpha": 1}, query_features[f"marqo__sort_field_weights_{0}"])
        self.assertEqual({}, query_features[f"marqo__sort_field_weights_{1}"])
        self.assertEqual({}, query_features[f"marqo__sort_field_weights_{2}"])

    def test_query_features_sort_field_weights_zero_fields(self):
        """A fuzzy test to ensure that query_features are correctly populated with sort field weights."""
        self.hybrid_query.sort_by = None

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        query_features = r["query_features"]

        for i in range(3):
            self.assertNotIn(f"marqo__sort_field_weights_{i}", query_features)