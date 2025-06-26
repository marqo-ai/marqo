from unittest import TestCase
from pydantic.v1 import ValidationError

from marqo.core.models.marqo_query import (
    MarqoTensorQuery, MarqoQuery, MarqoHybridQuery
)
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.core.search.search_filter import SearchFilter, EqualityTerm
from marqo.core.models.hybrid_parameters import (
    HybridParameters, RankingMethod, RetrievalMethod
)
from marqo.core.models.facets_parameters import (
    FacetsParameters, FieldFacetsConfiguration
)


class TestMarqoTensorQuery(TestCase):

    def test_creation_with_all_values(self):
        """Test creating MarqoTensorQuery with all possible values."""
        score_modifier = ScoreModifier(
            field="test_field",
            weight=1.5,
            type=ScoreModifierType.Multiply
        )
        
        filter_obj = SearchFilter(
            EqualityTerm("field1", "value1", "field1:value1")
        )
        
        query = MarqoTensorQuery(
            index_name="test_index",
            limit=20,
            offset=5,
            searchable_attributes=["field1", "field2"],
            attributes_to_retrieve=["field1", "field3"],
            filter=filter_obj,
            score_modifiers=[score_modifier],
            expose_facets=True,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            ef_search=100,
            approximate=False,
            approximate_threshold=0.95,
            rerank_depth_tensor=50
        )
        
        # Verify all fields are set correctly
        self.assertEqual("test_index", query.index_name)
        self.assertEqual(20, query.limit)
        self.assertEqual(5, query.offset)
        self.assertEqual(["field1", "field2"], query.searchable_attributes)
        self.assertEqual(["field1", "field3"], query.attributes_to_retrieve)
        self.assertEqual(filter_obj, query.filter)
        self.assertEqual([score_modifier], query.score_modifiers)
        self.assertTrue(query.expose_facets)
        self.assertEqual([0.1, 0.2, 0.3, 0.4], query.vector_query)
        self.assertEqual(100, query.ef_search)
        self.assertFalse(query.approximate)
        self.assertEqual(0.95, query.approximate_threshold)
        self.assertEqual(50, query.rerank_depth_tensor)
        
        # Test inheritance
        self.assertIsInstance(query, MarqoQuery)
        self.assertIsInstance(query, MarqoTensorQuery)

    def test_required_fields(self):
        """Test that all required fields must be provided."""
        base_params = {
            "index_name": "test_index",
            "limit": 10,
            "vector_query": [0.1, 0.2, 0.3]
        }
        
        required_fields = ["index_name", "limit", "vector_query"]
        
        for required_field in required_fields:
            with self.subTest(missing_field=required_field):
                params = base_params.copy()
                del params[required_field]
                
                with self.assertRaises(ValidationError) as context:
                    MarqoTensorQuery(**params)
                
                self.assertIn(required_field, str(context.exception))


class TestMarqoHybridQuery(TestCase):

    def test_creation_with_all_values(self):
        """Test creating MarqoHybridQuery with all possible values."""
        score_modifier = ScoreModifier(
            field="test_field",
            weight=1.5,
            type=ScoreModifierType.Multiply
        )
        
        filter_obj = SearchFilter(
            EqualityTerm("field1", "value1", "field1:value1")
        )
        
        hybrid_parameters = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.7,
            rrfK=100
        )
        
        facets = FacetsParameters(
            fields={
                "test_field": FieldFacetsConfiguration(type="string")
            },
            maxDepth=5,
            maxResults=100
        )
        
        query = MarqoHybridQuery(
            index_name="test_index",
            limit=20,
            offset=5,
            attributes_to_retrieve=["field1", "field3"],
            filter=filter_obj,
            expose_facets=True,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            ef_search=100,
            approximate=False,
            approximate_threshold=0.95,
            rerank_depth_tensor=50,
            or_phrases=["phrase1", "phrase2"],
            and_phrases=["phrase3"],
            hybrid_parameters=hybrid_parameters,
            score_modifiers_lexical=[score_modifier],
            score_modifiers_tensor=[score_modifier],
            global_rerank_depth=100,
            facets=facets,
            track_total_hits=True
        )
        
        # Verify all fields are set correctly
        self.assertEqual("test_index", query.index_name)
        self.assertEqual(20, query.limit)
        self.assertEqual(5, query.offset)
        self.assertEqual(["field1", "field3"], query.attributes_to_retrieve)
        self.assertEqual(filter_obj, query.filter)
        self.assertTrue(query.expose_facets)
        self.assertEqual([0.1, 0.2, 0.3, 0.4], query.vector_query)
        self.assertEqual(100, query.ef_search)
        self.assertFalse(query.approximate)
        self.assertEqual(0.95, query.approximate_threshold)
        self.assertEqual(50, query.rerank_depth_tensor)
        self.assertEqual(["phrase1", "phrase2"], query.or_phrases)
        self.assertEqual(["phrase3"], query.and_phrases)
        self.assertEqual(hybrid_parameters, query.hybrid_parameters)
        self.assertEqual([score_modifier], query.score_modifiers_lexical)
        self.assertEqual([score_modifier], query.score_modifiers_tensor)
        self.assertEqual(100, query.global_rerank_depth)
        self.assertEqual(facets, query.facets)
        self.assertTrue(query.track_total_hits)
        
        # Test inheritance
        self.assertIsInstance(query, MarqoQuery)
        self.assertIsInstance(query, MarqoHybridQuery)

    def test_required_fields(self):
        """Test that all required fields must be provided."""
        hybrid_parameters = HybridParameters()
        
        base_params = {
            "index_name": "test_index",
            "limit": 10,
            "or_phrases": ["phrase1"],
            "and_phrases": ["phrase2"],
            "hybrid_parameters": hybrid_parameters
        }
        
        required_fields = [
            "index_name", "limit", "or_phrases", "and_phrases",
            "hybrid_parameters"
        ]
        
        for required_field in required_fields:
            with self.subTest(missing_field=required_field):
                params = base_params.copy()
                del params[required_field]
                
                with self.assertRaises(ValidationError) as context:
                    MarqoHybridQuery(**params)
                
                self.assertIn(required_field, str(context.exception))

    def test_score_modifiers_validation_with_rrf(self):
        """Test that score_modifiers is allowed with RRF ranking method."""
        score_modifier = ScoreModifier(
            field="test_field",
            weight=1.5,
            type=ScoreModifierType.Multiply
        )
        
        hybrid_parameters = HybridParameters(
            rankingMethod=RankingMethod.RRF
        )
        
        # Should work with RRF
        query = MarqoHybridQuery(
            index_name="test_index",
            limit=10,
            or_phrases=["phrase1"],
            and_phrases=["phrase2"],
            hybrid_parameters=hybrid_parameters,
            score_modifiers=[score_modifier]
        )
        self.assertEqual([score_modifier], query.score_modifiers)

    def test_score_modifiers_validation_with_non_rrf(self):
        """Test that score_modifiers raises error with non-RRF ranking methods."""
        score_modifier = ScoreModifier(
            field="test_field",
            weight=1.5,
            type=ScoreModifierType.Multiply
        )
        
        non_rrf_methods = [RankingMethod.Tensor, RankingMethod.Lexical]
        
        for ranking_method in non_rrf_methods:
            with self.subTest(ranking_method=ranking_method):
                hybrid_parameters = HybridParameters(
                    retrievalMethod=RetrievalMethod.Tensor,
                    rankingMethod=ranking_method
                )
                
                with self.assertRaises(ValidationError) as context:
                    MarqoHybridQuery(
                        index_name="test_index",
                        limit=10,
                        or_phrases=["phrase1"],
                        and_phrases=["phrase2"],
                        hybrid_parameters=hybrid_parameters,
                        score_modifiers=[score_modifier]
                    )
                
                error_msg = ("'scoreModifiers' is only supported for hybrid "
                             "search if 'rankingMethod' is 'RRF'")
                self.assertIn(error_msg, str(context.exception))

    def test_searchable_attributes_validation_fails(self):
        """Test that searchable_attributes cannot be used in hybrid search."""
        hybrid_parameters = HybridParameters()
        
        with self.assertRaises(ValidationError) as context:
            MarqoHybridQuery(
                index_name="test_index",
                limit=10,
                or_phrases=["phrase1"],
                and_phrases=["phrase2"],
                hybrid_parameters=hybrid_parameters,
                searchable_attributes=["field1", "field2"]
            )
        
        self.assertIn(
            "'searchableAttributes' cannot be used for hybrid search",
            str(context.exception)
        )

    def test_marqo_hybrid_query_creates_the_same_hash_for_different_offset(self):
        query_without_offset = MarqoHybridQuery(
            index_name="test_index",
            limit=60,
            offset=0,
            or_phrases=[""],
            and_phrases=[""],
            vector_query=None,
            hybrid_parameters=HybridParameters(),
        )

        query_hash_before_offset = query_without_offset.get_query_hash_without_offset()

        query_with_offset = MarqoHybridQuery(
            index_name="test_index",
            limit=60,
            offset=60,
            or_phrases=[""],
            and_phrases=[""],
            vector_query=None,
            hybrid_parameters=HybridParameters(),
        )

        query_hash_after_offset = query_with_offset.get_query_hash_without_offset()

        assert query_hash_before_offset == query_hash_after_offset

    def test_marqo_hybrid_query_creates_different_hash_for_different_limit(self):
        query_with_limit_30 = MarqoHybridQuery(
            index_name="test_index",
            limit=30,
            offset=0,
            or_phrases=[""],
            and_phrases=[""],
            vector_query=None,
            hybrid_parameters=HybridParameters(),
        )

        query_hash_limit_30 = query_with_limit_30.get_query_hash_without_offset()

        query_with_limit_60 = MarqoHybridQuery(
            index_name="test_index",
            limit=60,
            offset=60,
            or_phrases=[""],
            and_phrases=[""],
            vector_query=None,
            hybrid_parameters=HybridParameters(),
        )

        query_hash_limit_60 = query_with_limit_60.get_query_hash_without_offset()

        assert query_hash_limit_30 != query_hash_limit_60
