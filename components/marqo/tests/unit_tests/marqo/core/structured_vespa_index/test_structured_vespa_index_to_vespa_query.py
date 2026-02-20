import time
import unittest
from typing import List

from marqo.core.constants import MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoHybridQuery
from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, Model, TextPreProcessing, TextSplitMethod,
    ImagePreProcessing, HnswConfig, DistanceMetric, Field, FieldType,
    FieldFeature, TensorField
)
from marqo.core.models.hybrid_parameters import (
    HybridParameters, RankingMethod, RetrievalMethod
)
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.exceptions import UnsupportedFeatureError
from marqo.exceptions import InternalError


class TestStructuredVespaIndexToVespaQuery(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with a structured index that supports both tensor and lexical search."""
        # Create a structured index with both tensor and lexical fields
        marqo_index = self._create_structured_marqo_index(
            name='test_index',
            text_field_names=['title', 'description'],
            tensor_field_names=['title', 'description']
        )
        self.vespa_index = StructuredVespaIndex(marqo_index)

    def _create_structured_marqo_index(
            self,
            name: str,
            text_field_names: List[str] = [],
            tensor_field_names: List[str] = []
    ) -> StructuredMarqoIndex:
        """Helper method to create a structured Marqo index for testing."""
        fields = []

        # Add text fields with lexical search and filter capabilities
        for field_name in text_field_names:
            fields.append(
                Field(
                    name=field_name,
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                    lexical_field_name=f'{field_name}_lexical',
                    filter_field_name=f'{field_name}_filter'
                )
            )

        # Add tensor fields
        tensor_fields = []
        for field_name in tensor_field_names:
            tensor_fields.append(
                TensorField(
                    name=field_name,
                    embeddings_field_name=f'{field_name}_embeddings',
                    chunk_field_name=f'{field_name}_chunks'
                )
            )

        return StructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all-MiniLM-L6-v2'),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type='float',
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version='2.12.0',  # Version that supports hybrid search
            created_at=time.time(),
            updated_at=time.time(),
            fields=fields,
            tensor_fields=tensor_fields,
            text_preprocessing=TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(
                patch_method=None
            )
        )

    def test_to_vespa_query_tensor_mode_approximate_threshold(self):
        """Test that to_vespa_query correctly sets approximate threshold for tensor queries."""
        threshold_values = [0.75, 0.85, 0.95, None]

        for threshold in threshold_values:
            with self.subTest(approximate_threshold=threshold):
                marqo_query = MarqoTensorQuery(
                    index_name='test_index',
                    limit=10,
                    offset=0,
                    vector_query=[0.1, 0.2, 0.3, 0.4],
                    approximate_threshold=threshold,
                    approximate=True
                )

                vespa_query = self.vespa_index.to_vespa_query(marqo_query)

                if threshold is not None:
                    # Verify approximate threshold is set correctly
                    self.assertEqual(vespa_query['ranking.matching.approximateThreshold'], threshold)
                else:
                    # When threshold is None, it should not be included in the query
                    self.assertNotIn('ranking.matching.approximateThreshold', vespa_query)

                # Verify other key fields are present
                self.assertIn('yql', vespa_query)
                self.assertIn('ranking', vespa_query)
                self.assertEqual(vespa_query['hits'], 10)

    def test_to_vespa_query_hybrid_mode_approximate_threshold(self):
        """Test that to_vespa_query correctly sets approximate threshold for hybrid queries."""
        threshold_values = [0.70, 0.80, 0.90, None]

        for threshold in threshold_values:
            with self.subTest(approximate_threshold=threshold):
                hybrid_parameters = HybridParameters(
                    retrievalMethod=RetrievalMethod.Disjunction,
                    rankingMethod=RankingMethod.RRF,
                    alpha=0.7,
                    rrfK=100
                )

                marqo_query = MarqoHybridQuery(
                    index_name='test_index',
                    limit=15,
                    offset=0,
                    vector_query=[0.2, 0.3, 0.4, 0.5],
                    or_phrases=['search', 'query'],
                    and_phrases=['required'],
                    hybrid_parameters=hybrid_parameters,
                    approximate_threshold=threshold,
                    approximate=True
                )

                vespa_query = self.vespa_index.to_vespa_query(marqo_query)

                if threshold is not None:
                    # Verify approximate threshold is set correctly
                    self.assertEqual(vespa_query['ranking.matching.approximateThreshold'], threshold)
                else:
                    # When threshold is None, it should not be included in the query
                    self.assertNotIn('ranking.matching.approximateThreshold', vespa_query)

                # Verify hybrid-specific fields are present
                self.assertEqual(vespa_query['hits'], 15)
                self.assertIn('searchChain', vespa_query)
                self.assertEqual(vespa_query['searchChain'], 'marqo')
                self.assertIn('marqo__hybrid.retrievalMethod', vespa_query)
                self.assertIn('marqo__hybrid.rankingMethod', vespa_query)

    def test_get_fields_to_bm25_rerank_by(self):
        """_get_fields_to_bm25_rerank_by returns fields or ['*'] for aggregate."""
        self.assertEqual(
            self.vespa_index._get_fields_to_bm25_rerank_by({'bm25_field_title'}),
            ['title'],
        )
        self.assertEqual(
            self.vespa_index._get_fields_to_bm25_rerank_by({'bm25_field_title', 'bm25_field_description'}),
            ['description', 'title'],
        )
        self.assertEqual(
            self.vespa_index._get_fields_to_bm25_rerank_by({'bm25_sum'}),
            ['*'],
        )
        self.assertEqual(
            self.vespa_index._get_fields_to_bm25_rerank_by({'bm25_field_title', 'bm25_max'}),
            ['*'],
        )

    def test_simplify_bm25_extra_fields_for_rank(self):
        """_simplify_bm25_extra_fields_for_rank removes redundancy with main lexical term."""
        # Empty bm25_fields -> no extra term
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank([], None),
            [],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank([], ['title']),
            [],
        )
        # Main uses default (None): no extra term needed for lexical
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['title'], None),
            [],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['*'], None),
            [],
        )
        # Main has specific fields; bm25 is default -> still need default for aggregate
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['*'], ['title']),
            ['*'],
        )
        # Main has fields; bm25 same set -> no extra
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['title', 'description'], ['title', 'description']),
            [],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['description', 'title'], ['title', 'description']),
            [],
        )
        # Main has subset; bm25 has extra -> only non-main fields
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['title', 'description'], ['title']),
            ['description'],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['title', 'description'], ['description']),
            ['title'],
        )
        # No overlap: main has one field, bm25 has another -> keep bm25 field
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['description'], ['title']),
            ['description'],
        )

    def test_get_fields_to_closeness_rerank_by(self):
        """_get_fields_to_closeness_rerank_by returns tensor fields or all for aggregate."""
        self.assertEqual(
            self.vespa_index._get_fields_to_closeness_rerank_by({'closeness_retrieval_vector_field_title'}),
            ['title'],
        )
        self.assertCountEqual(
            self.vespa_index._get_fields_to_closeness_rerank_by({'closeness_retrieval_vector_sum'}),
            ['description', 'title'],
        )

    def test_get_lexical_contains_term_with_attributes_to_search(self):
        """_get_lexical_contains_term with _is_ranking_term=True uses attributes_to_search."""
        term = self.vespa_index._get_lexical_contains_term(
            'hello', attributes_to_search=['*'], _is_ranking_term=True
        )
        self.assertEqual(term, 'default contains "hello"')
        term = self.vespa_index._get_lexical_contains_term(
            'hello', attributes_to_search=['title'], _is_ranking_term=True
        )
        self.assertIn('title_lexical', term)
        self.assertIn('hello', term)

    def test_get_individual_field_tensor_search_terms_ranking_term_uses_searchable_attributes(self):
        """With _is_ranking_term=True uses only searchable_attributes; term includes targetHits:1."""
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )
        q = MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['x'],
            and_phrases=[],
            hybrid_parameters=hybrid_params,
        )
        terms = self.vespa_index._get_individual_field_tensor_search_terms(
            q, searchable_attributes=['title'], _is_ranking_term=True
        )
        self.assertEqual(len(terms), 1)
        self.assertIn('title_embeddings', terms[0])
        self.assertIn('nearestNeighbor', terms[0])
        self.assertIn('targetHits:1', terms[0])

    def test_get_individual_field_tensor_search_terms_ranking_term_requires_searchable_attributes(self):
        """With _is_ranking_term=True and searchable_attributes=None raises."""
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )
        q = MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['x'],
            and_phrases=[],
            hybrid_parameters=hybrid_params,
        )
        with self.assertRaises(InternalError) as ctx:
            self.vespa_index._get_individual_field_tensor_search_terms(q, _is_ranking_term=True)
        self.assertIn('searchable_attributes', str(ctx.exception))

    def test_get_individual_field_tensor_search_terms_non_ranking_includes_target_hits(self):
        """With _is_ranking_term=False derives from query and term includes targetHits."""
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )
        q = MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['x'],
            and_phrases=[],
            hybrid_parameters=hybrid_params,
        )
        terms = self.vespa_index._get_individual_field_tensor_search_terms(q)
        self.assertGreater(len(terms), 0)
        self.assertIn('targetHits', terms[0])

    def test_generate_or_terms_ranking_term_no_target_hits(self):
        """_generate_or_terms with _is_ranking_term=True returns weakAnd without targetHits."""
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )
        q = MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['search'],
            and_phrases=[],
            hybrid_parameters=hybrid_params,
        )
        result = self.vespa_index._generate_or_terms(
            q, _is_ranking_term=True, attributes_to_search=['title']
        )
        self.assertIn('weakAnd', result)
        self.assertNotIn('targetHits', result)
        self.assertIn('search', result)

    def test_get_lexical_search_term_ranking_term(self):
        """_get_lexical_search_term with _is_ranking_term=True uses attributes_to_search, no targetHits."""
        hybrid_params = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )
        q = MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['hello'],
            and_phrases=[],
            hybrid_parameters=hybrid_params,
        )
        result = self.vespa_index._get_lexical_search_term(
            q, _is_ranking_term=True, attributes_to_search=['title']
        )
        self.assertNotEqual(result, 'false')
        self.assertNotIn('targetHits', result)
        self.assertIn('hello', result)

    def test_hybrid_query_with_custom_score_modifiers_raises_unsupported_on_structured(self):
        """Custom score reranking is only supported for semi-structured indexes; structured must raise."""
        hybrid_parameters = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )
        marqo_query = MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['search'],
            and_phrases=[],
            hybrid_parameters=hybrid_parameters,
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            self.vespa_index.to_vespa_query(marqo_query)
        self.assertIn("semi-structured", str(ctx.exception))
        self.assertIn("not for structured", str(ctx.exception))

    def test_hybrid_query_without_custom_score_modifiers_no_custom_score_query_inputs(self):
        """Without custom score modifiers, query_features must not contain custom score keys."""
        hybrid_parameters = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            alpha=0.5,
            rrfK=60,
        )
        marqo_query = MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['search'],
            and_phrases=[],
            hybrid_parameters=hybrid_parameters,
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)
        self.assertNotIn('marqo__custom_score_add_weights_global', str(vespa_query.get('query_features', {})))
        self.assertNotIn('marqo__custom_score_mult_weights_global', str(vespa_query.get('query_features', {})))


if __name__ == '__main__':
    unittest.main()
