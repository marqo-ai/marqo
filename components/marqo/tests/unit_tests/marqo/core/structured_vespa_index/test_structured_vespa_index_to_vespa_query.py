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

    def test_get_individual_field_tensor_search_terms_non_ranking_includes_target_hits(self):
        """With is_ranking_term=False derives from query and term includes targetHits."""
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

    def test_hybrid_query_with_custom_score_modifiers_raises_on_structured(self):
        """Custom score reranking is only supported for semi-structured indexes; structured has no such fields."""
        from marqo.core.exceptions import InvalidFieldNameError
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
        with self.assertRaises(InvalidFieldNameError) as ctx:
            self.vespa_index.to_vespa_query(marqo_query)
        self.assertIn("score modifier", str(ctx.exception).lower())

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

    def _hybrid_query(self, score_modifiers=None):
        return MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['search'],
            and_phrases=[],
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
                rrfK=60,
            ),
            score_modifiers=score_modifiers,
        )

    def test_expose_pre_rerank_score_set_when_global_score_modifiers_present(self):
        """marqo__expose_pre_rerank_score is True when global (non-custom) score modifiers are used."""
        marqo_query = self._hybrid_query(
            score_modifiers=[
                ScoreModifier(field='popularity', weight=1.0, type=ScoreModifierType.Add),
            ],
        )
        # Call _to_vespa_hybrid_query directly to bypass field-existence validation.
        vespa_query = self.vespa_index._to_vespa_hybrid_query(marqo_query)
        self.assertTrue(vespa_query.get('marqo__expose_pre_rerank_score'))

    def test_expose_pre_rerank_score_set_when_custom_score_rerank_modifiers_present(self):
        """marqo__expose_pre_rerank_score is True when custom score rerank modifiers are used."""
        marqo_query = self._hybrid_query(
            score_modifiers=[
                ScoreModifier(
                    field=f'{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_sum',
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        # Call _to_vespa_hybrid_query directly to bypass field-existence validation.
        vespa_query = self.vespa_index._to_vespa_hybrid_query(marqo_query)
        self.assertTrue(vespa_query.get('marqo__expose_pre_rerank_score'))

    def test_expose_pre_rerank_score_absent_when_no_score_modifiers(self):
        """marqo__expose_pre_rerank_score is not set when no score modifiers are used."""
        marqo_query = self._hybrid_query()
        vespa_query = self.vespa_index._to_vespa_hybrid_query(marqo_query)
        self.assertNotIn('marqo__expose_pre_rerank_score', vespa_query)


if __name__ == '__main__':
    unittest.main()
