import time
import unittest
from typing import List

from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoHybridQuery, MarqoLexicalQuery
from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, Model, TextPreProcessing, TextSplitMethod,
    ImagePreProcessing, HnswConfig, DistanceMetric, Field, FieldType,
    FieldFeature, TensorField
)
from marqo.core.models.hybrid_parameters import (
    HybridParameters, RankingMethod, RetrievalMethod
)
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex


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
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
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

    def test_to_vespa_query_lexical_all_inputs(self):
        """Test that to_vespa_query correctly handles all inputs for lexical queries including language."""

        # Test lexical query with language
        marqo_query = MarqoLexicalQuery(
            index_name='test_index',
            limit=20,
            offset=5,
            or_phrases=['machine learning', 'artificial intelligence'],
            and_phrases=['deep'],
            language='en'
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)

        expected_query = {
            'yql': 'select * from test_index where ((weakAnd(default contains "machine learning", default contains "artificial intelligence")) AND (default contains "deep"))',
            'model_restrict': 'test_index',
            'hits': 20,
            'offset': 5,
            'query_features': {
                'description_lexical': 1,
                'title_lexical': 1
            },
            'presentation.summary': 'all-non-vector-summary',
            'ranking': 'bm25',
            'language': 'en'
        }

        self.assertEqual(vespa_query, expected_query)

        # Test lexical query without language
        marqo_query = MarqoLexicalQuery(
            index_name='test_index',
            limit=15,
            offset=0,
            or_phrases=['test query'],
            and_phrases=[]
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)

        expected_query = {
            'yql': 'select * from test_index where (weakAnd(default contains "test query"))',
            'model_restrict': 'test_index',
            'hits': 15,
            'offset': 0,
            'query_features': {
                'description_lexical': 1,
                'title_lexical': 1
            },
            'presentation.summary': 'all-non-vector-summary',
            'ranking': 'bm25'
        }

        self.assertEqual(vespa_query, expected_query)
        self.assertNotIn('language', vespa_query)

        # Test lexical query with searchable attributes and language
        marqo_query = MarqoLexicalQuery(
            index_name='test_index',
            limit=25,
            offset=10,
            or_phrases=['specific field search'],
            and_phrases=[],
            searchable_attributes=['title'],
            language='es'
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)

        # With searchable attributes, only the specified field should be in query_features
        expected_query_features = {'title_lexical': 1}
        self.assertEqual(vespa_query['query_features'], expected_query_features)
        self.assertEqual(vespa_query['language'], 'es')
        self.assertEqual(vespa_query['hits'], 25)
        self.assertEqual(vespa_query['offset'], 10)

    def test_to_vespa_query_hybrid_all_inputs(self):
        """Test that to_vespa_query correctly handles all inputs for hybrid queries including language."""

        # Test hybrid query with language and RRF ranking
        marqo_query = MarqoHybridQuery(
            index_name='test_index',
            limit=30,
            offset=10,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['neural networks', 'deep learning'],
            and_phrases=['transformer'],
            language='en',
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
                rrfK=60
            ),
            approximate=True,
            approximate_threshold=0.85
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)

        expected_query = {
            'hits': 30,
            'language': 'en',
            'marqo__hybrid.alpha': 0.5,
            'marqo__hybrid.rankingMethod': 'rrf',
            'marqo__hybrid.retrievalMethod': 'disjunction',
            'marqo__hybrid.rrf_k': 60,
            'marqo__hybrid.verbose': False,
            'marqo__ranking.lexical.lexical': 'bm25',
            'marqo__ranking.lexical.tensor': 'hybrid_bm25_then_embedding_similarity',
            'marqo__ranking.tensor.lexical': 'hybrid_embedding_similarity_then_bm25',
            'marqo__ranking.tensor.tensor': 'embedding_similarity',
            'marqo__yql.lexical': 'select * from test_index where ((weakAnd(default contains "neural networks", default contains "deep learning")) AND (default contains "transformer"))',
            'marqo__yql.tensor': 'select * from test_index where (({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(title_embeddings, marqo__query_embedding)) OR ({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(description_embeddings, marqo__query_embedding)))',
            'model_restrict': 'test_index',
            'offset': 10,
            'presentation.summary': 'all-non-vector-summary',
            'query_features': {
                'marqo__fields_to_rank_lexical': {
                    'description_lexical': 1,
                    'title_lexical': 1
                },
                'marqo__fields_to_rank_tensor': {
                    'description_embeddings': 1,
                    'title_embeddings': 1
                },
                'marqo__query_embedding': [0.1, 0.2, 0.3, 0.4]
            },
            'ranking': 'hybrid_custom_searcher',
            'ranking.matching.approximateThreshold': 0.85,
            'ranking.rerankCount': 40,
            'searchChain': 'marqo',
            'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.'
        }

        self.assertEqual(vespa_query, expected_query)

        # Test hybrid query without language
        marqo_query = MarqoHybridQuery(
            index_name='test_index',
            limit=25,
            offset=0,
            vector_query=[0.3, 0.3, 0.3, 0.3],
            or_phrases=['general search'],
            and_phrases=[],
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.8,
                rrfK=100
            )
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)

        expected_query = {
            'hits': 25,
            'marqo__hybrid.alpha': 0.8,
            'marqo__hybrid.rankingMethod': 'rrf',
            'marqo__hybrid.retrievalMethod': 'disjunction',
            'marqo__hybrid.rrf_k': 100,
            'marqo__hybrid.verbose': False,
            'marqo__ranking.lexical.lexical': 'bm25',
            'marqo__ranking.lexical.tensor': 'hybrid_bm25_then_embedding_similarity',
            'marqo__ranking.tensor.lexical': 'hybrid_embedding_similarity_then_bm25',
            'marqo__ranking.tensor.tensor': 'embedding_similarity',
            'marqo__yql.lexical': 'select * from test_index where (weakAnd(default contains "general search"))',
            'marqo__yql.tensor': 'select * from test_index where (({targetHits:25, approximate:True, hnsw.exploreAdditionalHits:1975}nearestNeighbor(title_embeddings, marqo__query_embedding)) OR ({targetHits:25, approximate:True, hnsw.exploreAdditionalHits:1975}nearestNeighbor(description_embeddings, marqo__query_embedding)))',
            'model_restrict': 'test_index',
            'offset': 0,
            'presentation.summary': 'all-non-vector-summary',
            'query_features': {
                'marqo__fields_to_rank_lexical': {
                    'description_lexical': 1,
                    'title_lexical': 1
                },
                'marqo__fields_to_rank_tensor': {
                    'description_embeddings': 1,
                    'title_embeddings': 1
                },
                'marqo__query_embedding': [0.3, 0.3, 0.3, 0.3]
            },
            'ranking': 'hybrid_custom_searcher',
            'ranking.rerankCount': 25,
            'searchChain': 'marqo',
            'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.'
        }

        self.assertEqual(vespa_query, expected_query)
        self.assertNotIn('language', vespa_query)


if __name__ == '__main__':
    unittest.main()
