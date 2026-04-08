from unittest import TestCase

import random
import time
import unittest
from typing import List
from unittest.mock import MagicMock

from marqo.core.constants import MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod, WeakAndParameters
from marqo.core.models.score_modifier import ScoreModifier, ScoreModifierType
from marqo.core.models.marqo_index import (
    Model, TextPreProcessing, TextSplitMethod,
    ImagePreProcessing, HnswConfig, DistanceMetric, Field, FieldType,
    FieldFeature, TensorField, StringArrayField, CollapseField
)
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoHybridQuery, MarqoLexicalQuery
from marqo.core.models.marqo_query import MarqoTensorQuery
from marqo.core.search.search_filter import SearchFilter, EqualityTerm
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.tensor_search.models.relevance_cutoff_model import (
    RelevanceCutoffModel,
    RelevanceCutoffMethod,
    RelativeMaxScoreParameters,
    MeanStdParameters,
    ApplyInRetrieval
)
from marqo.tensor_search.models.sort_by_model import SortByModel
from marqo.version import get_version
from tests.unit_tests.marqo_test import MarqoTestCase
from marqo.tensor_search.models.collapse_model import CollapseModel, CollapseSortBy, CollapseSortByField

class TestSemiStructuredVespaIndexToVespaQuery(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with a semi-structured index that supports both tensor and lexical search."""
        # Create a semi-structured index with both lexical and tensor fields
        marqo_index = self._create_semi_structured_marqo_index(
            name='test_index',
            lexical_field_names=['title', 'description'], 
            tensor_field_names=['title', 'description'],
            string_array_field_names=['tags']
        )
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def _create_semi_structured_marqo_index(
        self, 
        name: str,
        lexical_field_names: List[str] = [],
        tensor_field_names: List[str] = [],
        string_array_field_names: List[str] = [],
        version: str = '2.16.0' # Version that supports hybrid search and partial updates
    ) -> SemiStructuredMarqoIndex:
        """Helper method to create a semi-structured Marqo index for testing."""
        
        # Create lexical fields
        lexical_fields = []
        for field_name in lexical_field_names:
            lexical_fields.append(
                Field(
                    name=field_name,
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                    lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}',
                    filter_field_name=f'{field_name}_filter'
                )
            )

        # Create tensor fields
        tensor_fields = []
        for field_name in tensor_field_names:
            tensor_fields.append(
                TensorField(
                    name=field_name,
                    embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
                    chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}'
                )
            )

        # Create string array fields
        string_array_fields = []
        for field_name in string_array_field_names:
            string_array_fields.append(
                StringArrayField(
                    name=field_name,
                    type=FieldType.ArrayText,
                    features=[FieldFeature.Filter],
                    string_array_field_name=f'{SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX}{field_name}'
                )
            )

        return SemiStructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all-MiniLM-L6-v2'),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type='float',
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version=version,
            created_at=time.time(),
            updated_at=time.time(),
            text_preprocessing=TextPreProcessing(
                split_length=2,
                split_overlap=0,
                split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(
                patch_method=None
            ),
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=50,
            lexical_fields=lexical_fields,
            tensor_fields=tensor_fields,
            string_array_fields=string_array_fields
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
                
                # Verify hybrid-specific fields are present for semi-structured indexes
                self.assertEqual(vespa_query['hits'], 15)
                self.assertIn('ranking', vespa_query)

    def test_to_vespa_query_lexical_all_inputs(self):
        """Test that to_vespa_query correctly handles all inputs for lexical queries including language."""
        test_cases = [
            {
                'name': 'with_language',
                'query_params': {
                    'index_name': 'test_index',
                    'limit': 20,
                    'offset': 5,
                    'or_phrases': ['machine learning', 'artificial intelligence'],
                    'and_phrases': ['deep'],
                    'language': 'en'
                },
                'expected_query': {
                    'yql': 'select * from test_index where ((weakAnd(default contains "machine learning", default contains "artificial intelligence")) AND (default contains "deep"))',
                    'model_restrict': 'test_index',
                    'hits': 20,
                    'offset': 5,
                    'query_features': {
                        'marqo__lexical_description': 1,
                        'marqo__lexical_title': 1
                    },
                    'presentation.summary': 'all-non-vector-summary',
                    'ranking': 'bm25',
                    'language': 'en'
                },
                'should_have_language': True
            },
            {
                'name': 'without_language',
                'query_params': {
                    'index_name': 'test_index',
                    'limit': 15,
                    'offset': 0,
                    'or_phrases': ['test query'],
                    'and_phrases': []
                },
                'expected_query': {
                    'yql': 'select * from test_index where (weakAnd(default contains "test query"))',
                    'model_restrict': 'test_index',
                    'hits': 15,
                    'offset': 0,
                    'query_features': {
                        'marqo__lexical_description': 1,
                        'marqo__lexical_title': 1
                    },
                    'presentation.summary': 'all-non-vector-summary',
                    'ranking': 'bm25'
                },
                'should_have_language': False
            },
            {
                'name': 'with_searchable_attributes_and_language',
                'query_params': {
                    'index_name': 'test_index',
                    'limit': 25,
                    'offset': 10,
                    'or_phrases': ['specific field search'],
                    'and_phrases': [],
                    'searchable_attributes': ['title'],
                    'language': 'es'
                },
                'expected_query_features': {'marqo__lexical_title': 1},
                'expected_language': 'es',
                'expected_hits': 25,
                'expected_offset': 10,
                'should_have_language': True
            }
        ]

        for test_case in test_cases:
            with self.subTest(case=test_case['name']):
                marqo_query = MarqoLexicalQuery(**test_case['query_params'])
                vespa_query = self.vespa_index.to_vespa_query(marqo_query)

                if 'expected_query' in test_case:
                    self.assertEqual(test_case['expected_query'], vespa_query)
                
                if 'expected_query_features' in test_case:
                    self.assertEqual(test_case['expected_query_features'], vespa_query['query_features'])
                    self.assertEqual(test_case['expected_language'], vespa_query['language'])
                    self.assertEqual(test_case['expected_hits'], vespa_query['hits'])
                    self.assertEqual(test_case['expected_offset'], vespa_query['offset'])
                
                if not test_case['should_have_language']:
                    self.assertNotIn('language', vespa_query)

    def test_to_vespa_query_hybrid_all_inputs(self):
        """Test that to_vespa_query correctly handles all inputs for hybrid queries including language."""
        test_cases = [
            {
                'name': 'with_language_and_rrf_ranking',
                'query_params': {
                    'index_name': 'test_index',
                    'limit': 30,
                    'offset': 10,
                    'vector_query': [0.1, 0.2, 0.3, 0.4],
                    'or_phrases': ['neural networks', 'deep learning'],
                    'and_phrases': ['transformer'],
                    'language': 'en',
                    'hybrid_parameters': HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        alpha=0.5,
                        rrfK=60
                    ),
                    'approximate': True,
                    'approximate_threshold': 0.85
                },
                'expected_query': {
                    'hits': 30,
                    'language': 'en',
                    'marqo__hybrid.alpha': 0.5,
                    'marqo__hybrid.rankingMethod': RankingMethod.RRF,
                    'marqo__hybrid.retrievalMethod': RetrievalMethod.Disjunction,
                    'marqo__hybrid.rrf_k': 60,
                    'marqo__hybrid.verbose': False,
                    'marqo__ranking.lexical.lexical': 'bm25',
                    'marqo__ranking.lexical.tensor': 'hybrid_bm25_then_embedding_similarity',
                    'marqo__ranking.tensor.lexical': 'hybrid_embedding_similarity_then_bm25',
                    'marqo__ranking.tensor.tensor': 'embedding_similarity',
                    'marqo__yql.lexical': 'select * from test_index where ((weakAnd(default contains "neural networks", default contains "deep learning")) AND (default contains "transformer"))',
                    'marqo__yql.tensor': 'select * from test_index where (({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) OR ({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(marqo__embeddings_description, marqo__query_embedding)))',
                    'model_restrict': 'test_index',
                    'offset': 10,
                    'presentation.summary': 'all-non-vector-summary',
                    'query_features': {
                        'marqo__fields_to_rank_lexical': {
                            'marqo__lexical_description': 1,
                            'marqo__lexical_title': 1
                        },
                        'marqo__fields_to_rank_tensor': {
                            'marqo__embeddings_description': 1,
                            'marqo__embeddings_title': 1
                        },
                        'marqo__query_embedding': [0.1, 0.2, 0.3, 0.4]
                    },
                    'ranking': 'hybrid_custom_searcher',
                    'ranking.matching.approximateThreshold': 0.85,
                    'ranking.rerankCount': 40,
                    'searchChain': 'marqo',
                    'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.'
                },
                'should_have_language': True
            },
            {
                'name': 'without_language',
                'query_params': {
                    'index_name': 'test_index',
                    'limit': 25,
                    'offset': 0,
                    'vector_query': [0.3, 0.3, 0.3, 0.3],
                    'or_phrases': ['general search'],
                    'and_phrases': [],
                    'hybrid_parameters': HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        alpha=0.8,
                        rrfK=100
                    )
                },
                'expected_query': {
                    'hits': 25,
                    'marqo__hybrid.alpha': 0.8,
                    'marqo__hybrid.rankingMethod': RankingMethod.RRF,
                    'marqo__hybrid.retrievalMethod': RetrievalMethod.Disjunction,
                    'marqo__hybrid.rrf_k': 100,
                    'marqo__hybrid.verbose': False,
                    'marqo__ranking.lexical.lexical': 'bm25',
                    'marqo__ranking.lexical.tensor': 'hybrid_bm25_then_embedding_similarity',
                    'marqo__ranking.tensor.lexical': 'hybrid_embedding_similarity_then_bm25',
                    'marqo__ranking.tensor.tensor': 'embedding_similarity',
                    'marqo__yql.lexical': 'select * from test_index where (weakAnd(default contains "general search"))',
                    'marqo__yql.tensor': 'select * from test_index where (({targetHits:25, approximate:True, hnsw.exploreAdditionalHits:1975}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) OR ({targetHits:25, approximate:True, hnsw.exploreAdditionalHits:1975}nearestNeighbor(marqo__embeddings_description, marqo__query_embedding)))',
                    'model_restrict': 'test_index',
                    'offset': 0,
                    'presentation.summary': 'all-non-vector-summary',
                    'query_features': {
                        'marqo__fields_to_rank_lexical': {
                            'marqo__lexical_description': 1,
                            'marqo__lexical_title': 1
                        },
                        'marqo__fields_to_rank_tensor': {
                            'marqo__embeddings_description': 1,
                            'marqo__embeddings_title': 1
                        },
                        'marqo__query_embedding': [0.3, 0.3, 0.3, 0.3]
                    },
                    'ranking': 'hybrid_custom_searcher',
                    'ranking.rerankCount': 25,
                    'searchChain': 'marqo',
                    'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.'
                },
                'should_have_language': False
            }
        ]

        for test_case in test_cases:
            with self.subTest(case=test_case['name']):
                marqo_query = MarqoHybridQuery(**test_case['query_params'])
                vespa_query = self.vespa_index.to_vespa_query(marqo_query)

                self.assertEqual(test_case['expected_query'], vespa_query)
                
                if not test_case['should_have_language']:
                    self.assertNotIn('language', vespa_query)

    def test_to_vespa_query_hybrid_lexical_with_rerankDepthLexical(self):
        """A test that to_vespa_query correctly handles lexical queries with rerankDepthLexical."""
        test_cases = [
            {
                'name': 'with_language_and_rrf_ranking',
                'query_params': {
                    'index_name': 'test_index',
                    'limit': 30,
                    'offset': 10,
                    'vector_query': [0.1, 0.2, 0.3, 0.4],
                    'or_phrases': ['neural networks', 'deep learning'],
                    'and_phrases': ['transformer'],
                    'language': 'en',
                    'hybrid_parameters': HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        alpha=0.5,
                        rrfK=60,
                        rerankDepthLexical=111,
                        rerankCount=222,
                        weakAndParameters=WeakAndParameters(
                            stopwordLimit=0.2,
                            adjustTarget=0.3,
                            allowDropAll=True,
                            filterThreshold=0.4
                        ),
                        secondPhaseModifier=True
                    ),
                    'approximate': True,
                    'approximate_threshold': 0.85,
                    'track_total_hits': True,
                },
                'expected_query': {
                    'hits': 30,
                    'language': 'en',
                    'marqo__hybrid.alpha': 0.5,
                    'marqo__hybrid.rankingMethod': RankingMethod.RRF,
                    'marqo__hybrid.retrievalMethod': RetrievalMethod.Disjunction,
                    'marqo__hybrid.rrf_k': 60,
                    'marqo__hybrid.verbose': False,
                    'marqo__ranking.lexical.lexical': 'hybrid_bm25_second_phase_modifiers',
                    'marqo__ranking.lexical.tensor': 'hybrid_bm25_then_embedding_similarity',
                    'marqo__ranking.tensor.lexical': 'hybrid_embedding_similarity_then_bm25',
                    'marqo__ranking.tensor.tensor': 'embedding_similarity',
                    'ranking.rerankCount': 222,
                    "ranking.matching.weakand.stopwordLimit": 0.2,
                    "ranking.matching.weakand.adjustTarget": 0.3,
                    "ranking.matching.weakand.allowDropAll": True,
                    "ranking.matching.filterThreshold": 0.4,
                    # Facets should still use the OR query structure
                    'marqo__yql.facets': 'select * from test_index where ((default contains "neural networks" OR default contains "deep learning") '
                                         'AND (default contains "transformer") OR '
                                         '(({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) '
                                         'OR ({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(marqo__embeddings_description, marqo__query_embedding)))) '
                                         'limit 0 | all(group(1.1) each(output(count())))',
                    'marqo__yql.lexical': 'select * from test_index where (({targetHits:111}weakAnd(default contains "neural networks", default contains "deep learning")) AND (default contains "transformer"))',
                    'marqo__yql.tensor': 'select * from test_index where (({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) OR ({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(marqo__embeddings_description, marqo__query_embedding)))',
                    'model_restrict': 'test_index',
                    'offset': 10,
                    'presentation.summary': 'all-non-vector-summary',
                    'query_features': {
                        'marqo__fields_to_rank_lexical': {
                            'marqo__lexical_description': 1,
                            'marqo__lexical_title': 1
                        },
                        'marqo__fields_to_rank_tensor': {
                            'marqo__embeddings_description': 1,
                            'marqo__embeddings_title': 1
                        },
                        'marqo__query_embedding': [0.1, 0.2, 0.3, 0.4]
                    },
                    'ranking': 'hybrid_custom_searcher',
                    'ranking.matching.approximateThreshold': 0.85,
                    'searchChain': 'marqo',
                    'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.'
                },
                'should_have_language': True
            },
            {
                'name': 'without_language',
                'query_params': {
                    'index_name': 'test_index',
                    'limit': 25,
                    'offset': 0,
                    'vector_query': [0.3, 0.3, 0.3, 0.3],
                    'or_phrases': ['general search'],
                    'and_phrases': [],
                    'hybrid_parameters': HybridParameters(
                        retrievalMethod=RetrievalMethod.Disjunction,
                        rankingMethod=RankingMethod.RRF,
                        alpha=0.8,
                        rrfK=100,
                        rerankDepthLexical=111,
                    )
                },
                'expected_query': {
                    'hits': 25,
                    'marqo__hybrid.alpha': 0.8,
                    'marqo__hybrid.rankingMethod': RankingMethod.RRF,
                    'marqo__hybrid.retrievalMethod': RetrievalMethod.Disjunction,
                    'marqo__hybrid.rrf_k': 100,
                    'marqo__hybrid.verbose': False,
                    'marqo__ranking.lexical.lexical': 'bm25',
                    'marqo__ranking.lexical.tensor': 'hybrid_bm25_then_embedding_similarity',
                    'marqo__ranking.tensor.lexical': 'hybrid_embedding_similarity_then_bm25',
                    'marqo__ranking.tensor.tensor': 'embedding_similarity',
                    'marqo__yql.lexical': 'select * from test_index where ({targetHits:111}weakAnd(default contains "general search"))',
                    'marqo__yql.tensor': 'select * from test_index where (({targetHits:25, approximate:True, hnsw.exploreAdditionalHits:1975}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) OR ({targetHits:25, approximate:True, hnsw.exploreAdditionalHits:1975}nearestNeighbor(marqo__embeddings_description, marqo__query_embedding)))',
                    'model_restrict': 'test_index',
                    'offset': 0,
                    'presentation.summary': 'all-non-vector-summary',
                    'query_features': {
                        'marqo__fields_to_rank_lexical': {
                            'marqo__lexical_description': 1,
                            'marqo__lexical_title': 1
                        },
                        'marqo__fields_to_rank_tensor': {
                            'marqo__embeddings_description': 1,
                            'marqo__embeddings_title': 1
                        },
                        'marqo__query_embedding': [0.3, 0.3, 0.3, 0.3]
                    },
                    'ranking': 'hybrid_custom_searcher',
                    'ranking.rerankCount': 25,
                    'searchChain': 'marqo',
                    'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.'
                },
                'should_have_language': False
            },
        ]

        for test_case in test_cases:
            with self.subTest(case=test_case['name']):
                self.maxDiff = None
                marqo_query = MarqoHybridQuery(**test_case['query_params'])
                vespa_query = self.vespa_index.to_vespa_query(marqo_query)

                self.assertEqual(test_case['expected_query'], vespa_query)

                if not test_case['should_have_language']:
                    self.assertNotIn('language', vespa_query)


class TestSemiStructuredIndexToVespaQuerySortBy(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.marqo_index = MagicMock(spec=SemiStructuredMarqoIndex)
        cls.marqo_index.parsed_marqo_version.return_value = get_version()
        cls.marqo_index.schema_name = "test_sort_by_index"
        cls.index = SemiStructuredVespaIndex(cls.marqo_index)

    def setUp(self):
        self.hybrid_query = MarqoHybridQuery(
            index_name = "test_index",
            vector_query = None,
            filter=None,
            limit=10,
            offset=0,
            attributes_to_retrieve=None,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF
            ),
            score_modifiers_lexical=None,
            score_modifiers_tensor=None,
            or_phrases=[],
            and_phrases=[],
            sort_by=None,
        )

    def test_sort_by_multiple_fields_desc_and_asc(self):
        """Test sorting by two fields with descending and ascending orders."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "price", "order": "desc"},
                {"field_name": "rating", "order": "asc"}
            ],
            sortDepth=3,
            minSortCandidates=50
        )

        r = self.index.to_vespa_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']

        self.assertEqual(2, len(sort_fields))
        self.assertEqual("price", sort_fields[0]["field_name"])
        self.assertEqual("desc", sort_fields[0]["order"])
        self.assertEqual("rating", sort_fields[1]["field_name"])
        self.assertEqual("asc", sort_fields[1]["order"])
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

        r = self.index.to_vespa_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']
        sort_depth = r["marqo__hybrid.sortBy.sortDepth"]
        sort_candidates = r["marqo__hybrid.sortBy.minSortCandidates"]

        self.assertEqual(1, len(sort_fields))
        self.assertEqual("title", sort_fields[0]["field_name"])
        self.assertEqual("asc", sort_fields[0]["order"])
        self.assertEqual(None, sort_depth)
        self.assertEqual(30, sort_candidates)

    def test_sort_by_with_missing_first(self):
        """Test a field with missing='first' and all optional params."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "description", "order": "asc", "missing": "first"}
            ],
            sortDepth=2,
            minSortCandidates=20
        )

        r = self.index.to_vespa_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']

        self.assertEqual(1, len(sort_fields))
        self.assertEqual("description", sort_fields[0]["field_name"])
        self.assertEqual("asc", sort_fields[0]["order"])
        self.assertEqual("first", sort_fields[0]["missing"])
        self.assertEqual(2, r['marqo__hybrid.sortBy.sortDepth'])
        self.assertEqual(20, r['marqo__hybrid.sortBy.minSortCandidates'])

    def test_sort_by_none(self):
        """Test that no sort_by results in no sort fields present."""
        self.hybrid_query.sort_by = None
        r = self.index.to_vespa_query(self.hybrid_query)

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

        r = self.index.to_vespa_query(self.hybrid_query)
        fields = r["marqo__hybrid.sortBy.fields"]

        self.assertEqual(3, len(fields))
        self.assertEqual("price", fields[0]["field_name"])
        self.assertEqual("desc", fields[0]["order"])
        self.assertEqual("last", fields[0]["missing"])

        self.assertEqual("rating", fields[1]["field_name"])
        self.assertEqual("asc", fields[1]["order"])
        self.assertEqual("last", fields[1]["missing"])  # Default missing policy

        self.assertEqual("stock", fields[2]["field_name"])
        self.assertEqual("desc", fields[2]["order"])
        self.assertEqual("first", fields[2]["missing"])

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

            r = self.index.to_vespa_query(self.hybrid_query)
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

            r = self.index.to_vespa_query(self.hybrid_query)
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

        r = self.index.to_vespa_query(self.hybrid_query)
        query_features = r["query_features"]

        self.assertEqual({"alpha": 1}, query_features[f"marqo__sort_field_weights_{0}"])
        self.assertEqual({}, query_features[f"marqo__sort_field_weights_{1}"])
        self.assertEqual({}, query_features[f"marqo__sort_field_weights_{2}"])

    def test_query_features_sort_field_weights_zero_fields(self):
        """A fuzzy test to ensure that query_features are correctly populated with sort field weights."""
        self.hybrid_query.sort_by = None

        r = self.index.to_vespa_query(self.hybrid_query)
        query_features = r["query_features"]

        for i in range(3):
            self.assertNotIn(f"marqo__sort_field_weights_{i}", query_features)


class TestSemiStructuredIndexToVespaQueryRelevanceCutoff(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.marqo_index = MagicMock(spec=SemiStructuredMarqoIndex)
        cls.marqo_index.parsed_marqo_version.return_value = get_version()
        cls.marqo_index.schema_name = "test_relevance_cutoff_index"
        cls.index = SemiStructuredVespaIndex(cls.marqo_index)

    def setUp(self):
        # Assign the basic hybrid query structure
        self.hybrid_query = MarqoHybridQuery(
            index_name="test_index",
            vector_query=None,
            filter=None,
            limit=10,
            offset=0,
            attributes_to_retrieve=None,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF
            ),
            score_modifiers_lexical=None,
            score_modifiers_tensor=None,
            or_phrases=[],
            and_phrases=[],
            sort_by=None,
            relevance_cutoff=None
        )

    def test_no_relevance_cutoff(self):
        """If relevance_cutoff is None, no cutoff keys should appear."""
        r = self.index.to_vespa_query(self.hybrid_query)
        for key in [
            "marqo__hybrid.relevanceCutoff.method",
            "marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor",
            "marqo__hybrid.relevanceCutoff.parameters.stdDevFactor",
            "marqo__hybrid.relevanceCutoff.probeDepth",
            "marqo__hybrid.relevanceCutoff.affectFacets",
            "marqo__hybrid.relevanceCutoff.overrideSortCandidates"
        ]:
            self.assertNotIn(key, r)

    def test_relative_max_score_default_probeDepth(self):
        """RelativeMaxScore should set method, relativeScoreFactor, and default probeDepth."""
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.8)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=params
        )

        r = self.index.to_vespa_query(self.hybrid_query)

        self.assertEqual(RelevanceCutoffMethod.RelativeMaxScore,
                         r["marqo__hybrid.relevanceCutoff.method"])
        self.assertAlmostEqual(0.8,
                               r["marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor"])
        # default probeDepth is 1000
        self.assertEqual(1000, r["marqo__hybrid.relevanceCutoff.probeDepth"])
        # default affectFacets is False
        self.assertEqual(False, r["marqo__hybrid.relevanceCutoff.affectFacets"])
        # default overrideSortCandidates is False
        self.assertEqual(False, r["marqo__hybrid.relevanceCutoff.overrideSortCandidates"])
        # no stdDevFactor for this method
        self.assertNotIn("marqo__hybrid.relevanceCutoff.parameters.stdDevFactor", r)

    def test_relative_max_score_custom_probeDepth(self):
        """Custom probeDepth should be honoured for RelativeMaxScore."""
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.3)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=params,
            probe_depth=5
        )

        r = self.index.to_vespa_query(self.hybrid_query)
        self.assertEqual(5, r["marqo__hybrid.relevanceCutoff.probeDepth"])

    def test_mean_std_dev_default_probeDepth(self):
        """MeanStdDev should set method, stdDevFactor, and default probeDepth."""
        params = MeanStdParameters(stdDevFactor=2.5)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.MeanStdDev,
            parameters=params
        )

        r = self.index.to_vespa_query(self.hybrid_query)

        self.assertEqual(RelevanceCutoffMethod.MeanStdDev,
                         r["marqo__hybrid.relevanceCutoff.method"])
        self.assertAlmostEqual(2.5,
                               r["marqo__hybrid.relevanceCutoff.parameters.stdDevFactor"])
        self.assertEqual(1000, r["marqo__hybrid.relevanceCutoff.probeDepth"])
        # no relativeScoreFactor for this method
        self.assertNotIn("marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor", r)

    def test_mean_std_dev_custom_probeDepth(self):
        """Custom probeDepth should be honoured for MeanStdDev."""
        params = MeanStdParameters(stdDevFactor=1.2)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.MeanStdDev,
            parameters=params,
            probe_depth=7
        )

        r = self.index.to_vespa_query(self.hybrid_query)
        self.assertEqual(7, r["marqo__hybrid.relevanceCutoff.probeDepth"])

    def test_gap_detection_default_and_custom_probeDepth(self):
        """GapDetection should set method, have no parameters, and honour probeDepth."""
        # default probeDepth
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection
        )
        r = self.index.to_vespa_query(self.hybrid_query)
        self.assertEqual(RelevanceCutoffMethod.GapDetection,
                         r["marqo__hybrid.relevanceCutoff.method"])
        self.assertEqual(1000, r["marqo__hybrid.relevanceCutoff.probeDepth"])
        self.assertNotIn("marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor", r)
        self.assertNotIn("marqo__hybrid.relevanceCutoff.parameters.stdDevFactor", r)

        # custom probeDepth
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            probe_depth=42
        )
        r2 = self.index.to_vespa_query(self.hybrid_query)
        self.assertEqual(42, r2["marqo__hybrid.relevanceCutoff.probeDepth"])

    def test_relevance_cutoff_edge_case_values(self):
        """Test relevance cutoff with edge case parameter values."""
        # Test minimum valid relativeScoreFactor
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.001)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=params,
            probe_depth=1  # minimum probe depth
        )
        
        r = self.index.to_vespa_query(self.hybrid_query)
        self.assertEqual(0.001,
                               r["marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor"])
        self.assertEqual(1, r["marqo__hybrid.relevanceCutoff.probeDepth"])
        
        # Test maximum valid relativeScoreFactor
        params = RelativeMaxScoreParameters(relativeScoreFactor=1.0)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=params,
            probe_depth=10000  # large probe depth
        )
        
        r = self.index.to_vespa_query(self.hybrid_query)
        self.assertAlmostEqual(1.0,
                               r["marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor"])
        self.assertEqual(10000, r["marqo__hybrid.relevanceCutoff.probeDepth"])

    def test_relevance_cutoff_std_dev_edge_cases(self):
        """Test MeanStdDev with edge case stdDevFactor values."""
        # Test small stdDevFactor
        params = MeanStdParameters(stdDevFactor=0.1)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.MeanStdDev,
            parameters=params,
            probe_depth=50
        )
        
        r = self.index.to_vespa_query(self.hybrid_query)
        self.assertEqual(RelevanceCutoffMethod.MeanStdDev,
                         r["marqo__hybrid.relevanceCutoff.method"])
        self.assertAlmostEqual(0.1,
                               r["marqo__hybrid.relevanceCutoff.parameters.stdDevFactor"])
        self.assertEqual(50, r["marqo__hybrid.relevanceCutoff.probeDepth"])
        
        # Test large stdDevFactor
        params = MeanStdParameters(stdDevFactor=10.0)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.MeanStdDev,
            parameters=params
        )
        
        r = self.index.to_vespa_query(self.hybrid_query)
        self.assertAlmostEqual(10.0,
                               r["marqo__hybrid.relevanceCutoff.parameters.stdDevFactor"])
        self.assertEqual(1000, r["marqo__hybrid.relevanceCutoff.probeDepth"])  # default

    def test_apply_in_retrieval_set_in_vespa_query(self):
        """applyInRetrieval should be passed to Vespa query when set."""
        for value in [ApplyInRetrieval.Tensor, ApplyInRetrieval.Both]:
            with self.subTest(value=value):
                self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
                    method=RelevanceCutoffMethod.GapDetection,
                    applyInRetrieval=value
                )
                r = self.index.to_vespa_query(self.hybrid_query)
                self.assertEqual(value, r["marqo__hybrid.relevanceCutoff.applyInRetrieval"])

    def test_apply_in_retrieval_both_by_default_in_vespa_query(self):
        """SearchQuery resolves applyInRetrieval=None to 'both'; the Vespa query receives 'both'."""
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            applyInRetrieval=ApplyInRetrieval.Both  # as resolved by SearchQuery validator
        )
        r = self.index.to_vespa_query(self.hybrid_query)
        self.assertEqual(ApplyInRetrieval.Both, r["marqo__hybrid.relevanceCutoff.applyInRetrieval"])


class TestSemiStructuredVespaIndexToVespaQueryCollapseFields(MarqoTestCase):

    def setUp(self):
        marqo_index = self.semi_structured_marqo_index("test_index",
                                                       collapse_fields=[CollapseField(name='parent_id')])

        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def test_hybrid_query_with_collapse_fields(self):
        marqo_query = MarqoHybridQuery(
            index_name="test_index",
            limit=10,
            offset=0,
            or_phrases=[],
            and_phrases=[],
            hybrid_parameters=HybridParameters(),
            collapse=CollapseModel(name="parent_id"),
            facets=FacetsParameters(
                fields={
                    "price": FieldFacetsConfiguration(type="number", ranges=[
                        {"from": 0, "to": 1},
                        {"from": 1, "to": 3},
                    ]),
                    "color": FieldFacetsConfiguration(type="string")
                }
            ),
            track_total_hits=True,
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)

        # assert collapsefield are populated
        self.assertEqual('parent_id', vespa_query['collapsefield'])
        self.assertEqual(1, vespa_query['collapsesize'])
        self.assertEqual('collapse-minimal-summary', vespa_query['collapse.summary'])
        self.assertTrue(vespa_query['FieldFiller.disable'])

        # assert rank profiles with '_diversity' suffix is used
        self.assertEqual(common.RANK_PROFILE_BM25 + '_diversity',
                         vespa_query['marqo__ranking.lexical.lexical'])
        self.assertEqual(common.RANK_PROFILE_EMBEDDING_SIMILARITY + '_diversity',
                         vespa_query['marqo__ranking.tensor.tensor'])
        self.assertEqual(common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY + '_diversity',
                         vespa_query['marqo__ranking.lexical.tensor'])
        self.assertEqual(common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25 + '_diversity',
                         vespa_query['marqo__ranking.tensor.lexical'])

        # assert facets query has an extra grouping
        self.assertEqual('select * from test_index where (false OR False) limit 0 | all(group(1.1) '
                         'each(group(parent_id) output(count())))\n'
                         '---MARQO-YQL-QUERY-DELIMITER---\n'
                         'select * from test_index where (false OR False) limit 0 | all( '
                         'all(group(predefined(marqo__int_fields{"price"}, bucket(0.0, 1.0), '
                         'bucket(1.0, 3.0))) max(100) order(-count()) each(group(parent_id) '
                         'output(count()))) all(group(predefined(marqo__float_fields{"price"}, '
                         'bucket(0.0, 1.0), bucket(1.0, 3.0))) max(100) order(-count()) '
                         'each(group(parent_id) output(count()))) '
                         'all(group(marqo__short_string_fields{"color"}) max(100) order(-count()) '
                         'each(group(parent_id) output(count()))) )', vespa_query['marqo__yql.facets'])

    def test_hybrid_query_without_collapse_fields(self):
        marqo_query = MarqoHybridQuery(
            index_name="test_index",
            limit=10,
            offset=0,
            or_phrases=[],
            and_phrases=[],
            hybrid_parameters=HybridParameters(),
            facets=FacetsParameters(
                fields={
                    "price": FieldFacetsConfiguration(type="number", ranges=[
                        {"from": 0, "to": 1},
                        {"from": 1, "to": 3},
                    ]),
                    "color": FieldFacetsConfiguration(type="string")
                }
            ),
            track_total_hits = True,
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)

        self.assertNotIn('collapsefield', vespa_query)
        self.assertNotIn('collapsesize', vespa_query)
        self.assertNotIn('collapse.summary', vespa_query)

        self.assertEqual(common.RANK_PROFILE_BM25,
                         vespa_query['marqo__ranking.lexical.lexical'])
        self.assertEqual(common.RANK_PROFILE_EMBEDDING_SIMILARITY,
                         vespa_query['marqo__ranking.tensor.tensor'])
        self.assertEqual(common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY,
                         vespa_query['marqo__ranking.lexical.tensor'])
        self.assertEqual(common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25,
                         vespa_query['marqo__ranking.tensor.lexical'])

        self.assertEqual('select * from test_index where (false OR False) limit 0 | all(group(1.1) '
                         'each(output(count())))\n'
                         '---MARQO-YQL-QUERY-DELIMITER---\n'
                         'select * from test_index where (false OR False) limit 0 | all( '
                         'all(group(predefined(marqo__int_fields{"price"}, bucket(0.0, 1.0), '
                         'bucket(1.0, 3.0))) max(100) order(-count()) '
                         'each(output(sum(marqo__int_fields{"price"}), '
                         'avg(marqo__int_fields{"price"}), min(marqo__int_fields{"price"}), '
                         'max(marqo__int_fields{"price"}), count()))) '
                         'all(group(predefined(marqo__float_fields{"price"}, bucket(0.0, 1.0), '
                         'bucket(1.0, 3.0))) max(100) order(-count()) '
                         'each(output(sum(marqo__float_fields{"price"}), '
                         'avg(marqo__float_fields{"price"}), min(marqo__float_fields{"price"}), '
                         'max(marqo__float_fields{"price"}), count()))) '
                         'all(group(marqo__short_string_fields{"color"}) max(100) order(-count()) '
                         'each(output(count()))) )', vespa_query['marqo__yql.facets'])

    def test_hybrid_query_with_collapse_fields_old_schema_version(self):
        """Test that collapse minimal summary params are NOT set for older schema versions."""
        marqo_index = self.semi_structured_marqo_index(
            "test_index_old",
            collapse_fields=[CollapseField(name='parent_id')],
            schema_template_version='2.24.5'  # Edge case: just below minimum 2.24.6
        )
        vespa_index = SemiStructuredVespaIndex(marqo_index)

        marqo_query = MarqoHybridQuery(
            index_name="test_index_old",
            limit=10,
            offset=0,
            or_phrases=[],
            and_phrases=[],
            hybrid_parameters=HybridParameters(),
            collapse=CollapseModel(name="parent_id"),
        )
        vespa_query = vespa_index.to_vespa_query(marqo_query)

        # Collapse field should still be set
        self.assertEqual('parent_id', vespa_query['collapsefield'])
        self.assertEqual(1, vespa_query['collapsesize'])

        # But minimal summary params should NOT be set for old schema versions
        self.assertNotIn('collapse.summary', vespa_query)
        self.assertNotIn('FieldFiller.disable', vespa_query)

    def test_hybrid_query_with_collapse_fields_and_second_phase_modifier(self):
        marqo_query = MarqoHybridQuery(
            index_name="test_index",
            limit=10,
            offset=0,
            or_phrases=[],
            and_phrases=[],
            hybrid_parameters=HybridParameters(secondPhaseModifier=True),
            collapse=CollapseModel(name="parent_id"),
            facets=FacetsParameters(
                fields={
                    "price": FieldFacetsConfiguration(type="number", ranges=[
                        {"from": 0, "to": 1},
                        {"from": 1, "to": 3},
                    ]),
                    "color": FieldFacetsConfiguration(type="string")
                }
            ),
            track_total_hits=True,
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)

        # assert collapsefield are populated
        self.assertEqual('parent_id', vespa_query['collapsefield'])
        self.assertEqual(1, vespa_query['collapsesize'])
        self.assertEqual('collapse-minimal-summary', vespa_query['collapse.summary'])
        self.assertTrue(vespa_query['FieldFiller.disable'])

        # assert rank profiles with '_diversity' suffix is used
        self.assertEqual(common.RANK_PROFILE_HYBRID_BM25_SECOND_PHASE_MODIFIERS + '_diversity',
                         vespa_query['marqo__ranking.lexical.lexical'])
        self.assertEqual(common.RANK_PROFILE_EMBEDDING_SIMILARITY + '_diversity',
                         vespa_query['marqo__ranking.tensor.tensor'])
        self.assertEqual(common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY + '_diversity',
                         vespa_query['marqo__ranking.lexical.tensor'])
        self.assertEqual(common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25 + '_diversity',
                         vespa_query['marqo__ranking.tensor.lexical'])

        # assert facets query has an extra grouping
        self.assertEqual('select * from test_index where (false OR False) limit 0 | all(group(1.1) '
                         'each(group(parent_id) output(count())))\n'
                         '---MARQO-YQL-QUERY-DELIMITER---\n'
                         'select * from test_index where (false OR False) limit 0 | all( '
                         'all(group(predefined(marqo__int_fields{"price"}, bucket(0.0, 1.0), '
                         'bucket(1.0, 3.0))) max(100) order(-count()) each(group(parent_id) '
                         'output(count()))) all(group(predefined(marqo__float_fields{"price"}, '
                         'bucket(0.0, 1.0), bucket(1.0, 3.0))) max(100) order(-count()) '
                         'each(group(parent_id) output(count()))) '
                         'all(group(marqo__short_string_fields{"color"}) max(100) order(-count()) '
                         'each(group(parent_id) output(count()))) )', vespa_query['marqo__yql.facets'])



class TestSemiStructuredVespaIndexToVespaQueryFacets(MarqoTestCase):

    def setUp(self):
        marqo_index = self.semi_structured_marqo_index("test_index")

        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def test_facets_query_with_multiple_or_phrases_and_filter_for_lexical_retriever(self):
        test_cases = [
            ("lexical", "lexical"),
            ("lexical", "tensor"),
        ]

        for retrieval_method, ranking_method in test_cases:
            with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                marqo_query = MarqoHybridQuery(
                    index_name="test_index",
                    limit=10,
                    offset=0,
                    or_phrases=["hello", "world"],
                    and_phrases=[],
                    hybrid_parameters=HybridParameters(
                        retrievalMethod=retrieval_method,
                        rankingMethod=ranking_method,
                    ),
                    filter=SearchFilter(root=EqualityTerm('a', 'n', 'a:n')),
                    facets=FacetsParameters(
                        fields={"color": FieldFacetsConfiguration(type="string")}
                    )
                )

            vespa_query = self.vespa_index.to_vespa_query(marqo_query)
            self.assertEqual('select * from test_index where (default contains "hello" OR default contains '
                             '"world") AND (((marqo__short_string_fields contains sameElement(key contains '
                             '"a", value contains "n")))) limit 0 | all( '
                             'all(group(marqo__short_string_fields{"color"}) max(100) order(-count()) '
                             'each(output(count()))) )', vespa_query['marqo__yql.facets'])


class TestSemiStructuredVespaIndexCollapseFieldAttributesToRetrieve(MarqoTestCase):

    def setUp(self):
        """Set up test fixtures with a semi-structured index that supports both tensor and lexical search."""
        # Create a semi-structured index with both lexical and tensor fields
        marqo_index = self.semi_structured_marqo_index(
            name='test_index',
            lexical_field_names=['title', 'description'], 
            tensor_field_names=['title', 'description'],
            string_array_field_names=['tags']
        )
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def test_to_vespa_query_adds_collapse_field_to_attributes_to_retrieve(self):
        """Test that collapse field is added to attributes_to_retrieve for hybrid queries"""
        hybrid_query = MarqoHybridQuery(
            index_name=self.vespa_index._marqo_index.name,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                searchableAttributesTensor=["title"],
                searchableAttributesLexical=["title"]
            ),
            or_phrases=["test query"],
            and_phrases=[],
            attributes_to_retrieve=["title", "description"],
            collapse=CollapseModel(name="parent_id"),
            limit=10,
            offset=0
        )
        
        # Call to_vespa_query to process the query
        vespa_query = self.vespa_index.to_vespa_query(hybrid_query)
        
        # Verify that collapse field was added to attributes_to_retrieve
        self.assertIn("parent_id", hybrid_query.attributes_to_retrieve)
        self.assertIn("parent_id", vespa_query["marqo__yql.tensor"])
        self.assertIn("parent_id", vespa_query["marqo__yql.lexical"])

        # Verify other expected attributes are still present
        self.assertIn("title", hybrid_query.attributes_to_retrieve)
        self.assertIn("description", hybrid_query.attributes_to_retrieve)

    def test_to_vespa_query_adds_collapse_field_to_empty_attributes_to_retrieve(self):
        """Test that collapse field is added to attributes_to_retrieve for hybrid queries"""
        hybrid_query = MarqoHybridQuery(
            index_name=self.vespa_index._marqo_index.name,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                searchableAttributesTensor=["title"],
                searchableAttributesLexical=["title"]
            ),
            or_phrases=["test query"],
            and_phrases=[],
            attributes_to_retrieve=[],
            collapse=CollapseModel(name="parent_id"),
            limit=10,
            offset=0
        )

        # Call to_vespa_query to process the query
        vespa_query = self.vespa_index.to_vespa_query(hybrid_query)

        # Verify that collapse field was added to attributes_to_retrieve
        self.assertIn("parent_id", hybrid_query.attributes_to_retrieve)
        self.assertIn("parent_id", vespa_query["marqo__yql.tensor"])
        self.assertIn("parent_id", vespa_query["marqo__yql.lexical"])

    def test_to_vespa_query_does_not_duplicate_collapse_field_in_attributes_to_retrieve(self):
        """Test that collapse field is not duplicated if already in attributes_to_retrieve"""
        hybrid_query = MarqoHybridQuery(
            index_name=self.vespa_index._marqo_index.name,
            vector_query=[0.1, 0.2, 0.3, 0.4], 
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                searchableAttributesTensor=["title"],
                searchableAttributesLexical=["title"]
            ),
            or_phrases=["test query"],
            and_phrases=[],
            attributes_to_retrieve=["title", "parent_id"],  # collapse field already present
            collapse=CollapseModel(name="parent_id"),
            limit=10,
            offset=0
        )
        
        # Call to_vespa_query to process the query
        vespa_query = self.vespa_index.to_vespa_query(hybrid_query)
        
        self.assertEqual(1, hybrid_query.attributes_to_retrieve.count("parent_id"))
        self.assertIn("parent_id", vespa_query["marqo__yql.tensor"])
        self.assertIn("parent_id", vespa_query["marqo__yql.lexical"])

    def test_to_vespa_query_does_not_add_collapse_field_to_attributes_if_not_provided(self):
        """Test that collapse field is not added to attributes_to_retrieve if not provided"""
        hybrid_query = MarqoHybridQuery(
            index_name=self.vespa_index._marqo_index.name,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                searchableAttributesTensor=["title"],
                searchableAttributesLexical=["title"]
            ),
            or_phrases=["test query"],
            and_phrases=[],
            attributes_to_retrieve=["title"],
            limit=10,
            offset=0
        )

        # Call to_vespa_query to process the query
        vespa_query = self.vespa_index.to_vespa_query(hybrid_query)

        self.assertNotIn("parent_id", hybrid_query.attributes_to_retrieve)
        self.assertNotIn("parent_id", vespa_query["marqo__yql.tensor"])
        self.assertNotIn("parent_id", vespa_query["marqo__yql.lexical"])

    def test_to_vespa_query_does_not_add_collapse_field_to_attributes_if_attributes_to_retrieve_is_none(self):
        """Test that collapse field is not added to attributes_to_retrieve if not provided"""
        hybrid_query = MarqoHybridQuery(
            index_name=self.vespa_index._marqo_index.name,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                searchableAttributesTensor=["title"],
                searchableAttributesLexical=["title"]
            ),
            or_phrases=["test query"],
            and_phrases=[],
            collapse=CollapseModel(name="parent_id"),
            limit=10,
            offset=0
        )

        # Call to_vespa_query to process the query
        vespa_query = self.vespa_index.to_vespa_query(hybrid_query)

        self.assertIsNone(hybrid_query.attributes_to_retrieve)
        self.assertNotIn("parent_id", vespa_query["marqo__yql.tensor"])
        self.assertNotIn("parent_id", vespa_query["marqo__yql.lexical"])

    def test_to_vespa_query_does_not_add_collapse_field_for_non_hybrid_queries(self):
        """Test that collapse field is not added for non-hybrid queries"""
        test_queries = [
            MarqoTensorQuery(
                index_name=self.vespa_index._marqo_index.name,
                limit=10, offset=0,
                attributes_to_retrieve=["title", "description"],
                vector_query=[0.1, 0.2, 0.3, 0.4]
            ),
            MarqoLexicalQuery(
                index_name=self.vespa_index._marqo_index.name,
                limit=10, offset=0,
                or_phrases=["test query"], and_phrases=[],
                attributes_to_retrieve=["title", "description"]
            )
        ]

        for marqo_query in test_queries:
            with self.subTest(type=type(marqo_query)):
                # MarqoQuery doesn't have collapse_field_name, so this should not affect attributes_to_retrieve
                vespa_query = self.vespa_index.to_vespa_query(marqo_query)

                self.assertNotIn("parent_id", vespa_query["yql"])
                self.assertNotIn("parent_id", marqo_query.attributes_to_retrieve)

class TestSemiStructuredVespaIndexToVespaQueryCollapseSortBy(MarqoTestCase):
    """Tests for the collapse sort_by code path in to_vespa_query.

    When collapse.sort_by is set and should_execute_sort() is True, the vespa query should:
    1. Override lexical ranking to 'collapse_to_sort_value'
    2. Set query_features with marqo__collapse_sort_weights (asc → -1, desc → 1)
    3. Set hits to COLLAPSE_SORT_BY_QUERY_LIMIT (9999)
    4. Optionally set ranking.matching.numThreadsPerSearch

    When should_execute_sort() is False, the query should use standard diversity ranking
    and not include any collapse sort parameters.
    """

    def setUp(self):
        marqo_index = self.semi_structured_marqo_index(
            "test_index",
            collapse_fields=[CollapseField(name='parent_id')]
        )
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def _build_query(self, collapse, hybrid_parameters=None):
        return MarqoHybridQuery(
            index_name="test_index", limit=10, offset=0,
            or_phrases=[], and_phrases=[],
            hybrid_parameters=hybrid_parameters or HybridParameters(),
            collapse=collapse,
        )

    def _make_collapse(self, field_name="price", order="asc", execute=False, num_threads=None):
        collapse = CollapseModel(
            name="parent_id",
            sort_by=CollapseSortBy(
                fields=[CollapseSortByField(fieldName=field_name, order=order)],
                numThreadsPerSearch=num_threads,
            )
        )
        if execute:
            collapse.sort_by.enable_execute_sort()
        return collapse

    def test_collapse_sort_by_executed(self):
        """When should_execute_sort() is True, verify sort-related keys in the full query."""
        # (name, field, order, num_threads, expected_weight)
        cases = [
            ('asc',              'price', 'asc',  None, {'price': -1}),
            ('desc',             'price', 'desc', None, {'price': 1}),
            ('desc_with_threads', 'price', 'desc', 4,   {'price': 1}),
            ('different_field',  'cost',  'asc',  None, {'cost': -1}),
        ]

        for name, field, order, threads, expected_weight in cases:
            with self.subTest(case=name):
                self.maxDiff = None
                collapse = self._make_collapse(field_name=field, order=order, execute=True, num_threads=threads)
                vespa_query = self.vespa_index.to_vespa_query(self._build_query(collapse))

                # Sort-by specific assertions
                self.assertEqual('collapse_to_sort_value', vespa_query['marqo__ranking.lexical.lexical'])
                self.assertEqual(expected_weight, vespa_query['query_features']['marqo__collapse_sort_weights'])
                self.assertEqual(9999, vespa_query['hits'])

                # numThreadsPerSearch
                if threads:
                    self.assertEqual(threads, vespa_query['ranking.matching.numThreadsPerSearch'])
                else:
                    self.assertNotIn('ranking.matching.numThreadsPerSearch', vespa_query)

                # Collapse field params always present
                self.assertEqual('parent_id', vespa_query['collapsefield'])
                self.assertEqual(1, vespa_query['collapsesize'])

    def test_collapse_sort_by_not_executed(self):
        """When should_execute_sort() is False, no sort params are added; diversity ranking is used."""
        collapse = self._make_collapse(execute=False)
        vespa_query = self.vespa_index.to_vespa_query(self._build_query(collapse))

        self.assertEqual('bm25_diversity', vespa_query['marqo__ranking.lexical.lexical'])
        self.assertNotIn('marqo__collapse_sort_weights', vespa_query.get('query_features', {}))
        self.assertEqual(10, vespa_query['hits'])
        self.assertNotIn('ranking.matching.numThreadsPerSearch', vespa_query)
        self.assertEqual('parent_id', vespa_query['collapsefield'])


class TestSemiStructuredCustomScoreRerankToVespaQuery(unittest.TestCase):
    """Unit tests for custom score reranking: rank() construction, facets and relevance_cutoff unchanged."""

    def setUp(self):
        marqo_index = self._create_semi_structured_marqo_index(
            name='test_index',
            lexical_field_names=['title', 'description'],
            tensor_field_names=['title', 'description'],
        )
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def _create_semi_structured_marqo_index(
        self,
        name: str,
        lexical_field_names: List[str],
        tensor_field_names: List[str],
        version: str = '2.16.0',
        distance_metric=DistanceMetric.Angular,
    ) -> SemiStructuredMarqoIndex:
        lexical_fields = []
        for field_name in lexical_field_names:
            lexical_fields.append(
                Field(
                    name=field_name,
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                    lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}',
                    filter_field_name=f'{field_name}_filter'
                )
            )
        tensor_fields = []
        for field_name in tensor_field_names:
            tensor_fields.append(
                TensorField(
                    name=field_name,
                    embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
                    chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}'
                )
            )
        return SemiStructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all-MiniLM-L6-v2'),
            normalize_embeddings=True,
            distance_metric=distance_metric,
            vector_numeric_type='float',
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version=version,
            created_at=time.time(),
            updated_at=time.time(),
            text_preprocessing=TextPreProcessing(
                split_length=2, split_overlap=0, split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(patch_method=None),
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=50,
            lexical_fields=lexical_fields,
            tensor_fields=tensor_fields,
            string_array_fields=[],
        )

    def _hybrid_query(self, score_modifiers=None, facets=None, relevance_cutoff=None, hybrid_parameters=None):
        # Set default hybrid parameters to use RRF
        if hybrid_parameters == None:
            hybrid_parameters = HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
            )

        return MarqoHybridQuery(
            index_name=self.vespa_index._marqo_index.name,
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['search'],
            and_phrases=[],
            hybrid_parameters=hybrid_parameters,
            score_modifiers=score_modifiers,
            facets=facets,
            relevance_cutoff=relevance_cutoff,
        )

    def test_global_score_modifiers_with_attributes_to_retrieve_does_not_force_select_star(self):
        """Global-only scoreModifiers must not widen hybrid YQL to select * (performance regression guard)."""
        hybrid_parameters = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            searchableAttributesLexical=['title'],
            searchableAttributesTensor=['title'],
        )
        marqo_query = MarqoHybridQuery(
            index_name=self.vespa_index._marqo_index.name,
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['search'],
            and_phrases=[],
            hybrid_parameters=hybrid_parameters,
            attributes_to_retrieve=['title'],
            score_modifiers=[
                ScoreModifier(field='popularity', weight=1.0, type=ScoreModifierType.Add),
            ],
        )
        q = self.vespa_index._get_base_vespa_hybrid_query(marqo_query)
        self.assertNotIn('select * from', q['marqo__yql.tensor'])
        self.assertNotIn('select * from', q['marqo__yql.lexical'])

    def test_hybrid_query_with_custom_score_rerank_full_query(self):
        """
        Full generated Vespa query with custom score rerank and with collapse, facets, track_total_hits
        (BM25 aggregate mult + closeness aggregate add).
        """
        marqo_index = MarqoTestCase.semi_structured_marqo_index(
            'test_index',
            collapse_fields=[CollapseField(name='parent_id')],
            lexical_field_names=('title', 'description'),
            tensor_field_names=('title', 'description'),
        )
        vespa_index = SemiStructuredVespaIndex(marqo_index)
        marqo_query = MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=['search'],
            and_phrases=[],
            hybrid_parameters=HybridParameters(retrievalMethod=RetrievalMethod.Disjunction),
            collapse=CollapseModel(name='parent_id'),
            facets=FacetsParameters(
                fields={
                    'price': FieldFacetsConfiguration(
                        type='number',
                        ranges=[{'from': 0, 'to': 1}, {'from': 1, 'to': 3}],
                    ),
                    'color': FieldFacetsConfiguration(type='string'),
                }
            ),
            track_total_hits=True,
            score_modifiers=[
                ScoreModifier(
                    field=f'{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_sum',
                    weight=2.0,
                    type=ScoreModifierType.Multiply,
                ),
                ScoreModifier(
                    field=f'{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_sum',
                    weight=0.5,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_query = vespa_index.to_vespa_query(marqo_query)

        self.assertEqual('parent_id', vespa_query['collapsefield'])
        self.assertEqual(1, vespa_query['collapsesize'])
        self.assertEqual('collapse-minimal-summary', vespa_query['collapse.summary'])
        self.assertTrue(vespa_query['FieldFiller.disable'])

        self.assertEqual(common.RANK_PROFILE_BM25 + '_diversity',
                         vespa_query['marqo__ranking.lexical.lexical'])
        self.assertEqual(common.RANK_PROFILE_EMBEDDING_SIMILARITY + '_diversity',
                         vespa_query['marqo__ranking.tensor.tensor'])
        self.assertEqual(common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY + '_diversity',
                         vespa_query['marqo__ranking.lexical.tensor'])
        self.assertEqual(common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25 + '_diversity',
                         vespa_query['marqo__ranking.tensor.lexical'])

        # Assert facets query does not have extra rank() term for custom score reranker
        self.assertEqual(
            'select * from test_index where (default contains "search" OR (({targetHits:10, approximate:True, '
            'hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) OR '
            '({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}nearestNeighbor('
            'marqo__embeddings_description, marqo__query_embedding)))) limit 0 | all(group(1.1) '
            'each(group(parent_id) output(count())))\n'
            '---MARQO-YQL-QUERY-DELIMITER---\n'
            'select * from test_index where (default contains "search" OR (({targetHits:10, approximate:True, '
            'hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) OR '
            '({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}nearestNeighbor('
            'marqo__embeddings_description, marqo__query_embedding)))) limit 0 | all( '
            'all(group(predefined(marqo__int_fields{"price"}, bucket(0.0, 1.0), bucket(1.0, 3.0))) max(100) '
            'order(-count()) each(group(parent_id) output(count()))) '
            'all(group(predefined(marqo__float_fields{"price"}, bucket(0.0, 1.0), bucket(1.0, 3.0))) max(100) '
            'order(-count()) each(group(parent_id) output(count()))) '
            'all(group(marqo__short_string_fields{"color"}) max(100) order(-count()) '
            'each(group(parent_id) output(count()))) )',
            vespa_query['marqo__yql.facets'],
        )

        # Assert lexical yql remains the same
        self.assertEqual(
            'select * from test_index where (weakAnd(default contains "search"))',
            vespa_query['marqo__yql.lexical'],
        )
        # Assert tensor yql has the extra weakAnd (for bm25 sum)
        self.assertEqual(
            'select * from test_index where rank((({targetHits:10, approximate:True, '
            'hnsw.exploreAdditionalHits:1990}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) OR '
            '({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}nearestNeighbor('
            'marqo__embeddings_description, marqo__query_embedding))), weakAnd(default contains "search"))',
            vespa_query['marqo__yql.tensor'],
        )

        self.assertEqual('hybrid_custom_searcher', vespa_query['ranking'])
        self.assertEqual(10, vespa_query['hits'])
        self.assertEqual(0, vespa_query['offset'])
        self.assertEqual('test_index', vespa_query['model_restrict'])
        self.assertEqual('all-non-vector-summary', vespa_query['presentation.summary'])
        self.assertEqual(10, vespa_query['ranking.rerankCount'])
        self.assertEqual(RetrievalMethod.Disjunction, vespa_query['marqo__hybrid.retrievalMethod'])
        self.assertEqual(RankingMethod.RRF, vespa_query['marqo__hybrid.rankingMethod'])
        self.assertEqual(0.5, vespa_query['marqo__hybrid.alpha'])
        self.assertEqual(60, vespa_query['marqo__hybrid.rrf_k'])

        # Assert hasRankingLexical and hasRankingVector flags are set
        self.assertTrue(vespa_query['marqo__hasRankingLexical'])
        self.assertTrue(vespa_query['marqo__hasRankingVector'])
        self.assertEqual(DistanceMetric.Angular.value, vespa_query['marqo__custom_score_closeness_distance_metric'])

        qf = vespa_query['query_features']
        self.assertEqual({'bm25_sum': 2.0}, qf['marqo__custom_score_mult_weights_global'])
        self.assertEqual({'closeness_retrieval_vector_sum': 0.5}, qf['marqo__custom_score_add_weights_global'])
        self.assertEqual(
            {'marqo__lexical_description': 1, 'marqo__lexical_title': 1},
            qf['marqo__fields_to_rank_lexical'],
        )
        self.assertEqual(
            {'marqo__embeddings_description': 1, 'marqo__embeddings_title': 1},
            qf['marqo__fields_to_rank_tensor'],
        )
        self.assertEqual([0.1, 0.2, 0.3, 0.4], qf['marqo__query_embedding'])

    def test_get_fields_to_bm25_rerank_by(self):
        """_get_fields_to_bm25_rerank_by returns fields or ['*'] for aggregate (internal suffix keys)."""
        self.assertEqual(
            self.vespa_index._get_fields_to_bm25_rerank_by({'bm25_field_title'}),
            ['title'],
        )
        self.assertEqual(
            self.vespa_index._get_fields_to_bm25_rerank_by(
                {'bm25_field_title', 'bm25_field_description'}
            ),
            ['description', 'title'],
        )
        self.assertEqual(
            self.vespa_index._get_fields_to_bm25_rerank_by({'bm25_sum'}),
            ['*'],
        )
        self.assertEqual(
            self.vespa_index._get_fields_to_bm25_rerank_by(
                {'bm25_field_title', 'bm25_max'}
            ),
            ['*'],
        )

    def test_simplify_bm25_extra_fields_for_rank(self):
        """_simplify_bm25_extra_fields_for_rank removes redundancy with main lexical term."""
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank([], None),
            [],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank([], ['title']),
            [],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['title'], None),
            [],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['*'], None),
            [],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['*'], ['title']),
            ['*'],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(
                ['title', 'description'], ['title', 'description']
            ),
            [],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(
                ['title', 'description'], ['title']
            ),
            ['description'],
        )
        self.assertEqual(
            self.vespa_index._simplify_bm25_extra_fields_for_rank(['description'], ['title']),
            ['description'],
        )

    def test_get_fields_to_closeness_rerank_by(self):
        """_get_fields_to_closeness_rerank_by returns tensor fields or all for aggregate (internal suffix keys)."""
        self.assertEqual(
            self.vespa_index._get_fields_to_closeness_rerank_by(
                {'closeness_retrieval_vector_field_title'}
            ),
            ['title'],
        )
        self.assertCountEqual(
            self.vespa_index._get_fields_to_closeness_rerank_by(
                {'closeness_retrieval_vector_sum'}
            ),
            ['description', 'title'],
        )

    def test_get_lexical_contains_term_with_attributes_to_search(self):
        """_get_lexical_contains_term with is_ranking_term=True uses attributes_to_search."""
        term = self.vespa_index._get_lexical_contains_term(
            'hello', attributes_to_search=['*'], is_ranking_term=True
        )
        self.assertEqual(term, 'default contains "hello"')
        term = self.vespa_index._get_lexical_contains_term(
            'hello', attributes_to_search=['title'], is_ranking_term=True
        )
        self.assertIn('marqo__lexical_title', term)
        self.assertIn('hello', term)

    def test_get_individual_field_tensor_search_terms_non_ranking_includes_target_hits(self):
        """Derives from query and term includes targetHits."""
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
        self.assertIn('marqo__embeddings', terms[0])

    def test_generate_or_terms_ranking_term_no_target_hits(self):
        """_generate_or_terms with is_ranking_term=True returns weakAnd without targetHits."""
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
            q, is_ranking_term=True, attributes_to_search=['title']
        )
        self.assertIn('weakAnd', result)
        self.assertNotIn('targetHits', result)
        self.assertIn('search', result)

    def test_get_lexical_search_term_ranking_term(self):
        """_get_lexical_search_term with is_ranking_term=True uses attributes_to_search, no targetHits."""
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
            q, is_ranking_term=True, attributes_to_search=['title']
        )
        self.assertNotEqual(result, 'false')
        self.assertNotIn('targetHits', result)
        self.assertIn('hello', result)

    def test_get_lexical_search_term_ranking_term_empty_attributes_returns_empty(self):
        """When is_ranking_term=True and attributes_to_search is empty, return "" so rank() is skipped."""
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
            q, is_ranking_term=True, attributes_to_search=[]
        )
        self.assertEqual(result, "")

    def test_get_lexical_search_term_ranking_term_no_lexical_fields_survive_returns_empty(self):
        """When is_ranking_term=True and no attributes have lexical_field_name (e.g. tags), return ""."""
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
        # 'tags' is a string-array field with no lexical_field_name in this index
        result = self.vespa_index._get_lexical_search_term(
            q, is_ranking_term=True, attributes_to_search=['tags']
        )
        self.assertEqual(result, "")

    def test_get_lexical_search_term_ranking_term_valid_attributes_returns_weak_and(self):
        """When is_ranking_term=True and attributes survive, return valid weakAnd (no invalid YQL)."""
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
            or_phrases=['hello', 'world'],
            and_phrases=[],
            hybrid_parameters=hybrid_params,
        )
        result = self.vespa_index._get_lexical_search_term(
            q, is_ranking_term=True, attributes_to_search=['title']
        )
        self.assertIn('weakAnd', result)
        self.assertIn('hello', result)
        self.assertIn('world', result)
        # Must not be invalid YQL (no empty slots like weakAnd(, , ))
        self.assertNotRegex(result, r'weakAnd\(\s*,\s*,')

    def test_generate_or_terms_ranking_term_all_empty_terms_returns_empty(self):
        """_generate_or_terms with is_ranking_term and all empty terms returns "" (defensive guard)."""
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
            or_phrases=['a', 'b'],
            and_phrases=[],
            hybrid_parameters=hybrid_params,
        )
        # attributes_to_search=['tags'] yields no lexical fields, so each term is ""
        result = self.vespa_index._generate_or_terms(
            q, is_ranking_term=True, attributes_to_search=['tags']
        )
        self.assertEqual(result, "")

    def test_hybrid_query_with_bm25_custom_score_includes_rank_in_yql(self):
        """With BM25 custom score modifiers, lexical and tensor YQL must wrap in rank() with extra BM25 term
        if not already included in the main lexical query."""
        marqo_query = self._hybrid_query(
            hybrid_parameters=HybridParameters(
                # Using description so title has to be used in rank()
                searchableAttributesLexical=["description"]
            ),
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        tensor_yql = vespa_query.get('marqo__yql.tensor', '')
        self.assertIn('rank(', lexical_yql, msg='Lexical YQL must use rank() when BM25 custom score is present')
        self.assertIn('rank(', tensor_yql, msg='Tensor YQL must use rank() when BM25 custom score is present')
        query_features = vespa_query.get('query_features', {})
        self.assertIn('marqo__custom_score_add_weights_global', str(query_features))

    def test_hybrid_query_with_only_closeness_custom_score_no_extra_rank_term(self):
        """With only closeness_retrieval_vector custom score (no BM25), no extra rank() term for BM25."""
        marqo_query = self._hybrid_query(
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        tensor_yql = vespa_query.get('marqo__yql.tensor', '')
        # rank() may still appear for lexical/tensor fusion (e.g. rank(tensor_term, lexical_term)), but we must not
        # add an extra BM25 ranking term. So the lexical term itself should not be rank(..., extra_bm25).
        # With only closeness, bm25_fields is empty so no extra_terms. So lexical_yql and tensor_yql
        # are the same as without custom score (no second rank for bm25). So we only check that custom score
        # query inputs are present and that we did not add a contains term for bm25 rerank.
        query_features = vespa_query.get('query_features', {})
        self.assertIn('marqo__custom_score_add_weights_global', str(query_features))
        # Lexical YQL with Disjunction is just the retrieval lexical term (no rank with extra bm25).
        self.assertIn('default contains "search"', lexical_yql)
        self.assertNotIn('rank(', lexical_yql, msg='No rank for BM25 when only closeness is used')

    def test_facets_query_unchanged_with_custom_score_modifiers(self):
        """Facets YQL must be identical with and without custom score modifiers."""
        facets = FacetsParameters(fields={"title": FieldFacetsConfiguration(type="string", maxResults=5)})
        query_without = self._hybrid_query(facets=facets)
        query_with = self._hybrid_query(
            facets=facets,
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_without = self.vespa_index.to_vespa_query(query_without)
        vespa_with = self.vespa_index.to_vespa_query(query_with)
        self.assertIn('marqo__yql.facets', vespa_without)
        self.assertIn('marqo__yql.facets', vespa_with)
        self.assertEqual(
            vespa_without['marqo__yql.facets'],
            vespa_with['marqo__yql.facets'],
            msg='Facets query must be unchanged by custom score modifiers',
        )

    def test_relevance_cutoff_probe_lexical_yql_excludes_custom_score_extra_rank_terms(self):
        """With relevance_cutoff and BM25 custom score, marqo__yql.lexical.probe is sent and has no extra rank() terms; main lexical has them."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=RelativeMaxScoreParameters(relative_score_factor=0.5),
            probe_depth=100,
        )
        # Use searchableAttributesLexical=["description"] so main lexical is description-only;
        # BM25 custom score is for title, so an extra rank(..., title_term) is added to main.
        marqo_query = self._hybrid_query(
            relevance_cutoff=relevance_cutoff,
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        marqo_query.hybrid_parameters.searchableAttributesLexical = ["description"]
        marqo_query.hybrid_parameters.searchableAttributesTensor = ["title", "description"]
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)
        self.assertIn(
            "marqo__yql.lexical.probe",
            vespa_query,
            msg="Probe lexical YQL must be sent when relevance_cutoff is set",
        )
        main_lexical = vespa_query.get("marqo__yql.lexical", "")
        probe_lexical = vespa_query.get("marqo__yql.lexical.probe", "")
        self.assertIn("rank(", main_lexical, msg="Main lexical YQL must include extra rank() term for BM25 custom score")
        self.assertNotEqual(
            main_lexical,
            probe_lexical,
            msg="Probe must not include custom-score extra rank() terms; it must be base lexical only",
        )
        # Probe must be the base lexical (no second rank for BM25); main is rank(base, extra).
        self.assertNotIn("rank(", probe_lexical,
                         "Probe YQL must not contain rank from custom score")
        self.assertNotIn("marqo__lexical_title", probe_lexical,
                         "Probe YQL must not contain anything related to title")

    def test_relevance_cutoff_params_unchanged_with_custom_score_modifiers(self):
        """Relevance cutoff query params must be identical with and without custom score modifiers."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=RelativeMaxScoreParameters(relative_score_factor=0.5),
            probe_depth=100,
        )
        query_without = self._hybrid_query(relevance_cutoff=relevance_cutoff)
        query_with = self._hybrid_query(
            relevance_cutoff=relevance_cutoff,
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_without = self.vespa_index.to_vespa_query(query_without)
        vespa_with = self.vespa_index.to_vespa_query(query_with)
        for key in ('marqo__hybrid.relevanceCutoff.method', 'marqo__hybrid.relevanceCutoff.probeDepth',
                    'marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor'):
            if key in vespa_without:
                self.assertIn(key, vespa_with, msg=f'{key} should be present when relevance_cutoff is set')
                self.assertEqual(
                    vespa_without[key], vespa_with[key],
                    msg=f'Relevance cutoff param {key} must be unchanged by custom score modifiers',
                )

    def test_hybrid_query_with_closeness_sets_distance_metric_in_query(self):
        """When closeness custom score is used, query must include marqo__custom_score_closeness_distance_metric with index's metric."""
        marqo_query = self._hybrid_query(
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_query = self.vespa_index.to_vespa_query(marqo_query)
        self.assertIn("marqo__custom_score_closeness_distance_metric", vespa_query)
        self.assertEqual(
            vespa_query["marqo__custom_score_closeness_distance_metric"],
            self.vespa_index._marqo_index.distance_metric.value,
            msg="Query must pass index distance metric so searcher can decide whether to min-max normalize (dotproduct only)",
        )
        self.assertEqual(vespa_query["marqo__custom_score_closeness_distance_metric"], "angular")

    def test_hybrid_query_with_closeness_and_dotproduct_index_sets_dotproduct(self):
        """When index uses dot product and closeness custom score, query must pass distance_metric 'dotproduct'."""
        marqo_index = self._create_semi_structured_marqo_index(
            name="test_dotproduct",
            lexical_field_names=["title", "description"],
            tensor_field_names=["title", "description"],
            distance_metric=DistanceMetric.DotProduct,
        )
        vespa_index = SemiStructuredVespaIndex(marqo_index)
        marqo_query = MarqoHybridQuery(
            index_name=marqo_index.name,
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=["search"],
            and_phrases=[],
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF,
                alpha=0.5,
                rrfK=60,
            ),
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}closeness_retrieval_vector_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_query = vespa_index.to_vespa_query(marqo_query)
        self.assertEqual(vespa_query["marqo__custom_score_closeness_distance_metric"], "dotproduct")

    def test_expose_pre_rerank_score_set_when_global_score_modifiers_present(self):
        """marqo__expose_pre_rerank_score is True when global (non-custom) score modifiers are used."""
        marqo_query = self._hybrid_query(
            score_modifiers=[
                ScoreModifier(field='popularity', weight=1.0, type=ScoreModifierType.Add),
            ],
        )
        vespa_query = self.vespa_index._get_base_vespa_hybrid_query(marqo_query)
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
        vespa_query = self.vespa_index._get_base_vespa_hybrid_query(marqo_query)
        self.assertTrue(vespa_query.get('marqo__expose_pre_rerank_score'))

    def test_expose_pre_rerank_score_absent_when_no_score_modifiers(self):
        """marqo__expose_pre_rerank_score is not set when no score modifiers are used."""
        marqo_query = self._hybrid_query()
        vespa_query = self.vespa_index._get_base_vespa_hybrid_query(marqo_query)
        self.assertNotIn('marqo__expose_pre_rerank_score', vespa_query)


class TestAppendCustomScoreRerankTerms(unittest.TestCase):
    """Unit tests for _append_custom_score_rerank_terms."""

    def setUp(self):
        marqo_index = self._create_semi_structured_marqo_index(
            name='test_index',
            lexical_field_names=['title', 'description'],
            tensor_field_names=['title', 'description'],
        )
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)

    def _create_semi_structured_marqo_index(
        self,
        name: str,
        lexical_field_names: List[str],
        tensor_field_names: List[str],
        version: str = '2.16.0',
    ) -> SemiStructuredMarqoIndex:
        lexical_fields = []
        for field_name in lexical_field_names:
            lexical_fields.append(
                Field(
                    name=field_name,
                    type=FieldType.Text,
                    features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                    lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}',
                    filter_field_name=f'{field_name}_filter'
                )
            )
        tensor_fields = []
        for field_name in tensor_field_names:
            tensor_fields.append(
                TensorField(
                    name=field_name,
                    embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
                    chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}'
                )
            )
        return SemiStructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all-MiniLM-L6-v2'),
            normalize_embeddings=True,
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type='float',
            hnsw_config=HnswConfig(ef_construction=100, m=16),
            marqo_version=version,
            created_at=time.time(),
            updated_at=time.time(),
            text_preprocessing=TextPreProcessing(
                split_length=2, split_overlap=0, split_method=TextSplitMethod.Sentence
            ),
            image_preprocessing=ImagePreProcessing(patch_method=None),
            treat_urls_and_pointers_as_images=False,
            treat_urls_and_pointers_as_media=False,
            filter_string_max_length=50,
            lexical_fields=lexical_fields,
            tensor_fields=tensor_fields,
            string_array_fields=[],
        )

    def _hybrid_query(
        self,
        ranking_method=RankingMethod.RRF,
        searchable_attributes_lexical=None,
        retrieval_method=RetrievalMethod.Disjunction,
    ):
        return MarqoHybridQuery(
            index_name=self.vespa_index._marqo_index.name,
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2],
            or_phrases=['q'],
            and_phrases=[],
            hybrid_parameters=HybridParameters(
                retrievalMethod=retrieval_method,
                rankingMethod=ranking_method,
                searchableAttributesLexical=searchable_attributes_lexical,
            ),
        )

    def test_returns_unchanged_when_custom_score_keys_empty(self):
        """Empty custom_score_keys returns (lexical_term, tensor_term) unchanged."""
        marqo_query = self._hybrid_query()
        lexical_term, tensor_term = self.vespa_index._append_custom_score_rerank_terms(
            marqo_query, 'base_lex', 'base_tensor', set()
        )
        self.assertEqual(lexical_term, 'base_lex')
        self.assertEqual(tensor_term, 'base_tensor')

    def test_returns_unchanged_when_ranking_method_not_rrf(self):
        """When ranking method is not RRF, terms are returned unchanged."""
        marqo_query = self._hybrid_query(
            ranking_method=RankingMethod.Lexical,
            retrieval_method=RetrievalMethod.Lexical,
        )
        custom_score_keys = {'bm25_field_title'}
        lexical_term, tensor_term = self.vespa_index._append_custom_score_rerank_terms(
            marqo_query, 'base_lex', 'base_tensor', custom_score_keys
        )
        self.assertEqual(lexical_term, 'base_lex')
        self.assertEqual(tensor_term, 'base_tensor')

    def test_appends_rank_terms_when_rrf_and_bm25_fields(self):
        """With RRF and BM25 custom score keys, both terms get an extra rank(..., extra_bm25) term."""
        marqo_query = self._hybrid_query(
            ranking_method=RankingMethod.RRF,
            searchable_attributes_lexical=['description'],  # title not in main lexical -> extra term for title
        )
        custom_score_keys = {'bm25_field_title'}
        lexical_term, tensor_term = self.vespa_index._append_custom_score_rerank_terms(
            marqo_query, 'base_lex', 'base_tensor', custom_score_keys
        )
        self.assertIn('rank(', lexical_term, 'Lexical term should wrap in rank() with extra BM25 term')
        self.assertIn('rank(', tensor_term, 'Tensor term should wrap in rank() with extra BM25 term')
        self.assertTrue(lexical_term.startswith('rank(base_lex,'), lexical_term)
        self.assertTrue(tensor_term.startswith('rank(base_tensor,'), tensor_term)

    def test_returns_unchanged_when_only_closeness_keys(self):
        """When custom score keys contain only closeness (no BM25), no extra rank terms are added."""
        marqo_query = self._hybrid_query(ranking_method=RankingMethod.RRF)
        custom_score_keys = {'closeness_retrieval_vector_field_title'}
        lexical_term, tensor_term = self.vespa_index._append_custom_score_rerank_terms(
            marqo_query, 'base_lex', 'base_tensor', custom_score_keys
        )
        self.assertEqual(lexical_term, 'base_lex')
        self.assertEqual(tensor_term, 'base_tensor')


class TestLexicalOperandSemiStructured(TestSemiStructuredVespaIndexToVespaQuery):
    """Tests for the lexicalOperand parameter in semi-structured index."""

    # Sentinel object used as the default value for or_phrases so we can distinguish
    # "caller didn't pass or_phrases" from "caller explicitly passed None or []".
    # Using None would be ambiguous since None/[] are valid values a caller might pass.
    _SENTINEL = object()

    def _make_hybrid_query(self, lexical_operand=None, rerank_depth_lexical=None,
                           or_phrases=_SENTINEL, and_phrases=None, relevance_cutoff=None,
                           score_modifiers=None, searchable_attributes_lexical=None):
        """Helper to create a MarqoHybridQuery with the given lexicalOperand."""
        if or_phrases is self._SENTINEL:
            or_phrases = ['search']
        hp = HybridParameters(
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            lexicalOperand=lexical_operand,
            rerankDepthLexical=rerank_depth_lexical,
            searchableAttributesLexical=searchable_attributes_lexical,
        )
        return MarqoHybridQuery(
            index_name='test_index',
            limit=10,
            offset=0,
            vector_query=[0.1, 0.2, 0.3, 0.4],
            or_phrases=or_phrases,
            and_phrases=and_phrases or [],
            hybrid_parameters=hp,
            relevance_cutoff=relevance_cutoff,
            score_modifiers=score_modifiers,
        )

    def test_lexical_operand_or_uses_or_join(self):
        """When lexicalOperand='or', OR terms should use OR join."""
        q = self._make_hybrid_query(lexical_operand='or', or_phrases=['term1', 'term2'])
        result = self.vespa_index._generate_or_terms(q)
        self.assertEqual(result, 'default contains "term1" OR default contains "term2"')

    def test_lexical_operand_and_uses_and_join(self):
        """When lexicalOperand='and', OR terms should use AND join."""
        q = self._make_hybrid_query(lexical_operand='and', or_phrases=['term1', 'term2'])
        result = self.vespa_index._generate_or_terms(q)
        self.assertEqual(result, 'default contains "term1" AND default contains "term2"')

    def test_lexical_operand_weakand_uses_weakand(self):
        """When lexicalOperand='weakAnd', OR terms should use weakAnd."""
        q = self._make_hybrid_query(lexical_operand='weakAnd')
        result = self.vespa_index._generate_or_terms(q)
        self.assertEqual(result, 'weakAnd(default contains "search")')

    def test_lexical_operand_none_uses_default_logic(self):
        """When lexicalOperand=None, default logic applies (weakAnd for no modifiers)."""
        q = self._make_hybrid_query()
        self.assertIsNone(q.hybrid_parameters.lexicalOperand)
        result = self.vespa_index._generate_or_terms(q)
        self.assertEqual(result, 'weakAnd(default contains "search")')

    def test_lexical_operand_or_with_rerank_depth(self):
        """When lexicalOperand='or' with rerankDepthLexical, should use OR (no targetHits wrapping)."""
        q = self._make_hybrid_query(lexical_operand='or', rerank_depth_lexical=100,
                                    or_phrases=['term1', 'term2'])
        result = self.vespa_index._generate_or_terms(q)
        self.assertEqual(result, 'default contains "term1" OR default contains "term2"')

    def test_lexical_operand_weakand_with_rerank_depth(self):
        """When lexicalOperand='weakAnd' with rerankDepthLexical, should use weakAnd with targetHits."""
        q = self._make_hybrid_query(lexical_operand='weakAnd', rerank_depth_lexical=100,
                                    or_phrases=['term1', 'term2'])
        result = self.vespa_index._generate_or_terms(q)
        self.assertEqual(result, '{targetHits:100}weakAnd(default contains "term1", default contains "term2")')

    def test_lexical_operand_in_full_vespa_query(self):
        """lexicalOperand='or' should produce OR-based lexical YQL in full vespa query."""
        q = self._make_hybrid_query(lexical_operand='or', or_phrases=['hello', 'world'])
        vespa_query = self.vespa_index.to_vespa_query(q)
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected = ('select * from test_index where '
                    '(default contains "hello" OR default contains "world")')
        self.assertEqual(expected, lexical_yql)

    def test_relevance_cutoff_lexical_operand_overrides_for_probe(self):
        """relevanceCutoff.lexicalOperand overrides the outer lexicalOperand for the probe query."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=RelativeMaxScoreParameters(relativeScoreFactor=0.5),
            lexicalOperand='or'
        )
        q = self._make_hybrid_query(
            lexical_operand='weakAnd',
            or_phrases=['hello', 'world'],
            relevance_cutoff=relevance_cutoff
        )
        vespa_query = self.vespa_index.to_vespa_query(q)

        # Main lexical YQL should use weakAnd (outer operand)
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected_lexical = ('select * from test_index where '
                            '(weakAnd(default contains "hello", default contains "world"))')
        self.assertEqual(expected_lexical, lexical_yql)

        # Probe YQL should use OR (relevanceCutoff operand override)
        probe_yql = vespa_query.get('marqo__yql.lexical.probe', '')
        expected_probe = ('select * from test_index where '
                          '(default contains "hello" OR default contains "world")')
        self.assertEqual(expected_probe, probe_yql)

    def test_sentence_query_and_outside_or_inside_probe(self):
        """For query 'this is a sentence' with AND as outer operand and OR in relevanceCutoff,
        the main lexical YQL should use AND and the probe lexical YQL should use OR."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=RelativeMaxScoreParameters(relativeScoreFactor=0.5),
            lexicalOperand='or'
        )
        q = self._make_hybrid_query(
            lexical_operand='and',
            or_phrases=['this', 'is', 'a', 'sentence'],
            relevance_cutoff=relevance_cutoff
        )
        vespa_query = self.vespa_index.to_vespa_query(q)

        # Main lexical YQL should use AND to join terms
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected_lexical_yql = ('select * from test_index where '
                                '(default contains "this" AND default contains "is" '
                                'AND default contains "a" AND default contains "sentence")')
        self.assertEqual(expected_lexical_yql, lexical_yql)

        # Probe YQL should use OR to join terms
        probe_yql = vespa_query.get('marqo__yql.lexical.probe', '')
        expected_lexical_yql = ('select * from test_index where (default contains "this" OR default contains "is" '
                                'OR default contains "a" OR default contains "sentence")')
        self.assertEqual(expected_lexical_yql, probe_yql)

    def test_relevance_cutoff_no_lexical_operand_no_probe_yql(self):
        """When relevanceCutoff.lexicalOperand is None, no separate probe YQL is set."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=RelativeMaxScoreParameters(relativeScoreFactor=0.5),
        )
        q = self._make_hybrid_query(
            lexical_operand='or',
            or_phrases=['hello', 'world'],
            relevance_cutoff=relevance_cutoff
        )
        vespa_query = self.vespa_index.to_vespa_query(q)

        # Main YQL should use OR
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected = ('select * from test_index where '
                    '(default contains "hello" OR default contains "world")')
        self.assertEqual(expected, lexical_yql)

        probe_lexical_yql = vespa_query.get('marqo__yql.lexical.probe', '')
        expected = ('select * from test_index where '
                    '(default contains "hello" OR default contains "world")')
        self.assertEqual(
            expected, probe_lexical_yql
        )

    def test_all_quoted_terms_with_lexical_operand_or_still_uses_and(self):
        """When the query is fully quoted like '"this" "is" "a" "sentence"', all terms become and_phrases.
        Even with lexicalOperand='or', the and_phrases are always joined with AND because
        lexicalOperand only affects or_phrases (unquoted terms)."""
        for lexical_operand in ['or', 'and', 'weakAnd']:
            with self.subTest(lexical_operand=lexical_operand):
                q = self._make_hybrid_query(
                    lexical_operand=lexical_operand,
                    or_phrases=[],
                    and_phrases=['this', 'is', 'a', 'sentence'],
                )
                vespa_query = self.vespa_index.to_vespa_query(q)

                lexical_yql = vespa_query.get('marqo__yql.lexical', '')
                expected = ('select * from test_index where '
                            '(default contains "this" AND default contains "is" '
                            'AND default contains "a" AND default contains "sentence")')
                self.assertEqual(expected, lexical_yql)

    def test_all_quoted_terms_with_lexical_operand_or_generate_or_terms_is_empty(self):
        """When all terms are quoted (and_phrases only), _generate_or_terms returns empty string
        regardless of lexicalOperand, since lexicalOperand only controls or_phrases."""
        for lexical_operand in ['or', 'and', 'weakAnd']:
            with self.subTest(lexical_operand=lexical_operand):
                q = self._make_hybrid_query(
                    lexical_operand=lexical_operand,
                    or_phrases=[],
                    and_phrases=['this', 'is', 'a', 'sentence'],
                )
                result = self.vespa_index._generate_or_terms(q)
                self.assertEqual(result, '')

    def test_mixed_quoted_and_unquoted_with_lexical_operand_or(self):
        """When query has both quoted and unquoted terms like 'hello "exact phrase" world',
        unquoted terms use OR (from lexicalOperand) while quoted terms always use AND."""
        q = self._make_hybrid_query(
            lexical_operand='or',
            or_phrases=['hello', 'world'],
            and_phrases=['exact phrase'],
        )
        vespa_query = self.vespa_index.to_vespa_query(q)

        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected = ('select * from test_index where '
                    '((default contains "hello" OR default contains "world") '
                    'AND (default contains "exact phrase"))')
        self.assertEqual(expected, lexical_yql)

    def test_all_quoted_terms_with_lexical_operand_or_and_relevance_cutoff(self):
        """When all terms are quoted and lexicalOperand='or' with relevanceCutoff override,
        both main and probe YQL should use AND since all terms are in and_phrases.
        The relevanceCutoff.lexicalOperand override has no effect when or_phrases is empty."""
        relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=RelativeMaxScoreParameters(relativeScoreFactor=0.5),
            lexicalOperand='or'
        )
        q = self._make_hybrid_query(
            lexical_operand='and',
            or_phrases=[],
            and_phrases=['this', 'is', 'a', 'sentence'],
            relevance_cutoff=relevance_cutoff
        )
        vespa_query = self.vespa_index.to_vespa_query(q)

        # Main lexical YQL should use AND (and_phrases are always AND-joined)
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected = ('select * from test_index where '
                    '(default contains "this" AND default contains "is" '
                    'AND default contains "a" AND default con'
                    'tains "sentence")')
        self.assertEqual(expected, lexical_yql)

        # Probe YQL: even with relevanceCutoff.lexicalOperand='or', it only affects or_phrases.
        # Since or_phrases is empty, the probe YQL is identical to main.
        probe_yql = vespa_query.get('marqo__yql.lexical.probe', '')
        if probe_yql:
            self.assertEqual(expected, probe_yql)


    def test_custom_score_ranking_term_always_weakand_regardless_of_lexical_operand(self):
        """Custom score modifier ranking terms (is_ranking_term=True) must always use weakAnd,
        regardless of the lexicalOperand setting. lexicalOperand only affects the main retrieval query."""
        expected = ('weakAnd((marqo__lexical_title contains "hello"), '
                    '(marqo__lexical_title contains "world"))')
        for lexical_operand in ['or', 'and', 'weakAnd']:
            with self.subTest(lexical_operand=lexical_operand):
                q = self._make_hybrid_query(
                    lexical_operand=lexical_operand,
                    or_phrases=['hello', 'world'],
                )
                result = self.vespa_index._generate_or_terms(
                    q, is_ranking_term=True, attributes_to_search=['title']
                )
                self.assertEqual(expected, result)

    def test_custom_score_ranking_term_weakand_with_lexical_operand_or_full_query(self):
        """End-to-end: with lexicalOperand='or' and BM25 custom score, the main retrieval uses OR
        but the extra rank() term must use weakAnd."""
        q = self._make_hybrid_query(
            lexical_operand='or',
            or_phrases=['hello', 'world'],
            searchable_attributes_lexical=['description'],
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_query = self.vespa_index.to_vespa_query(q)
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected_lexical = (
            'select * from test_index where '
            '(rank('
            '(marqo__lexical_description contains "hello") OR (marqo__lexical_description contains "world"), '
            'weakAnd((marqo__lexical_title contains "hello"), (marqo__lexical_title contains "world"))))')
        self.assertEqual(expected_lexical, lexical_yql)

    def test_custom_score_ranking_term_weakand_with_lexical_operand_and_full_query(self):
        """End-to-end: with lexicalOperand='and' and BM25 custom score, the main retrieval uses AND
        but the extra rank() term must use weakAnd."""
        q = self._make_hybrid_query(
            lexical_operand='and',
            or_phrases=['hello', 'world'],
            searchable_attributes_lexical=['description'],
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_query = self.vespa_index.to_vespa_query(q)
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected_lexical = (
            'select * from test_index where '
            '(rank('
            '(marqo__lexical_description contains "hello") AND (marqo__lexical_description contains "world"), '
            'weakAnd((marqo__lexical_title contains "hello"), (marqo__lexical_title contains "world"))))')
        self.assertEqual(expected_lexical, lexical_yql)

    def test_custom_score_ranking_term_weakand_with_lexical_operand_weakand_full_query(self):
        """End-to-end: with lexicalOperand='weakAnd' and BM25 custom score, both the main retrieval
        and the extra rank() term use weakAnd."""
        q = self._make_hybrid_query(
            lexical_operand='weakAnd',
            or_phrases=['hello', 'world'],
            searchable_attributes_lexical=['description'],
            score_modifiers=[
                ScoreModifier(
                    field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                    weight=1.0,
                    type=ScoreModifierType.Add,
                ),
            ],
        )
        vespa_query = self.vespa_index.to_vespa_query(q)
        lexical_yql = vespa_query.get('marqo__yql.lexical', '')
        expected_lexical = (
            'select * from test_index where '
            '(rank('
            'weakAnd((marqo__lexical_description contains "hello"), (marqo__lexical_description contains "world")), '
            'weakAnd((marqo__lexical_title contains "hello"), (marqo__lexical_title contains "world"))))')
        self.assertEqual(expected_lexical, lexical_yql)

    def test_custom_score_tensor_ranking_term_always_weakand_with_lexical_operand(self):
        """Tensor YQL's extra BM25 ranking term must use weakAnd regardless of lexicalOperand.
        The tensor YQL is identical across all lexicalOtest_no_lexical_fields_returns_false_regardless_of_lexical_operandperand values since it only affects lexical retrieval."""
        expected_tensor = (
            'select * from test_index where '
            'rank(('
            '({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}'
            'nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) OR '
            '({targetHits:10, approximate:True, hnsw.exploreAdditionalHits:1990}'
            'nearestNeighbor(marqo__embeddings_description, marqo__query_embedding))), '
            'weakAnd((marqo__lexical_title contains "hello"), (marqo__lexical_title contains "world")))')
        for lexical_operand in ['or', 'and', 'weakAnd']:
            with self.subTest(lexical_operand=lexical_operand):
                q = self._make_hybrid_query(
                    lexical_operand=lexical_operand,
                    or_phrases=['hello', 'world'],
                    score_modifiers=[
                        ScoreModifier(
                            field=f"{MARQO_CUSTOM_SCORE_RERANK_INPUT_PREFIX}bm25_field_title",
                            weight=1.0,
                            type=ScoreModifierType.Add,
                        ),
                    ],
                )
                vespa_query = self.vespa_index.to_vespa_query(q)
                tensor_yql = vespa_query.get('marqo__yql.tensor', '')
                self.assertEqual(expected_tensor, tensor_yql)

    def test_no_lexical_fields_returns_false_regardless_of_lexical_operand(self):
        """When the index has no lexical fields, the lexical YQL must be 'False' regardless of lexicalOperand."""
        no_lex_index = self._create_semi_structured_marqo_index(
            name='test_index',
            lexical_field_names=[],
            tensor_field_names=['title', 'description'],
        )
        vi_no_lex = SemiStructuredVespaIndex(no_lex_index)
        expected = 'select * from test_index where (False)'
        for lexical_operand in ['or', 'and', 'weakAnd', None]:
            with self.subTest(lexical_operand=lexical_operand):
                q = self._make_hybrid_query(
                    lexical_operand=lexical_operand,
                    or_phrases=['hello', 'world'],
                )
                vespa_query = vi_no_lex.to_vespa_query(q)
                lexical_yql = vespa_query.get('marqo__yql.lexical', '')
                self.assertEqual(expected, lexical_yql)


if __name__ == '__main__':
    unittest.main()