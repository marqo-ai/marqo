from unittest import TestCase

import random
import time
import unittest
from typing import List
from unittest.mock import MagicMock

from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
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
    MeanStdParameters
)
from marqo.tensor_search.models.sort_by_model import SortByModel
from marqo.version import get_version
from tests.unit_tests.marqo_test import MarqoTestCase


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
        string_array_field_names: List[str] = []
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
            marqo_version='2.16.0',  # Version that supports hybrid search and partial updates
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
                    'marqo__ranking.lexical.lexical': 'bm25',
                    'marqo__ranking.lexical.tensor': 'hybrid_bm25_then_embedding_similarity',
                    'marqo__ranking.tensor.lexical': 'hybrid_embedding_similarity_then_bm25',
                    'marqo__ranking.tensor.tensor': 'embedding_similarity',
                    # Facets should still use the OR query structure
                    'marqo__yql.facets': 'select * from test_index where ((default contains "neural networks" OR default contains "deep learning") '
                                         'AND (default contains "transformer") OR '
                                         '(({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(marqo__embeddings_title, marqo__query_embedding)) '
                                         'OR ({targetHits:40, approximate:True, hnsw.exploreAdditionalHits:1960}nearestNeighbor(marqo__embeddings_description, marqo__query_embedding)))) '
                                         'limit 0 | all(group(1.1) each(output(count())))',
                    'marqo__yql.lexical': 'select * from test_index where (({targetHits:111}weakAnd(default contains "neural networks",default contains "deep learning")) AND (default contains "transformer"))',
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
                        rrfK=100,
                        rerankDepthLexical=111
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
            }
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
            "marqo__hybrid.relevanceCutoff.probeDepth"
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
            collapse_field_name='parent_id',
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
            collapse_field_name='parent_id',
        )
        vespa_query = vespa_index.to_vespa_query(marqo_query)

        # Collapse field should still be set
        self.assertEqual('parent_id', vespa_query['collapsefield'])
        self.assertEqual(1, vespa_query['collapsesize'])

        # But minimal summary params should NOT be set for old schema versions
        self.assertNotIn('collapse.summary', vespa_query)
        self.assertNotIn('FieldFiller.disable', vespa_query)


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
            collapse_field_name="parent_id",
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
            collapse_field_name="parent_id",
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
            collapse_field_name="parent_id",
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
            collapse_field_name="parent_id",
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


if __name__ == '__main__':
    unittest.main() 