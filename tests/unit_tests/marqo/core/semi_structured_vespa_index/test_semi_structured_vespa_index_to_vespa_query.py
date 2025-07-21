from unittest import TestCase

import random
import time
import unittest
from typing import List
from unittest.mock import MagicMock

from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.marqo_index import (
    Model, TextPreProcessing, TextSplitMethod,
    ImagePreProcessing, HnswConfig, DistanceMetric, Field, FieldType,
    FieldFeature, TensorField, StringArrayField
)
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoHybridQuery, MarqoLexicalQuery
from marqo.core.models.marqo_query import MarqoTensorQuery
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
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
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
                    'marqo__hybrid.pagination_limit_cutoff': 600,
                    'marqo__hybrid.pagination_schema': 'marqo__pagination',
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
                    'marqo__hybrid.pagination_hash': 'b9ab5ca824a5086c862f17766572162cd6c7d15b5ac5fbb4b6634e7f6ba20cb4',
                    'marqo__hybrid.pagination_limit_cutoff': 600,
                    'marqo__hybrid.pagination_schema': 'marqo__pagination',
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

    def test_sort_by_multiple_fields_desc_and_asc(self):
        """Test sorting by two fields with descending and ascending orders."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "price", "order": "desc"},
                {"field_name": "rating", "order": "asc"}
            ],
            sortDepth=3,
            sortCandidates=50
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']

        self.assertEqual(2, len(sort_fields))
        self.assertEqual("price", sort_fields[0]["field_name"])
        self.assertEqual("desc", sort_fields[0]["order"].value)
        self.assertEqual("rating", sort_fields[1]["field_name"])
        self.assertEqual("asc", sort_fields[1]["order"].value)
        self.assertEqual(3, r['marqo__hybrid.sortBy.sortDepth'])
        self.assertEqual(50, r['marqo__hybrid.sortBy.sortCandidates'])

    def test_sort_by_single_field_no_optional(self):
        """Test sorting by a single field with no optional params."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "title", "order": "asc"}
            ],
            sortCandidates=30
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']
        sort_depth = r["marqo__hybrid.sortBy.sortDepth"]
        sort_candidates = r["marqo__hybrid.sortBy.sortCandidates"]

        self.assertEqual(1, len(sort_fields))
        self.assertEqual("title", sort_fields[0]["field_name"])
        self.assertEqual("asc", sort_fields[0]["order"].value)
        self.assertEqual(None, sort_depth)
        self.assertEqual(30, sort_candidates)

    def test_sort_by_with_missing_first(self):
        """Test a field with missing='first' and all optional params."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "description", "order": "asc", "missing": "first"}
            ],
            sortDepth=2,
            sortCandidates=20
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        sort_fields = r['marqo__hybrid.sortBy.fields']

        self.assertEqual(1, len(sort_fields))
        self.assertEqual("description", sort_fields[0]["field_name"])
        self.assertEqual("asc", sort_fields[0]["order"].value)
        self.assertEqual("first", sort_fields[0]["missing"].value)
        self.assertEqual(2, r['marqo__hybrid.sortBy.sortDepth'])
        self.assertEqual(20, r['marqo__hybrid.sortBy.sortCandidates'])

    def test_sort_by_none(self):
        """Test that no sort_by results in no sort fields present."""
        self.hybrid_query.sort_by = None
        r = self.index._to_vespa_hybrid_query(self.hybrid_query)

        self.assertNotIn("marqo__hybrid.sortBy.fields", r)
        self.assertNotIn("marqo__hybrid.sortBy.sortDepth", r)
        self.assertNotIn("marqo__hybrid.sortBy.sortCandidates", r)

    def test_sort_by_three_fields_mixed_order_and_missing(self):
        """Test three fields with mixed order and missing policies."""
        self.hybrid_query.sort_by = SortByModel(
            fields=[
                {"field_name": "price", "order": "desc", "missing": "last"},
                {"field_name": "rating", "order": "asc"},
                {"field_name": "stock", "order": "desc", "missing": "first"}
            ],
            sortDepth=4,
            sortCandidates=100
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
        self.assertEqual(100, r["marqo__hybrid.sortBy.sortCandidates"])

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
            attributes_to_retrieve=["title", "description", "price"],
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
        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        for key in [
            "marqo__hybrid.relevanceCutoff.method",
            "marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor",
            "marqo__hybrid.relevanceCutoff.parameters.meanStdDevFactor",
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

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)

        self.assertEqual(RelevanceCutoffMethod.RelativeMaxScore,
                         r["marqo__hybrid.relevanceCutoff.method"])
        self.assertAlmostEqual(0.8,
                               r["marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor"])
        # default probeDepth is 1000
        self.assertEqual(1000, r["marqo__hybrid.relevanceCutoff.probeDepth"])
        # no meanStdDevFactor for this method
        self.assertNotIn("marqo__hybrid.relevanceCutoff.parameters.meanStdDevFactor", r)

    def test_relative_max_score_custom_probeDepth(self):
        """Custom probeDepth should be honoured for RelativeMaxScore."""
        params = RelativeMaxScoreParameters(relativeScoreFactor=0.3)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.RelativeMaxScore,
            parameters=params,
            probe_depth=5
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        self.assertEqual(5, r["marqo__hybrid.relevanceCutoff.probeDepth"])

    def test_mean_std_dev_default_probeDepth(self):
        """MeanStdDev should set method, meanStdDevFactor, and default probeDepth."""
        params = MeanStdParameters(stdDevFactor=2.5)
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.MeanStdDev,
            parameters=params
        )

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)

        self.assertEqual(RelevanceCutoffMethod.MeanStdDev,
                         r["marqo__hybrid.relevanceCutoff.method"])
        self.assertAlmostEqual(2.5,
                               r["marqo__hybrid.relevanceCutoff.parameters.meanStdDevFactor"])
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

        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        self.assertEqual(7, r["marqo__hybrid.relevanceCutoff.probeDepth"])

    def test_gap_detection_default_and_custom_probeDepth(self):
        """GapDetection should set method, have no parameters, and honour probeDepth."""
        # default probeDepth
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection
        )
        r = self.index._to_vespa_hybrid_query(self.hybrid_query)
        self.assertEqual(RelevanceCutoffMethod.GapDetection,
                         r["marqo__hybrid.relevanceCutoff.method"])
        self.assertEqual(1000, r["marqo__hybrid.relevanceCutoff.probeDepth"])
        self.assertNotIn("marqo__hybrid.relevanceCutoff.parameters.relativeScoreFactor", r)
        self.assertNotIn("marqo__hybrid.relevanceCutoff.parameters.meanStdDevFactor", r)

        # custom probeDepth
        self.hybrid_query.relevance_cutoff = RelevanceCutoffModel(
            method=RelevanceCutoffMethod.GapDetection,
            probe_depth=42
        )
        r2 = self.index._to_vespa_hybrid_query(self.hybrid_query)
        self.assertEqual(42, r2["marqo__hybrid.relevanceCutoff.probeDepth"])


if __name__ == '__main__':
    unittest.main() 