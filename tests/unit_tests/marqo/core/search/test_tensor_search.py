import unittest
from unittest.mock import patch, MagicMock

from marqo.tensor_search import tensor_search
from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, Model, TextPreProcessing, ImagePreProcessing,
    DistanceMetric, VectorNumericType, HnswConfig, FieldType, FieldFeature, IndexType, Field, TensorField
)
from marqo.config import Config
from marqo.tensor_search.telemetry import RequestMetricsStore

class TensorSearchTest(unittest.TestCase):


    def test_tensor_search(self):
        tensor_search.search(self.config, "index_name", "query", search_method="tensor")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], (
             'select * from test_schema where (({targetHits:3, approximate:True, '
             'hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_1, '
             'marqo__query_embedding)) OR ({targetHits:3, approximate:True, '
             'hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_2, '
             'marqo__query_embedding)) OR ({targetHits:3, approximate:True, '
             'hnsw.exploreAdditionalHits:1997}nearestNeighbor(multimodal_combo_field, '
             'marqo__query_embedding)) OR ({targetHits:3, approximate:True, '
             'hnsw.exploreAdditionalHits:1997}nearestNeighbor(custom_vector_field, '
             'marqo__query_embedding)))'
            )
        )
        self.assertEqual(call_args['model_restrict'], 'test_schema')
        self.assertEqual(call_args['hits'], '3')
        self.assertEqual(call_args['offset'], '0')

    def test_tensor_search_with_target_hits(self):
        tensor_search.search(self.config, "index_name", "query", search_method="tensor", target_hits=5)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], (
             'select * from test_schema where (({targetHits:5, approximate:True, '
             'hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_1, '
             'marqo__query_embedding)) OR ({targetHits:5, approximate:True, '
             'hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_2, '
             'marqo__query_embedding)) OR ({targetHits:5, approximate:True, '
             'hnsw.exploreAdditionalHits:1997}nearestNeighbor(multimodal_combo_field, '
             'marqo__query_embedding)) OR ({targetHits:5, approximate:True, '
             'hnsw.exploreAdditionalHits:1997}nearestNeighbor(custom_vector_field, '
             'marqo__query_embedding)))'
            )
        )

    def test_lexical_search(self):
        tensor_search.search(self.config, "index_name", "query", search_method="lexical")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], 'select * from test_schema where (weakAnd(default contains "query"))')
        self.assertEqual(call_args['query_features'], {'text_field_2': 1, 'text_field_1': 1})
        self.assertEqual(call_args['ranking'], 'bm25')
        self.assertEqual(call_args['hits'], 3)
        self.assertEqual(call_args['offset'], 0)
        self.assertEqual(call_args['model_restrict'], 'test_schema')
        self.assertEqual(call_args['presentation.summary'], 'all-non-vector-summary')

    def test_hybrid_search(self):
        tensor_search.search(self.config, "index_name", "query", search_method="hybrid")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.tensor'],
            'select * from test_schema where (({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_1, marqo__query_embedding)) OR ({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_2, marqo__query_embedding)) OR ({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(multimodal_combo_field, marqo__query_embedding)) OR ({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(custom_vector_field, marqo__query_embedding)))'
        )
        self.assertEqual(
            call_args['marqo__yql.lexical'], 'select * from test_schema where (weakAnd(default contains "query"))'
        )

    def test_hybrid_search_with_target_hits(self):
        tensor_search.search(self.config, "index_name", "query", search_method="hybrid", target_hits=5)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.tensor'],
            'select * from test_schema where (({targetHits:5, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_1, marqo__query_embedding)) OR ({targetHits:5, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_2, marqo__query_embedding)) OR ({targetHits:5, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(multimodal_combo_field, marqo__query_embedding)) OR ({targetHits:5, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(custom_vector_field, marqo__query_embedding)))'
        )
        self.assertEqual(
            call_args['marqo__yql.lexical'], 'select * from test_schema where (weakAnd(default contains "query"))'
        )