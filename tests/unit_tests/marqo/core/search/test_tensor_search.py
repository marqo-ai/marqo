import unittest
from unittest.mock import patch, MagicMock, ANY

from marqo.tensor_search import tensor_search
from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, Model, TextPreProcessing, ImagePreProcessing,
    DistanceMetric, VectorNumericType, HnswConfig, FieldType, FieldFeature, IndexType, Field, TensorField
)
from marqo.config import Config
from marqo.tensor_search.telemetry import RequestMetricsStore

class TensorSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Mock RequestMetricsStore to avoid complications with not having TelemetryMiddleware configuring metrics.
        """
        cls.mock_request = MagicMock()
        cls.metrics_store_patcher = patch('marqo.tensor_search.telemetry.RequestMetricsStore._get_request')
        cls.mock_get_request = cls.metrics_store_patcher.start()
        cls.mock_get_request.return_value = cls.mock_request
        RequestMetricsStore.set_in_request(cls.mock_request)

    @classmethod
    def tearDownClass(cls):
        cls.metrics_store_patcher.stop()

    def setUp(self):
        # Create a real StructuredMarqoIndex instance with the same schema as `default_text_index`
        self.model = Model(name="hf/all_datasets_v4_MiniLM-L6")

        self.index = StructuredMarqoIndex(
            name="index_name",
            schema_name="test_schema",
            type=IndexType.Structured,
            model=self.model,
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(split_length=5, split_overlap=2, split_method="word"),
            image_preprocessing=ImagePreProcessing(patch_method=None),
            distance_metric=DistanceMetric.Euclidean,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=200, m=16),
            marqo_version="2.16.0",
            created_at=1234567890,
            updated_at=1234567890,
            fields=[
                Field(name="text_field_1", type=FieldType.Text,
                      features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                      lexical_field_name="text_field_1", filter_field_name="text_field_1"
                ),
                Field(name="text_field_2", type=FieldType.Text,
                      features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                      lexical_field_name="text_field_2", filter_field_name="text_field_2"
                ),
                Field(
                    name="int_field_1", type=FieldType.Int, features=[FieldFeature.Filter],
                    filter_field_name="text_field_1"
                ),
                Field(
                    name="float_field_1", type=FieldType.Float, features=[FieldFeature.Filter],
                    filter_field_name="text_field_1"
                ),
                Field(name="multimodal_combo_field", type=FieldType.MultimodalCombination, dependent_fields={
                    "text_field_1": 1.0, "text_field_2": 2.0
                }),
                Field(name="custom_vector_field", type=FieldType.CustomVector,
                      )
            ],
            tensor_fields=[
                TensorField(name="text_field_1", chunk_field_name="text_field_1", embeddings_field_name="text_field_1"),
                TensorField(name="text_field_2", chunk_field_name="text_field_2", embeddings_field_name="text_field_2"),
                TensorField(
                    name="multimodal_combo_field",
                    chunk_field_name="multimodal_combo_field",
                    embeddings_field_name="multimodal_combo_field"
                ),
                TensorField(
                    name="custom_vector_field",
                    chunk_field_name="custom_vector_field",
                    embeddings_field_name="custom_vector_field"
                )
            ]
        )

        self.vespa_client_mock = MagicMock()
        self.config = Config(self.vespa_client_mock)
        self.logger_mock = MagicMock()

        self.get_index_patcher = patch(
            "marqo.tensor_search.tensor_search.index_meta_cache.get_index",
            return_value=self.index
        )
        self.logger_patcher = patch(
            "marqo.tensor_search.tensor_search.logger",
            self.logger_mock
        )

        self.get_index_patcher.start()
        self.logger_patcher.start()

    def tearDown(self):
        self.get_index_patcher.stop()
        self.logger_patcher.stop()

    def test_tensor_search(self):
        tensor_search.search(self.config, "index_name", "query")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        print(call_args)
        self.assertEqual(call_args['yql'], ('select * from test_schema where (({targetHits:3, approximate:True, '
 'hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_1, '
 'marqo__query_embedding)) OR ({targetHits:3, approximate:True, '
 'hnsw.exploreAdditionalHits:1997}nearestNeighbor(text_field_2, '
 'marqo__query_embedding)) OR ({targetHits:3, approximate:True, '
 'hnsw.exploreAdditionalHits:1997}nearestNeighbor(multimodal_combo_field, '
 'marqo__query_embedding)) OR ({targetHits:3, approximate:True, '
 'hnsw.exploreAdditionalHits:1997}nearestNeighbor(custom_vector_field, '
 'marqo__query_embedding)))'))

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