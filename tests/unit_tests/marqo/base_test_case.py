import unittest
from unittest.mock import patch, MagicMock

from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, Model, TextPreProcessing, ImagePreProcessing,
    DistanceMetric, VectorNumericType, HnswConfig, FieldType, FieldFeature, IndexType, Field, TensorField
)
from marqo.config import Config
from marqo.tensor_search.telemetry import RequestMetricsStore

class BaseUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Mock RequestMetricsStore to avoid complications with not having TelemetryMiddleware configuring metrics.
        """
        cls.mock_request = MagicMock()
        cls.metrics_store_patcher = patch('marqo.tensor_search.telemetry.RequestMetricsStore._get_request')
        cls.mock_get_request = cls.metrics_store_patcher.start()
        cls.mock_get_request.return_value = cls.mock_request
        RequestMetricsStore.set_in_request(cls.mock_request)

        cls.model = Model(name="hf/all_datasets_v4_MiniLM-L6")

        cls.structured_index = StructuredMarqoIndex(
            name="index_name", schema_name="test_schema", type=IndexType.Structured, model=cls.model,
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(split_length=5, split_overlap=2, split_method="word"),
            image_preprocessing=ImagePreProcessing(patch_method=None), distance_metric=DistanceMetric.Euclidean,
            vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=200, m=16),
            marqo_version="2.16.0", created_at=1234567890, updated_at=1234567890, fields=[Field(
                name="text_field_1", type=FieldType.Text, features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                lexical_field_name="text_field_1", filter_field_name="text_field_1"
            ), Field(
                name="text_field_2", type=FieldType.Text, features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                lexical_field_name="text_field_2", filter_field_name="text_field_2"
            ), Field(
                name="int_field_1", type=FieldType.Int, features=[FieldFeature.Filter], filter_field_name="text_field_1"
            ), Field(
                name="float_field_1", type=FieldType.Float, features=[FieldFeature.Filter],
                filter_field_name="text_field_1"
            ), Field(
                name="multimodal_combo_field", type=FieldType.MultimodalCombination, dependent_fields={
                    "text_field_1": 1.0,
                    "text_field_2": 2.0
                }
            ), Field(
                name="custom_vector_field", type=FieldType.CustomVector, )], tensor_fields=[
                TensorField(name="text_field_1", chunk_field_name="text_field_1", embeddings_field_name="text_field_1"),
                TensorField(name="text_field_2", chunk_field_name="text_field_2", embeddings_field_name="text_field_2"),
                TensorField(
                    name="multimodal_combo_field", chunk_field_name="multimodal_combo_field",
                    embeddings_field_name="multimodal_combo_field"
                ), TensorField(
                    name="custom_vector_field", chunk_field_name="custom_vector_field",
                    embeddings_field_name="custom_vector_field"
                )]
        )

        cls.vespa_client_mock = MagicMock()
        cls.config = Config(cls.vespa_client_mock)
        cls.logger_mock = MagicMock()

        cls.get_index_patcher = patch(
            "marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=cls.structured_index
        )
        cls.logger_patcher = patch(
            "marqo.tensor_search.tensor_search.logger", cls.logger_mock
        )

        cls.get_index_patcher.start()
        cls.logger_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.metrics_store_patcher.stop()

    def get_expected_tensor_yql(cls, index=None, target_hits=3):
        if index is None:
            index = cls.structured_index
        yql = f"select * from {index.schema_name} where ("
        for field in index.fields:
            if field.type in (FieldType.Float, FieldType.Int):
                continue
            yql += (
                f"({{targetHits:{target_hits}, approximate:True, hnsw.exploreAdditionalHits:1997}}"
                f"nearestNeighbor({field.name}, marqo__query_embedding)) OR "
            )
        return yql[:-4] + ")"

    def get_expected_lexical_yql(self, query, index=None):
        if index is None:
            index = self.structured_index
        return f'select * from {index.schema_name} where (weakAnd(default contains "{query}"))'

    def set_index_to_return(self, index):
        self.get_index_patcher.stop()
        self.get_index_patcher = patch(
            "marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=index
        )
        self.get_index_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.get_index_patcher.stop()
        cls.logger_patcher.stop()

    def tearDown(self):
        self.get_index_patcher.stop()
        self.get_index_patcher = patch(
            "marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=self.structured_index
        )
        self.get_index_patcher.start()