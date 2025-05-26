import unittest
from unittest.mock import patch, MagicMock

from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, Model, TextPreProcessing, ImagePreProcessing,
    DistanceMetric, VectorNumericType, HnswConfig, FieldType, FieldFeature, IndexType, Field, TensorField
)
from marqo.config import Config
from marqo.core.search.hybrid_search import HybridSearch
from marqo.version import get_version
from marqo.tensor_search import tensor_search


class TestApproximateThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a model
        cls.model = Model(name="hf/all_datasets_v4_MiniLM-L6")

        # Structured index with fields for testing
        cls.structured_index = StructuredMarqoIndex(
            name="index_name", schema_name="test_schema", type=IndexType.Structured, model=cls.model,
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(split_length=5, split_overlap=2, split_method="word"),
            image_preprocessing=ImagePreProcessing(patch_method=None), distance_metric=DistanceMetric.Euclidean,
            vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=200, m=16),
            marqo_version=get_version(), created_at=1234567890, updated_at=1234567890, fields=[Field(
                name="text_field_1", type=FieldType.Text, features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                lexical_field_name="text_field_1", filter_field_name="text_field_1"
            ), Field(
                name="text_field_2", type=FieldType.Text, features=[FieldFeature.LexicalSearch, FieldFeature.Filter],
                lexical_field_name="text_field_2", filter_field_name="text_field_2"
            )], tensor_fields=[
                TensorField(name="text_field_1", chunk_field_name="text_field_1", embeddings_field_name="text_field_1"),
                TensorField(name="text_field_2", chunk_field_name="text_field_2", embeddings_field_name="text_field_2"),
            ]
        )

        # Mock VespaClient and Config
        cls.vespa_client_mock = MagicMock()
        cls.inference_mock = MagicMock()
        cls.config = Config(cls.vespa_client_mock, cls.inference_mock)

        # Patch the get_index method to return the structured index
        cls.get_index_patcher = patch(
            "marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=cls.structured_index
        )
        # Start the patchers
        cls.get_index_patcher.start()

    @classmethod
    def tearDownClass(cls):
        # Stop the patchers
        cls.get_index_patcher.stop()

    def test_tensor_search_with_approximate_threshold(self):
        """Test that approximate_threshold parameter is passed to the Vespa query in tensor search"""
        tensor_search.search(
            self.config, 
            "index_name", 
            "query", 
            search_method="tensor",
            approximate=True,
            approximate_threshold=0.75
        )
        
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['ranking.matching.approximateThreshold'], 0.75)

    def test_tensor_search_default_approximate_threshold(self):
        """Test that approximate_threshold is None by default in tensor search"""
        tensor_search.search(
            self.config, 
            "index_name", 
            "query", 
            search_method="tensor",
            approximate=True
        )
        
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertIsNone(call_args.get('ranking.matching.approximateThreshold'))

    @patch('marqo.core.search.hybrid_search.HybridSearch.search')
    def test_hybrid_search_with_approximate_threshold(self, mock_hybrid_search):
        """Test that approximate_threshold parameter is passed to HybridSearch in hybrid search method"""
        tensor_search.search(
            self.config, 
            "index_name", 
            "query", 
            search_method="hybrid",
            approximate=True,
            approximate_threshold=0.85
        )
        
        mock_hybrid_search.assert_called_once()
        # Check that approximate_threshold was passed to HybridSearch.search
        call_kwargs = mock_hybrid_search.call_args[1]
        self.assertEqual(call_kwargs['approximate_threshold'], 0.85)

    @patch('marqo.core.structured_vespa_index.structured_vespa_index.StructuredVespaIndex._tensor_search')
    def test_structured_vespa_index_tensor_search_with_threshold(self, mock_tensor_search):
        """Test that approximate_threshold is passed to _tensor_search in StructuredVespaIndex"""
        # Create a mock MarqoTensorQuery with approximate_threshold
        mock_query = MagicMock()
        mock_query.approximate_threshold = 0.9
        
        # Create a StructuredVespaIndex instance using our structured_index
        index = MagicMock()
        index._marqo_index = self.structured_index
        
        # Call _tensor_search with the mock query
        from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
        StructuredVespaIndex._tensor_search(index, mock_query)
        
        # Check that approximate_threshold was included in the query parameters
        mock_tensor_search.assert_called_once()
        call_args, call_kwargs = mock_tensor_search.call_args
        self.assertEqual(call_args[0].approximate_threshold, 0.9) 