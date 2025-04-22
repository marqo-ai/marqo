import numpy as np

from marqo.core.inference.api import InferenceResult
from marqo.core.models.hybrid_parameters import HybridParameters, RankingMethod, RetrievalMethod
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.tensor_search import tensor_search
import unittest
from unittest.mock import patch, MagicMock, ANY

from marqo.core.models.marqo_index import (
    StructuredMarqoIndex, Model, TextPreProcessing, ImagePreProcessing,
    DistanceMetric, VectorNumericType, HnswConfig, FieldType, FieldFeature, IndexType, Field, TensorField,
    UnstructuredMarqoIndex, SemiStructuredMarqoIndex
)

from marqo.config import Config
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.version import get_version

class SearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock RequestMetricsStore to avoid complications with not having TelemetryMiddleware configuring metrics.
        cls.mock_request = MagicMock()
        cls.metrics_store_patcher = patch('marqo.tensor_search.telemetry.RequestMetricsStore._get_request')
        cls.mock_get_request = cls.metrics_store_patcher.start()
        cls.mock_get_request.return_value = cls.mock_request
        RequestMetricsStore.set_in_request(cls.mock_request)

        # Create a model
        cls.model = Model(name="hf/all_datasets_v4_MiniLM-L6")

        # Structured index with multimodal fields
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

        cls.legacy_unstructured_index = UnstructuredMarqoIndex(
            name="legacy_unstructured_index", schema_name="legacy_test_schema", model=cls.model, normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(split_length=5, split_overlap=2, split_method="word"),
            image_preprocessing=ImagePreProcessing(patch_method=None), distance_metric=DistanceMetric.Euclidean,
            vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=200, m=16),
            marqo_version=get_version(), created_at=1234567890, updated_at=1234567890,
            treat_urls_and_pointers_as_images=True, filter_string_max_length=1000
        )

        cls.unstructured_index = SemiStructuredMarqoIndex(
            name="unstructured_index", schema_name="unstructured_test_schema", model=cls.model, normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(split_length=5, split_overlap=2, split_method="word"),
            image_preprocessing=ImagePreProcessing(patch_method=None), distance_metric=DistanceMetric.Euclidean,
            vector_numeric_type=VectorNumericType.Float, hnsw_config=HnswConfig(ef_construction=200, m=16),
            marqo_version=get_version(), created_at=1234567890, updated_at=1234567890,
            treat_urls_and_pointers_as_images=True, filter_string_max_length=1000,
            tensor_fields=[
                TensorField(name="text_field_1", chunk_field_name="text_field_1", embeddings_field_name="text_field_1"),
                TensorField(name="text_field_2", chunk_field_name="text_field_2", embeddings_field_name="text_field_2"),
            ],
            lexical_fields=[
                Field(name="text_field_1", type=FieldType.Text, features=[FieldFeature.Filter], filter_field_name="text_field_1"),
                Field(name="text_field_2", type=FieldType.Text, features=[FieldFeature.Filter], filter_field_name="text_field_2")
            ]
        )

        # Mock VespaClient and Config
        cls.vespa_client_mock = MagicMock()
        cls.inference_mock = MagicMock()
        def make_mock_vectorise_result(contents):
            return InferenceResult(
                result=[
                    [("chunk", np.random.rand(5))] for _ in contents  # assuming 5-dim vector per content
                ]
            )

        cls.inference_mock.vectorise.side_effect = lambda req: make_mock_vectorise_result(req.contents)
        cls.config = Config(cls.vespa_client_mock, cls.inference_mock)
        cls.logger_mock = MagicMock()

        # Patch the get_index method to return the structured index
        cls.get_index_patcher = patch(
            "marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=cls.structured_index
        )
        cls.current_index = cls.structured_index
        cls.logger_patcher = patch(
            "marqo.tensor_search.tensor_search.logger", cls.logger_mock
        )

        # Start the patchers
        cls.get_index_patcher.start()
        cls.logger_patcher.start()

    @classmethod
    def tearDownClass(cls):
        # Stop the patchers
        cls.get_index_patcher.stop()
        cls.logger_patcher.stop()
        cls.metrics_store_patcher.stop()

    def get_expected_tensor_yql(self, rerank_depth=3, additional_hits=1997):
        yql = f"select * from {self.current_index.schema_name} where ("
        for field in self.current_index.fields:
            if field.type in (FieldType.Float, FieldType.Int):
                continue
            yql += (
                f"({{targetHits:{rerank_depth}, approximate:True, hnsw.exploreAdditionalHits:{additional_hits}}}"
                f"nearestNeighbor({field.name}, marqo__query_embedding)) OR "
            )
        return yql[:-4] + ")"

    def get_expected_tensor_yql_unstructured(self, rerank_depth=3, additional_hits=1997, include_select=True):
        yql = ""
        if include_select:
            yql = f"select * from {self.current_index.schema_name} where ("
        for field in self.current_index.tensor_fields:
            yql += (
                f"({{targetHits:{rerank_depth}, approximate:True, hnsw.exploreAdditionalHits:{additional_hits}}}"
                f"nearestNeighbor({field.name}, marqo__query_embedding)) OR "
            )
        return yql[:-4] + ")"

    def get_expected_lexical_yql(self, query):
        return f'select * from {self.current_index.schema_name} where weakAnd(default contains "{query}")'

    def get_expected_lexical_yql2(self, query):
        return f'select * from {self.current_index.schema_name} where (weakAnd(default contains "{query}"))'

    def get_expected_lexical_yql_with_or(self, query, include_select=True):
        query_strings = query.split(" ")
        query_string = " OR ".join([f"default contains \"{q}\"" for q in query_strings])
        return f'select * from {self.current_index.schema_name} where {query_string}' if include_select else query_string

    def set_index_to_return(self, index):
        self.get_index_patcher.stop()
        self.get_index_patcher = patch(
            "marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=index
        )
        self.current_index = index
        self.get_index_patcher.start()

    def tearDown(self):
        # Reset the index to the structured index (Default)
        self.get_index_patcher.stop()
        self.get_index_patcher = patch(
            "marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=self.structured_index
        )
        self.current_index = self.structured_index
        self.get_index_patcher.start()

        # Reset the mocks
        self.logger_mock.reset_mock()
        self.vespa_client_mock.reset_mock()

    def test_tensor_search(self):
        tensor_search.search(self.config, "index_name", "query", search_method="tensor")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], self.get_expected_tensor_yql())
        self.assertEqual(call_args['model_restrict'], 'test_schema')
        self.assertEqual(call_args['hits'], 3)
        self.assertEqual(call_args['offset'], 0)

    def test_tensor_search_with_rerank_depth(self):
        tensor_search.search(self.config, "index_name", "query", search_method="tensor", rerank_depth=5)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], self.get_expected_tensor_yql(rerank_depth=5, additional_hits=1995))

    def test_lexical_search(self):
        tensor_search.search(self.config, "index_name", "query", search_method="lexical")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(call_args['yql'], self.get_expected_lexical_yql2("query"))
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
            self.get_expected_tensor_yql(),
            call_args['marqo__yql.tensor']
        )
        self.assertEqual(
            self.get_expected_lexical_yql("query"), call_args['marqo__yql.lexical']
        )

    def test_hybrid_search_with_rerank_depth_tensor(self):
        tensor_search.search(self.config, "index_name", "query", search_method="hybrid", hybrid_parameters=HybridParameters(rerankDepthTensor=5))
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.tensor'],
            self.get_expected_tensor_yql(rerank_depth=5, additional_hits=1995),
        )
        self.assertEqual(
            call_args['marqo__yql.lexical'], self.get_expected_lexical_yql("query")
        )

    def test_hybrid_search_with_rerank_depth_and_rerank_depth_tensor(self):
        tensor_search.search(self.config, "index_name", "query", search_method="hybrid", rerank_depth=15, hybrid_parameters=HybridParameters(rerankDepthTensor=5))
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.tensor'],
            self.get_expected_tensor_yql(rerank_depth=5, additional_hits=1995)
        )
        self.assertEqual(
            call_args['marqo__yql.lexical'], self.get_expected_lexical_yql("query")
        )
        self.assertEqual(call_args['marqo__hybrid.rerankDepthGlobal'], 15)

    def test_hybrid_search_with_rerank_depth_and_no_rerank_depth_tensor(self):
        tensor_search.search(self.config, "index_name", "query", search_method="hybrid", rerank_depth=15)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.tensor'],
            self.get_expected_tensor_yql(),
        )
        self.assertEqual(
            call_args['marqo__yql.lexical'], self.get_expected_lexical_yql("query")
        )
        self.assertEqual(call_args['marqo__hybrid.rerankDepthGlobal'], 15)

    def test_rerank_depth_higher_than_default_ef_search_overrides_it(self):
        tensor_search.search(self.config, "index_name", "query", search_method="tensor", rerank_depth=3000)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['yql'],
            self.get_expected_tensor_yql(rerank_depth=3000, additional_hits=0),
        )

    def test_ef_search_higher_than_default_ef_search_overrides_it(self):
        tensor_search.search(self.config, "index_name", "query", search_method="tensor", ef_search=3000)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['yql'],
            self.get_expected_tensor_yql(rerank_depth=3, additional_hits=2997),
        )

    def test_ef_search_and_rerank_depth_specified_minimal_is_selected(self):
        scenarios = [
            (1500, 1000),
            (1000, 1500),
            (1500, 1500),
        ]
        expected_results = [
            self.get_expected_tensor_yql(rerank_depth=1000, additional_hits=500),
            self.get_expected_tensor_yql(rerank_depth=1000, additional_hits=0),
            self.get_expected_tensor_yql(rerank_depth=1500, additional_hits=0),
        ]
        for i in range(len(scenarios)):
            with self.subTest(ef_search=scenarios[i][0], rerank_depth=scenarios[i][1]):
                tensor_search.search(
                    self.config, "index_name", "query", search_method="tensor",
                    ef_search=scenarios[i][0], rerank_depth=scenarios[i][1]
                )
                call_args = self.vespa_client_mock.query.call_args[1]
                self.assertEqual(
                    call_args['yql'],
                    expected_results[i],
                )

    def test_legacy_unstructured_ef_search(self):
        self.set_index_to_return(self.legacy_unstructured_index)
        tensor_search.search(self.config, "legacy_unstructured_index", "query", search_method="tensor", ef_search=3000)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertIn(
            "{targetHits:3, approximate:True, hnsw.exploreAdditionalHits:2997}",
            call_args['yql']
        )

    def test_legacy_structured_default_ef_search(self):
        self.set_index_to_return(self.legacy_unstructured_index)
        tensor_search.search(self.config, "index_name", "query", search_method="tensor")
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertIn(
            "{targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}",
            call_args['yql']
        )

    def test_legacy_structured_default_ef_search_with_limit(self):
        self.set_index_to_return(self.legacy_unstructured_index)
        tensor_search.search(self.config, "index_name", "query", search_method="tensor", result_count=100)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertIn(
            "{targetHits:100, approximate:True, hnsw.exploreAdditionalHits:1900}",
            call_args['yql']
        )

    def test_legacy_structured_ef_search_with_limit_higher_than_ef_search(self):
        self.set_index_to_return(self.legacy_unstructured_index)
        tensor_search.search(self.config, "index_name", "query", search_method="tensor", result_count=100, ef_search=50)
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertIn(
            "{targetHits:50, approximate:True, hnsw.exploreAdditionalHits:0}",
            call_args['yql']
        )

    def test_hybrid_search_with_facets_lexical(self):
        self.set_index_to_return(self.unstructured_index)
        tensor_search.search(self.config, "unstructured_index", "query", search_method="hybrid",
                             facets=FacetsParameters(
                                 fields={"text_field_1": {"type": "string"}, "text_field_2": {"type": "string"}},
                             ),
                             hybrid_parameters=HybridParameters(
                                 retrievalMethod=RetrievalMethod.Lexical,
                                 rankingMethod=RankingMethod.Lexical,
                             )
                             )
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.facets'].split('|')[0].strip(" "),
            self.get_expected_lexical_yql_with_or("query") + " limit 0"
        )

    def test_hybrid_search_with_facets_tensor(self):
        self.set_index_to_return(self.unstructured_index)
        tensor_search.search(self.config, "unstructured_index", "query", search_method="hybrid",
                             facets=FacetsParameters(
                                 fields={"text_field_1": {"type": "string"}, "text_field_2": {"type": "string"}},
                             ),
                             hybrid_parameters=HybridParameters(
                                 retrievalMethod=RetrievalMethod.Tensor,
                                 rankingMethod=RankingMethod.Tensor,
                             )
                             )
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.facets'].split('|')[0].strip(" "),
            self.get_expected_tensor_yql_unstructured() + " limit 0"
        )

    def test_hybrid_search_with_facets_rrf(self):
        self.set_index_to_return(self.unstructured_index)
        tensor_search.search(self.config, "unstructured_index", "query", search_method="hybrid",
                             facets=FacetsParameters(
                                    fields={"text_field_1": {"type": "string"}, "text_field_2": {"type": "string"}},
                             )
                         )
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertEqual(
            call_args['marqo__yql.facets'].split('|')[0].strip(" "),
            f"select * from {self.current_index.schema_name} where (" +
            self.get_expected_lexical_yql_with_or("query", include_select=False) +
            " OR " +
            f"({self.get_expected_tensor_yql_unstructured(include_select=False)})" +
            " limit 0"
        )

    def test_hybrid_search_with_facets_and_filter(self):
        self.set_index_to_return(self.unstructured_index)
        tensor_search.search(self.config, "unstructured_index", "query", search_method="hybrid",
                             facets=FacetsParameters(
                                 fields={"text_field_1": {"type": "string"}, "text_field_2": {"type": "string"}},
                             ),
                             filter="text_field_1:test"
                             )
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        self.assertNotIn("\n---MARQO-YQL-QUERY-DELIMITER---\n", call_args['marqo__yql.facets'])

    def test_hybrid_search_with_facets_and_exclude_terms(self):
        self.set_index_to_return(self.unstructured_index)
        tensor_search.search(self.config, "unstructured_index", "query", search_method="hybrid",
                             facets=FacetsParameters(
                                 fields={
                                     "text_field_1": {
                                         "type": "string",
                                         "excludeTerms": ["text_field_2:test2"]
                                     },
                                     "text_field_2": {
                                         "type": "string",
                                        "excludeTerms": ["text_field_1:test"]
                                     }
                                 },
                             ),
                             filter="text_field_1:test AND text_field_2:test2"
                             )
        self.vespa_client_mock.query.assert_called_once()
        call_args = self.vespa_client_mock.query.call_args[1]
        facet_queries = call_args['marqo__yql.facets'].split('\n---MARQO-YQL-QUERY-DELIMITER---\n')

        self.assertEqual(len(facet_queries), 2)  # One query per excluded term

        self.assertIn('(marqo__short_string_fields contains sameElement(key contains "text_field_1", value contains "test")', facet_queries[0])
        self.assertIn("text_field_1", facet_queries[0].split('|')[1]) # getting facets for it
        self.assertNotIn("text_field_2", facet_queries[0].split('|')[1]) # not getting facets for it

        self.assertIn('(marqo__short_string_fields contains sameElement(key contains "text_field_2", value contains "test2")', facet_queries[1])
        self.assertIn("text_field_2", facet_queries[1].split('|')[1]) # getting facets for it
        self.assertNotIn("text_field_1", facet_queries[1].split('|')[1]) # not getting facets for it

if __name__ == '__main__':
    unittest.main()