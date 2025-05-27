import hashlib
import unittest
from typing import Any

import numpy as np
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.export import InMemoryMetricReader
from opentelemetry.sdk.metrics._internal.point import MetricsData
from opentelemetry.test.globals_test import reset_metrics_globals
from orjson import orjson

from marqo.core.inference.api import InferenceRequest, Modality, ModelConfig, TextPreprocessingConfig, Inference, \
    InferenceResult, InferenceErrorModel
from marqo.inference.inference_cache.caching_inference import CachingInference


class RandomInferenceStub(Inference):
    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        dimension = request.model_config.model_properties["dimensions"]
        model_key = hashlib.md5(orjson.dumps(request.model_config.model_properties)).hexdigest()

        def random_ndarray(content: str):
            seed = int(hashlib.sha256(f'{model_key}||{content}'.encode("utf-8")).hexdigest(), 16) % 2 ** 32
            arr = np.random.default_rng(seed).random((dimension,), dtype=np.float32)
            return arr / np.linalg.norm(arr)

        return InferenceResult(
            result=[InferenceErrorModel(error_message=content) if content.startswith("error:") else
                    [(content, random_ndarray(content))] for content in request.contents])


class TestInferenceCache(unittest.TestCase):
    def setUp(self):
        self.inference_local = RandomInferenceStub()

        self.base_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["a"],
            model_config=ModelConfig(
                model_name="hf/all_datasets_v4_MiniLM-L6",
                model_properties={
                    "name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                    "dimensions": 384,
                    "tokens": 128,
                    "type": "hf"
                }
            ),
            preprocessing_config=TextPreprocessingConfig(should_chunk=False),
            use_inference_cache=True
        )

    def test_caching_inference_should_return_same_result_as_its_delegate(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                caching_inference = CachingInference(self.inference_local, 10, "LRU")

                req = self.base_request.copy(update={"contents": ["a", "b", "error:c"]})

                result_from_local_inference = self.inference_local.vectorise(req)
                result_from_caching_inference = caching_inference.vectorise(req)

                model_key = caching_inference.model_cache_key(req.model_config.model_properties)

                self.assertEqual(len(result_from_local_inference.result), len(result_from_caching_inference.result))
                for i in range(len(result_from_local_inference.result)):
                    # assert return the same inference error
                    if isinstance(result_from_local_inference.result[i], InferenceErrorModel):
                        self.assertEqual(result_from_local_inference.result[i], result_from_caching_inference.result[i])
                        continue

                    # assert return the same embeddings
                    content1, embedding1 = result_from_local_inference.result[i][0]
                    content2, embedding2 = result_from_caching_inference.result[i][0]
                    self.assertEqual(content1, content2)
                    self.assertTrue(np.array_equal(embedding1, embedding2))

                    # assert that the embeddings are cached
                    cached_embedding = caching_inference.inference_cache.get(model_key, content1)
                    self.assertTrue(np.array_equal(embedding1, cached_embedding))

    def test_caching_inference_should_not_exceed_max_cache_size(self):
        with self.subTest(cache_type="LRU"):
            caching_inference = CachingInference(self.inference_local, 2, "LRU")

            result = caching_inference.vectorise(self.base_request.copy(update={"contents": ["1", "2", "3"]}))

            model_key = caching_inference.model_cache_key(self.base_request.model_config.model_properties)
            self.assertEqual(len(result.result), 3)
            self.assertEqual(caching_inference.inference_cache._cache.currsize, 2)
            self.assertIsNone(caching_inference.inference_cache.get(model_key, "1"))
            self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "2"))
            self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "3"))

        with self.subTest(cache_type="LFU"):
            caching_inference = CachingInference(self.inference_local, 2, "LFU")

            caching_inference.vectorise(self.base_request.copy(update={"contents": ["1", "2"]}))
            caching_inference.vectorise(self.base_request.copy(update={"contents": ["1"]}))
            result = caching_inference.vectorise(self.base_request.copy(update={"contents": ["1", "2", "3"]}))

            model_key = caching_inference.model_cache_key(self.base_request.model_config.model_properties)
            self.assertEqual(len(result.result), 3)
            self.assertEqual(caching_inference.inference_cache._cache.currsize, 2)
            self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "1"))
            # 2 is evicted because it's less frequently accessed
            self.assertIsNone(caching_inference.inference_cache.get(model_key, "2"))
            self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "3"))

    def test_caching_inference_should_support_multiple_models(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                caching_inference = CachingInference(self.inference_local, 10, "LRU")

                caching_inference.vectorise(self.base_request)
                model_key1 = caching_inference.model_cache_key(self.base_request.model_config.model_properties)

                req_with_new_model = self.base_request.copy(update={"model_config": ModelConfig(
                    model_name="hf/all-mpnet-base-v2",
                    model_properties={
                       "name": "sentence-transformers/all-mpnet-base-v2",
                       "dimensions": 768, "tokens": 128, "type": "hf"
                    }
                )})
                caching_inference.vectorise(req_with_new_model)
                model_key2 = caching_inference.model_cache_key(req_with_new_model.model_config.model_properties)

                cached_embedding_model_1 = caching_inference.inference_cache.get(model_key1, "a")
                cached_embedding_model_2 = caching_inference.inference_cache.get(model_key2, "a")

                self.assertIsNotNone(cached_embedding_model_1)
                self.assertIsNotNone(cached_embedding_model_2)
                self.assertNotEqual(cached_embedding_model_1, cached_embedding_model_2)

    def test_caching_inference_should_capture_key_metrics(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                reset_metrics_globals()
                reader = InMemoryMetricReader()
                provider = MeterProvider(metric_readers=[reader])
                metrics.set_meter_provider(provider)

                caching_inference = CachingInference(self.inference_local, 12, "LRU")

                req1 = self.base_request.copy(update={"contents": ["1", "2", "3"]})  # misses: 3
                caching_inference.vectorise(req1)

                self._assert_metric_value(reader.get_metrics_data(), 'cache_miss_total', 3)
                self._assert_metric_value(reader.get_metrics_data(), 'cache_size_curr', 3)

                req2 = self.base_request.copy(update={"contents": ["1", "2", "4", "error:5"]})  # hits 2, misses: 2
                caching_inference.vectorise(req2)
                self._assert_metric_value(reader.get_metrics_data(), 'cache_miss_total', 5)
                self._assert_metric_value(reader.get_metrics_data(), 'cache_hit_total', 2)
                self._assert_metric_value(reader.get_metrics_data(), 'cache_size_curr', 4)  # error result not cached

                provider.shutdown()

    def _assert_metric_value(self, metric_data: MetricsData, name: str, expected_value: Any):
        cache_metrics = metric_data.resource_metrics[0].scope_metrics[0].metrics
        metric = next((metric for metric in cache_metrics if metric.name == name), None)
        self.assertIsNotNone(metric, f'metric {name} not found')
        self.assertEqual(expected_value, metric.data.data_points[0].value)
