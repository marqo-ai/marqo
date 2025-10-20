import hashlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.export import InMemoryMetricReader
from opentelemetry.sdk.metrics._internal.point import MetricsData
from opentelemetry.test.globals_test import reset_metrics_globals
from orjson import orjson

from inference_orchestrator.schemas.api import (
    EmbeddingModelConfig,
    ImagePreprocessingConfig,
    Inference,
    InferenceErrorModel,
    InferenceRequest,
    InferenceResult,
    Modality,
    TextPreprocessingConfig,
)
from inference_orchestrator.services.inference_cache.caching_inference import (
    CachingInference,
)
from inference_orchestrator.services.triton_inference.embedding_models.marqo_model_registry import (
    get_model_properties,
)
from tests.integration_tests.test_case import InferenceTestCase


class RandomInferenceStub(Inference):
    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        dimension = request.embedding_model_config.model_properties["dimensions"]
        model_key = hashlib.md5(
            orjson.dumps(request.embedding_model_config.model_properties)
        ).hexdigest()

        def random_ndarray(content: str):
            seed = (
                int(
                    hashlib.sha256(
                        f"{model_key}||{content}".encode("utf-8")
                    ).hexdigest(),
                    16,
                )
                % 2**32
            )
            arr = np.random.default_rng(seed).random((dimension,), dtype=np.float32)
            return arr / np.linalg.norm(arr)

        return InferenceResult(
            result=[
                InferenceErrorModel(error_message=content)
                if content.startswith("error:")
                else [(content, random_ndarray(content))]
                for content in request.contents
            ]
        )


class TestInferenceCache(InferenceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.eject_all_models()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.eject_all_models()

    def setUp(self):
        self.inference_local = RandomInferenceStub()
        self.model_name = "hf/all-MiniLM-L6-v2"
        self.model_properties = get_model_properties(self.model_name)

        self.base_request = InferenceRequest(
            modality=Modality.TEXT,
            contents=["a"],
            embedding_model_config=EmbeddingModelConfig(
                model_name=self.model_name, model_properties=self.model_properties
            ),
            preprocessing_config=TextPreprocessingConfig(should_chunk=False),
            use_inference_cache=True,
        )

    def test_caching_inference_should_return_same_result_as_its_delegate(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                caching_inference = CachingInference(
                    self.inference_local, 10, cache_type
                )

                req = self.base_request.model_copy(
                    update={"contents": ["a", "b", "error:c"]}
                )

                result_from_local_inference = self.inference_local.vectorise(req)
                result_from_caching_inference = caching_inference.vectorise(req)

                model_key = caching_inference.model_cache_key(
                    req.embedding_model_config.model_properties
                )

                self.assertEqual(
                    len(result_from_local_inference.result),
                    len(result_from_caching_inference.result),
                )
                for i in range(len(result_from_local_inference.result)):
                    # assert return the same inference error
                    if isinstance(
                        result_from_local_inference.result[i], InferenceErrorModel
                    ):
                        self.assertEqual(
                            result_from_local_inference.result[i],
                            result_from_caching_inference.result[i],
                        )
                        continue

                    # assert return the same embeddings
                    content1, embedding1 = result_from_local_inference.result[i][0]
                    content2, embedding2 = result_from_caching_inference.result[i][0]
                    self.assertEqual(content1, content2)
                    self.assertTrue(np.array_equal(embedding1, embedding2))

                    # assert that the embeddings are cached
                    cached_embedding = caching_inference.inference_cache.get(
                        model_key, content1
                    )
                    self.assertTrue(np.array_equal(embedding1, cached_embedding))

    def test_caching_inference_should_not_exceed_max_cache_size(self):
        with self.subTest(cache_type="LRU"):
            caching_inference = CachingInference(
                self.inference_local, 2, cache_type="LRU"
            )

            result = caching_inference.vectorise(
                self.base_request.model_copy(update={"contents": ["1", "2", "3"]})
            )

            model_key = caching_inference.model_cache_key(
                self.base_request.embedding_model_config.model_properties
            )
            self.assertEqual(len(result.result), 3)
            self.assertEqual(caching_inference.inference_cache._cache.currsize, 2)
            self.assertIsNone(caching_inference.inference_cache.get(model_key, "1"))
            self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "2"))
            self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "3"))

        with self.subTest(cache_type="LFU"):
            caching_inference = CachingInference(self.inference_local, 2, "LFU")

            caching_inference.vectorise(
                self.base_request.model_copy(update={"contents": ["1", "2"]})
            )
            caching_inference.vectorise(
                self.base_request.model_copy(update={"contents": ["1"]})
            )
            result = caching_inference.vectorise(
                self.base_request.model_copy(update={"contents": ["1", "2", "3"]})
            )

            model_key = caching_inference.model_cache_key(
                self.base_request.embedding_model_config.model_properties
            )
            self.assertEqual(len(result.result), 3)
            self.assertEqual(caching_inference.inference_cache._cache.currsize, 2)
            self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "1"))
            # 2 is evicted because it's less frequently accessed
            self.assertIsNone(caching_inference.inference_cache.get(model_key, "2"))
            self.assertIsNotNone(caching_inference.inference_cache.get(model_key, "3"))

    def test_caching_inference_should_support_multiple_models(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                caching_inference = CachingInference(
                    self.inference_local, 10, cache_type
                )

                caching_inference.vectorise(self.base_request)
                model_key1 = caching_inference.model_cache_key(
                    self.base_request.embedding_model_config.model_properties
                )

                req_with_new_model = self.base_request.model_copy(
                    update={
                        "embedding_model_config": EmbeddingModelConfig(
                            model_name="hf/e5-small-v2",
                            model_properties=get_model_properties("hf/e5-small-v2"),
                        )
                    }
                )
                caching_inference.vectorise(req_with_new_model)
                model_key2 = caching_inference.model_cache_key(
                    req_with_new_model.embedding_model_config.model_properties
                )

                cached_embedding_model_1 = caching_inference.inference_cache.get(
                    model_key1, "a"
                )
                cached_embedding_model_2 = caching_inference.inference_cache.get(
                    model_key2, "a"
                )

                self.assertIsNotNone(cached_embedding_model_1)
                self.assertIsNotNone(cached_embedding_model_2)
                self.assertNotEqual(
                    cached_embedding_model_1[0], cached_embedding_model_2[0]
                )

    def test_inference_cache_is_thread_safe(self):
        """Test if the cache is thread-safe by simulating concurrent reads and writes."""
        ITERATIONS = 10_000
        FREQUENT_ACCESS_RATIO = 0.5
        FREQUENT_ACCESS_SUBSET_SIZE = 5000
        TOTAL_QUERY_SET_SIZE = 100_000
        CACHE_SIZE = 1_000

        texts = [f"text{i}" for i in range(TOTAL_QUERY_SET_SIZE)]
        frequent_texts = random.sample(texts, FREQUENT_ACCESS_SUBSET_SIZE)

        def read_write_cache(caching_inference):
            if random.random() < FREQUENT_ACCESS_RATIO:
                text = random.choice(frequent_texts)
            else:
                text = random.choice(texts)
            req = self.base_request.model_copy(update={"contents": [text]})
            res = caching_inference.vectorise(req)
            res_skipping_cache = self.inference_local.vectorise(req)
            # test if the cached embedding is the same as the original
            self.assertTrue(
                np.array_equal(res.result[0][0][1], res_skipping_cache.result[0][0][1])
            )

        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                caching_inference = CachingInference(
                    self.inference_local, CACHE_SIZE, cache_type
                )
                errors = []

                # Using ThreadPoolExecutor to simulate concurrent access to the cache
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [
                        executor.submit(read_write_cache, caching_inference)
                        for _ in range(ITERATIONS)
                    ]

                    # Collect results or errors from the futures
                    for future in as_completed(futures):
                        try:
                            future.result()  # Raises exception if one occurred in the thread
                        except Exception as e:
                            errors.append(e)

                # Assert no errors were encountered
                self.assertEqual(
                    len(errors), 0, f"Thread safety issues encountered: {errors}"
                )

    def test_caching_inference_should_capture_key_metrics(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                reset_metrics_globals()
                reader = InMemoryMetricReader()
                provider = MeterProvider(metric_readers=[reader])
                metrics.set_meter_provider(provider)

                caching_inference = CachingInference(
                    self.inference_local, 12, cache_type
                )

                req1 = self.base_request.model_copy(
                    update={"contents": ["1", "2", "3"]}
                )  # misses: 3
                caching_inference.vectorise(req1)

                self._assert_metric_value(
                    reader.get_metrics_data(), "cache_miss_total", 3
                )
                self._assert_metric_value(
                    reader.get_metrics_data(), "cache_size_curr", 3
                )

                req2 = self.base_request.model_copy(
                    update={"contents": ["1", "2", "4", "error:5"]}
                )  # hits 2, misses: 2
                caching_inference.vectorise(req2)
                self._assert_metric_value(
                    reader.get_metrics_data(), "cache_miss_total", 5
                )
                self._assert_metric_value(
                    reader.get_metrics_data(), "cache_hit_total", 2
                )
                self._assert_metric_value(
                    reader.get_metrics_data(), "cache_size_curr", 4
                )  # error result not cached

                provider.shutdown()

    def test_base64_image_selective_caching(self):
        """Test selective caching: base64 images cached, URL images processed normally."""
        caching_inference = CachingInference(self.inference_local, 10, "LRU")

        base64_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        base64_jpeg = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
        url_image = "https://example.com/image.jpg"

        # Mixed request: 2 base64 images + 1 URL
        mixed_request = InferenceRequest(
            modality=Modality.IMAGE,
            contents=[base64_png, url_image, base64_jpeg],
            embedding_model_config=EmbeddingModelConfig(
                model_name="test/clip-model",
                model_properties={
                    "name": "test-clip-model",
                    "dimensions": 512,
                    "type": "clip",
                },
            ),
            preprocessing_config=ImagePreprocessingConfig(should_chunk=False),
            use_inference_cache=True,
        )

        # First call - base64 images should be cached, URL processed normally
        result1 = caching_inference.vectorise(mixed_request)

        # Verify cache contains blake3 keys for both base64 images
        import blake3

        model_key = caching_inference.model_cache_key(
            mixed_request.embedding_model_config.model_properties
        )

        hash1 = blake3.blake3(base64_png.encode()).hexdigest()
        hash2 = blake3.blake3(base64_jpeg.encode()).hexdigest()
        cache_key1 = f"blake3:{hash1}"
        cache_key2 = f"blake3:{hash2}"

        cached_embedding1 = caching_inference.inference_cache.get(model_key, cache_key1)
        cached_embedding2 = caching_inference.inference_cache.get(model_key, cache_key2)

        self.assertIsNotNone(cached_embedding1, "First base64 image should be cached")
        self.assertIsNotNone(cached_embedding2, "Second base64 image should be cached")

        # Verify URL image is NOT cached
        url_cached_embedding = caching_inference.inference_cache.get(
            model_key, url_image
        )
        self.assertIsNone(url_cached_embedding, "URL image should not be cached")

        # Verify cache size (only 2 base64 images cached)
        self.assertEqual(caching_inference.inference_cache._cache.currsize, 2)

        # Second call with same mixed content
        result2 = caching_inference.vectorise(mixed_request)

        # Base64 results should return original base64 content (not blake3 keys)
        png_content1, png_embedding1 = result1.result[0][0]
        png_content2, png_embedding2 = result2.result[0][0]

        # Content should be original base64, embeddings should be identical (from cache)
        self.assertEqual(png_content1, base64_png)
        self.assertEqual(png_content2, base64_png)
        self.assertTrue(np.array_equal(png_embedding1, png_embedding2))

        jpeg_content1, jpeg_embedding1 = result1.result[2][0]
        jpeg_content2, jpeg_embedding2 = result2.result[2][0]

        # Content should be original base64, embeddings should be identical (from cache)
        self.assertEqual(jpeg_content1, base64_jpeg)
        self.assertEqual(jpeg_content2, base64_jpeg)
        self.assertTrue(np.array_equal(jpeg_embedding1, jpeg_embedding2))

        # URL results should be unchanged (original URL returned)
        url_content1, url_embedding1 = result1.result[1][0]
        url_content2, url_embedding2 = result2.result[1][0]
        self.assertEqual(url_content1, url_image)  # Original URL unchanged
        self.assertEqual(url_content2, url_image)  # Original URL unchanged
        self.assertTrue(np.array_equal(url_embedding1, url_embedding2))

    def _assert_metric_value(
        self, metric_data: MetricsData, name: str, expected_value: Any
    ):
        cache_metrics = metric_data.resource_metrics[0].scope_metrics[0].metrics
        metric = next((metric for metric in cache_metrics if metric.name == name), None)
        self.assertIsNotNone(metric, f"metric {name} not found")
        self.assertEqual(expected_value, metric.data.data_points[0].value)
