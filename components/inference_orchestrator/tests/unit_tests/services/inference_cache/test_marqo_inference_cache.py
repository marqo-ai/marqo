import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from inference_orchestrator.core.enum import MarqoCacheType
from inference_orchestrator.services.inference_cache.marqo_inference_cache import (
    MarqoInferenceCache,
)


class TestMarqoInferenceCache(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "inference_orchestrator.services.inference_cache.marqo_inference_cache.OTELCacheStatsCollector"
        )
        self.mock_collector_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_collector = self.mock_collector_class.return_value

    def test_cache_initializationCacheType_success(self):
        """Test if the cache initializes with the correct size and type."""
        test_cases = [
            {"cache_size": 10, "cache_type": "LRU", "expected": "LRU"},
            {"cache_size": 10, "cache_type": "LFU", "expected": "LFU"},
        ]
        for test_case in test_cases:
            with self.subTest(test_case):
                cache = MarqoInferenceCache(
                    cache_size=test_case["cache_size"],
                    cache_type=test_case["cache_type"],
                )
                self.assertEqual(test_case["cache_size"], cache._cache.maxsize)
                self.assertTrue(
                    isinstance(
                        cache._cache,
                        MarqoInferenceCache._CACHE_TYPES_MAPPING[test_case["expected"]],
                    )
                )
                self.assertEqual(0, cache._cache.currsize)

    # Test generate keys
    def test_generate_valid_key(self):
        cache = MarqoInferenceCache(cache_size=1)
        key = cache._generate_key("model1", "content1")
        self.assertEqual(key, "model1||content1")

    def test_generate_key_raises_error_when_model_key_is_not_str(self):
        cache = MarqoInferenceCache(cache_size=1)
        with self.assertRaises(TypeError):
            cache._generate_key(123, "content")

    def test_generate_key_raises_error_when_content_key_is_not_str(self):
        cache = MarqoInferenceCache(cache_size=1)
        with self.assertRaises(TypeError):
            cache._generate_key("model", 456)

    # Test set
    def test_set_records_size_and_duration_when_cache_is_enabled(self):
        cache = MarqoInferenceCache(cache_size=10)
        value = [1, 2, 3]

        with mock.patch("time.perf_counter", side_effect=[1.0, 1.5]):
            cache.set("m", "c", value)
            expected_size = sys.getsizeof(value) + sys.getsizeof("m||c")
            self.mock_collector.record_set.assert_called_once_with(expected_size, 0.5)

        # should be able to get the value back
        self.assertEqual(cache.get("m", "c"), value)

    # Test get
    def test_get_records_miss_and_returns_default(self):
        cache = MarqoInferenceCache(cache_size=10)
        default = object()
        with mock.patch("time.perf_counter", side_effect=[1.0, 1.5]):
            result = cache.get("m", "c", default=default)
            self.assertIs(result, default)
            self.mock_collector.record_get.assert_called_once_with(False, 0.5)

    def test_get_records_hit_and_returns_value(self):
        cache = MarqoInferenceCache(cache_size=10)
        value = object()
        cache.set("m", "c", value)

        default = object()
        with mock.patch("time.perf_counter", side_effect=[1.0, 1.5]):
            result = cache.get("m", "c", default=default)
            self.assertIs(result, value)
            self.mock_collector.record_get.assert_called_once_with(True, 0.5)

    # Test clear
    def test_clear_cache(self):
        cache = MarqoInferenceCache(cache_size=1)
        cache.set("m", "c", 789)
        cache.clear()
        self.assertIsNone(cache.get("m", "c"))

    # Test eviction strategy
    def test_lru_eviction(self):
        cache = MarqoInferenceCache(cache_size=2, cache_type=MarqoCacheType.LRU)
        cache.set("m", "a", 1)
        cache.set("m", "b", 2)
        cache.set("m", "c", 3)

        # a is evicted, b still remains
        self.assertIsNone(cache.get("m", "a"))
        self.assertEqual(cache.get("m", "b"), 2)
        self.assertEqual(cache.get("m", "c"), 3)

    def test_lfu_eviction(self):
        cache = MarqoInferenceCache(cache_size=2, cache_type=MarqoCacheType.LFU)
        cache.set("m", "a", 1)
        cache.get("m", "a")  # bump freq of a to 2
        cache.set("m", "b", 2)
        cache.set("m", "c", 3)

        # b is evicted, a still remains
        self.assertIsNone(cache.get("m", "b"))
        self.assertEqual(cache.get("m", "a"), 1)
        self.assertEqual(cache.get("m", "c"), 3)

    # Test concurrent read/write
    def test_cache_concurrent_reads(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                cache = MarqoInferenceCache(cache_size=10, cache_type=cache_type)
                cache.set("test-model-cache-key", "test-content", [1.0])
                # Use ThreadPoolExecutor to simulate concurrent reads
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(
                            lambda: cache.get("test-model-cache-key", "test-content")
                        )
                        for _ in range(10)
                    ]
                    results = [future.result() for future in futures]

                # Verify all reads were successful and returned the correct data
                self.assertTrue(all(result == [1.0] for result in results))

    def test_cache_concurrent_writes(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                cache = MarqoInferenceCache(cache_size=10, cache_type=cache_type)
                # Use ThreadPoolExecutor to simulate concurrent writes
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(
                            lambda i=i: cache.set(
                                "test-model-cache-key", f"test-content-{i}", [float(i)]
                            )
                        )
                        for i in range(10)
                    ]
                    # Ensure all futures complete
                    for future in futures:
                        future.result()

                # Verify all writes were successful
                for i in range(10):
                    self.assertEqual(
                        cache.get("test-model-cache-key", f"test-content-{i}"),
                        [float(i)],
                    )

    def test_cache_read_write_lock(self):
        """Test that read-write lock ensures consistency between concurrent reads and writes."""
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                cache = MarqoInferenceCache(cache_size=10, cache_type=cache_type)
                cache.set("block-key", "block-content", [99.0])
                # Simulate a concurrent read and write to test the read-write lock behavior
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_write = executor.submit(
                        lambda: cache.set("block-key", "block-content", [100.0])
                    )
                    future_read = executor.submit(
                        lambda: cache.get("block-key", "block-content")
                    )

                # Ensure both operations complete
                write_result = future_write.result()
                read_result = future_read.result()

                self.assertIsNone(write_result)  # set operation returns None
                # Read should return either old or new value depending on execution order
                self.assertIn(
                    read_result, [[99.0], [100.0]], "Read did not return a valid value"
                )

    def test_cache_concurrent_writes_to_same_key(self):
        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                cache = MarqoInferenceCache(cache_size=10, cache_type=cache_type)
                model_cache_key = "shared-model-cache-key"
                content = "shared-content"
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(
                            lambda value=i: cache.set(model_cache_key, content, [value])
                        )
                        for i in range(10)
                    ]

                    for future in futures:
                        future.result()
                    final_value = cache.get(model_cache_key, content)
                    self.assertIn(
                        final_value,
                        [[i] for i in range(10)],
                        f"Final value {final_value} is not an expected value under {cache_type} policy",
                    )

    def test_cache_size_increases_linearly_with_items(self):
        def get_cache_size(cache):
            # please note this is just a rough estimation. other data structures in the cache might take extra spaces
            return sum(
                [
                    sys.getsizeof(key) + sys.getsizeof(value)
                    for key, value in cache.items()
                ]
            )

        for cache_type in ["LRU", "LFU"]:
            with self.subTest(cache_type=cache_type):
                cache = MarqoInferenceCache(cache_size=100, cache_type=cache_type)
                cache.set("test-model-cache-key", "query 000", [100.0] * 768)

                size = get_cache_size(cache._cache._cache)

                # verify the cache size increases linearly with the item count
                for i in range(1, 100):
                    cache.set("test-model-cache-key", f"query {i:3d}", [100.0] * 768)
                    self.assertEqual(
                        get_cache_size(cache._cache._cache),
                        size * (i + 1),
                        f"size does not increase linearly on item {i + 1}",
                    )

                # Verify the cache size stops increasing after the cache is full
                for i in range(100, 120):
                    cache.set("test-model-cache-key", f"query {i:3d}", [100.0] * 768)
                    self.assertEqual(
                        get_cache_size(cache._cache._cache),
                        size * 100,
                        f"size keeps increasing after cache is full on item {i + 1}",
                    )
