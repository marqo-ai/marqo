import time
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import random
from marqo.api.exceptions import EnvVarError
from marqo.s2_inference.inference_cache.inference_cache import InferenceCache
from queue import Queue


class TestInferenceCache(unittest.TestCase):
    """A test suite for the InferenceCache class outside marqo.s2_inference.s2_inference.vectorise function"""
    def setUp(self):
        self.cache_size = 10

    def test_cache_initializationCacheType_success(self):
        """Test if the cache initializes with the correct size and type."""
        test_cases = [
            {"cache_size": 10, "cache_type": "LRU", "expected": "LRU"},
            {"cache_size": 10, "cache_type": "LFU", "expected": "LFU"}
        ]
        for test_case in test_cases:
            with self.subTest(test_case):
                cache = InferenceCache(cache_size=test_case["cache_size"], cache_type=test_case["cache_type"])
                self.assertEqual(test_case["cache_size"], cache.max_size)
                self.assertEqual(test_case["expected"], cache.cache_type)
                self.assertEqual(0, cache.current_size)

    def test_cache_initializationCacheType_fail(self):
        """Test if the cache raises an error for an invalid cache type."""
        test_cases = [
            {"cache_size": 10, "cache_type": "INVALID"},  # Invalid cache type
            {"cache_size": 10, "cache_type": 1},  # Invalid cache type
            {"cache_size": 1.4, "cache_type": "LFU"},  # Invalid cache size
            {"cache_size": -1, "cache_type": "LRU"}  # Invalid cache size
        ]
        for test_case in test_cases:
            with self.subTest(test_case):
                with self.assertRaises(EnvVarError):
                    InferenceCache(cache_size=test_case["cache_size"], cache_type=test_case["cache_type"])

    def test_cache_setAndGetItems(self):
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                cache = InferenceCache(cache_size=self.cache_size, cache_type=cache_type)
                cache.set("key1", [1.0])
                self.assertTrue("key1" in cache)
                self.assertEqual(cache.get("key1"), [1.0])

    def test_cache_getNoneExistItems(self):
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                cache = InferenceCache(cache_size=self.cache_size, cache_type=cache_type)
                self.assertIsNone(cache.get("nonexistent_key"))
                self.assertEqual(cache.get("nonexistent_key", default=[2.0]), [2.0])

    def test_cache_itemOverRide(self):
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                cache = InferenceCache(cache_size=self.cache_size, cache_type=cache_type)
                cache.set("key1", [1.0])
                cache.set("key1", [2.0])
                self.assertEqual(cache.get("key1"), [2.0])

    def test_cache_evictionPolicy(self):
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                cache = InferenceCache(cache_size=self.cache_size, cache_type=cache_type)
                # Fill up the cache
                for i in range(self.cache_size):
                    cache.set(f"key{i}", [float(i)])

                if cache_type == "LRU":
                    # Access the first key to update its LRU position
                    cache.get("key0")
                    # Adding a new key should evict the least recently used key, which is now "key1"
                    evicted_key = "key1"
                else:  # LFU
                    # Increase the access frequency of all but the last key
                    for i in range(self.cache_size - 1):
                        cache.get(f"key{i}")
                    # Adding a new key should evict the least frequently used key, which is "key9"
                    evicted_key = f"key{self.cache_size - 1}"

                cache.set("key_new", [100.0])
                self.assertTrue("key_new" in cache)
                self.assertFalse(evicted_key in cache, f"{evicted_key} was not evicted under {cache_type} policy")

    def test_cache_concurrentReads(self):
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                cache = InferenceCache(cache_size=self.cache_size, cache_type=cache_type)
                cache.set("key", [1.0])
                # Use ThreadPoolExecutor to simulate concurrent reads
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(lambda: cache.get("key")) for _ in range(10)]
                    results = [future.result() for future in futures]

                # Verify all reads were successful and returned the correct data
                self.assertTrue(all(result == [1.0] for result in results))

    def test_cache_concurrentWrites(self):
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                cache = InferenceCache(cache_size=self.cache_size, cache_type=cache_type)
                # Use ThreadPoolExecutor to simulate concurrent writes
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(lambda i=i: cache.set(f"key_{i}", [float(i)])) for i in range(10)]
                    # Ensure all futures complete
                    for future in futures:
                        future.result()

                # Verify all writes were successful
                for i in range(10):
                    self.assertEqual(cache.get(f"key_{i}"), [float(i)])

    def test_cache_readWriteLock(self):
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                cache = InferenceCache(cache_size=self.cache_size, cache_type=cache_type)
                cache.set("blocking_key", [99.0])
                # Simulate a concurrent read and write to test the read-write lock behavior
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_write = executor.submit(lambda: cache.set("blocking_key", [100.0]))
                    time.sleep(0.1)  # Small delay to ensure write starts first
                    future_read = executor.submit(lambda: cache.get("blocking_key"))

                # Ensure write has completed before read
                write_result = future_write.result()
                read_result = future_read.result()

                self.assertIsNone(write_result)  # set operation returns None
                self.assertEqual(read_result, [100.0], "Read did not return the updated value after write")

    def test_cache_concurrentWritesToSameKey(self):
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                cache = InferenceCache(cache_size=self.cache_size, cache_type=cache_type)
                key = "shared_key"
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(lambda value=i: cache.set(key, [value]),) for i in range(10)]

                    for future in futures:
                        future.result()
                    final_value = cache.get(key)
                    self.assertIn(final_value, [[i] for i in range(10)],
                                  f"Final value {final_value} is not an expected value under {cache_type} policy")

    def test_cache_isEnabled(self):
        """Test if the cache is enabled or disabled based on the cache size."""
        # Test with a positive cache size
        cache = InferenceCache(cache_size=10, cache_type="LRU")
        self.assertTrue(cache.is_enabled(), "Cache should be enabled with a positive cache_size.")

        # Test with cache_size 0, which should disable the cache
        cache = InferenceCache(cache_size=0, cache_type="LRU")
        self.assertFalse(cache.is_enabled(), "Cache should be disabled with cache_size 0.")

    def test_cache_threadSafety(self):
        """Test if the cache is thread-safe by simulating concurrent reads and writes."""
        DATA_DIMENSIONS = 768
        SIZE = 16384
        ITERATIONS = 100_000
        FREQUENT_ACCESS_RATIO = 0.5
        FREQUENT_ACCESS_SUBSET_SIZE = 5000
        TOTAL_QUERY_SET_SIZE = 1_000_000

        hits = Queue()
        misses = Queue()

        def read_write_cache(cache):
            if random.random() < FREQUENT_ACCESS_RATIO:
                text = random.choice(frequent_texts)
            else:
                text = random.choice(texts)
            cache_value = cache.get(text)
            if cache_value is None:
                cache_value = np.random.rand(1, DATA_DIMENSIONS).astype(np.float32).tolist()
                cache.set(text, cache_value)
                misses.put(1)
                return cache_value
            else:
                hits.put(1)
                return cache_value
        texts = [f"text{i} " * 5 for i in range(TOTAL_QUERY_SET_SIZE)]
        frequent_texts = random.sample(texts, FREQUENT_ACCESS_SUBSET_SIZE)
        for cache_type in ['LRU', 'LFU']:
            with self.subTest(cache_type=cache_type):
                test_cache = InferenceCache(cache_size=SIZE, cache_type="LRU")
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(read_write_cache, test_cache) for _ in
                               range(ITERATIONS)]
                    result = [future.result() for future in futures]
                    self.assertEqual(ITERATIONS, len(result))
                    self.assertTrue(hits.qsize() > 0)
                    self.assertTrue(misses.qsize() > 0)