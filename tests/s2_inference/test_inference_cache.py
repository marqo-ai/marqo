import unittest
import threading

from marqo.s2_inference.inference_cache.inference_cache import InferenceCache


class TestInferenceCache(unittest.TestCase):

    def test_cache_initialization(self):
        """Test if the cache initializes with the correct size and type."""
        cache_size = 5
        cache = InferenceCache(cache_size=cache_size, cache_type="LRU")
        self.assertEqual(cache.cache_size, cache_size)
        self.assertIsInstance(cache.cache, cache._CACHE_TYPE_MAPPINGS["LRU"])

    def test_write_and_read_cache(self):
        """Test writing to the cache and reading from it."""
        cache = InferenceCache(cache_size=2, cache_type="LRU")
        test_key = "test_key"
        test_value = [1.0, 2.0, 3.0]
        cache.write_cache(test_key, test_value)
        self.assertEqual(cache.read_cache(test_key), test_value)

    def test_cache_eviction_policy(self):
        """Test the cache's eviction policy (LRU) by exceeding its size."""
        cache = InferenceCache(cache_size=2, cache_type="LRU")
        cache.write_cache("key1", [1.0])
        cache.write_cache("key2", [2.0])
        # This should cause key1 to be evicted (LRU policy)
        cache.write_cache("key3", [3.0])

        self.assertIsNone(cache.read_cache("key1"))
        self.assertIsNotNone(cache.read_cache("key2"))
        self.assertIsNotNone(cache.read_cache("key3"))

    def test_non_existent_key(self):
        """Test reading a key that doesn't exist in the cache."""
        cache = InferenceCache(cache_size=2)
        self.assertIsNone(cache.read_cache("non_existent_key"))
        self.assertEqual(cache.read_cache("non_existent_key", default="default_value"), "default_value")

    def test_lfu_eviction_policy(self):
        """Test the cache's eviction policy (LFU) by managing access frequency."""
        cache = InferenceCache(cache_size=2, cache_type="LFU")
        cache.write_cache("key1", [1.0])
        cache.write_cache("key2", [2.0])

        # Access key1 twice, making key2 the least frequently accessed.
        cache.read_cache("key1")
        cache.read_cache("key1")

        # Adding a new item should evict key2.
        cache.write_cache("key3", [3.0])

        self.assertIsNotNone(cache.read_cache("key1"))
        self.assertIsNone(cache.read_cache("key2"))
        self.assertIsNotNone(cache.read_cache("key3"))