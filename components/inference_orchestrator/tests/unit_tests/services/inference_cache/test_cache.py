import unittest

from inference_orchestrator.services.inference_cache.marqo_lfu_cache import (
    MarqoLFUCache,
)
from inference_orchestrator.services.inference_cache.marqo_lru_cache import (
    MarqoLRUCache,
)


class TestCache(unittest.TestCase):
    """This class tests the LRU and LFU cache implementations."""

    def setUp(self):
        # Instantiate both cache types with a common maxsize for testing
        self.caches = {"LFU": MarqoLFUCache(maxsize=2), "LRU": MarqoLRUCache(maxsize=2)}

    def test_setAndGetItem(self):
        """Test setting an item and then getting it for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set("key1", "value1")
                self.assertEqual(
                    cache.get("key1"), "value1", f"Failed in {cache_type} cache."
                )

    def test_cache_evictionPolicy(self):
        """Test that the correct item is evicted according to the cache's policy."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set("key1", "value1")
                cache.set("key2", "value2")
                # Access key1 again to ensure we evict 'key2'
                cache.get("key1")
                cache.set("key3", "value3")
                self.assertFalse(
                    "key2" in cache,
                    f"{cache_type} cache did not evict the least used item.",
                )
                self.assertTrue(
                    "key1" in cache and "key3" in cache,
                    f"{cache_type} cache eviction did not work as expected.",
                )

    def test_cache_Concurrency(self):
        """Test that the cache handles concurrent access correctly for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                import threading

                def set_items():
                    for i in range(10):
                        cache.set(f"key{i}", f"value{i}")

                def get_items():
                    for i in range(10):
                        _ = cache.get(f"key{i}", None)

                threads = []
                for _ in range(5):
                    t1 = threading.Thread(target=set_items)
                    t2 = threading.Thread(target=get_items)
                    t1.start()
                    t2.start()
                    threads.extend([t1, t2])

                for t in threads:
                    t.join()

                self.assertEqual(
                    len(cache),
                    2,
                    f"{cache_type} cache size does not match after concurrent access.",
                )

    def test_cache_length(self):
        """Test the length of the cache for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set("key1", "value1")
                cache.set("key2", "value2")
                self.assertEqual(
                    len(cache), 2, f"{cache_type} cache length is incorrect."
                )

    def test_cache_maxsizeAndCurrsize(self):
        """Test the maxsize and currsize properties for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                self.assertEqual(
                    cache.maxsize, 2, f"{cache_type} Maxsize property incorrect."
                )
                cache.set("key1", "value1")
                cache.set("key2", "value2")
                self.assertEqual(
                    cache.currsize,
                    2,
                    f"{cache_type} Currsize property incorrect after adding items.",
                )

    def test_cache_getitem_operator(self):
        """Test the __getitem__ operator for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set("key1", "value1")
                # Use the [] operator to access item
                result = cache["key1"]
                self.assertEqual("value1", result, f"{cache_type} __getitem__ failed.")

    def test_cache_getitem_operator_missing_key_raises_error(self):
        """Test that __getitem__ raises KeyError for missing keys."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                with self.assertRaises(KeyError):
                    _ = cache["nonexistent_key"]

    def test_cache_popitem(self):
        """Test the popitem method for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set("key1", "value1")
                cache.set("key2", "value2")
                self.assertEqual(2, len(cache))

                # Remove one item using popitem
                cache.popitem()
                self.assertEqual(
                    1, len(cache), f"{cache_type} popitem did not remove an item."
                )

    def test_cache_clear(self):
        """Test the clear method for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set("key1", "value1")
                cache.set("key2", "value2")
                cache.clear()
                self.assertEqual(len(cache), 0, f"{cache_type} cache did not clear.")
                self.assertEqual(
                    cache.currsize, 0, f"{cache_type} cache did not clear currsize."
                )
                self.assertEqual(
                    cache.maxsize, 2, f"{cache_type} cache maxsize changed after clear."
                )
