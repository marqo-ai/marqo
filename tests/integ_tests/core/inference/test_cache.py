import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np

from marqo.api.exceptions import EnvVarError
from marqo.inference.inference_cache.marqo_lfu_cache import MarqoLFUCache
from marqo.inference.inference_cache.marqo_lru_cache import MarqoLRUCache


class TestLFUCache(unittest.TestCase):
    """This class tests the LRU and LFU cache implementations."""

    def setUp(self):
        # Instantiate both cache types with a common maxsize for testing
        self.caches = {
            'LFU': MarqoLFUCache(maxsize=2),
            'LRU': MarqoLRUCache(maxsize=2)
        }

    def test_setAndGetItem(self):
        """Test setting an item and then getting it for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set('key1', 'value1')
                self.assertEqual(cache.get('key1'), 'value1', f"Failed in {cache_type} cache.")

    def test_cache_evictionPolicy(self):
        """Test that the correct item is evicted according to the cache's policy."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set('key1', 'value1')
                cache.set('key2', 'value2')
                # Access key1 again to ensure we evict 'key2'
                cache.get('key1')
                cache.set('key3', 'value3')
                self.assertFalse('key2' in cache, f"{cache_type} cache did not evict the least used item.")
                self.assertTrue('key1' in cache and 'key3' in cache,
                                f"{cache_type} cache eviction did not work as expected.")

    def test_cache_Concurrency(self):
        """Test that the cache handles concurrent access correctly for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                import threading

                def set_items():
                    for i in range(10):
                        cache.set(f'key{i}', f'value{i}')

                def get_items():
                    for i in range(10):
                        value = cache.get(f'key{i}', None)

                threads = []
                for _ in range(5):
                    t1 = threading.Thread(target=set_items)
                    t2 = threading.Thread(target=get_items)
                    t1.start()
                    t2.start()
                    threads.extend([t1, t2])

                for t in threads:
                    t.join()

                self.assertEqual(len(cache), 2, f"{cache_type} cache size does not match after concurrent access.")

    def test_cache_length(self):
        """Test the length of the cache for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set('key1', 'value1')
                cache.set('key2', 'value2')
                self.assertEqual(len(cache), 2, f"{cache_type} cache length is incorrect.")

    def test_cache_maxsizeAndCurrsize(self):
        """Test the maxsize and currsize properties for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                self.assertEqual(cache.maxsize, 2, f"{cache_type} Maxsize property incorrect.")
                cache.set('key1', 'value1')
                cache.set('key2', 'value2')
                self.assertEqual(cache.currsize, 2, f"{cache_type} Currsize property incorrect after adding items.")

    def test_cache_clear(self):
        """Test the clear method for both cache types."""
        for cache_type, cache in self.caches.items():
            with self.subTest(cache_type=cache_type):
                cache.set('key1', 'value1')
                cache.set('key2', 'value2')
                cache.clear()
                self.assertEqual(len(cache), 0, f"{cache_type} cache did not clear.")
                self.assertEqual(cache.currsize, 0, f"{cache_type} cache did not clear currsize.")
                self.assertEqual(cache.maxsize, 2, f"{cache_type} cache maxsize changed after clear.")