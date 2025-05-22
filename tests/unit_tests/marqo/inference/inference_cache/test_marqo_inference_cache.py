import sys
import unittest
from unittest import mock

from marqo.api.exceptions import EnvVarError
from marqo.inference.inference_cache.enums import MarqoCacheType
from marqo.inference.inference_cache.marqo_inference_cache import MarqoInferenceCache
from marqo.inference.inference_cache.marqo_lfu_cache import MarqoLFUCache
from marqo.inference.inference_cache.marqo_lru_cache import MarqoLRUCache


class TestMarqoInferenceCache(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch('marqo.inference.inference_cache.marqo_inference_cache.OTELCacheStatsCollector')
        self.mock_collector_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_collector = self.mock_collector_class.return_value

    # Test constructor
    def test_constructor_error_non_int_size(self):
        with self.assertRaises(EnvVarError):
            MarqoInferenceCache(cache_size="not_int")

    def test_constructor_error_negative_size(self):
        with self.assertRaises(EnvVarError):
            MarqoInferenceCache(cache_size=-1)

    def test_constructor_error_invalid_type(self):
        with self.assertRaises(EnvVarError):
            MarqoInferenceCache(cache_size=1, cache_type="UNKNOWN")

    def test_constructor_zero_cache_disables(self):
        cache = MarqoInferenceCache(cache_size=0)
        self.assertFalse(cache.is_enabled())
        self.assertIsNone(cache._cache)

    def test_constructor_valid_lru_instantiation(self):
        cache = MarqoInferenceCache(cache_size=5, cache_type=MarqoCacheType.LRU)
        self.assertTrue(cache.is_enabled())
        self.assertIsInstance(cache._cache, MarqoLRUCache)

    def test_constructor_valid_lfu_instantiation(self):
        cache = MarqoInferenceCache(cache_size=5, cache_type=MarqoCacheType.LFU)
        self.assertTrue(cache.is_enabled())
        self.assertIsInstance(cache._cache, MarqoLFUCache)

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

    # Test is_enabled
    def test_is_enabled_return_fase_when_disabled(self):
        cache = MarqoInferenceCache(cache_size=0)
        self.assertFalse(cache.is_enabled())

    def test_is_enabled_return_true_when_enabled(self):
        cache = MarqoInferenceCache(cache_size=1)
        self.assertTrue(cache.is_enabled())

    # Test set
    def test_set_does_noop_when_cache_is_disabled(self):
        cache = MarqoInferenceCache(cache_size=0)
        # Should not raise
        cache.set("m", "c", 123)
        self.assertIsNone(cache.get("m", "c"))
        self.mock_collector.record_set.assert_not_called()

    def test_set_records_size_and_duration_when_cache_is_enabled(self):
        cache = MarqoInferenceCache(cache_size=10)
        value = [1, 2, 3]

        with mock.patch('time.perf_counter', side_effect=[1.0, 1.5]):
            cache.set("m", "c", value)
            expected_size = sys.getsizeof(value) + sys.getsizeof("m||c")
            self.mock_collector.record_set.assert_called_once_with(expected_size, 0.5)

        # should be able to get the value back
        self.assertEqual(cache.get("m", "c"), value)

    # Test get
    def test_get_returns_default_value_when_cache_is_disabled(self):
        cache = MarqoInferenceCache(cache_size=0)
        default = object()
        self.assertIs(cache.get("m", "c", default=default), default)

    def test_get_records_miss_and_returns_default(self):
        cache = MarqoInferenceCache(cache_size=10)
        default = object()
        with mock.patch('time.perf_counter', side_effect=[1.0, 1.5]):
            result = cache.get("m", "c", default=default)
            self.assertIs(result, default)
            self.mock_collector.record_get.assert_called_once_with(False, 0.5)

    def test_get_records_hit_and_returns_value(self):
        cache = MarqoInferenceCache(cache_size=10)
        value = object()
        cache.set("m", "c", value)

        default = object()
        with mock.patch('time.perf_counter', side_effect=[1.0, 1.5]):
            result = cache.get("m", "c", default=default)
            self.assertIs(result, value)
            self.mock_collector.record_get.assert_called_once_with(True, 0.5)

    # Test clear
    def test_clear_cache(self):
        cache = MarqoInferenceCache(cache_size=1)
        cache.set("m", "c", 789)
        cache.clear()
        self.assertIsNone(cache.get("m", "c"))

    def test_clear_does_noop_when_cache_is_disabled(self):
        cache = MarqoInferenceCache(cache_size=0)
        # Should not raise
        cache.clear()

    # Test eviction strategy
    def test_lru_eviction(self):
        cache = MarqoInferenceCache(cache_size=1, cache_type=MarqoCacheType.LRU)
        cache.set("m", "a", 1)
        cache.set("m", "b", 2)
        self.assertIsNone(cache.get("m", "a"))
        self.assertEqual(cache.get("m", "b"), 2)

    def test_lfu_eviction(self):
        cache = MarqoInferenceCache(cache_size=1, cache_type=MarqoCacheType.LFU)
        cache.set("m", "a", 1)
        cache.get("m", "a")  # bump freq
        cache.set("m", "b", 2)
        remaining = cache.get("m", "a") or cache.get("m", "b")
        self.assertIn(remaining, (1, 2))

