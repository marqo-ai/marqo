import sys
import threading
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, TypeVar, Dict, Callable

from marqo import logging
from marqo.api.exceptions import EnvVarError
from marqo.inference.inference_cache.abstract_cache import MarqoAbstractCache
from marqo.inference.inference_cache.enums import MarqoCacheType
from marqo.inference.inference_cache.marqo_lfu_cache import MarqoLFUCache
from marqo.inference.inference_cache.marqo_lru_cache import MarqoLRUCache

T = TypeVar("T")
logger = logging.get_logger(__name__)


class CacheStatsCollector(ABC):
    """Abstract interface for collecting cache metrics."""

    @abstractmethod
    def record_hit(self) -> None:
        ...

    @abstractmethod
    def record_miss(self) -> None:
        ...

    @abstractmethod
    def record_insert(self, size_bytes: int = 1) -> None:
        """Called when an item is inserted; `size_bytes` is optional."""
        ...

    @abstractmethod
    def snapshot_total(self) -> Dict[str, float]:
        """Return cumulative metrics including size."""
        ...

    @abstractmethod
    def snapshot_interval(self) -> Dict[str, float]:
        """Return interval metrics including size."""
        ...


class RawCacheStatsCollector(CacheStatsCollector):
    """
    A simple in‐process collector that keeps cumulative counters
    plus a background logger every `interval` seconds, reporting both
    total and interval stats.
    """

    def __init__(self, interval: float = 10.0,
                 currsize_fn: Optional[Callable[[], int]] = None,
                 maxsize_fn: Optional[Callable[[], int]] = None,
                 ):
        self._hits = 0
        self._misses = 0
        self._inserts = 0
        self._insert_bytes = 0
        self._lock = threading.Lock()
        self._interval = interval
        self._currsize_fn = currsize_fn
        self._maxsize_fn = maxsize_fn

        # For computing interval deltas:
        self._last_hits = 0
        self._last_misses = 0
        self._last_inserts = 0
        self._last_insert_bytes = 0

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._log_loop, daemon=True)
        self._thread.start()

    def record_hit(self) -> None:
        with self._lock:
            self._hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self._misses += 1

    def record_insert(self, size_bytes: int = 1) -> None:
        with self._lock:
            self._inserts += 1
            self._insert_bytes += size_bytes

    def snapshot_total(self) -> Dict[str, float]:
        with self._lock:
            hits = self._hits
            misses = self._misses
            inserts = self._inserts
            insert_bytes = self._insert_bytes

        total_ops = hits + misses
        hit_rate = hits / total_ops if total_ops > 0 else 0.0

        stats = {
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "inserts": inserts,
            "insert_bytes": insert_bytes,
        }

        if self._currsize_fn:
            stats["currsize"] = self._currsize_fn()
        if self._maxsize_fn:
            stats["maxsize"] = self._maxsize_fn()

        return stats

    def snapshot_interval(self) -> Dict[str, float]:
        with self._lock:
            h = self._hits - self._last_hits
            m = self._misses - self._last_misses
            ins = self._inserts - self._last_inserts
            in_bytes = self._insert_bytes - self._last_insert_bytes

            # update last counters
            self._last_hits = self._hits
            self._last_misses = self._misses
            self._last_inserts = self._inserts
            self._last_insert_bytes = self._insert_bytes

        total = h + m
        hit_rate = h / total if total > 0 else 0.0
        stats = {
            "hits": h,
            "misses": m,
            "hit_rate": hit_rate,
            "inserts": ins,
            "insert_bytes": in_bytes,
        }

        if self._currsize_fn:
            stats["currsize"] = self._currsize_fn()
        if self._maxsize_fn:
            stats["maxsize"] = self._maxsize_fn()

        return stats

    def _log_loop(self):
        while not self._stop.wait(self._interval):
            total = self.snapshot_total()
            interval = self.snapshot_interval()
            logger.info(
                "Cache total stats: hits=%d misses=%d hit_rate=%.2f inserts=%d insert_bytes=%d currsize=%s maxsize=%s",
                total["hits"], total["misses"], total["hit_rate"],
                total["inserts"], total["insert_bytes"], total.get("currsize"), total.get("maxsize"),
            )
            logger.info(
                "Cache interval stats (last %.1fs): hits=%d misses=%d hit_rate=%.2f inserts=%d insert_bytes=%d currsize=%s maxsize=%s",
                self._interval,
                interval["hits"], interval["misses"], interval["hit_rate"],
                interval["inserts"], interval["insert_bytes"], interval.get("currsize"), interval.get("maxsize"),
            )

    def shutdown(self):
        self._stop.set()
        self._thread.join()


class MarqoInferenceCache:
    """MarqoInferenceCache is a thread-safe cache implementation for storing embeddings.

    The key is a string consisting of model_cache_key and content to identify the cache.
    The value is a list of floats representing the embeddings.
    """

    _CACHE_TYPES_MAPPING = {
        MarqoCacheType.LRU: MarqoLRUCache,
        MarqoCacheType.LFU: MarqoLFUCache,
    }

    def __init__(self, cache_size: int = 0, cache_type: Union[None, str, MarqoCacheType] = MarqoCacheType.LRU,
                 stats_collector: Optional[CacheStatsCollector] = None,
                 value_size_lambda: Callable[[T], int] = lambda v: sys.getsizeof(v)):

        self._cache = self._build_cache(cache_size, cache_type)

        if self.is_enabled():
            self._stats = stats_collector or RawCacheStatsCollector(
                currsize_fn=lambda: self._cache.currsize,
                maxsize_fn=lambda: self._cache.maxsize,
            )
            self._value_size_lambda = value_size_lambda

    def _build_cache(self, cache_size: int, cache_type: MarqoCacheType) -> Optional[MarqoAbstractCache]:
        """Return a cache instance based on the cache type and size.

        Args:
            cache_size: The maximum size of the cache.
            cache_type: The type of the cache.

        Returns:
            A cache instance based on the cache type and size. None if the cache_size is 0.

        Raises:
            EnvVarError: If the cache size or type is invalid.
        """
        if not isinstance(cache_size, int) or cache_size < 0:
            raise EnvVarError(f"Invalid cache size: {cache_size}. "
                              f"Must be a non-negative integer. ")
        elif cache_size == 0:
            return None

        if cache_type not in self._CACHE_TYPES_MAPPING:
            raise EnvVarError(f"Invalid cache type: {cache_type}. "
                              f"Must be one of {self._CACHE_TYPES_MAPPING.keys()}.")
        return self._CACHE_TYPES_MAPPING[cache_type](maxsize=cache_size)

    def get(self, model_cache_key: str, content: str, default=None) -> Optional[T]:
        if not self.is_enabled():
            return default

        key = self._generate_key(model_cache_key, content)
        cache = self._cache

        if key in cache:
            self._stats.record_hit()
            return cache[key]
        else:
            self._stats.record_miss()
            return default

    def set(self, model_cache_key: str, content: str, value: T) -> None:
        if not self.is_enabled():
            return

        key = self._generate_key(model_cache_key, content)
        self._cache[key] = value

        # estimate size in bytes (shallow)
        size = len(key) + self._value_size_lambda(value)
        self._stats.record_insert(size)

    # def __getitem__(self, model_cache_key: str, content: str, key: str) -> T:
    #     key = self._generate_key(model_cache_key, content)
    #     return self._cache[key]
    #
    # def __setitem__(self, model_cache_key: str, content: str, value: T) -> None:
    #     key = self._generate_key(model_cache_key, content)
    #     self._cache[key] = value

    def __contains__(self, item: Tuple) -> bool:
        if len(item) != 2:
            raise ValueError("MarqoInferenceCache received an unsupported input for 'in' operation. "
                             "Expected input is a tuple with 'model-cache-key' and 'content'. "
                             "E.g., ('my-model-cache-key', 'content'). ")
        model_cache_key, content = item
        key = self._generate_key(model_cache_key, content)
        return key in self._cache

    def _generate_key(self, model_cache_key: str, content: str) -> str:
        if not isinstance(model_cache_key, str):
            raise TypeError(f"model_cache_key must be a string, not {type(model_cache_key)}")
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, not {type(content)}")
        return f"{model_cache_key}||{content}"

    def clear(self) -> None:
        """Clear the cache."""
        if self._cache is not None:
            self._cache.clear()

    def is_enabled(self) -> bool:
        """Return True if the cache is enabled, else False."""
        return self._cache is not None

    def shutdown(self) -> None:
        """Clean up background threads in the collector."""
        if self._stats and hasattr(self._stats, "shutdown"):
            self._stats.shutdown()
