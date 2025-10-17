import sys
import time
from typing import Optional, TypeVar, Union

from inference_orchestrator.core.enum import MarqoCacheType
from inference_orchestrator.core.logging import get_logger
from inference_orchestrator.core.settings import get_settings
from inference_orchestrator.services.inference_cache.abstract_cache import (
    MarqoAbstractCache,
)
from inference_orchestrator.services.inference_cache.marqo_lfu_cache import (
    MarqoLFUCache,
)
from inference_orchestrator.services.inference_cache.marqo_lru_cache import (
    MarqoLRUCache,
)
from inference_orchestrator.services.inference_cache.monitoring import (
    CacheStatsCollector,
    OTELCacheStatsCollector,
)

T = TypeVar("T")
logger = get_logger(__name__)
settings = get_settings()


class MarqoInferenceCache:
    """MarqoInferenceCache is a thread-safe cache implementation for storing embeddings.

    The key is a string consisting of model_cache_key and content to identify the cache.
    The value is a list of floats representing the embeddings.
    """

    _CACHE_TYPES_MAPPING = {
        MarqoCacheType.LRU: MarqoLRUCache,
        MarqoCacheType.LFU: MarqoLFUCache,
    }

    def __init__(
        self,
        cache_size: int,
        cache_type: Union[None, str, MarqoCacheType] = MarqoCacheType.LRU,
    ):
        self._cache = self._build_cache(cache_size, cache_type)
        self._stats: CacheStatsCollector = OTELCacheStatsCollector(
            curr_size_fn=lambda: self._cache.currsize,
            max_size_fn=lambda: self._cache.maxsize,
        )

    def _build_cache(
        self, cache_size: int, cache_type: MarqoCacheType
    ) -> Optional[MarqoAbstractCache]:
        """Return a cache instance based on the cache type and size.

        Args:
            cache_size: The maximum size of the cache.
            cache_type: The type of the cache.

        Returns:
            A cache instance based on the cache type and size. None if the cache_size is 0.

        Raises:
            EnvironmentVariableParsingError: If the cache size or type is invalid.
        """

        cache = self._CACHE_TYPES_MAPPING[cache_type](maxsize=cache_size)
        logger.info(
            f"Built inference cache with type {cache_type} and size {cache_size}"
        )
        return cache

    def get(self, model_cache_key: str, content: str, default=None) -> Optional[T]:
        key = self._generate_key(model_cache_key, content)
        cache = self._cache

        now = time.perf_counter()
        value = cache.get(key)
        elapsed = time.perf_counter() - now

        if value is None:
            self._stats.record_get(False, elapsed)
            return default
        else:
            self._stats.record_get(True, elapsed)
            return value

    def set(self, model_cache_key: str, content: str, value: T) -> None:
        key = self._generate_key(model_cache_key, content)

        now = time.perf_counter()
        self._cache[key] = value
        elapsed = time.perf_counter() - now

        item_size = sys.getsizeof(value) + sys.getsizeof(key)
        self._stats.record_set(item_size, elapsed)

    def _generate_key(self, model_cache_key: str, content: str) -> str:
        if not isinstance(model_cache_key, str):
            raise TypeError(
                f"model_cache_key must be a string, not {type(model_cache_key)}"
            )
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, not {type(content)}")
        return f"{model_cache_key}||{content}"

    def clear(self) -> None:
        """Clear the cache."""
        if self._cache is not None:
            self._cache.clear()
