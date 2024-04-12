from typing import List, Optional

from readerwriterlock import rwlock

from marqo.api.exceptions import EnvVarError
from marqo.s2_inference.inference_cache.abstract_cache import MarqoAbstractCache
from marqo.s2_inference.inference_cache.marqo_lfu_cache import MarqoLFUCache
from marqo.s2_inference.inference_cache.marqo_lru_cache import MarqoLRUCache


class InferenceCache:
    """Inference cache class that wraps MarqoAbstractCache and LRUCache with a read-write lock.

    The key is a string consisting of model_cache_key and content to identify the cache.
    The value is a list of floats representing the embeddings.
    """
    _CACHE_TYPE_MAPPINGS = {
        "LFU": MarqoLFUCache,
        "LRU": MarqoLRUCache
    }

    def __init__(self, cache_size: int, cache_type: str = "LRU"):
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.cache = self._build_cache()

        self.lock = rwlock.RWLockFair()

    def _build_cache(self) -> MarqoAbstractCache:
        if self.cache_type not in self._CACHE_TYPE_MAPPINGS:
            raise EnvVarError(f"Invalid cache type: {self.cache_type}. "
                              f"Must be one of {self._CACHE_TYPE_MAPPINGS.keys()}."
                              f"Please set the 'MARQO_INFERENCE_CACHE_TYPE' "
                              f"environment variable to one of the valid cache types.")

        if not isinstance(self.cache_size, int) or self.cache_size < 0:
            raise EnvVarError(f"Invalid cache size: {self.cache_size}. "
                              f"Must be a non-negative integer. "
                              f"Please set the 'MARQO_INFERENCE_CACHE_SIZE' "
                              f"environment variable to a non-negative integer.")

        else:
            return self._CACHE_TYPE_MAPPINGS[self.cache_type](maxsize=self.cache_size)

    def get(self, key: str, default=None) -> Optional[List[float]]:
        with self.lock.gen_rlock():
            return self.cache.get(key, default)

    def set(self, key: str, value: List[float]) -> None:
        with self.lock.gen_wlock():
            self.cache[key] = value

    def __getitem__(self, key: str) -> List[float]:
        with self.lock.gen_rlock():
            return self.cache[key]

    def __setitem__(self, key: str, value: List[float]) -> None:
        with self.lock.gen_wlock():
            self.cache[key] = value

    def __contains__(self, key: str) -> bool:
        with self.lock.gen_rlock():
            return key in self.cache

    def is_enabled(self) -> bool:
        """Return True if the cache is enabled, else False."""
        return self.cache_size > 0

    @property
    def max_size(self) -> int:
        """Return the maximum size of the cache."""
        return self.cache_size

    @property
    def current_size(self) -> int:
        """Return the current size of the cache."""
        return len(self.cache)