from typing import List, Optional
from cachetools import LRUCache, LFUCache
from readerwriterlock import rwlock


class InferenceCache:
    _CACHE_TYPE_MAPPINGS = {
        "LFU": LFUCache,
        "LRU": LRUCache
    }

    def __init__(self, cache_size: int, cache_type: str = "LRU"):
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.cache = self._build_cache()

        self.lock = rwlock.RWLockFair()

    def _build_cache(self):
        if self.cache_type not in self._CACHE_TYPE_MAPPINGS:
            raise ValueError(f"Invalid cache type: {self.cache_type}")
        else:
            return self._CACHE_TYPE_MAPPINGS[self.cache_type](maxsize=self.cache_size)

    def get(self, key: str, default=None) -> Optional[List[List[float]]]:
        with self.lock.gen_rlock():
            return self.cache.get(key, default)

    def __getitem__(self, key: str) -> List[float]:
        with self.lock.gen_rlock():
            return self.cache[key]

    def __setitem__(self, key: str, value: List[List[float]]) -> None:
        with self.lock.gen_wlock():
            self.cache[key] = value

    def __contains__(self, key: str) -> bool:
        with self.lock.gen_rlock():
            return key in self.cache

    def is_enabled(self) -> bool:
        return self.cache_size > 0

    @property
    def currsize(self) -> int:
        return self.cache.currsize

    @property
    def maxsize(self) -> int:
        return self.cache.maxsize

