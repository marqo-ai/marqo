from typing import Hashable, Any

from cachetools import LFUCache
from readerwriterlock import rwlock

from marqo.inference.inference_cache.abstract_cache import MarqoAbstractCache


class MarqoLFUCache(MarqoAbstractCache):
    """A thread-safe Least Frequently Used (LFU) cache implementation with a read-write lock.

    This class is currently implemented using cachetools.LFUCache, but it can be replaced with any other LFU cache.
    """

    def __init__(self, maxsize: int):
        self._cache = LFUCache(maxsize=maxsize)
        self.lock = rwlock.RWLockFair()

    def get(self, key: Hashable, default=None) -> Any:
        with self.lock.gen_rlock():
            return self._cache.get(key, default)

    def set(self, key: Hashable, value: Any):
        """The lock is implemented in the __setitem__ method to avoid double locking when setting a value."""
        self.__setitem__(key, value)

    def __contains__(self, key: Hashable) -> bool:
        with self.lock.gen_rlock():
            return key in self._cache

    def __setitem__(self, key: Hashable, value: Any):
        with self.lock.gen_wlock():
            self._cache[key] = value

    def __getitem__(self, key: Hashable) -> Any:
        with self.lock.gen_rlock():
            return self._cache[key]

    def __len__(self) -> int:
        return len(self._cache)

    def popitem(self) -> None:
        with self.lock.gen_wlock():
            self._cache.popitem()

    def clear(self) -> None:
        with self.lock.gen_wlock():
            self._cache.clear()

    @property
    def maxsize(self) -> int:
        """Return the maximum size of the cache."""
        return int(self._cache.maxsize)

    @property
    def currsize(self) -> int:
        """Return the current size of the cache."""
        return int(self._cache.currsize)